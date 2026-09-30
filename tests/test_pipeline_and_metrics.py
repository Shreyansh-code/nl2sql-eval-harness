"""Cache keying, the scoring pipeline, and the statistics.

The kappa and bootstrap tests are here because a wrong denominator or a wrong interval
is the easiest way to publish a number that does not survive review.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from harness.config import Settings
from harness.generation.cache import GenerationCache, cache_key
from harness.generation.port import Generation, GenerationStatus
from harness.metrics.report import bootstrap_interval, build_report, render_markdown
from harness.metrics.verify import compare_to_report, recompute_from_matches
from harness.pipeline import score_generations
from harness.runs import open_run


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        generator_model="gpt-6-luna",
        judge_model="gpt-6-luna",
        generator_temperature=None,
        generator_mode="standard",
        judge_mode="batch",
        openai_base_url=None,
        openai_api_key="sk-test",
        data_dir=tmp_path,
        subset_path=tmp_path / "subset.yaml",
        max_concurrency=2,
        sql_timeout_seconds=5.0,
        max_rows=1000,
        label_sample_size=50,
        langsmith_tracing=False,
    )


def _generation(question, sql: str | None, status=GenerationStatus.OK) -> Generation:
    return Generation(
        question_id=question.question_id,
        db_id=question.db_id,
        status=status,
        sql=sql,
        raw_response=f"```sql\n{sql}\n```" if sql else "",
        model="gpt-6-luna",
        prompt_version="v1",
        cache_key="k",
    )


class TestCacheKey:
    def _key(self, **overrides) -> str:
        base = {
            "model": "gpt-6-luna",
            "prompt_version": "v1",
            "question_id": "q0",
            "schema_ddl": "CREATE TABLE t (a)",
            "question": "how many?",
            "evidence": "a means b",
        }
        return cache_key(**{**base, **overrides})

    def test_same_inputs_give_the_same_key(self) -> None:
        assert self._key() == self._key()

    @pytest.mark.parametrize(
        "field",
        ["model", "prompt_version", "question_id", "schema_ddl", "question", "evidence"],
    )
    def test_every_input_participates_in_the_key(self, field: str) -> None:
        assert self._key(**{field: "changed"}) != self._key()

    def test_key_is_short_and_hex(self) -> None:
        key = self._key()
        assert len(key) == 32
        int(key, 16)


class TestGenerationCache:
    def test_round_trips_through_a_run_file(self, tmp_path: Path, question) -> None:
        run = open_run(_settings(tmp_path), "test", root=tmp_path / "runs")
        generation = _generation(question, "SELECT 1")
        run.append_rows("generations.jsonl", [generation.to_row()])

        cache = GenerationCache(run.path / "generations.jsonl")
        assert len(cache) == 1
        assert cache.get("k") is not None
        assert cache.get("missing") is None

    def test_write_is_append_only_so_a_resumed_run_keeps_both(
        self, tmp_path: Path, question
    ) -> None:
        run = open_run(_settings(tmp_path), "test", root=tmp_path / "runs")
        first = _generation(question, "SELECT 1")
        second = Generation(**{**first.__dict__, "sql": "SELECT 2"})
        run.append_rows("generations.jsonl", [first.to_row()])
        run.append_rows("generations.jsonl", [second.to_row()])

        rows = run.read_rows("generations.jsonl")
        assert len(rows) == 2
        assert [r["sql"] for r in rows] == ["SELECT 1", "SELECT 2"]

    def test_generation_row_survives_json(self, question) -> None:
        restored = Generation.from_row(
            json.loads(json.dumps(_generation(question, "SELECT 1").to_row()))
        )
        assert restored == _generation(question, "SELECT 1")


class TestScoringPipeline:
    def test_correct_query_is_scored_correct(self, loaded_questions, tmp_path: Path) -> None:
        question = loaded_questions[0]
        good = "SELECT count(*) FROM player WHERE position = 'M'"
        scored = score_generations([question], [_generation(question, good)], _settings(tmp_path))
        assert scored[0].scored
        assert scored[0].correct
        assert scored[0].skip_reason is None

    def test_wrong_query_is_scored_incorrect(self, loaded_questions, tmp_path: Path) -> None:
        question = loaded_questions[0]
        scored = score_generations(
            [question], [_generation(question, "SELECT count(*) FROM player")], _settings(tmp_path)
        )
        assert scored[0].scored
        assert not scored[0].correct

    def test_unparseable_generation_counts_as_a_failure_not_a_skip(
        self, loaded_questions, tmp_path: Path
    ) -> None:
        """A model that returns prose has failed. It must not leave the denominator."""
        question = loaded_questions[0]
        generation = _generation(question, None, GenerationStatus.UNPARSEABLE)
        scored = score_generations([question], [generation], _settings(tmp_path))
        assert scored[0].scored is True
        assert scored[0].correct is False
        assert "unparseable" in (scored[0].execution.detail or "")

    def test_broken_gold_query_excludes_the_question(self, db_path, tmp_path: Path) -> None:
        from dataclasses import replace

        from harness.dataset.models import HarnessQuestion

        broken = HarnessQuestion(
            question_id="q0",
            db_id="league",
            question="?",
            evidence="",
            gold_sql="SELECT FROM WHERE",
            difficulty="easy",
            db_path=db_path,
            schema=replace(
                __import__("harness.dataset.models", fromlist=["x"]).SchemaPayload(
                    db_id="league", tables=("player",), ddl="CREATE TABLE player (name TEXT)"
                ),
                column_values={},
            ),
        )
        scored = score_generations(
            [broken], [_generation(broken, "SELECT name FROM player")], _settings(tmp_path)
        )
        assert scored[0].scored is False
        assert scored[0].skip_reason == "gold_did_not_execute"


class TestBootstrap:
    def test_all_correct_gives_a_degenerate_interval(self) -> None:
        interval = bootstrap_interval([True] * 10, resamples=200)
        assert interval is not None
        assert interval.point == 1.0
        assert interval.low == interval.high == 1.0

    def test_empty_input_returns_none(self) -> None:
        assert bootstrap_interval([]) is None

    def test_interval_brackets_the_point_estimate(self) -> None:
        flags = [True] * 7 + [False] * 3
        interval = bootstrap_interval(flags, resamples=500)
        assert interval is not None
        assert interval.low <= interval.point <= interval.high

    def test_wider_sample_narrows_the_interval(self) -> None:
        flags = [True, False] * 25
        small = bootstrap_interval(flags[:20], resamples=500)
        large = bootstrap_interval(flags, resamples=500)
        assert small is not None and large is not None
        assert (large.high - large.low) < (small.high - small.low)

    def test_is_deterministic(self) -> None:
        flags = [True, False, True, True, False, False]
        first = bootstrap_interval(flags, resamples=300)
        second = bootstrap_interval(flags, resamples=300)
        assert (first.low, first.high) == (second.low, second.high)

    def test_percentage_rendering_carries_the_interval(self) -> None:
        interval = bootstrap_interval([True] * 8 + [False] * 2, resamples=200)
        assert interval is not None
        text = interval.as_percent()
        assert "95% CI" in text
        assert math.isclose(float(text.split("%")[0]), 80.0, abs_tol=0.05)


class TestReport:
    def test_report_counts_the_denominator_correctly(
        self, loaded_questions, tmp_path: Path
    ) -> None:
        questions = loaded_questions
        generations = [
            _generation(questions[0], "SELECT count(*) FROM player WHERE position = 'M'"),
            _generation(questions[1], "SELECT tournament FROM match"),
        ]
        scored = score_generations(questions, generations, _settings(tmp_path))
        report = build_report(scored)

        assert report.n_total == 2
        assert report.n_scored == 2
        assert report.accuracy is not None
        assert report.accuracy.point == 0.5
        assert set(report.per_db) == {"league"}

    def test_report_serializes_for_the_readme(self, loaded_questions, tmp_path: Path) -> None:
        scored = score_generations(
            loaded_questions,
            [_generation(loaded_questions[0], "SELECT count(*) FROM player WHERE position = 'M'")],
            _settings(tmp_path),
        )
        payload = build_report(scored).to_dict()
        assert json.loads(json.dumps(payload))["n_scored"] == 1
        assert payload["exec_accuracy"]["point"] == 1.0

    def test_markdown_states_n_and_says_what_is_not_yet_measured(
        self, loaded_questions, tmp_path: Path
    ) -> None:
        scored = score_generations(
            loaded_questions[:1],
            [_generation(loaded_questions[0], "SELECT count(*) FROM player WHERE position = 'M'")],
            _settings(tmp_path),
        )
        report = build_report(scored)
        text = render_markdown(report, {"run_id": "r", "git_sha": "abc", "config": {}})
        assert "n=1" in text
        assert "Not yet measured" in text
        assert "No diagnostic claim" in text


class TestRunValidity:
    """A transport failure must never be reportable as model accuracy."""

    def _api_error_generation(self, question) -> Generation:
        return Generation(
            question_id=question.question_id,
            db_id=question.db_id,
            status=GenerationStatus.API_ERROR,
            sql=None,
            raw_response="",
            model="gpt-6-luna",
            prompt_version="v1",
            cache_key="k",
            error="http_400: temperature unsupported",
        )

    def test_api_errors_invalidate_the_run(self, loaded_questions, tmp_path: Path) -> None:
        generations = [self._api_error_generation(q) for q in loaded_questions]
        report = build_report(score_generations(loaded_questions, generations, _settings(tmp_path)))
        assert report.api_errors == len(loaded_questions)
        assert not report.valid
        assert "not a valid measurement" in (report.invalid_reason() or "")

    def test_invalid_run_says_so_in_the_markdown(self, loaded_questions, tmp_path: Path) -> None:
        generations = [self._api_error_generation(q) for q in loaded_questions]
        report = build_report(score_generations(loaded_questions, generations, _settings(tmp_path)))
        text = render_markdown(report, {"run_id": "r", "git_sha": "a", "config": {}})
        assert "INVALID RUN" in text

    def test_validity_is_serialized(self, loaded_questions, tmp_path: Path) -> None:
        generations = [self._api_error_generation(q) for q in loaded_questions]
        payload = build_report(
            score_generations(loaded_questions, generations, _settings(tmp_path))
        ).to_dict()
        assert payload["valid"] is False
        assert payload["api_errors"] == len(loaded_questions)

    def test_healthy_run_is_valid(self, loaded_questions, tmp_path: Path) -> None:
        generations = [
            _generation(loaded_questions[0], "SELECT count(*) FROM player WHERE position = 'M'")
        ]
        report = build_report(
            score_generations(loaded_questions[:1], generations, _settings(tmp_path))
        )
        assert report.valid
        assert report.invalid_reason() is None


class TestTemperatureIsOptional:
    def test_temperature_is_omitted_unless_configured(self, tmp_path: Path) -> None:
        """gpt-6-luna rejects temperature=0; sending it by default 400s every request."""
        from harness.generation.standard import StandardGenerator

        settings = replace(_settings(tmp_path), generator_temperature=None)
        generator = StandardGenerator(settings)
        assert "temperature" not in generator._request_body([{"role": "user", "content": "hi"}])

    def test_temperature_is_sent_when_configured(self, tmp_path: Path) -> None:
        from harness.generation.standard import StandardGenerator

        settings = replace(_settings(tmp_path), generator_temperature=0.0)
        generator = StandardGenerator(settings)
        body = generator._request_body([{"role": "user", "content": "hi"}])
        assert body["temperature"] == 0.0
        assert body["model"] == "gpt-6-luna"


class TestVerify:
    """A published number must be recomputable without re-running a model."""

    def _published(self, tmp_path: Path) -> Path:
        run = open_run(_settings(tmp_path), "test", root=tmp_path / "runs")
        rows = [{"question_id": f"q{i}", "scored": True, "correct": i < 6} for i in range(10)]
        run.append_rows("match.jsonl", rows)
        report = {"n_total": 10, "n_scored": 10, "exec_accuracy": {"point": 0.6}}
        run.write_json("report.json", report)
        return run.path

    def test_recompute_matches_the_stored_report(self, tmp_path: Path) -> None:
        target = self._published(tmp_path)
        report = json.loads((target / "report.json").read_text())
        recomputed = recompute_from_matches(target / "match.jsonl")
        assert compare_to_report(recomputed, report) == []
        assert recomputed.n_correct == 6
        assert recomputed.accuracy == 0.6

    def test_excluded_rows_leave_the_denominator(self, tmp_path: Path) -> None:
        target = self._published(tmp_path)
        with (target / "match.jsonl").open("a") as handle:
            handle.write(
                json.dumps({"question_id": "qX", "scored": False, "correct": False}) + "\n"
            )
        recomputed = recompute_from_matches(target / "match.jsonl")
        assert recomputed.n_total == 11
        assert recomputed.n_scored == 10
        assert recomputed.accuracy == 0.6

    def test_a_doctored_report_is_detected(self, tmp_path: Path) -> None:
        target = self._published(tmp_path)
        report = json.loads((target / "report.json").read_text())
        report["exec_accuracy"]["point"] = 0.95
        problems = compare_to_report(recompute_from_matches(target / "match.jsonl"), report)
        assert any("accuracy" in p for p in problems)

    def test_a_doctored_denominator_is_detected(self, tmp_path: Path) -> None:
        target = self._published(tmp_path)
        report = json.loads((target / "report.json").read_text())
        report["n_scored"] = 8
        problems = compare_to_report(recompute_from_matches(target / "match.jsonl"), report)
        assert any("n_scored" in p for p in problems)

    def test_a_report_with_no_accuracy_is_rejected(self, tmp_path: Path) -> None:
        target = self._published(tmp_path)
        problems = compare_to_report(
            recompute_from_matches(target / "match.jsonl"), {"n_total": 10, "n_scored": 10}
        )
        assert problems

    def test_all_excluded_gives_zero_not_a_division_error(self, tmp_path: Path) -> None:
        target = tmp_path / "empty"
        target.mkdir()
        (target / "match.jsonl").write_text(
            json.dumps({"question_id": "q0", "scored": False, "correct": False}) + "\n"
        )
        recomputed = recompute_from_matches(target / "match.jsonl")
        assert recomputed.n_scored == 0
        assert recomputed.accuracy == 0.0
