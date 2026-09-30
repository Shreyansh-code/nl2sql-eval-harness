"""Judge validation, taxonomy, sampling, kappa, and the graph's retry/degrade path.

The judge is measured by whether its output survives scrutiny, so the tests here are
mostly adversarial: fabricated citations, invented categories, hedged verdicts, and
transports that fail. All of it runs against a fake client, so none of this needs an API
key and the retry path is deterministic.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from harness.judge.graph import JudgeGraph
from harness.judge.pipeline import run_judge
from harness.judge.port import Judgement, JudgeStatus, Verdict
from harness.judge.prompts import PROMPT_VERSION, build_user_prompt
from harness.judge.sampling import plan
from harness.judge.taxonomy import FailureMode, coerce, is_valid
from harness.judge.validate import validate
from harness.metrics.agreement import binary_agreement, cohen_kappa, confusion

CONTEXT = "## Schema\nCREATE TABLE player (name TEXT, height_cm INTEGER)\n## Question\nTallest?"


def _response(**overrides) -> str:
    payload = {
        "verdict": "generator_error",
        "failure_mode": "WRONG_FILTER",
        "confidence": 0.8,
        "rationale": "The filter used >= instead of >.",
        "evidence_span": "CREATE TABLE player",
    }
    payload.update(overrides)
    return json.dumps(payload)


class TestTaxonomy:
    def test_eleven_codes_are_fixed(self) -> None:
        assert len(list(FailureMode)) == 11

    def test_every_code_has_a_description(self) -> None:
        from harness.judge.taxonomy import DESCRIPTIONS

        assert set(DESCRIPTIONS) == set(FailureMode)
        assert all(len(text) > 30 for text in DESCRIPTIONS.values())

    @pytest.mark.parametrize("code", [mode.value for mode in FailureMode])
    def test_coerce_round_trips_every_code(self, code: str) -> None:
        assert coerce(code) is FailureMode(code)

    @pytest.mark.parametrize(
        "raw", ["wrong_filter", "WRONG FILTER", "  wrong-filter  ", "WRONG_FILTER (threshold)"]
    )
    def test_coerce_tolerates_decoration(self, raw: str) -> None:
        assert coerce(raw) is FailureMode.WRONG_FILTER

    @pytest.mark.parametrize("raw", ["", None, "SOMETHING_ELSE", "HALLUCINATED_MODE", 42])
    def test_coerce_refuses_to_guess(self, raw) -> None:
        assert coerce(raw) is None

    def test_is_valid_matches_coerce(self) -> None:
        assert is_valid("PRECISION")
        assert not is_valid("precision_error")


class TestValidation:
    def test_a_well_formed_response_is_accepted(self) -> None:
        result = validate(_response(), CONTEXT)
        assert result.ok
        assert result.verdict is Verdict.GENERATOR_ERROR
        assert result.failure_mode is FailureMode.WRONG_FILTER
        assert result.confidence == 0.8

    def test_fenced_json_is_accepted(self) -> None:
        assert validate(f"```json\n{_response()}\n```", CONTEXT).ok

    def test_json_with_surrounding_prose_is_accepted(self) -> None:
        assert validate(f"Here is my analysis:\n{_response()}\nHope that helps.", CONTEXT).ok

    def test_prose_only_is_unparseable(self) -> None:
        result = validate("I think it filtered the wrong column.", CONTEXT)
        assert result.status is JudgeStatus.UNPARSEABLE
        assert result.complaint

    def test_empty_response_is_unparseable(self) -> None:
        assert validate("", CONTEXT).status is JudgeStatus.UNPARSEABLE

    def test_invented_category_is_rejected_not_bucketed(self) -> None:
        result = validate(_response(failure_mode="VIBES_ERROR"), CONTEXT)
        assert result.status is JudgeStatus.OUT_OF_TAXONOMY
        assert "eleven codes" in (result.complaint or "")

    def test_fabricated_citation_is_rejected(self) -> None:
        """The core guard: a judge that cannot quote is not reasoning."""
        result = validate(_response(evidence_span="the model was being silly"), CONTEXT)
        assert result.status is JudgeStatus.UNCITED
        assert "verbatim" in (result.complaint or "")

    def test_empty_citation_is_rejected(self) -> None:
        assert validate(_response(evidence_span=""), CONTEXT).status is JudgeStatus.UNCITED

    def test_trivially_short_citation_is_rejected(self) -> None:
        assert validate(_response(evidence_span="CT"), CONTEXT).status is JudgeStatus.UNCITED

    def test_sql_ok_requires_the_none_label(self) -> None:
        good = validate(_response(verdict="sql_ok", failure_mode="NONE"), CONTEXT)
        assert good.ok
        assert good.failure_mode is None
        assert good.verdict is Verdict.SQL_OK

    def test_sql_ok_with_a_real_failure_mode_is_rejected_as_hedging(self) -> None:
        result = validate(_response(verdict="sql_ok", failure_mode="WRONG_FILTER"), CONTEXT)
        assert result.status is JudgeStatus.OUT_OF_TAXONOMY

    def test_unknown_verdict_is_rejected(self) -> None:
        result = validate(_response(verdict="probably_fine"), CONTEXT)
        assert result.status is JudgeStatus.UNPARSEABLE

    @pytest.mark.parametrize("value", [-0.1, 1.5, "high", None, True])
    def test_out_of_range_confidence_is_rejected(self, value) -> None:
        result = validate(_response(confidence=value), CONTEXT)
        assert result.status is JudgeStatus.UNPARSEABLE

    def test_empty_rationale_is_rejected(self) -> None:
        assert validate(_response(rationale="  "), CONTEXT).status is JudgeStatus.UNPARSEABLE

    def test_missing_fields_are_rejected(self) -> None:
        assert validate('{"verdict": "sql_ok"}', CONTEXT).status is JudgeStatus.UNPARSEABLE

    def test_rationale_is_capped_at_four_sentences(self) -> None:
        long = " ".join(f"Sentence {i}." for i in range(10))
        result = validate(_response(rationale=long), CONTEXT)
        assert result.ok
        assert result.rationale.count(".") <= 4


class FakeClient:
    """Returns queued responses, recording what it was asked."""

    def __init__(self, responses: list[str], model: str = "gpt-6-luna") -> None:
        self.responses = list(responses)
        self.model = model
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], **options: object) -> str:
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("client called more times than the test expected")
        return self.responses.pop(0)


class ExplodingClient:
    model = "gpt-6-luna"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[dict[str, str]], **options: object) -> str:
        self.calls += 1
        raise RuntimeError("http_500: upstream is down")


class TestGraph:
    def test_a_good_response_is_persisted_on_the_first_attempt(self, loaded_questions) -> None:
        item = _scored(loaded_questions[0], correct_sql=True)
        client = FakeClient([_response()])
        graph = JudgeGraph(client=client, model="gpt-6-luna")
        judgement = graph.judge(item)
        assert judgement.status is JudgeStatus.OK
        assert judgement.attempts == 1
        assert len(client.calls) == 1

    def test_a_bad_response_is_retried_once_with_the_defect_named(self, loaded_questions) -> None:
        item = _scored(loaded_questions[0], correct_sql=True)
        bad = _response(evidence_span="something I invented")
        good = _response()
        client = FakeClient([bad, good])
        graph = JudgeGraph(client=client, model="gpt-6-luna")
        judgement = graph.judge(item)
        assert judgement.status is JudgeStatus.OK
        assert judgement.attempts == 2
        # The retry must name the specific defect, not just "try again".
        assert "not a verbatim substring" in client.calls[1][-1]["content"]

    def test_two_bad_responses_degrade_to_unscored_rather_than_forcing_a_label(
        self, loaded_questions
    ) -> None:
        item = _scored(loaded_questions[0], correct_sql=True)
        bad = _response(evidence_span="invented")
        client = FakeClient([bad, bad])
        graph = JudgeGraph(client=client, model="gpt-6-luna")
        judgement = graph.judge(item)
        assert judgement.status is not JudgeStatus.OK
        assert judgement.failure_mode is None
        assert len(client.calls) == 2  # bounded: exactly one retry, never a loop

    def test_a_dead_transport_degrades_instead_of_raising(self, loaded_questions) -> None:
        item = _scored(loaded_questions[0], correct_sql=True)
        graph = JudgeGraph(client=ExplodingClient(), model="gpt-6-luna")
        judgement = graph.judge(item)
        assert judgement.status is JudgeStatus.UNPARSEABLE
        assert judgement.failure_mode is None

    def test_repeat_judging_is_served_from_cache(self, loaded_questions) -> None:
        item = _scored(loaded_questions[0], correct_sql=True)
        client = FakeClient([_response()])
        graph = JudgeGraph(client=client, model="gpt-6-luna")
        first = graph.judge(item)
        second = graph.judge(item)
        assert first is second
        assert len(client.calls) == 1

    def test_prompt_shows_the_judge_the_gold_query_and_the_difference(
        self, loaded_questions
    ) -> None:
        """The judge sees gold SQL by decision; the diff is what it must explain."""
        item = _scored(loaded_questions[0], correct_sql=False)
        prompt = build_user_prompt(item)
        assert "## Gold SQL" in prompt
        assert "## Generated SQL" in prompt
        assert "## Observed difference" in prompt
        assert item.question.gold_sql in prompt

    def test_prompt_cites_the_taxonomy_verbatim(self) -> None:
        from harness.judge.prompts import SYSTEM
        from harness.judge.taxonomy import describe_all

        assert describe_all() in SYSTEM


class TestSampling:
    def _scored(self, loaded_questions, n_correct: int):
        return [
            _scored(q, correct_sql=index < n_correct) for index, q in enumerate(loaded_questions)
        ]

    def test_every_failure_is_selected(self, loaded_questions) -> None:
        items = self._scored(loaded_questions, n_correct=1)
        judge_plan = plan(items, audit_fraction=0.2, seed=1)
        assert len(judge_plan.failures) == len(items) - 1

    def test_successes_are_audited_not_ignored(self, loaded_questions) -> None:
        items = self._scored(loaded_questions, n_correct=4)
        judge_plan = plan(items, audit_fraction=0.5, seed=1)
        assert len(judge_plan.audit) == 2
        assert not set(judge_plan.audit) & set(judge_plan.failures)

    def test_audit_is_seeded_and_reproducible(self, loaded_questions) -> None:
        items = self._scored(loaded_questions, n_correct=4)
        first = plan(items, audit_fraction=0.5, seed=7)
        second = plan(items, audit_fraction=0.5, seed=7)
        assert first.audit == second.audit

    def test_different_seeds_can_differ(self, loaded_questions) -> None:
        items = self._scored(loaded_questions, n_correct=8)
        assert (
            plan(items, audit_fraction=0.5, seed=1).audit
            != plan(items, audit_fraction=0.5, seed=2).audit
        )

    def test_audit_size_follows_the_fraction(self, loaded_questions) -> None:
        items = self._scored(loaded_questions, n_correct=5)
        assert len(plan(items, audit_fraction=0.2, seed=1).audit) == 1
        assert len(plan(items, audit_fraction=0.4, seed=1).audit) == 2

    def test_no_successes_means_no_audit_rather_than_an_error(self, loaded_questions) -> None:
        items = self._scored(loaded_questions, n_correct=0)
        assert plan(items, audit_fraction=0.2, seed=1).audit == ()


class TestKappa:
    def test_perfect_agreement_gives_kappa_one(self) -> None:
        labels = ["A", "B", "A", "B", "C", "C"]
        result = cohen_kappa(labels, labels)
        assert result.kappa == pytest.approx(1.0)
        assert result.observed_agreement == 1.0

    def test_matches_a_hand_computed_table(self) -> None:
        """2x2 table, judge x human, n=45.

        judge: 25 correct / 20 incorrect.   human: 30 correct / 15 incorrect.
        Cells: a=25, b=0, c=5, d=15.

        po = (25 + 15)/45 = 0.8889
        pe = (25/45)(30/45) + (20/45)(15/45) = 0.3704 + 0.1481 = 0.5185
        kappa = (0.8889 - 0.5185) / (1 - 0.5185) = 0.7692
        """
        judge = ["correct"] * 25 + ["incorrect"] * 20
        human = ["correct"] * 30 + ["incorrect"] * 15
        result = cohen_kappa(judge, human)
        assert result.n == 45
        assert result.observed_agreement == pytest.approx(40 / 45)
        assert result.expected_agreement == pytest.approx(0.5185, abs=1e-3)
        assert result.kappa == pytest.approx(0.7692, abs=1e-3)

    def test_chance_level_agreement_gives_kappa_near_zero(self) -> None:
        # Balanced marginals, agreement exactly at chance.
        result = cohen_kappa(["A", "A", "B", "B"], ["A", "B", "A", "B"])
        assert abs(result.kappa) < 1e-9

    def test_single_category_is_reported_undefined_not_zero(self) -> None:
        """Undefined and zero are different claims and must not be conflated."""
        result = cohen_kappa(["A"] * 5, ["A"] * 5)
        assert result.kappa is None
        assert result.degenerate == "both_raters_used_one_category"
        assert "undefined" in result.summary()

    def test_empty_input_is_reported_undefined(self) -> None:
        assert cohen_kappa([], []).degenerate == "no_labelled_items"

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="label counts differ"):
            cohen_kappa(["A"], ["A", "B"])

    def test_categories_used_by_only_one_rater_are_kept(self) -> None:
        table = confusion(["A", "B"], ["A", "C"])
        assert set(table) == {"A", "B", "C"}
        assert table["B"]["C"] == 1

    def test_kappa_is_penalised_by_prevalence(self) -> None:
        """High agreement, rare category: kappa must still fall below observed agreement."""
        skewed = ["A"] * 90 + ["B"] * 10
        judge = ["A"] * 88 + ["B"] * 10 + ["A", "A"]
        human = skewed
        result = cohen_kappa(judge, human)
        # Positional agreement is 96/100 (the judge is right on the first 88 and on
        # positions 90-97); chance is 0.82, so kappa lands well below the raw agreement.
        assert result.observed_agreement == pytest.approx(0.96)
        assert result.kappa == pytest.approx(0.7778, abs=1e-3)
        assert result.kappa < result.observed_agreement

    def test_binary_agreement_maps_flags_to_labels(self) -> None:
        result = binary_agreement([True, True, False, False], [True, False, False, True])
        assert result.n == 4
        assert result.observed_agreement == 0.5

    def test_summary_always_shows_n(self) -> None:
        assert (
            "n=4"
            in binary_agreement([True, True, False, False], [True, False, False, False]).summary()
        )

    def test_undefined_summary_says_undefined_not_zero(self) -> None:
        """A single-category table has no kappa; reporting 0.0 would be a false claim."""
        result = binary_agreement([True, True], [True, True])
        assert result.kappa is None
        assert "undefined" in result.summary()


class TestJudgementSerialisation:
    def test_row_round_trips(self) -> None:
        judgement = Judgement(
            question_id="q1",
            db_id="league",
            status=JudgeStatus.OK,
            verdict=Verdict.GENERATOR_ERROR,
            failure_mode=FailureMode.WRONG_JOIN,
            confidence=0.5,
            rationale="because",
            evidence_span="x = 1",
            model="gpt-6-luna",
            prompt_version=PROMPT_VERSION,
            cache_key="abc",
        )
        assert Judgement.from_row(json.loads(json.dumps(judgement.to_row()))) == judgement

    def test_unscored_row_round_trips_without_a_mode(self) -> None:
        judgement = Judgement(
            question_id="q1", db_id="league", status=JudgeStatus.UNCITED, error="nope"
        )
        restored = Judgement.from_row(json.loads(json.dumps(judgement.to_row())))
        assert restored.failure_mode is None
        assert restored.status is JudgeStatus.UNCITED


class TestJudgePipeline:
    def test_run_judge_covers_failures_and_audit(self, many_questions) -> None:
        scored = [_scored(q, correct_sql=index < 8) for index, q in enumerate(many_questions)]
        failures = sum(1 for s in scored if not s.correct)
        client = FakeClient([_response() for _ in range(20)])
        graph = JudgeGraph(client=client, model="gpt-6-luna")
        outcome = run_judge(scored, graph, audit_fraction=0.5, seed=1)
        assert len(outcome.judgements) == failures + 4  # 12 failures + 4 audited successes
        assert outcome.plan.describe()["n_failures"] == failures
        assert outcome.plan.describe()["n_audit"] == 4

    def test_unusable_judgements_are_counted_not_dropped(self, many_questions) -> None:
        scored = [_scored(q, correct_sql=False) for q in many_questions]
        client = FakeClient([_response(evidence_span="invented")] * 60)
        graph = JudgeGraph(client=client, model="gpt-6-luna")
        outcome = run_judge(scored, graph, audit_fraction=0.0, seed=1)
        assert outcome.usable == ()
        assert len(outcome.unscored) == len(scored)
        assert sum(outcome.status_counts().values()) == len(scored)


def _scored(question, *, correct_sql: bool):
    """A minimal ScoredQuestion for judge tests, with a real execution result."""
    from harness.execution.executor import ExecResult, Outcome
    from harness.generation.port import Generation, GenerationStatus
    from harness.pipeline import ScoredQuestion
    from harness.scoring.exec_match import Verdict as MatchVerdict
    from harness.scoring.exec_match import compare

    question = replace(question, gold_sql="SELECT name FROM player WHERE height_cm > 170")
    sql = (
        "SELECT name FROM player WHERE height_cm > 170"
        if correct_sql
        else "SELECT name FROM player WHERE height_cm > 999"
    )
    generation = Generation(
        question_id=question.question_id,
        db_id=question.db_id,
        status=GenerationStatus.OK,
        sql=sql,
        raw_response=sql,
        model="gpt-6-luna",
        prompt_version="v1",
        cache_key="k",
    )
    gold = ExecResult(outcome=Outcome.OK, rows=(("ana",), ("cy",)))
    execution = ExecResult(
        outcome=Outcome.OK if correct_sql else Outcome.EMPTY,
        rows=gold.rows if correct_sql else (),
    )
    match = compare(question.gold_sql, gold.rows, execution.rows)
    assert (match.is_match) is correct_sql
    assert match.verdict in {MatchVerdict.MATCH, MatchVerdict.MISMATCH}
    return ScoredQuestion(question, generation, execution, match, gold)


class TestJudgeStandardAccuracy:
    """The dual-standard number is only meaningful on a full-run judging pass."""

    def _outcome(self, scored, judgements):
        from harness.judge.pipeline import JudgeOutcome
        from harness.judge.sampling import JudgePlan

        return JudgeOutcome(
            judgements=tuple(judgements),
            plan=JudgePlan(failures=(), audit=(), seed=1, audit_fraction=1.0),
        )

    def _judgement(self, question, *, says_ok: bool) -> Judgement:
        return Judgement(
            question_id=question.question_id,
            db_id=question.db_id,
            status=JudgeStatus.OK,
            verdict=Verdict.SQL_OK if says_ok else Verdict.GENERATOR_ERROR,
            failure_mode=None if says_ok else FailureMode.WRONG_FILTER,
            confidence=0.9,
        )

    def test_quoted_when_every_scored_question_was_judged(self, many_questions) -> None:
        from harness.metrics.judge_report import build_judge_report

        scored = [_scored(q, correct_sql=i < 5) for i, q in enumerate(many_questions)]
        judgements = [self._judgement(item.question, says_ok=item.correct) for item in scored]
        report = build_judge_report(self._outcome(scored, judgements), scored)
        assert report.covers_whole_run
        assert report.judge_standard_accuracy == 5 / len(scored)

    def test_withheld_on_a_failure_weighted_sample(self, many_questions) -> None:
        """A failures-plus-audit sample would make the number look like run accuracy."""
        from harness.metrics.judge_report import build_judge_report

        scored = [_scored(q, correct_sql=i < 2) for i, q in enumerate(many_questions)]
        subset = scored[:6]
        judgements = [self._judgement(item.question, says_ok=item.correct) for item in subset]
        report = build_judge_report(self._outcome(scored, judgements), scored)
        assert not report.covers_whole_run
        assert report.judge_standard_accuracy is None

    def test_the_report_explains_why_it_is_withheld(self, many_questions) -> None:
        from harness.metrics.judge_report import build_judge_report, render_judge_markdown

        scored = [_scored(q, correct_sql=i < 2) for i, q in enumerate(many_questions)]
        judgements = [self._judgement(item.question, says_ok=item.correct) for item in scored[:5]]
        report = build_judge_report(self._outcome(scored, judgements), scored)
        text = render_judge_markdown(report, {"n_failures": 3, "n_audit": 2}, {})
        assert "failure-weighted" in text
        assert "--all" in text

    def test_disagreement_is_split_by_direction(self, many_questions) -> None:
        from harness.metrics.judge_report import build_judge_report

        scored = [_scored(q, correct_sql=i < 4) for i, q in enumerate(many_questions)]
        judgements = [
            # Judge is right on the first four, and lenient on one genuine failure.
            self._judgement(item.question, says_ok=(item.correct or i == 5))
            for i, item in enumerate(scored)
        ]
        report = build_judge_report(self._outcome(scored, judgements), scored)
        assert report.false_accusations == 0
        assert report.missed_failures == 1


class TestJudgementPersistence:
    """A judging pass is a complete set, so re-running must overwrite, not accumulate."""

    def test_rewriting_a_pass_does_not_duplicate_rows(self, many_questions, tmp_path) -> None:
        import json

        from harness.config import Settings
        from harness.judge.pipeline import persist
        from harness.runs import open_run

        settings = Settings(
            generator_model="gpt-6-luna",
            judge_model="gpt-6-luna",
            generator_temperature=None,
            generator_mode="standard",
            judge_mode="batch",
            openai_base_url=None,
            openai_api_key="sk-test",
            data_dir=tmp_path,
            subset_path=tmp_path / "subset.yaml",
            max_concurrency=1,
            sql_timeout_seconds=1.0,
            max_rows=10,
            label_sample_size=50,
            langsmith_tracing=False,
        )
        scored = [_scored(q, correct_sql=False) for q in many_questions[:4]]
        graph = JudgeGraph(client=FakeClient([_response() for _ in range(8)]), model="gpt-6-luna")
        outcome = run_judge(scored, graph, audit_fraction=0.0, seed=1)

        run = open_run(settings, "judge", root=tmp_path / "runs")
        assert persist(run, outcome) == 4
        # Re-judging the same inputs must leave the file identical in size.
        assert persist(run, outcome) == 4
        rows = [
            json.loads(line) for line in (run.path / "judgements.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 4
        assert len({r["cache_key"] for r in rows}) == 4
