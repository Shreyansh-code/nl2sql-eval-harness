from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from harness.dataset.loader import DataMissingError, available_db_ids, load_questions
from harness.dataset.schema import introspect
from harness.dataset.subset import SubsetSpec
from harness.scoring.baseline import gate, run_baseline
from harness.scoring.exec_match import Verdict

SPEC = SubsetSpec(name="t", source="bird-dev", db_ids=("league",), limit_per_db=10, sampling="f")


def test_loader_reads_dev_json_and_filters_by_db(loaded_questions) -> None:
    """Other databases are excluded; the first rows are the hand-written ones."""
    ids = [q.question_id for q in loaded_questions]
    assert ids[:2] == ["league_0", "league_1"]
    assert "other_0" not in ids
    assert {q.db_id for q in loaded_questions} == {"league"}


def test_loader_attaches_a_real_schema(bird_layout: Path, loaded_questions) -> None:
    schema = loaded_questions[0].schema
    assert "player" in schema.tables and "match" in schema.tables
    assert "CREATE TABLE player" in schema.ddl
    assert loaded_questions[0].db_path == bird_layout / "dev_databases/league/league.sqlite"


def test_schema_samples_coded_columns_so_they_can_be_decoded(db_path: Path) -> None:
    result = introspect(db_path)
    # position is a coded field: its values are the only way to read the question.
    assert set(result.column_values["player"]["position"]) == {"M", "F"}


def test_schema_skips_high_cardinality_columns(high_cardinality: Path) -> None:
    result = introspect(high_cardinality)
    assert "body" not in result.column_values["note"]
    assert result.column_values["note"]["tag"] == ["t"]


def test_prompt_block_includes_ddl_and_value_samples(schema) -> None:
    block = schema.to_prompt_block()
    assert "CREATE TABLE player" in block
    assert "player.position in ('M'" in block


def test_loader_respects_limit_per_db(bird_layout: Path) -> None:
    spec = SubsetSpec(
        name="t", source="bird-dev", db_ids=("league",), limit_per_db=1, sampling="prefix"
    )
    assert len(load_questions(bird_layout, spec)) == 1


def test_loader_pins_explicit_question_ids(bird_layout: Path) -> None:
    spec = SubsetSpec(
        name="t",
        source="bird-dev",
        db_ids=("league",),
        limit_per_db=99,
        sampling="explicit",
        question_ids=("league_1",),
    )
    assert [q.question_id for q in load_questions(bird_layout, spec)] == ["league_1"]


def test_loader_accepts_both_gold_sql_key_spellings(tmp_path: Path) -> None:
    """BIRD ships `SQL`; the parquet re-dump we fetch ships `sql`."""
    data_dir = tmp_path / "bird"
    db_dir = data_dir / "dev_databases" / "league"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "league.sqlite")
    conn.execute("CREATE TABLE player (name TEXT)")
    conn.commit()
    conn.close()
    (data_dir / "dev.json").write_text(
        json.dumps([{"question_id": "q1", "db_id": "league", "question": "?", "sql": "SELECT 1"}]),
        encoding="utf-8",
    )
    assert load_questions(data_dir, SPEC)[0].gold_sql == "SELECT 1"


class TestStratifiedSampling:
    """A prefix would take whichever difficulties sit first in the dev.json group."""

    def _layout(self, tmp_path: Path) -> Path:
        data_dir = tmp_path / "bird"
        db_dir = data_dir / "dev_databases" / "league"
        db_dir.mkdir(parents=True)
        conn = sqlite3.connect(db_dir / "league.sqlite")
        conn.execute("CREATE TABLE player (name TEXT)")
        conn.commit()
        conn.close()
        rows = [
            {
                "question_id": f"q{i}",
                "db_id": "league",
                "question": "?",
                "evidence": "",
                "SQL": "SELECT 1",
                "difficulty": "simple" if i < 4 else "challenging",
            }
            for i in range(6)
        ]
        (data_dir / "dev.json").write_text(json.dumps(rows), encoding="utf-8")
        return data_dir

    def test_prefix_takes_one_difficulty_only(self, tmp_path: Path) -> None:
        spec = SubsetSpec(
            name="t",
            source="bird-dev",
            db_ids=("league",),
            limit_per_db=4,
            sampling="contiguous_prefix",
        )
        assert {q.difficulty for q in load_questions(self._layout(tmp_path), spec)} == {"simple"}

    def test_stratified_sampling_spreads_the_cap(self, tmp_path: Path) -> None:
        spec = SubsetSpec(
            name="t",
            source="bird-dev",
            db_ids=("league",),
            limit_per_db=4,
            sampling="stratified_by_difficulty",
        )
        difficulties = [q.difficulty for q in load_questions(self._layout(tmp_path), spec)]
        assert difficulties.count("simple") == difficulties.count("challenging") == 2

    def test_stratified_sampling_is_deterministic(self, tmp_path: Path) -> None:
        spec = SubsetSpec(
            name="t",
            source="bird-dev",
            db_ids=("league",),
            limit_per_db=3,
            sampling="stratified_by_difficulty",
        )
        data_dir = self._layout(tmp_path)
        first = [q.question_id for q in load_questions(data_dir, spec)]
        second = [q.question_id for q in load_questions(data_dir, spec)]
        assert first == second


def test_loader_explains_an_empty_selection(bird_layout: Path) -> None:
    spec = SubsetSpec(name="t", source="bird-dev", db_ids=("nope",), limit_per_db=1, sampling="x")
    try:
        load_questions(bird_layout, spec)
    except DataMissingError as exc:
        assert "0 questions" in str(exc)
    else:
        raise AssertionError("expected DataMissingError")


def test_loader_points_at_the_missing_download(tmp_path: Path) -> None:
    try:
        load_questions(tmp_path / "empty", SPEC)
    except DataMissingError as exc:
        assert "dev.json" in str(exc)
    else:
        raise AssertionError("expected DataMissingError")


def test_available_db_ids_lists_disk_layouts(bird_layout: Path) -> None:
    assert available_db_ids(bird_layout) == ["league"]


def test_baseline_passes_on_a_healthy_subset(loaded_questions) -> None:
    report = run_baseline(loaded_questions)
    assert report.total == len(loaded_questions)
    assert report.passed == report.total
    gate(report)  # must not raise
    assert all(row.match.verdict is Verdict.MATCH for row in report.rows)


def test_baseline_gate_fails_loudly_on_a_broken_query(db_path: Path, loaded_questions) -> None:
    broken = loaded_questions[0]
    object.__setattr__(broken, "gold_sql", "SELECT FROM WHERE")
    report = run_baseline([broken])
    assert report.passed == 0
    assert report.failures[0].first.outcome.value == "error"
    try:
        gate(report)
    except AssertionError as exc:
        assert "baseline gate failed" in str(exc)
    else:
        raise AssertionError("gate should have raised")


class TestNesting:
    """A widened subset must contain the published one, not re-roll it.

    Round-robin stratification is what makes this true, and it is what lets the
    90-question M2 result and the wider M3/M4 result be compared instead of merely
    coexisting. Note the property is per database, not a global list prefix: the loader
    groups questions by db_id, so containment is checked per group. A test asserts that
    distinction too, because a global-prefix claim would be wrong.
    """

    def test_widening_the_cap_keeps_the_earlier_questions(self, tmp_path: Path) -> None:
        data_dir = _mixed_difficulty_layout(tmp_path)
        small = _ids(data_dir, limit=30)
        large = _ids(data_dir, limit=60)
        assert len(small) == 30 and len(large) == 60
        assert small == large[:30]

    def test_nesting_holds_per_database(self, tmp_path: Path) -> None:
        data_dir = _multi_db_layout(tmp_path)
        for db in ("alpha", "beta"):
            assert _ids(data_dir, limit=10, db=db) == _ids(data_dir, limit=25, db=db)[:10]

    def test_containment_is_per_database_not_a_global_prefix(self, tmp_path: Path) -> None:
        """Documents the real shape: grouped by db_id, so not a global prefix."""
        data_dir = _multi_db_layout(tmp_path)
        small = _ids(data_dir, limit=10, db="alpha") + _ids(data_dir, limit=10, db="beta")
        spec = SubsetSpec(
            name="t",
            source="bird-dev",
            db_ids=("alpha", "beta"),
            limit_per_db=25,
            sampling="stratified_by_difficulty",
        )
        wide = [q.question_id for q in load_questions(data_dir, spec)]
        assert set(small) <= set(wide)  # contained
        assert small != wide[: len(small)]  # but not a global prefix

    def test_nesting_holds_across_four_databases(self, tmp_path: Path) -> None:
        data_dir = _multi_db_layout(tmp_path, dbs=("alpha", "beta", "gamma", "delta"))
        spec = SubsetSpec(
            name="t",
            source="bird-dev",
            db_ids=("alpha", "beta", "gamma", "delta"),
            limit_per_db=20,
            sampling="stratified_by_difficulty",
        )
        load_questions(data_dir, spec)


def _spec(db: str, limit: int) -> SubsetSpec:
    return SubsetSpec(
        name="t",
        source="bird-dev",
        db_ids=(db,),
        limit_per_db=limit,
        sampling="stratified_by_difficulty",
    )


def _ids(data_dir: Path, *, limit: int, db: str = "league") -> list[str]:
    return [q.question_id for q in load_questions(data_dir, _spec(db, limit))]


def _write_dev_json(data_dir: Path, rows: list[dict]) -> None:
    (data_dir / "dev.json").write_text(json.dumps(rows), encoding="utf-8")


def _mixed_difficulty_layout(tmp_path: Path) -> Path:
    data_dir = _empty_db(tmp_path, "league")
    rows = [
        {
            "question_id": f"q{i}",
            "db_id": "league",
            "question": "?",
            "evidence": "",
            "SQL": "SELECT 1",
            "difficulty": ["simple", "moderate", "challenging"][i % 3],
        }
        for i in range(90)
    ]
    _write_dev_json(data_dir, rows)
    return data_dir


def _multi_db_layout(tmp_path: Path, dbs: tuple[str, ...] = ("alpha", "beta")) -> Path:
    data_dir = tmp_path / "multi"
    rows = []
    for db in dbs:
        _empty_db(data_dir, db)
        for i in range(30):
            rows.append(
                {
                    "question_id": f"{db}_{i}",
                    "db_id": db,
                    "question": "?",
                    "evidence": "",
                    "SQL": "SELECT 1",
                    "difficulty": ["simple", "moderate", "challenging"][i % 3],
                }
            )
    _write_dev_json(data_dir, rows)
    return data_dir


def _empty_db(data_dir: Path, db_id: str) -> Path:
    db_dir = data_dir / "dev_databases" / db_id
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_dir / f"{db_id}.sqlite")
    conn.execute("CREATE TABLE t (a INTEGER)")
    conn.commit()
    conn.close()
    return data_dir
