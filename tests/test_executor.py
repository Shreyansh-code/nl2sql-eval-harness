"""The executor runs model output, so its guardrails are tests, not assumptions."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from harness.execution.executor import Outcome, _readonly_connection, execute
from harness.execution.guard import check


class TestGuard:
    def test_allows_a_plain_select(self) -> None:
        assert check("SELECT 1")

    def test_allows_a_cte_select(self) -> None:
        assert check("WITH t AS (SELECT 1 AS x) SELECT x FROM t")

    def test_allows_a_trailing_semicolon(self) -> None:
        assert check("SELECT 1;")

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE player",
            "DELETE FROM player",
            "UPDATE player SET name = 'x'",
            "INSERT INTO player VALUES (9, 'z', 'M', 1)",
            "PRAGMA table_info(player)",
            "ATTACH DATABASE 'other.sqlite' AS other",
            "",
            "   ",
        ],
    )
    def test_rejects_non_read_only(self, sql: str) -> None:
        assert not check(sql)

    def test_rejects_multi_statement_injection(self) -> None:
        verdict = check("SELECT 1; DROP TABLE player")
        assert not verdict
        assert verdict.reason == "multi_statement"

    def test_rejects_a_cte_that_writes(self) -> None:
        verdict = check("WITH t AS (DELETE FROM player RETURNING *) SELECT * FROM t")
        assert not verdict
        assert verdict.reason == "forbidden_keyword:DELETE"

    def test_semicolon_inside_a_string_literal_is_not_a_statement_break(self) -> None:
        assert check("SELECT 'a; DROP TABLE player' AS s")

    def test_comment_hiding_a_payload_is_rejected(self) -> None:
        assert not check("SELECT 1 /* harmless */; -- tail\nDELETE FROM player")

    def test_replace_string_function_is_not_treated_as_replace_into(self) -> None:
        assert check("SELECT REPLACE(name, 'a', 'b') FROM player")

    def test_replace_into_is_rejected(self) -> None:
        assert not check("REPLACE INTO player VALUES (9, 'z', 'M', 1)")

    def test_column_named_like_a_keyword_is_allowed(self) -> None:
        assert check("SELECT update_time, create_date FROM events")


class TestExecutor:
    def test_executes_a_select(self, db_path: Path) -> None:
        result = execute("SELECT name FROM player ORDER BY name", db_path)
        assert result.outcome is Outcome.OK
        assert result.rows == (("ana",), ("bo",), ("cy",))

    def test_empty_result_is_reported_as_empty_not_error(self, db_path: Path) -> None:
        result = execute("SELECT name FROM player WHERE height_cm > 999", db_path)
        assert result.outcome is Outcome.EMPTY
        assert result.produced_rows is False

    def test_syntax_error_is_captured_as_data(self, db_path: Path) -> None:
        result = execute("SELECT FROM WHERE", db_path)
        assert result.outcome is Outcome.ERROR
        assert result.detail

    def test_rejected_sql_never_reaches_the_database(self, db_path: Path) -> None:
        result = execute("DROP TABLE player", db_path)
        assert result.outcome is Outcome.REJECTED
        assert execute("SELECT count(*) FROM player", db_path).rows == ((3,),)

    def test_row_cap_sets_truncated(self, wide_table: Path) -> None:
        result = execute("SELECT i FROM t", wide_table, max_rows=10)
        assert result.truncated is True
        assert len(result.rows) == 10

    def test_row_cap_not_hit_when_result_fits(self, db_path: Path) -> None:
        assert execute("SELECT name FROM player", db_path, max_rows=10).truncated is False

    def test_slow_query_is_interrupted_at_the_deadline(self, db_path: Path) -> None:
        slow = (
            "WITH RECURSIVE spin(x) AS ("
            "  SELECT 1 UNION ALL SELECT x + 1 FROM spin WHERE x < 4000000000"
            ") SELECT sum(x) FROM spin"
        )
        result = execute(slow, db_path, timeout_seconds=0.3)
        assert result.outcome is Outcome.TIMEOUT

    def test_missing_database_is_captured(self, tmp_path: Path) -> None:
        result = execute("SELECT 1", tmp_path / "nope.sqlite")
        assert result.outcome is Outcome.ERROR
        assert "missing_database" in (result.detail or "")


class TestReadOnlyBoundary:
    """The guard is the fast gate; this connection is the actual boundary."""

    def test_connection_refuses_writes(self, db_path: Path) -> None:
        conn = _readonly_connection(db_path)
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("INSERT INTO player VALUES (9, 'z', 'M', 1)")
        finally:
            conn.close()

    def test_database_file_is_untouched_after_a_write_attempt(self, db_path: Path) -> None:
        execute("DROP TABLE player", db_path)
        execute("DELETE FROM player", db_path)
        assert execute("SELECT count(*) FROM player", db_path).rows == ((3,),)
