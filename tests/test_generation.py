"""Response parsing. A harness that guesses here inflates accuracy."""

from __future__ import annotations

from dataclasses import replace

from harness.generation.parsing import parse_sql


class TestAcceptsRealisticResponses:
    def test_sql_fence(self) -> None:
        result = parse_sql("Here you go:\n```sql\nSELECT name FROM player\n```\n")
        assert result.sql == "SELECT name FROM player"

    def test_bare_fence_without_language_tag(self) -> None:
        assert parse_sql("```\nSELECT 1\n```").sql == "SELECT 1"

    def test_leading_sql_label(self) -> None:
        assert parse_sql("SQL: SELECT 1").sql == "SELECT 1"

    def test_trailing_semicolon_is_stripped(self) -> None:
        assert parse_sql("```sql\nSELECT 1;\n```").sql == "SELECT 1"

    def test_multiline_query_keeps_its_shape(self) -> None:
        raw = "```sql\nSELECT a\nFROM t\nWHERE x = 1\n```"
        assert parse_sql(raw).sql == "SELECT a\nFROM t\nWHERE x = 1"

    def test_unfenced_sql_is_accepted(self) -> None:
        assert parse_sql("SELECT name FROM player").sql == "SELECT name FROM player"

    def test_unfenced_sql_with_trailing_prose_stops_at_the_semicolon(self) -> None:
        raw = "SELECT name FROM player;\nThis lists the players."
        assert parse_sql(raw).sql == "SELECT name FROM player"

    def test_cte_is_accepted(self) -> None:
        raw = "```sql\nWITH t AS (SELECT 1 AS x) SELECT x FROM t\n```"
        assert parse_sql(raw).sql.startswith("WITH")

    def test_leading_comment_banner_is_dropped(self) -> None:
        raw = "```sql\n-- count the players\nSELECT count(*) FROM player\n```"
        assert parse_sql(raw).sql == "SELECT count(*) FROM player"

    def test_string_literal_containing_a_semicolon_survives(self) -> None:
        raw = "```sql\nSELECT 'a; b' AS s FROM t\n```"
        assert parse_sql(raw).sql == "SELECT 'a; b' AS s FROM t"


class TestRejectsRatherThanGuesses:
    def test_empty_response(self) -> None:
        result = parse_sql("")
        assert result.sql is None
        assert result.reason == "empty_response"

    def test_whitespace_only(self) -> None:
        assert parse_sql("   \n ").reason == "empty_response"

    def test_prose_without_sql(self) -> None:
        result = parse_sql("I cannot answer that without more context.")
        assert result.sql is None
        assert result.reason in {"not_a_select", "no_sql_found"}

    def test_destructive_statement_is_rejected_by_the_guard(self) -> None:
        result = parse_sql("```sql\nDROP TABLE player\n```")
        assert result.sql is None
        assert result.reason is not None
        assert "guard:" in result.reason or result.reason == "not_a_select"

    def test_multi_statement_response_is_rejected(self) -> None:
        result = parse_sql("```sql\nSELECT 1; DROP TABLE player\n```")
        assert result.sql is None
        assert result.reason == "multi_statement"

    def test_explanation_inside_the_fence_is_not_treated_as_sql(self) -> None:
        result = parse_sql("```sql\nThis query finds the tallest player.\n```")
        assert result.sql is None

    def test_counts_blocks_for_the_report(self) -> None:
        raw = "```sql\nSELECT 1\n```\nand also\n```sql\nSELECT 2\n```"
        assert parse_sql(raw).blocks_seen == 2

    def test_first_valid_block_wins_when_several_are_present(self) -> None:
        raw = "```sql\nSELECT 1\n```\n```sql\nSELECT 2\n```"
        assert parse_sql(raw).sql == "SELECT 1"


class TestPromptContent:
    def test_prompt_includes_schema_question_and_evidence(self, loaded_questions) -> None:
        from harness.generation.prompts import build_user_prompt

        prompt = build_user_prompt(loaded_questions[0])
        assert "CREATE TABLE player" in prompt
        assert "player.position in ('M'" in prompt  # decoded coded column
        assert loaded_questions[0].question in prompt
        assert loaded_questions[0].evidence in prompt

    def test_evidence_section_omitted_when_absent(self, question) -> None:
        from harness.generation.prompts import build_user_prompt

        assert "## Evidence" in build_user_prompt(question)  # fixture carries evidence
        bare = replace(question, evidence="  ")
        assert "## Evidence" not in build_user_prompt(bare)

    def test_system_prompt_forbids_prose(self) -> None:
        from harness.generation.prompts import SYSTEM

        assert "No prose" in SYSTEM
        assert "single" in SYSTEM
