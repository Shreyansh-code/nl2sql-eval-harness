"""ExecMatch: the objective half of the metric, so it gets the most tests."""

from __future__ import annotations

import pytest

from harness.scoring.exec_match import Verdict, compare, normalize_cell

GOLD_UNORDERED = "SELECT name FROM player"
GOLD_ORDERED = "SELECT name FROM player ORDER BY name"


class TestNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (1, 1),
            (1.0, 1),
            (1.5, 1.5),
            (10.0, 10),
            (-0.0, 0),
            (True, 1),
            (False, 0),
            (None, None),
            (b"\x00\xff", "00ff"),
            ("M", "M"),
        ],
    )
    def test_cells_collapse_to_meaningful_values(self, raw: object, expected: object) -> None:
        assert normalize_cell(raw) == expected

    def test_int_and_float_results_agree(self) -> None:
        result = compare(GOLD_UNORDERED, ((1,),), ((1.0,),))
        assert result.verdict is Verdict.MATCH


class TestRowComparison:
    def test_identical_rows_match(self) -> None:
        result = compare(GOLD_UNORDERED, (("a",), ("b",)), (("a",), ("b",)))
        assert result.verdict is Verdict.MATCH
        assert result.is_match and result.scored

    def test_order_ignored_when_gold_has_no_order_by(self) -> None:
        result = compare(GOLD_UNORDERED, (("a",), ("b",)), (("b",), ("a",)))
        assert result.verdict is Verdict.MATCH
        assert result.order_sensitive is False

    def test_order_enforced_when_gold_orders(self) -> None:
        result = compare(GOLD_ORDERED, (("a",), ("b",)), (("b",), ("a",)))
        assert result.verdict is Verdict.MISMATCH
        assert result.order_sensitive is True

    def test_duplicate_multiplicity_matters(self) -> None:
        assert compare(GOLD_UNORDERED, (("a",), ("a",)), (("a",),)).verdict is Verdict.MISMATCH

    def test_wrong_column_count_is_a_mismatch(self) -> None:
        result = compare(GOLD_UNORDERED, (("a", "b"),), (("a",),))
        assert result.verdict is Verdict.MISMATCH
        assert result.first_difference["reason"] == "column_count"

    def test_empty_gold_against_empty_generated_matches(self) -> None:
        assert compare(GOLD_UNORDERED, (), ()).verdict is Verdict.MATCH

    def test_null_is_not_equal_to_empty_string(self) -> None:
        assert compare(GOLD_UNORDERED, ((None,),), (("",),)).verdict is Verdict.MISMATCH

    def test_difference_is_reported_for_the_report(self) -> None:
        result = compare(GOLD_UNORDERED, (("a",), ("b",)), (("a",), ("z",)))
        detail = result.first_difference or {}
        assert detail["missing_from_generated"] == [{"col0": "b"}]
        assert detail["unexpected_in_generated"] == [{"col0": "z"}]


class TestTruncation:
    def test_truncated_generated_result_is_inconclusive_not_a_match(self) -> None:
        result = compare(GOLD_UNORDERED, (("a",),), (("a",),), truncated=True)
        assert result.verdict is Verdict.INCONCLUSIVE
        assert not result.scored

    def test_truncation_does_not_hide_an_observed_mismatch(self) -> None:
        result = compare(GOLD_UNORDERED, (("a",),), (("z",),), truncated=True)
        assert result.verdict is Verdict.MISMATCH


class TestRowCounts:
    def test_counts_are_reported_for_every_verdict(self) -> None:
        result = compare(GOLD_UNORDERED, (("a",), ("b",)), (("a",),))
        assert (result.gold_row_count, result.generated_row_count) == (2, 1)


class TestNumericTextEquivalence:
    """SQLite returns TEXT or REAL depending on affinity and casts.

    Found in the wild: gold returned '202.484' and the generated query returned 202.484 for
    the same lap time, and the harness scored it wrong. The rule is narrow on purpose.
    """

    def test_a_number_and_its_plain_string_form_are_the_same_value(self) -> None:
        result = compare(GOLD_UNORDERED, (("202.484",),), ((202.484,),))
        assert result.verdict is Verdict.MATCH

    def test_it_works_with_column_headers_in_the_aggregate(self) -> None:
        result = compare(GOLD_UNORDERED, ((1, "2.5"),), ((1.0, 2.5),))
        assert result.verdict is Verdict.MATCH

    def test_a_genuinely_different_number_is_still_a_mismatch(self) -> None:
        assert compare(GOLD_UNORDERED, (("202.484",),), ((202.5,),)).verdict is Verdict.MISMATCH

    def test_zero_padded_identifiers_stay_distinct_from_numbers(self) -> None:
        """The guard that stops this convenience rule from masking a real difference."""
        assert compare(GOLD_UNORDERED, (("007",),), ((7,),)).verdict is Verdict.MISMATCH

    def test_non_canonical_spellings_are_not_numbers(self) -> None:
        for text in ("1e3", "1.50", " 5", "+5", "1_000"):
            result = compare(GOLD_UNORDERED, ((text,),), ((1000 if text == "1e3" else 0,),))
            assert result.verdict is Verdict.MISMATCH, text

    def test_a_date_string_is_not_a_number(self) -> None:
        assert compare(GOLD_UNORDERED, (("2024-01-01",),), ((2024,),)).verdict is Verdict.MISMATCH

    def test_the_diff_still_shows_readable_values(self) -> None:
        """Canonical rows decide the verdict; the report must not leak ('num', Decimal(..))."""
        result = compare(GOLD_UNORDERED, (("202.484",),), ((9.9,),))
        detail = result.first_difference or {}
        rendered = str(detail.get("missing_from_generated"))
        assert "202.484" in rendered
        assert "Decimal" not in rendered
        assert "num" not in rendered

    def test_matching_still_ignores_row_order(self) -> None:
        result = compare(GOLD_UNORDERED, (("1",), ("2",)), ((2.0,), (1.0,)))
        assert result.verdict is Verdict.MATCH
