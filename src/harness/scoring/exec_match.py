"""Result-set comparison. The objective half of the metric.

Matching rules, stated once (HLD 5.4):

* rows compare as multisets of tuples, so duplicates and ordering both matter;
* comparison is order-*sensitive* only when the gold query has ORDER BY;
* NULL, booleans, and numeric scale are normalized before comparison;
* a truncated result set cannot be proven equal, so it yields INCONCLUSIVE rather
  than a silent match.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from ..execution.sql_text import has_order_by

_MAX_REPORTED_ROWS = 5


class Verdict(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class MatchResult:
    verdict: Verdict
    gold_row_count: int
    generated_row_count: int
    order_sensitive: bool
    truncated: bool = False
    first_difference: dict[str, object] | None = None

    @property
    def is_match(self) -> bool:
        return self.verdict is Verdict.MATCH

    @property
    def scored(self) -> bool:
        """Whether this question contributes to the accuracy denominator."""
        return self.verdict is not Verdict.INCONCLUSIVE


def normalize_cell(value: object) -> object:
    """Collapse values that SQLite reports differently but that mean the same thing."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return _normalize_decimal(Decimal(repr(value)))
        except (InvalidOperation, ValueError):
            return value
    if isinstance(value, Decimal):
        return _normalize_decimal(value)
    if isinstance(value, bytes):
        return value.hex()
    return value


def canonical_cell(value: object) -> tuple[str, object]:
    """Hashable, comparison-grade form of a cell.

    Adds one rule on top of `normalize_cell`: a number and a *numerically identical*
    string are the same value. SQLite hands back TEXT or REAL depending on the column's
    affinity and on whether the query casts, so `SELECT duration FROM lap` and
    `SELECT CAST(duration AS TEXT) FROM lap` return 202.484 and '202.484' for the same
    fact. Counting that as a wrong answer is a defect in the comparator, not in the model.

    Guarded by a round-trip check rather than a bare `float()` parse, so a zero-padded
    identifier ('007') stays distinct from the number 7. That guard is what stops this
    convenience rule from quietly merging two genuinely different values.
    """
    normalized = normalize_cell(value)
    if isinstance(normalized, (int, float)) and not isinstance(normalized, bool):
        return ("num", Decimal(str(normalized)))
    if isinstance(normalized, str):
        decimal = _round_trips_to_number(normalized)
        if decimal is not None:
            return ("num", decimal)
    return ("str", normalized)


def _round_trips_to_number(text: str) -> Decimal | None:
    """Parse `text` as a number only if it is the plain rendering of that number."""
    stripped = text.strip()
    if not stripped or stripped != text:
        return None
    if len(stripped) > 1 and stripped[0] in "+-" and stripped[1] == "0":
        return None  # -0.5, +0.75 are fine; -007 is an identifier
    if len(stripped) > 1 and stripped[0] == "0" and stripped[1] not in ".eE":
        return None  # zero-padded code
    try:
        value = Decimal(stripped)
    except (InvalidOperation, ValueError):
        return None
    if not value.is_finite():
        return None
    if format(value, "f") != stripped and str(value) != stripped:
        # Reject '1e3', '1.50', ' 1' and anything else that is not the canonical spelling.
        return None
    return value


def canonical_row(row: tuple[object, ...]) -> tuple[tuple[str, object], ...]:
    return tuple(canonical_cell(cell) for cell in row)


def _normalize_decimal(value: Decimal) -> object:
    if value.is_nan() or value.is_infinite():
        return str(value)
    if value == 0:
        return 0  # collapse -0.0 and 0
    normalized = value.normalize()
    as_int = normalized.to_integral_value()
    if normalized == as_int:
        return int(as_int)  # 1.0 -> 1, 10.0 -> 10
    return float(normalized)


def normalize_rows(rows: tuple[tuple[object, ...], ...]) -> tuple[tuple[object, ...], ...]:
    return tuple(tuple(normalize_cell(cell) for cell in row) for row in rows)


def canonical_rows(
    rows: tuple[tuple[object, ...], ...],
) -> tuple[tuple[tuple[str, object], ...], ...]:
    """Comparison-grade rows. Used for the verdict, not for the diff we show a human."""
    return tuple(canonical_row(row) for row in rows)


def _shape(rows: tuple[tuple[tuple[str, object], ...], ...]) -> list[int]:
    return sorted({len(row) for row in rows})


def compare(
    gold_sql: str,
    gold_rows: tuple[tuple[object, ...], ...],
    generated_rows: tuple[tuple[object, ...], ...],
    *,
    truncated: bool = False,
) -> MatchResult:
    """Compare a generated result set against the gold result set.

    The verdict is decided on canonical rows, but the diff attached to the result is built
    from the display rows. Those are two different jobs: canonical rows exist so that a
    number and its string form compare equal, and nobody wants to read `('num', Decimal(
    '202.484'))` in a report.
    """
    order_sensitive = has_order_by(gold_sql)
    gold = canonical_rows(gold_rows)
    generated = canonical_rows(generated_rows)
    shown_gold = normalize_rows(gold_rows)
    shown_generated = normalize_rows(generated_rows)

    if _shape(gold) != _shape(generated):
        return _mismatch(
            gold,
            generated,
            shown_gold,
            shown_generated,
            order_sensitive,
            truncated,
            "column_count",
        )

    if order_sensitive:
        equal = gold == generated
    else:
        equal = Counter(gold) == Counter(generated)

    if truncated and equal:
        # Equal up to the row cap, but the generated set may have more rows than we kept.
        return MatchResult(
            verdict=Verdict.INCONCLUSIVE,
            gold_row_count=len(gold),
            generated_row_count=len(generated),
            order_sensitive=order_sensitive,
            truncated=True,
            first_difference={"reason": "generated_result_truncated_at_row_cap"},
        )

    if equal:
        return MatchResult(
            verdict=Verdict.MATCH,
            gold_row_count=len(gold),
            generated_row_count=len(generated),
            order_sensitive=order_sensitive,
            truncated=truncated,
        )

    return _mismatch(
        gold, generated, shown_gold, shown_generated, order_sensitive, truncated, "row_multiset"
    )


def _mismatch(
    gold: tuple[tuple[tuple[str, object], ...], ...],
    generated: tuple[tuple[tuple[str, object], ...], ...],
    shown_gold: tuple[tuple[object, ...], ...],
    shown_generated: tuple[tuple[object, ...], ...],
    order_sensitive: bool,
    truncated: bool,
    reason: str,
) -> MatchResult:
    return MatchResult(
        verdict=Verdict.MISMATCH,
        gold_row_count=len(gold),
        generated_row_count=len(generated),
        order_sensitive=order_sensitive,
        truncated=truncated,
        first_difference=_describe(shown_gold, shown_generated, order_sensitive, reason),
    )


def _describe(
    gold: tuple[tuple[tuple[str, object], ...], ...],
    generated: tuple[tuple[tuple[str, object], ...], ...],
    order_sensitive: bool,
    reason: str,
) -> dict[str, object]:
    detail: dict[str, object] = {"reason": reason, "order_sensitive": order_sensitive}

    gold_counter = Counter(gold)
    generated_counter = Counter(generated)
    missing = list((gold_counter - generated_counter).elements())[:_MAX_REPORTED_ROWS]
    extra = list((generated_counter - gold_counter).elements())[:_MAX_REPORTED_ROWS]
    detail["missing_from_generated"] = [_render(r) for r in missing]
    detail["unexpected_in_generated"] = [_render(r) for r in extra]

    if order_sensitive and gold and generated and gold[0] != generated[0]:
        detail["first_gold_row"] = _render(gold[0])
        detail["first_generated_row"] = _render(generated[0])
    return detail


def _render(row: tuple[object, ...]) -> dict[str, object]:
    """Rows must stay JSON-serializable to land in match.jsonl."""
    return {f"col{i}": cell for i, cell in enumerate(row)}
