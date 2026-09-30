"""The failure taxonomy.

Fixed **before** any labelling happens, and closed: exactly one label per failure. That
ordering is not incidental. A taxonomy revised after seeing the judge's output would make
kappa uninterpretable, because the categories would have been fitted to the thing being
measured.

Eleven codes is a compromise found by asking what a human and a model can both apply
reliably. Finer distinctions (a wrong date format versus a wrong comparison operator)
would read as more rigorous and measure noise instead.
"""

from __future__ import annotations

from enum import StrEnum


class FailureMode(StrEnum):
    SCHEMA_MISREAD = "SCHEMA_MISREAD"
    WRONG_JOIN = "WRONG_JOIN"
    WRONG_FILTER = "WRONG_FILTER"
    MISSING_CONDITION = "MISSING_CONDITION"
    AGGREGATION_ERROR = "AGGREGATION_ERROR"
    SELECTION_ERROR = "SELECTION_ERROR"
    UNIT_CONVERSION = "UNIT_CONVERSION"
    PRECISION = "PRECISION"
    SQL_SYNTAX = "SQL_SYNTAX"
    TIMEOUT = "TIMEOUT"
    EMPTY_WRONG = "EMPTY_WRONG"


DESCRIPTIONS: dict[FailureMode, str] = {
    FailureMode.SCHEMA_MISREAD: (
        "Used a table or column that does not exist, or exists but means something else "
        "than the question implies (e.g. treating a name column as an identifier)."
    ),
    FailureMode.WRONG_JOIN: (
        "Chose the right tables but joined them wrongly: missing join, wrong key, or a "
        "join that multiplies rows when it should not."
    ),
    FailureMode.WRONG_FILTER: (
        "Filtered on the right column with the wrong comparison, threshold, or literal "
        "value (>, >=, <, <=, or an equality against the wrong value)."
    ),
    FailureMode.MISSING_CONDITION: (
        "Omitted a condition the question or the evidence field requires, e.g. dropping a "
        "second filter or a restriction stated in the evidence hint."
    ),
    FailureMode.AGGREGATION_ERROR: (
        "Wrong aggregate, wrong GROUP BY, or aggregation applied at the wrong level "
        "(per-row instead of per-group, or counting the wrong thing)."
    ),
    FailureMode.SELECTION_ERROR: (
        "Filtered and joined correctly but returned the wrong columns, or the wrong "
        "number of columns."
    ),
    FailureMode.UNIT_CONVERSION: (
        "Value scale or format error: percent versus fraction, epoch versus date string, "
        "unit conversion, or rounding."
    ),
    FailureMode.PRECISION: (
        "Right answer to the question, wrong only on ordering, LIMIT, or ties."
    ),
    FailureMode.SQL_SYNTAX: "The query did not execute at all.",
    FailureMode.TIMEOUT: "The query was too expensive to finish inside the time budget.",
    FailureMode.EMPTY_WRONG: (
        "The query ran and returned no rows, but the gold query returned rows."
    ),
}


def is_valid(code: str) -> bool:
    try:
        FailureMode(code)
    except ValueError:
        return False
    return True


def describe_all() -> str:
    """The taxonomy as it is given to the judge, and as it appears in the report."""
    return "\n".join(f"- {mode.value}: {DESCRIPTIONS[mode]}" for mode in FailureMode)


def coerce(code: str | None) -> FailureMode | None:
    """Map a judge's label onto the taxonomy, tolerating case and surrounding text.

    Returns None when nothing matches, so an unmappable label is *rejected* rather than
    silently bucketed — a silent fallback would inflate agreement by absorbing the
    judge's mistakes into the nearest category.
    """
    if not isinstance(code, str):
        return None
    candidate = code.strip().strip(".").upper().replace(" ", "_").replace("-", "_")
    if is_valid(candidate):
        return FailureMode(candidate)
    # Tolerate a decorated label such as "WRONG_FILTER (threshold)" or "WRONG_FILTER_".
    head = candidate.split("(")[0].strip().rstrip("_")
    return FailureMode(head) if is_valid(head) else None
