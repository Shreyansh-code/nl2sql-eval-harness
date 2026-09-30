"""Pre-execution admission checks for model-authored SQL.

The read-only connection in executor.py is the actual security boundary. This module
is the cheap, well-instrumented first gate: it rejects bad SQL before we spend a
timeout on it, and it records *why* a query was rejected so the report can distinguish
`SQL_SYNTAX` from `TIMEOUT` from `EMPTY_WRONG`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .sql_text import READ_ONLY_LEADERS, forbidden_keywords, is_single_statement, leading_keyword


@dataclass(frozen=True)
class GuardVerdict:
    allowed: bool
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.allowed


def check(sql: str) -> GuardVerdict:
    """Admit only a single, read-only statement."""
    if not sql or not sql.strip():
        return GuardVerdict(False, "empty")
    if not is_single_statement(sql):
        return GuardVerdict(False, "multi_statement")
    keyword = leading_keyword(sql)
    if keyword not in READ_ONLY_LEADERS:
        return GuardVerdict(False, f"not_read_only:{keyword or 'UNKNOWN'}")
    banned = forbidden_keywords(sql)
    if banned:
        return GuardVerdict(False, f"forbidden_keyword:{','.join(banned)}")
    return GuardVerdict(True)
