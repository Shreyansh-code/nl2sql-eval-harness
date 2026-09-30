"""Pull exactly one SQL statement out of a model response.

Strict on purpose. A harness that silently accepts the first plausible-looking fragment
inflates accuracy, and when a number is going on a resume it has to survive someone
reading this function. Anything ambiguous is reported as a failure with a reason rather
than guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..execution.guard import check
from ..execution.sql_text import is_single_statement, leading_keyword

_FENCE_RE = re.compile(r"```(?:sql|sqlite)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_LEADING_LABEL_RE = re.compile(r"^\s*(?:sql|sqlite|query|answer)\s*:\s*", re.IGNORECASE)
_BARE_START_RE = re.compile(r"^\s*(?:SELECT|WITH|VALUES)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ParseResult:
    sql: str | None
    reason: str | None = None
    blocks_seen: int = 0

    @property
    def ok(self) -> bool:
        return self.sql is not None


def parse_sql(response: str) -> ParseResult:
    """Extract a single read-only statement, or explain why we could not."""
    if not response or not response.strip():
        return ParseResult(sql=None, reason="empty_response")

    blocks = [block.strip() for block in _FENCE_RE.findall(response)]
    blocks = [block for block in blocks if block]
    candidates = blocks if blocks else [_from_bare(response)]

    for candidate in candidates:
        cleaned = _clean(candidate)
        if not cleaned:
            continue
        if not _BARE_START_RE.match(cleaned):
            return ParseResult(sql=None, reason="not_a_select", blocks_seen=len(blocks))
        if not is_single_statement(cleaned):
            return ParseResult(sql=None, reason="multi_statement", blocks_seen=len(blocks))
        verdict = check(cleaned)
        if not verdict:
            return ParseResult(sql=None, reason=f"guard:{verdict.reason}", blocks_seen=len(blocks))
        return ParseResult(sql=cleaned, blocks_seen=len(blocks))

    return ParseResult(sql=None, reason="no_sql_found", blocks_seen=len(blocks))


def _from_bare(response: str) -> str:
    """Fallback for a response with no fence: take the leading statement, if any."""
    cleaned = _clean(response)
    if not _BARE_START_RE.match(cleaned):
        return ""
    lines: list[str] = []
    for line in cleaned.splitlines():
        lines.append(line)
        if line.rstrip().endswith(";"):
            break
    return "\n".join(lines)


def _clean(candidate: str) -> str:
    text = _LEADING_LABEL_RE.sub("", candidate.strip())
    # Drop a leading SQL comment banner the model sometimes adds.
    while text.startswith("--"):
        _, _, text = text.partition("\n")
    text = text.strip()
    if text.endswith(";"):
        text = text[:-1]
    return text.strip()


def summarize_parse_failure(reason: str | None) -> str:
    """Short label for the report; keeps free-text reasons out of the metrics."""
    if not reason:
        return "unknown"
    if reason.startswith("guard:"):
        return f"guard_rejected:{reason.split(':', 1)[1]}"
    return reason


def is_executable(sql: str) -> bool:
    return leading_keyword(sql) in {"SELECT", "WITH", "VALUES", "EXPLAIN"}
