"""M1 gate: prove the harness scores a known-correct query at 100%.

If this fails, every later number is untrustworthy, so it runs before any LLM call.
Three properties are checked per question:

1. the gold query executes at all (`ok` or `empty`);
2. executing it twice yields byte-identical rows (the executor is deterministic);
3. gold vs gold compares as MATCH (the comparator does not lie).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..dataset.models import HarnessQuestion
from ..execution.executor import ExecResult, Outcome, execute
from .exec_match import MatchResult, compare


@dataclass(frozen=True)
class BaselineRow:
    question_id: str
    db_id: str
    first: ExecResult
    second: ExecResult
    match: MatchResult

    @property
    def executed(self) -> bool:
        return self.first.outcome in {Outcome.OK, Outcome.EMPTY}

    @property
    def deterministic(self) -> bool:
        return self.first.rows == self.second.rows and self.first.outcome == self.second.outcome

    @property
    def self_match(self) -> bool:
        return self.match.is_match

    @property
    def passed(self) -> bool:
        return self.executed and self.deterministic and self.self_match


@dataclass(frozen=True)
class BaselineReport:
    rows: tuple[BaselineRow, ...]

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def passed(self) -> int:
        return sum(1 for row in self.rows if row.passed)

    @property
    def failures(self) -> tuple[BaselineRow, ...]:
        return tuple(row for row in self.rows if not row.passed)

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failures": [
                {
                    "question_id": row.question_id,
                    "db_id": row.db_id,
                    "outcome": str(row.first.outcome),
                    "executed": row.executed,
                    "deterministic": row.deterministic,
                    "self_match": row.self_match,
                    "detail": row.first.detail,
                }
                for row in self.failures
            ],
        }


def run_baseline(
    questions: Sequence[HarnessQuestion],
    *,
    timeout_seconds: float = 5.0,
    max_rows: int = 1000,
) -> BaselineReport:
    rows: list[BaselineRow] = []
    for question in questions:
        first = execute(
            question.gold_sql, question.db_path, timeout_seconds=timeout_seconds, max_rows=max_rows
        )
        second = execute(
            question.gold_sql, question.db_path, timeout_seconds=timeout_seconds, max_rows=max_rows
        )
        rows.append(
            BaselineRow(
                question_id=question.question_id,
                db_id=question.db_id,
                first=first,
                second=second,
                match=compare(
                    question.gold_sql, first.rows, second.rows, truncated=first.truncated
                ),
            )
        )
    return BaselineReport(rows=tuple(rows))


def gate(report: BaselineReport) -> None:
    """Raise unless every gold query executed, was deterministic, and self-matched."""
    if report.passed == report.total and report.total:
        return
    detail = "\n".join(
        f"  {row.question_id} ({row.db_id}): outcome={row.first.outcome} "
        f"deterministic={row.deterministic} self_match={row.self_match} detail={row.first.detail}"
        for row in report.failures
    )
    raise AssertionError(
        f"baseline gate failed: {report.passed}/{report.total} gold queries did not "
        f"round-trip cleanly\n{detail}"
    )
