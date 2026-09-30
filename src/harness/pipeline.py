"""Score generated SQL: execute it, compare it, and write the stage artifacts.

This is where a run stops being a pile of model output and becomes a measurement. The
only correctness signal is `ExecMatch` over actually-executed SQL; nothing here consults
a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .config import Settings
from .dataset.models import HarnessQuestion
from .execution.executor import ExecResult, Outcome, execute
from .generation.port import Generation
from .runs import Run
from .scoring.exec_match import MatchResult, Verdict, compare

GENERATIONS = "generations.jsonl"
EXECUTIONS = "executions.jsonl"
MATCHES = "match.jsonl"


@dataclass(frozen=True)
class ScoredQuestion:
    question: HarnessQuestion
    generation: Generation
    execution: ExecResult
    match: MatchResult
    gold: ExecResult | None = None

    @property
    def gold_broken(self) -> bool:
        """Gold failed to run, so the question cannot be scored at all."""
        return self.gold is not None and self.gold.outcome in {Outcome.ERROR, Outcome.REJECTED}

    @property
    def scored(self) -> bool:
        """Whether this question belongs in the accuracy denominator.

        Two exclusions, both from the HLD: a gold query that does not run, and a
        result truncated at the row cap that we cannot prove equal.
        """
        return not self.gold_broken and self.match.scored

    @property
    def correct(self) -> bool:
        return self.scored and self.match.is_match

    @property
    def skip_reason(self) -> str | None:
        if self.gold_broken:
            return "gold_did_not_execute"
        if not self.match.scored:
            return "inconclusive_result"
        return None


def score_generations(
    questions: Sequence[HarnessQuestion],
    generations: Sequence[Generation],
    settings: Settings,
) -> list[ScoredQuestion]:
    """Execute each generated query and compare its rows to the gold rows.

    Gold is executed too, in the same way, so a question whose gold query breaks gets
    excluded from the denominator instead of being counted as a model failure.
    """
    by_id = {q.question_id: q for q in questions}
    scored: list[ScoredQuestion] = []

    for generation in generations:
        question = by_id[generation.question_id]
        if not generation.usable:
            # Never executed. An empty result with a recorded reason is the honest
            # representation, and it is scored as a failure: the model did not produce
            # a runnable query, which is a real failure and must stay in the denominator.
            execution = ExecResult(
                outcome=Outcome.EMPTY if not generation.sql else Outcome.ERROR,
                detail=f"generation_{generation.status}:{generation.error or 'unspecified'}",
            )
            scored.append(ScoredQuestion(question, generation, execution, _failed_match(execution)))
            continue

        execution = execute(
            generation.sql or "",
            question.db_path,
            timeout_seconds=settings.sql_timeout_seconds,
            max_rows=settings.max_rows,
        )
        gold = execute(
            question.gold_sql,
            question.db_path,
            timeout_seconds=settings.sql_timeout_seconds,
            max_rows=settings.max_rows,
        )
        match = compare(
            question.gold_sql,
            gold.rows,
            execution.rows,
            truncated=execution.truncated,
        )
        scored.append(ScoredQuestion(question, generation, execution, match, gold))

    return scored


def _failed_match(execution: ExecResult) -> MatchResult:
    return MatchResult(
        verdict=Verdict.MISMATCH,
        gold_row_count=0,
        generated_row_count=0,
        order_sensitive=False,
        first_difference={"reason": execution.detail or "not_executed"},
    )


def persist(run: Run, scored: Sequence[ScoredQuestion]) -> dict[str, int]:
    run.append_rows(GENERATIONS, (item.generation.to_row() for item in scored))
    run.append_rows(
        EXECUTIONS,
        (
            {
                "question_id": item.question.question_id,
                "db_id": item.question.db_id,
                "outcome": str(item.execution.outcome),
                "row_count": len(item.execution.rows),
                "truncated": item.execution.truncated,
                "elapsed_ms": item.execution.elapsed_ms,
                "detail": item.execution.detail,
                "sample_rows": [list(row[:8]) for row in item.execution.rows[:5]],
            }
            for item in scored
        ),
    )
    run.append_rows(
        MATCHES,
        (
            {
                "question_id": item.question.question_id,
                "db_id": item.question.db_id,
                "difficulty": item.question.difficulty,
                "verdict": str(item.match.verdict),
                "correct": item.correct,
                "scored": item.scored,
                "order_sensitive": item.match.order_sensitive,
                "gold_row_count": item.match.gold_row_count,
                "generated_row_count": item.match.generated_row_count,
                "difference": item.match.first_difference,
            }
            for item in scored
        ),
    )
    return {
        "generations": len(scored),
        "executions": len(scored),
        "matches": len(scored),
    }
