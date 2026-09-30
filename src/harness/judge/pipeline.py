"""Run the judge over a scored run and write judgements.jsonl.

Correctness ground truth enters here from execution, never from the judge: the judge's
`verdict` is *compared against* `ScoredQuestion.correct`, and that comparison is a metric.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ..config import Settings
from ..dataset.models import HarnessQuestion
from ..execution.executor import execute
from ..generation.port import Generation
from ..pipeline import ScoredQuestion
from ..runs import Run
from ..scoring.exec_match import compare
from .graph import JudgeGraph
from .port import Judgement, JudgeStatus
from .sampling import JudgePlan
from .sampling import plan as plan_judging

JUDGEMENTS = "judgements.jsonl"


@dataclass(frozen=True)
class JudgeOutcome:
    judgements: tuple[Judgement, ...]
    plan: JudgePlan

    @property
    def usable(self) -> tuple[Judgement, ...]:
        return tuple(j for j in self.judgements if j.status is JudgeStatus.OK)

    @property
    def unscored(self) -> tuple[Judgement, ...]:
        return tuple(j for j in self.judgements if j.status is not JudgeStatus.OK)

    def status_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(str(j.status) for j in self.judgements).items()))


def run_judge(
    scored: Sequence[ScoredQuestion],
    graph: JudgeGraph,
    *,
    audit_fraction: float = 0.20,
    seed: int = 20260930,
) -> JudgeOutcome:
    """Judge every scored failure plus the seeded audit sample of successes."""
    judge_plan = plan_judging(scored, audit_fraction=audit_fraction, seed=seed)
    wanted = set(judge_plan.question_ids)
    targets = [item for item in scored if item.question.question_id in wanted]
    return JudgeOutcome(judgements=tuple(graph.judge(item) for item in targets), plan=judge_plan)


def persist(run: Run, outcome: JudgeOutcome) -> int:
    """Write judgements as a deduplicated snapshot, not an append log.

    Unlike generations, a judging pass is a *complete set* over its inputs, so re-running
    it must overwrite rather than accumulate. Appending produced two copies of every row
    on a re-judge, which silently doubled every count in the report — the kind of bug that
    `harness verify` exists to catch, and which it did catch.
    """
    by_key: dict[str, dict] = {}
    for judgement in outcome.judgements:
        by_key[judgement.cache_key] = judgement.to_row()
    rows = list(by_key.values())
    run.path.mkdir(parents=True, exist_ok=True)
    (run.path / JUDGEMENTS).write_text(
        "".join(json.dumps(row, default=str) + "\n" for row in rows), encoding="utf-8"
    )
    return len(rows)


def reload_scored(
    questions: Sequence[HarnessQuestion],
    generations: Sequence[Generation],
    matches: Sequence[dict],
    settings: Settings,
) -> list[ScoredQuestion]:
    """Rebuild ScoredQuestions from a finished run's artifacts, for judging it.

    Both queries are executed again rather than trusting `executions.jsonl`: the row
    samples stored there are truncated for readability, and the judge is shown real rows.
    The stored `correct` flag is compared against the recomputed verdict, and a
    disagreement invalidates the run — a judge shown a diff that does not match the
    recorded one is being shown a fiction.
    """
    by_question = {q.question_id: q for q in questions}
    by_generation = {g.question_id: g for g in generations}

    rebuilt: list[ScoredQuestion] = []
    disagreements: list[str] = []
    for row in matches:
        question_id = row["question_id"]
        question = by_question.get(question_id)
        generation = by_generation.get(question_id)
        if question is None or generation is None:
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
        match = compare(question.gold_sql, gold.rows, execution.rows, truncated=execution.truncated)
        if bool(row.get("correct")) != match.is_match:
            disagreements.append(question_id)
        rebuilt.append(ScoredQuestion(question, generation, execution, match, gold))

    if disagreements:
        raise ValueError(
            f"{len(disagreements)} recomputed verdicts disagree with match.jsonl "
            f"(first: {disagreements[:5]}). Judging a run whose stored results do not "
            "reproduce would be measuring a fiction."
        )
    return rebuilt
