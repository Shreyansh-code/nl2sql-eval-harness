"""Which questions get judged.

The policy is the HLD's, and the reason for it is that a judge pass rate computed only on
failures is unfalsifiable: a judge that labels everything `sql_error` and explains nothing
would look excellent on that population alone. So the judge also sees a random sample of
the queries that *passed*, and its `verdict` on those is checked against execution ground
truth. Agreement there is the number that says whether the judge can be trusted at all.

Selection is seeded and recorded, so the audit sample is reproducible and not a
convenient-looking slice.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from ..pipeline import ScoredQuestion

DEFAULT_AUDIT_FRACTION = 0.20
DEFAULT_SEED = 20260930


@dataclass(frozen=True)
class JudgePlan:
    failures: tuple[str, ...]
    audit: tuple[str, ...]
    seed: int
    audit_fraction: float

    @property
    def question_ids(self) -> tuple[str, ...]:
        return self.failures + self.audit

    def describe(self) -> dict[str, object]:
        return {
            "n_failures": len(self.failures),
            "n_audit": len(self.audit),
            "seed": self.seed,
            "audit_fraction": self.audit_fraction,
        }


def plan(
    scored: Sequence[ScoredQuestion],
    *,
    audit_fraction: float = DEFAULT_AUDIT_FRACTION,
    seed: int = DEFAULT_SEED,
) -> JudgePlan:
    """Judge every scored failure, plus a seeded sample of the successes."""
    failures = [item.question.question_id for item in scored if item.scored and not item.correct]
    successes = [item.question.question_id for item in scored if item.scored and item.correct]

    count = round(len(successes) * audit_fraction)
    rng = random.Random(seed)
    audit = tuple(sorted(rng.sample(successes, count))) if successes else ()

    return JudgePlan(
        failures=tuple(failures), audit=audit, seed=seed, audit_fraction=audit_fraction
    )
