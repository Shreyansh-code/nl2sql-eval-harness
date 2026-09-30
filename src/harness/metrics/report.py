"""Metrics and the report.

Everything here is derived from the run's JSONL artifacts. No number is ever hand-entered
— the README quotes `report.json`, and `report.json` quotes these files. Confidence
intervals are included because a 90-question accuracy reported as a bare percentage is a
marketing claim, not a measurement.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from ..generation.port import GenerationStatus
from ..pipeline import ScoredQuestion

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260930


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def as_percent(self) -> str:
        return f"{self.point * 100:.1f}% (95% CI {self.low * 100:.1f}-{self.high * 100:.1f})"

    def to_dict(self) -> dict[str, float]:
        return {"point": self.point, "ci_low": self.low, "ci_high": self.high}


def bootstrap_interval(
    flags: Sequence[bool], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> Interval | None:
    """Percentile bootstrap CI for a proportion.

    Seeded, so the same run always prints the same interval: a report whose confidence
    bound moves between renders is not reproducible.
    """
    n = len(flags)
    if n == 0:
        return None
    point = sum(1 for f in flags if f) / n
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        draw = sum(1 for _ in range(n) if flags[rng.randrange(n)])
        means.append(draw / n)
    means.sort()
    low = means[int(0.025 * (resamples - 1))]
    high = means[int(0.975 * (resamples - 1))]
    return Interval(point=point, low=low, high=high)


@dataclass(frozen=True)
class SliceStat:
    """Accuracy for one slice, with the n that produced it.

    n travels with the interval on purpose: a percentage without its denominator is the
    thing this whole project exists to avoid.
    """

    n: int
    interval: Interval | None

    def to_dict(self) -> dict[str, object]:
        return {"n": self.n, "accuracy": self.interval.to_dict() if self.interval else None}


@dataclass(frozen=True)
class Report:
    n_total: int
    n_scored: int
    n_skipped: int
    accuracy: Interval | None
    per_db: dict[str, SliceStat]
    per_difficulty: dict[str, SliceStat]
    outcome_counts: dict[str, int]
    generation_status_counts: dict[str, int]
    token_totals: dict[str, int]
    skip_reasons: dict[str, int]
    api_errors: int

    @property
    def valid(self) -> bool:
        """Whether this run may be reported as a measurement at all.

        A transport failure is not a model failure. If any request failed, the accuracy
        number is measuring the harness, not the generator, and must not be quoted.
        """
        return self.api_errors == 0

    def invalid_reason(self) -> str | None:
        if self.valid:
            return None
        return (
            f"{self.api_errors} of {self.n_total} generations failed at the API. "
            "Accuracy from this run is not a valid measurement."
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "invalid_reason": self.invalid_reason(),
            "n_total": self.n_total,
            "n_scored": self.n_scored,
            "n_skipped": self.n_skipped,
            "api_errors": self.api_errors,
            "exec_accuracy": self.accuracy.to_dict() if self.accuracy else None,
            "per_db": {k: v.to_dict() for k, v in self.per_db.items()},
            "per_difficulty": {k: v.to_dict() for k, v in self.per_difficulty.items()},
            "execution_outcomes": self.outcome_counts,
            "generation_statuses": self.generation_status_counts,
            "tokens": self.token_totals,
            "skipped": self.skip_reasons,
        }


def build_report(scored: Sequence[ScoredQuestion]) -> Report:
    evaluable = [item for item in scored if item.scored]
    flags = [item.correct for item in evaluable]

    per_db: dict[str, list[bool]] = defaultdict(list)
    per_difficulty: dict[str, list[bool]] = defaultdict(list)
    for item in evaluable:
        per_db[item.question.db_id].append(item.correct)
        per_difficulty[item.question.difficulty or "unknown"].append(item.correct)

    return Report(
        n_total=len(scored),
        n_scored=len(evaluable),
        n_skipped=len(scored) - len(evaluable),
        accuracy=bootstrap_interval(flags),
        per_db={k: SliceStat(len(v), bootstrap_interval(v)) for k, v in sorted(per_db.items())},
        per_difficulty={
            k: SliceStat(len(v), bootstrap_interval(v)) for k, v in sorted(per_difficulty.items())
        },
        outcome_counts=dict(Counter(str(i.execution.outcome) for i in scored).most_common()),
        generation_status_counts=dict(
            Counter(str(i.generation.status) for i in scored).most_common()
        ),
        token_totals={
            "input": sum(i.generation.input_tokens for i in scored),
            "output": sum(i.generation.output_tokens for i in scored),
        },
        skip_reasons=dict(Counter(i.skip_reason for i in scored if i.skip_reason).most_common()),
        api_errors=sum(1 for i in scored if i.generation.status is GenerationStatus.API_ERROR),
    )


def render_markdown(report: Report, meta: dict) -> str:
    """The human-readable report. This is what the README quotes."""
    lines: list[str] = []
    config = meta.get("config", {})

    lines.append("# Run report")
    lines.append("")
    lines.append(f"- run: `{meta.get('run_id')}`")
    lines.append(f"- git: `{meta.get('git_sha')}`")
    lines.append(f"- generator: `{config.get('generator_model')}` ({config.get('generator_mode')})")
    lines.append(f"- subset: `{meta.get('subset')}`")
    lines.append("")

    if not report.valid:
        lines.append("> **INVALID RUN.** " + (report.invalid_reason() or ""))
        lines.append("")

    lines.append("## Execution accuracy")
    lines.append("")
    if report.accuracy is None:
        lines.append("No scored questions.")
    else:
        lines.append(
            f"**{report.accuracy.as_percent()}** on n={report.n_scored} "
            f"({report.n_skipped} excluded)."
        )
        lines.append("")
        lines.append("| Slice | Accuracy | n |")
        lines.append("|---|---|---|")
        for label, table in (("database", report.per_db), ("difficulty", report.per_difficulty)):
            for key, stat in table.items():
                if stat.interval is None:
                    continue
                lines.append(f"| {label}: {key} | {stat.interval.as_percent()} | {stat.n} |")
    lines.append("")

    lines.append("## Generation outcomes")
    lines.append("")
    for status, count in report.generation_status_counts.items():
        lines.append(f"- {status}: {count}")
    lines.append("")

    lines.append("## Execution outcomes")
    lines.append("")
    for outcome, count in report.outcome_counts.items():
        lines.append(f"- {outcome}: {count}")
    lines.append("")

    if report.skip_reasons:
        lines.append("## Excluded from the denominator")
        lines.append("")
        for reason, count in report.skip_reasons.items():
            lines.append(f"- {reason}: {count}")
        lines.append("")

    lines.append("## Tokens")
    lines.append("")
    lines.append(f"- input: {report.token_totals['input']}")
    lines.append(f"- output: {report.token_totals['output']}")
    lines.append("")

    lines.append("## Not yet measured")
    lines.append("")
    lines.append(
        "Failure-mode classification and judge-vs-human agreement (M3/M4) are absent. "
        "No diagnostic claim is made by this report."
    )
    lines.append("")
    return "\n".join(lines)
