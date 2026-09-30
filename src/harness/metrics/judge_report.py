"""Judge metrics: the failure histogram and the judge's own reliability.

Two different claims live here and must not be confused:

* the **failure histogram** is the project's finding — what the generator actually gets
  wrong, in the generator's own voice, not the judge's;
* the **agreement statistics** are the credibility layer — how much that finding can be
  trusted, measured two ways: judge verdict against execution ground truth (available as
  soon as M3 runs) and judge failure mode against human labels (M4).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..judge.pipeline import JudgeOutcome
from ..judge.port import Judgement, JudgeStatus, Verdict
from ..judge.taxonomy import DESCRIPTIONS, FailureMode
from ..pipeline import ScoredQuestion
from .agreement import KappaResult, binary_agreement, cohen_kappa, confusion


@dataclass(frozen=True)
class JudgeReport:
    n_judged: int
    n_usable: int
    n_unscored: int
    status_counts: dict[str, int]
    failure_histogram: dict[str, int]
    failure_share: dict[str, float]
    n_failures_judged: int
    verdict_kappa: KappaResult | None
    mode_kappa: KappaResult | None
    confusion_table: dict[str, dict[str, int]] = field(default_factory=dict)
    low_confidence: int = 0
    false_accusations: int = 0
    missed_failures: int = 0
    n_judge_says_ok: int = 0
    n_execution_says_ok: int = 0
    n_scored_total: int | None = None

    @property
    def covers_whole_run(self) -> bool:
        """Whether every scored question was judged, or only failures plus an audit sample."""
        return self.n_scored_total is not None and self.n_judged == self.n_scored_total

    @property
    def judge_standard_accuracy(self) -> float | None:
        """Accuracy the judge's own standard would give, on the judged items.

        A second number next to execution accuracy, not a replacement for it. The gap
        between the two *is* the finding: it measures how much the choice of correctness
        definition moves the score.

        Returned only for a full-run pass. On a failures-plus-audit sample the population
        is deliberately failure-weighted, so an "accuracy" computed on it would not be
        comparable to the run's accuracy and would invite exactly the wrong reading.
        """
        if not self.n_judged or not self.covers_whole_run:
            return None
        return self.n_judge_says_ok / self.n_judged

    def to_dict(self) -> dict[str, object]:
        return {
            "n_judged": self.n_judged,
            "n_usable": self.n_usable,
            "n_unscored": self.n_unscored,
            "statuses": self.status_counts,
            "failure_histogram": self.failure_histogram,
            "failure_share": self.failure_share,
            "n_failures_judged": self.n_failures_judged,
            "judge_vs_execution_kappa": (
                self.verdict_kappa.describe() if self.verdict_kappa else None
            ),
            "judge_vs_human_mode_kappa": (self.mode_kappa.describe() if self.mode_kappa else None),
            "confusion": self.confusion_table,
            "low_confidence": self.low_confidence,
            "disagreement": {
                "false_accusations": self.false_accusations,
                "missed_failures": self.missed_failures,
            },
            "judge_standard_accuracy": self.judge_standard_accuracy,
            "covers_whole_run": self.covers_whole_run,
        }


def build_judge_report(
    outcome: JudgeOutcome,
    scored: Sequence[ScoredQuestion],
    *,
    human_labels: dict[str, str] | None = None,
    low_confidence_below: float = 0.5,
) -> JudgeReport:
    correct_by_id = {item.question.question_id: item.correct for item in scored}
    usable = outcome.usable

    histogram = Counter(
        str(j.failure_mode)
        for j in usable
        if j.failure_mode and not _judged_success(j, correct_by_id)
    )
    total = sum(histogram.values())
    ordered = {
        mode.value: histogram.get(mode.value, 0)
        for mode in FailureMode
        if histogram.get(mode.value)
    }

    return JudgeReport(
        n_judged=len(outcome.judgements),
        n_usable=len(usable),
        n_unscored=len(outcome.unscored),
        status_counts=outcome.status_counts(),
        failure_histogram=ordered,
        failure_share={key: (value / total if total else 0.0) for key, value in ordered.items()},
        n_failures_judged=total,
        verdict_kappa=_verdict_kappa(usable, correct_by_id),
        mode_kappa=_mode_kappa(usable, correct_by_id, human_labels or {}),
        confusion_table=_confusion_for_human(usable, correct_by_id, human_labels or {}),
        low_confidence=sum(
            1 for j in usable if j.confidence is not None and j.confidence < low_confidence_below
        ),
        n_scored_total=sum(1 for item in scored if item.scored),
        **_disagreement(usable, correct_by_id),
    )


def _disagreement(usable: Sequence[Judgement], correct_by_id: dict[str, bool]) -> dict[str, int]:
    """Split the judge's errors by direction.

    Direction is the whole story. A judge that cries wolf on correct queries is
    unusable; a judge that is merely lenient on wrong ones is a calibration problem with a
    known direction, and the two need completely different fixes. Reporting a single
    accuracy would hide that.
    """
    judged = [j for j in usable if j.verdict is not None and j.question_id in correct_by_id]
    return {
        "false_accusations": sum(
            1 for j in judged if j.verdict is not Verdict.SQL_OK and correct_by_id[j.question_id]
        ),
        "missed_failures": sum(
            1 for j in judged if j.verdict is Verdict.SQL_OK and not correct_by_id[j.question_id]
        ),
        "n_judge_says_ok": sum(1 for j in judged if j.verdict is Verdict.SQL_OK),
        "n_execution_says_ok": sum(1 for j in judged if correct_by_id[j.question_id]),
    }


def _judged_success(judgement: Judgement, correct_by_id: dict[str, bool]) -> bool:
    """True when this judged question was actually a success (the audit sample)."""
    return bool(correct_by_id.get(judgement.question_id))


def _verdict_kappa(
    usable: Sequence[Judgement], correct_by_id: dict[str, bool]
) -> KappaResult | None:
    """Does the judge agree with execution about correctness?

    This is the load-bearing reliability number and it needs no human labels: the judge is
    asked whether the query is correct, execution knows, and the two are compared on the
    audit sample of successes plus every failure.
    """
    judged = [j for j in usable if j.verdict is not None and j.question_id in correct_by_id]
    if not judged:
        return None
    # `sql_ok` is the only verdict that means correct, so this is the whole mapping. Written
    # as a negation it would be a silent sign error waiting to flatter the judge.
    judge_says_correct = [j.verdict is Verdict.SQL_OK for j in judged]
    truth = [correct_by_id[j.question_id] for j in judged]
    return binary_agreement(judge_says_correct, truth)


def _mode_kappa(
    usable: Sequence[Judgement],
    correct_by_id: dict[str, bool],
    human_labels: dict[str, str],
) -> KappaResult | None:
    """Judge failure mode vs human failure mode. Needs M4's labels; None until then."""
    if not human_labels:
        return None
    paired = [
        (str(j.failure_mode), human_labels[j.question_id])
        for j in usable
        if j.failure_mode is not None
        and not _judged_success(j, correct_by_id)
        and j.question_id in human_labels
    ]
    if not paired:
        return None
    judge_labels, human = zip(*paired, strict=True)
    return cohen_kappa(list(judge_labels), list(human))


def _confusion_for_human(
    usable: Sequence[Judgement],
    correct_by_id: dict[str, bool],
    human_labels: dict[str, str],
) -> dict[str, dict[str, int]]:
    if not human_labels:
        return {}
    judge_labels, human = [], []
    for j in usable:
        if (
            j.failure_mode is not None
            and not _judged_success(j, correct_by_id)
            and j.question_id in human_labels
        ):
            judge_labels.append(str(j.failure_mode))
            human.append(human_labels[j.question_id])
    return confusion(judge_labels, human) if judge_labels else {}


def render_judge_markdown(report: JudgeReport, plan_description: dict, meta: dict) -> str:
    lines: list[str] = []
    lines.append("# Judge report")
    lines.append("")
    lines.append(f"- run: `{meta.get('run_id')}`")
    lines.append(f"- judge: `{meta.get('judge_model')}`")
    lines.append(
        f"- judged: {report.n_judged} "
        f"({plan_description.get('n_failures')} failures + "
        f"{plan_description.get('n_audit')} audited successes)"
    )
    lines.append("")

    lines.append("## Judge output health")
    lines.append("")
    for status, count in report.status_counts.items():
        lines.append(f"- {status}: {count}")
    lines.append("")
    if report.n_unscored:
        lines.append(
            f"{report.n_unscored} judgement(s) were excluded from every statistic below. "
            "An unusable judge output is counted, never quietly dropped."
        )
        lines.append("")

    lines.append("## Failure modes (generator's failures only)")
    lines.append("")
    if not report.failure_histogram:
        lines.append("No classified failures.")
    else:
        lines.append("| Mode | Count | Share | Meaning |")
        lines.append("|---|---|---|---|")
        for mode, count in report.failure_histogram.items():
            share = report.failure_share.get(mode, 0.0)
            meaning = DESCRIPTIONS[FailureMode(mode)].split(".")[0]
            lines.append(f"| `{mode}` | {count} | {share:.0%} | {meaning} |")
    lines.append("")

    lines.append("## Is the judge trustworthy?")
    lines.append("")
    lines.append(
        "Judge verdict vs execution ground truth, on every failure plus the audit sample "
        "of successes. No human labels needed."
    )
    lines.append("")
    if report.verdict_kappa is None:
        lines.append("Not enough judged items to compare against execution ground truth.")
    else:
        lines.append(f"- {report.verdict_kappa.summary()}")
        lines.append("")
        lines.append("### The disagreement is one-directional")
        lines.append("")
        lines.append("| | execution says correct | execution says wrong |")
        lines.append("|---|---|---|")
        agreed_correct = report.n_execution_says_ok - report.false_accusations
        lines.append(
            f"| judge says correct (`sql_ok`) | {agreed_correct} | {report.missed_failures} |"
        )
        lines.append(
            f"| judge says wrong | {report.false_accusations} | "
            f"{report.n_judged - report.n_execution_says_ok - report.missed_failures} |"
        )
        lines.append("")
        lines.append(
            f"**{report.false_accusations} false accusation(s)** — the judge never called a "
            f"correct query wrong. **{report.missed_failures} missed failure(s)** — the judge "
            "called a wrong query correct."
        )
        lines.append("")
        lines.append(
            "Direction is the actionable part. A judge that accused correct queries would be "
            "unusable; a judge that is only lenient on wrong ones is a calibration problem "
            "with a known direction, and the fix is different. Note also that the failure "
            "histogram above is built only from the failures the judge *did* label, so it "
            f"describes {report.n_failures_judged} of the failures, not all of them."
        )
        lines.append("")
        if report.judge_standard_accuracy is not None:
            lines.append(
                f"Every scored question was judged, so the two standards can be compared "
                f"directly on the same population: strict execution accuracy is "
                f"{report.n_execution_says_ok / report.n_judged:.1%}, and the judge's own "
                f"standard would give {report.judge_standard_accuracy:.1%}. The gap is the "
                "cost of choosing strict result-set equality over semantic equivalence."
            )
            lines.append("")
        else:
            lines.append(
                "Only failures plus an audit sample were judged, which is deliberately "
                "failure-weighted, so no judge-standard accuracy is quoted here: it would "
                "not be comparable to the run's accuracy. Re-run with `--all` to compare "
                "the two standards on the same population."
            )
            lines.append("")
        lines.append(
            "**Read this narrowly.** The judge is shown the gold SQL and the gold rows, so "
            "recognising a correct query is easier than detecting a wrong one blind. This "
            "number says the judge's verdicts are consistent with ground truth when it can "
            "see ground truth; it is not a blind error-detection score, and a high value "
            "here is not evidence that the judge could diagnose errors unaided."
        )
    lines.append("")

    lines.append("## Judge vs human (failure mode)")
    lines.append("")
    if report.mode_kappa is None:
        lines.append(
            "Not yet measured: this needs the 50 blinded human labels from M4. "
            "No judge-vs-human agreement is claimed."
        )
    else:
        lines.append(f"- {report.mode_kappa.summary()}")
        lines.append("")
        lines.append("| judge \\ human | " + " | ".join(sorted(report.confusion_table)) + " |")
        for judge_label, row in sorted(report.confusion_table.items()):
            cells = " | ".join(str(row.get(h, 0)) for h in sorted(report.confusion_table))
            lines.append(f"| `{judge_label}` | {cells} |")
    lines.append("")

    if report.low_confidence:
        lines.append(f"{report.low_confidence} usable judgement(s) came back below 0.5 confidence.")
        lines.append("")

    return "\n".join(lines)


def unscored_reasons(outcome: JudgeOutcome) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                j.error or "unknown" for j in outcome.judgements if j.status is not JudgeStatus.OK
            ).items()
        )
    )
