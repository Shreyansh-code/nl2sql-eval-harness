"""Recompute a report's headline numbers from the raw artifacts.

The point of this module is that nobody has to take `report.json` on faith. Accuracy is
recomputed from `match.jsonl` by an independent path, and `harness verify` fails loudly
if the two disagree — so a published number can be checked by a reviewer, or by me, six
months later, without re-running a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Recomputed:
    n_total: int
    n_scored: int
    n_correct: int
    accuracy: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_total": self.n_total,
            "n_scored": self.n_scored,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
        }


def recompute_from_matches(match_path: Path) -> Recomputed:
    """Derive accuracy from match.jsonl alone.

    A row counts toward the denominator when it was not excluded. Exclusions are
    identifiable because a scored row always carries a boolean `scored`, and the run
    records the reason for every row it dropped.
    """
    total = 0
    scored = 0
    correct = 0
    with match_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            if not row.get("scored", True):
                continue
            scored += 1
            if row.get("correct"):
                correct += 1
    accuracy = correct / scored if scored else 0.0
    return Recomputed(n_total=total, n_scored=scored, n_correct=correct, accuracy=accuracy)


def compare_to_report(recomputed: Recomputed, report: dict) -> list[str]:
    """Differences between the recomputed numbers and the stored report. Empty means agree."""
    problems: list[str] = []
    stored = report.get("exec_accuracy") or {}

    if report.get("n_total") != recomputed.n_total:
        problems.append(
            f"n_total: report {report.get('n_total')} vs recomputed {recomputed.n_total}"
        )
    if report.get("n_scored") != recomputed.n_scored:
        problems.append(
            f"n_scored: report {report.get('n_scored')} vs recomputed {recomputed.n_scored}"
        )
    if not stored:
        problems.append("report has no exec_accuracy to compare against")
        return problems

    reported = float(stored.get("point", -1.0))
    if abs(reported - recomputed.accuracy) > 1e-9:
        problems.append(f"accuracy: report {reported:.6f} vs recomputed {recomputed.accuracy:.6f}")
    return problems
