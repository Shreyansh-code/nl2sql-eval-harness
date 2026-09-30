"""Cohen's kappa and the agreement statistics built on it.

Kappa is the credibility layer of this project, so it is implemented from the definition
rather than pulled from a library, and tested against a hand-computed 2x2 table. It is
easy to get wrong in a way that flatters the judge: off-by-one in the diagonal, or
silently dropping the categories that only one rater used.

Observed agreement is always reported next to kappa. Kappa is unreadable on its own and is
depressed by prevalence — with 11 categories and a skewed distribution, a judge that is
mostly right can still post a low kappa, and that has to be visible rather than hidden.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class KappaResult:
    kappa: float | None
    observed_agreement: float
    expected_agreement: float
    n: int
    n_categories: int
    degenerate: str | None = None

    def describe(self) -> dict[str, object]:
        return {
            "kappa": self.kappa,
            "observed_agreement": self.observed_agreement,
            "expected_agreement": self.expected_agreement,
            "n": self.n,
            "n_categories": self.n_categories,
            "degenerate": self.degenerate,
        }

    def summary(self) -> str:
        if self.kappa is None:
            return (
                f"undefined ({self.degenerate}); observed agreement {self.observed_agreement:.1%}"
            )
        return (
            f"kappa {self.kappa:.3f} (observed agreement {self.observed_agreement:.1%}, "
            f"expected {self.expected_agreement:.1%}, n={self.n})"
        )


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> KappaResult:
    """Cohen's kappa for two raters over a shared label set.

    Returns `kappa=None` with a reason when it is undefined, rather than dividing by zero
    or silently returning 0.0 — "undefined" and "no agreement" are different claims.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(f"label counts differ: {len(labels_a)} vs {len(labels_b)}")
    n = len(labels_a)
    if n == 0:
        return KappaResult(None, 0.0, 0.0, 0, 0, "no_labelled_items")

    categories = sorted(set(labels_a) | set(labels_b))
    index = {label: i for i, label in enumerate(categories)}
    size = len(categories)

    observed = 0
    for a, b in zip(labels_a, labels_b, strict=True):
        if a == b:
            observed += 1
    observed /= n

    counts_a = Counter(index[a] for a in labels_a)
    counts_b = Counter(index[b] for b in labels_b)
    expected = sum((counts_a[i] / n) * (counts_b[i] / n) for i in range(size))

    if math.isclose(expected, 1.0):
        return KappaResult(None, observed, expected, n, size, "both_raters_used_one_category")

    kappa = (observed - expected) / (1.0 - expected)
    return KappaResult(
        kappa=kappa,
        observed_agreement=observed,
        expected_agreement=expected,
        n=n,
        n_categories=size,
    )


def confusion(labels_a: Sequence[str], labels_b: Sequence[str]) -> dict[str, dict[str, int]]:
    """Full label x label table, including categories only one rater used.

    Kept in the report: with 11 categories, the off-diagonal cells are the entire story.
    """
    categories = sorted(set(labels_a) | set(labels_b))
    table = {a: {b: 0 for b in categories} for a in categories}
    for a, b in zip(labels_a, labels_b, strict=True):
        table[a][b] += 1
    return table


def binary_agreement(
    correct_flags_a: Sequence[bool], correct_flags_b: Sequence[bool]
) -> KappaResult:
    """Kappa over correctness, for judge verdict vs execution ground truth."""
    return cohen_kappa(
        ["correct" if flag else "incorrect" for flag in correct_flags_a],
        ["correct" if flag else "incorrect" for flag in correct_flags_b],
    )
