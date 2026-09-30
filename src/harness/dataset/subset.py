"""The subset manifest: the reproducibility contract for every number we publish."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SubsetSpec:
    name: str
    source: str
    db_ids: tuple[str, ...]
    limit_per_db: int
    sampling: str
    question_ids: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source": self.source,
            "db_ids": list(self.db_ids),
            "limit_per_db": self.limit_per_db,
            "sampling": self.sampling,
            "question_ids": list(self.question_ids),
        }


def load_subset(path: Path) -> SubsetSpec:
    if not path.exists():
        raise FileNotFoundError(
            f"subset manifest not found: {path}\n"
            "Copy subset.example.yaml to subset.yaml and set the db ids you want."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [key for key in ("name", "source", "db_ids") if key not in raw]
    if missing:
        raise ValueError(f"subset manifest {path} is missing keys: {missing}")

    return SubsetSpec(
        name=str(raw["name"]),
        source=str(raw["source"]),
        db_ids=tuple(str(db) for db in raw["db_ids"]),
        limit_per_db=int(raw.get("limit_per_db", 0)) or 10**9,
        sampling=str(raw.get("sampling", "contiguous_prefix_unspecified")),
        question_ids=tuple(str(q) for q in raw.get("question_ids", []) or []),
        notes=str(raw.get("notes", "")),
    )
