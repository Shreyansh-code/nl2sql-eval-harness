"""Append-only JSONL run artifacts, one file per pipeline stage.

Runs are immutable and self-describing: `meta.json` plus these files are enough to
regenerate the report without calling a model again. That is what makes a published
number checkable, and it is why writes are append-only and keyed by run id.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings

RUNS_DIRNAME = "runs"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def git_sha(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


@dataclass(frozen=True)
class Run:
    run_id: str
    path: Path

    def write_json(self, name: str, payload: Any) -> Path:
        target = self.path / name
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return target

    def read_json(self, name: str) -> Any:
        target = self.path / name
        if not target.exists():
            return None
        return json.loads(target.read_text(encoding="utf-8"))

    def append_rows(self, name: str, rows: Iterable[dict]) -> int:
        """Append rows to a stage file, creating it if needed. Returns rows written."""
        target = self.path / name
        with target.open("a", encoding="utf-8") as handle:
            count = 0
            for row in rows:
                handle.write(json.dumps(row, default=str) + "\n")
                count += 1
        return count

    def read_rows(self, name: str) -> list[dict]:
        return list(iter_rows(self.path / name))

    def has(self, name: str) -> bool:
        return (self.path / name).exists()


def iter_rows(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def open_run(
    settings: Settings,
    stage: str,
    *,
    run_id: str | None = None,
    root: Path | None = None,
) -> Run:
    """Create (or reopen) a run directory and record what produced it.

    `stage` is a suffix on the id so the generation of a run and its judging pass are
    distinguishable at a glance in `runs/`.
    """
    base = root or (Path(__file__).resolve().parents[2] / RUNS_DIRNAME)
    identifier = run_id or f"{_now()}-{stage}"
    path = base / identifier
    path.mkdir(parents=True, exist_ok=True)

    run = Run(run_id=identifier, path=path)
    if not run.has("meta.json"):
        run.write_json(
            "meta.json",
            {
                "run_id": identifier,
                "created_at": datetime.now(UTC).isoformat(),
                "git_sha": git_sha(path.parent.parent),
                "config": settings.redacted(),
            },
        )
    return run
