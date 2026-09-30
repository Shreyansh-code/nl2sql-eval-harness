"""Execute a single read-only SQL statement against a SQLite file, under a budget."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from . import guard


class Outcome(StrEnum):
    OK = "ok"
    EMPTY = "empty"
    REJECTED = "rejected"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class ExecResult:
    outcome: Outcome
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[object, ...], ...] = ()
    truncated: bool = False
    elapsed_ms: int = 0
    detail: str | None = None
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def produced_rows(self) -> bool:
        return bool(self.rows)


def _readonly_connection(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=1.0)
    conn.execute("PRAGMA query_only = ON")
    return conn


def execute(
    sql: str,
    db_path: Path,
    *,
    timeout_seconds: float = 5.0,
    max_rows: int = 1000,
) -> ExecResult:
    """Run `sql` read-only, capturing failure as data rather than raising.

    A malformed or expensive model query must never abort a 100-question run, so every
    failure path returns an ExecResult carrying the reason.
    """
    verdict = guard.check(sql)
    if not verdict:
        return ExecResult(outcome=Outcome.REJECTED, detail=verdict.reason)

    started = time.monotonic()
    deadline = started + timeout_seconds
    conn: sqlite3.Connection | None = None
    try:
        conn = _readonly_connection(db_path)
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 2000)

        cursor = conn.cursor()
        try:
            cursor.execute(sql)
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower():
                return _finish(Outcome.TIMEOUT, started, detail="interrupted_by_deadline")
            return _finish(Outcome.ERROR, started, detail=str(exc))
        except sqlite3.Error as exc:
            return _finish(Outcome.ERROR, started, detail=f"{type(exc).__name__}: {exc}")

        columns = tuple(d[0] for d in cursor.description or ())
        # Fetch one extra row to detect truncation without paying for the full result.
        fetched = cursor.fetchmany(max_rows + 1)
        truncated = len(fetched) > max_rows
        rows = tuple(_materialize(r) for r in fetched[:max_rows])

        if not rows:
            return _finish(Outcome.EMPTY, started, columns=columns, detail="no rows")
        return _finish(Outcome.OK, started, columns=columns, rows=rows, truncated=truncated)
    except FileNotFoundError as exc:
        return _finish(Outcome.ERROR, started, detail=f"missing_database:{exc}")
    except sqlite3.Error as exc:
        return _finish(Outcome.ERROR, started, detail=f"{type(exc).__name__}: {exc}")
    finally:
        if conn is not None:
            conn.set_progress_handler(None, 0)
            conn.close()


def _materialize(row: Sequence[object]) -> tuple[object, ...]:
    return tuple(row)


def _finish(
    outcome: Outcome,
    started: float,
    *,
    columns: tuple[str, ...] = (),
    rows: tuple[tuple[object, ...], ...] = (),
    truncated: bool = False,
    detail: str | None = None,
) -> ExecResult:
    return ExecResult(
        outcome=outcome,
        columns=columns,
        rows=rows,
        truncated=truncated,
        elapsed_ms=int((time.monotonic() - started) * 1000),
        detail=detail,
    )
