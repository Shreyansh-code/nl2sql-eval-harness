"""Load BIRD dev questions and attach an introspected schema to each one."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .models import HarnessQuestion, SchemaPayload
from .schema import introspect
from .subset import SubsetSpec

BIRD_DEV_JSON = "dev.json"
BIRD_DB_DIRNAME = "dev_databases"

# BIRD's dev.json spells the gold query key `SQL`. Accept the snake_case spelling too so
# the loader survives a re-dump of the dataset.
_GOLD_KEYS = ("SQL", "sql", "gold_sql")


class DataMissingError(RuntimeError):
    pass


@lru_cache(maxsize=32)
def _schema_for(db_id: str, db_path_str: str) -> SchemaPayload:
    result = introspect(Path(db_path_str))
    return SchemaPayload(
        db_id=db_id, tables=result.tables, ddl=result.ddl, column_values=result.column_values
    )


def database_path(data_dir: Path, db_id: str) -> Path:
    return data_dir / BIRD_DB_DIRNAME / db_id / f"{db_id}.sqlite"


def load_questions(data_dir: Path, subset: SubsetSpec) -> list[HarnessQuestion]:
    dev_json = data_dir / BIRD_DEV_JSON
    if not dev_json.exists():
        raise DataMissingError(
            f"BIRD dev.json not found at {dev_json}\n"
            f"Expected BIRD dev under {data_dir}: {BIRD_DEV_JSON} + "
            f"{BIRD_DB_DIRNAME}/<db_id>/<db_id>.sqlite.\n"
            "Download BIRD dev from the official benchmark site and set DATA_DIR."
        )

    rows = json.loads(dev_json.read_text(encoding="utf-8"))
    wanted = set(subset.db_ids)
    explicit = set(subset.question_ids)

    selected: list[dict] = []
    if explicit:
        selected = [row for row in rows if str(row.get("question_id")) in explicit]
    else:
        selected = _sample(rows, wanted, subset)

    if not selected:
        raise DataMissingError(
            f"Subset {subset.name!r} selected 0 questions from {dev_json}. "
            "Check db_ids against the dataset."
        )

    questions = [_to_question(row, data_dir) for row in selected]
    return questions


def _sample(rows: list[dict], wanted: set[str], subset: SubsetSpec) -> list[dict]:
    """Draw the subset, deterministically, using the method named in the manifest.

    BIRD's dev.json is grouped by database, so a plain prefix takes whichever
    difficulties happen to sit first in each group. `stratified_by_difficulty` exists
    to make that explicit: the cap is applied per difficulty, in dataset order, so the
    sample spans simple/moderate/challenging and the failure histogram cannot be an
    artifact of dataset ordering.
    """
    by_db: dict[str, list[dict]] = {db: [] for db in wanted}
    for row in rows:
        db_id = str(row.get("db_id"))
        if db_id in by_db:
            by_db[db_id].append(row)

    selected: list[dict] = []
    for db_id in subset.db_ids:
        candidates = by_db.get(db_id) or []
        if subset.sampling == "stratified_by_difficulty":
            selected.extend(_stratify(candidates, subset.limit_per_db))
        else:
            selected.extend(candidates[: subset.limit_per_db])
    return selected


def _stratify(rows: list[dict], limit_per_db: int) -> list[dict]:
    """Round-robin across difficulty groups so the cap is spread, not front-loaded."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("difficulty") or "unknown"), []).append(row)

    selected: list[dict] = []
    while len(selected) < limit_per_db and any(groups.values()):
        for group in list(groups):
            if groups[group] and len(selected) < limit_per_db:
                selected.append(groups[group].pop(0))
    return selected


def _to_question(row: dict, data_dir: Path) -> HarnessQuestion:
    db_id = str(row["db_id"])
    gold_sql = next((row[key] for key in _GOLD_KEYS if row.get(key)), None)
    if not gold_sql:
        raise DataMissingError(f"question {row.get('question_id')} has no gold SQL")
    db_path = database_path(data_dir, db_id)
    return HarnessQuestion(
        question_id=str(row["question_id"]),
        db_id=db_id,
        question=str(row.get("question", "")),
        evidence=str(row.get("evidence", "") or ""),
        gold_sql=str(gold_sql),
        difficulty=str(row.get("difficulty", "")),
        db_path=db_path,
        schema=_schema_for(db_id, str(db_path)),
    )


def available_db_ids(data_dir: Path) -> list[str]:
    """Database ids present on disk. Used to help pick a subset before one exists."""
    db_root = data_dir / BIRD_DB_DIRNAME
    if not db_root.exists():
        return []
    return sorted(p.name for p in db_root.iterdir() if p.is_dir() and not p.name.startswith("_"))
