from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from harness.dataset.loader import load_questions
from harness.dataset.models import HarnessQuestion, SchemaPayload
from harness.dataset.subset import SubsetSpec

DDL = [
    "CREATE TABLE player (player_id INTEGER, name TEXT, position TEXT, height_cm INTEGER)",
    "CREATE TABLE match (match_id INTEGER, player_id INTEGER, tournament TEXT, score INTEGER)",
]


def _seed(path: Path) -> None:
    conn = sqlite3.connect(path)
    for statement in DDL:
        conn.execute(statement)
    conn.executemany(
        "INSERT INTO player VALUES (?,?,?,?)",
        [(1, "ana", "M", 180), (2, "bo", "F", 165), (3, "cy", "M", 175)],
    )
    conn.executemany(
        "INSERT INTO match VALUES (?,?,?,?)",
        [(1, 1, "open", 10), (2, 2, "open", 12), (3, 1, "cup", 8)],
    )
    conn.commit()
    conn.close()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "league.sqlite"
    _seed(path)
    return path


@pytest.fixture
def schema() -> SchemaPayload:
    return SchemaPayload(
        db_id="league",
        tables=("match", "player"),
        ddl="\n".join(f"{s};" for s in DDL),
        column_values={"player": {"position": ["M", "F"]}},
    )


@pytest.fixture
def question(db_path: Path, schema: SchemaPayload) -> HarnessQuestion:
    return HarnessQuestion(
        question_id="q0",
        db_id="league",
        question="List the name of players taller than 170 cm.",
        evidence="height refers to height_cm",
        gold_sql="SELECT name FROM player WHERE height_cm > 170 ORDER BY name",
        difficulty="easy",
        db_path=db_path,
        schema=schema,
    )


@pytest.fixture
def bird_layout(tmp_path: Path) -> Path:
    """A BIRD-shaped data dir: dev.json + dev_databases/<db>/<db>.sqlite."""
    data_dir = tmp_path / "bird"
    db_dir = data_dir / "dev_databases" / "league"
    db_dir.mkdir(parents=True)
    _seed(db_dir / "league.sqlite")

    rows = [
        {
            "question_id": "league_0",
            "db_id": "league",
            "question": "How many players are male?",
            "evidence": "male means position = 'M'",
            "SQL": "SELECT count(*) FROM player WHERE position = 'M'",
            "difficulty": "easy",
        },
        {
            "question_id": "league_1",
            "db_id": "league",
            "question": "List the tournament of every match.",
            "evidence": "",
            "SQL": "SELECT tournament FROM match ORDER BY tournament",
            "difficulty": "simple",
        },
        *[
            {
                "question_id": f"league_x{i}",
                "db_id": "league",
                "question": f"Question {i}?",
                "evidence": "",
                "SQL": "SELECT name FROM player",
                "difficulty": "simple",
            }
            for i in range(30)
        ],
        {
            "question_id": "other_0",
            "db_id": "other",
            "question": "ignored",
            "evidence": "",
            "SQL": "SELECT 1",
            "difficulty": "easy",
        },
    ]
    (data_dir / "dev.json").write_text(json.dumps(rows), encoding="utf-8")
    return data_dir


@pytest.fixture
def loaded_questions(bird_layout: Path) -> list[HarnessQuestion]:
    spec = SubsetSpec(
        name="test",
        source="bird-dev",
        db_ids=("league",),
        limit_per_db=10,
        sampling="fixture",
    )
    return load_questions(bird_layout, spec)


@pytest.fixture
def wide_table(tmp_path: Path) -> Path:
    path = tmp_path / "wide.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (i INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(50)])
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def high_cardinality(tmp_path: Path) -> Path:
    """A text column with more distinct values than we are willing to sample."""
    path = tmp_path / "cards.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE note (body TEXT, tag TEXT)")
    conn.executemany("INSERT INTO note VALUES (?,?)", [(f"note-{i}", "t") for i in range(50)])
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def many_questions(bird_layout: Path) -> list[HarnessQuestion]:
    """20 questions, for tests that need a population to sample from.

    The two-question `loaded_questions` fixture is fine for scoring tests and useless for
    anything that samples or computes agreement.
    """
    spec = SubsetSpec(
        name="many",
        source="bird-dev",
        db_ids=("league",),
        limit_per_db=20,
        sampling="contiguous_prefix",
    )
    return load_questions(bird_layout, spec)
