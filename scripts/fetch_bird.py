"""Fetch BIRD dev questions and only the databases a subset actually needs.

The official dev bundle is ~4GB of SQLite files; a mini subset needs a few hundred MB.
This pulls the questions table plus the specific `<db_id>.sqlite` files requested, and
lays them out exactly as BIRD does (`dev.json` + `dev_databases/<db_id>/<db_id>.sqlite`)
so the loader cannot tell the difference.

    python scripts/fetch_bird.py --dbs superhero formula_1 thrombosis_prediction

Nothing is modified: rows are copied verbatim, and `SQL` is written under BIRD's own
key name so the harness sees the release's schema, not a re-dump of it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_REPO = "EuricoGVP/birdsql_complete_devset"
QUESTIONS_FILE = "questions/dev-00000-of-00001.parquet"
# dev_20240627, the cleaned release also used by the official dev.zip.
RELEASE = "bird dev 20240627"


def fetch_questions() -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(QUESTIONS_REPO, QUESTIONS_FILE, repo_type="dataset")
    rows = pq.read_table(path).to_pylist()
    # Normalize to the release's own column names so the on-disk layout is BIRD-shaped.
    return [
        {
            "question_id": row["question_id"],
            "db_id": row["db_id"],
            "question": row["question"],
            "evidence": row.get("evidence") or "",
            "SQL": row["sql"],
            "difficulty": row.get("difficulty") or "",
        }
        for row in rows
    ]


def fetch_database(db_id: str, target: Path) -> Path:
    from huggingface_hub import hf_hub_download

    target.mkdir(parents=True, exist_ok=True)
    destination = target / f"{db_id}.sqlite"
    if destination.exists() and destination.stat().st_size:
        return destination
    source = hf_hub_download(
        QUESTIONS_REPO, f"databases/{db_id}/{db_id}.sqlite", repo_type="dataset"
    )
    destination.write_bytes(Path(source).read_bytes())
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbs", nargs="+", required=True, help="db_ids to download")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "bird"))
    parser.add_argument(
        "--questions-only", action="store_true", help="write dev.json, skip database files"
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = fetch_questions()
    known = {row["db_id"] for row in rows}
    unknown = [db for db in args.dbs if db not in known]
    if unknown:
        print(f"error: unknown db_ids in {QUESTIONS_REPO}: {unknown}", file=sys.stderr)
        print(f"available: {sorted(known)}", file=sys.stderr)
        return 2

    (data_dir / "dev.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    counts = Counter(row["db_id"] for row in rows)
    print(f"wrote {data_dir / 'dev.json'}: {len(rows)} questions ({RELEASE})")
    for db in args.dbs:
        print(f"  {db}: {counts[db]} questions available in the release")

    if args.questions_only:
        return 0

    for db in args.dbs:
        path = fetch_database(db, data_dir / "dev_databases" / db)
        size_mb = path.stat().st_size / 1e6
        print(f"  {db}: {path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
