"""CLI. Every stage of the pipeline is a subcommand so a run is reproducible by hand.

Commands land in order of the milestones: `dbs` / `baseline` are M1, the rest arrive
with their milestones. Kept dependency-free so M1 runs with only pyyaml installed.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_settings
from .dataset.loader import DataMissingError, available_db_ids, load_questions
from .dataset.subset import load_subset
from .scoring.baseline import gate, run_baseline


def cmd_dbs(args: argparse.Namespace) -> int:
    settings = load_settings()
    ids = available_db_ids(settings.data_dir)
    if not ids:
        print(f"No BIRD databases found under {settings.data_dir / 'dev_databases'}")
        print("Download BIRD dev and point DATA_DIR at it.")
        return 1
    print(f"{len(ids)} databases under {settings.data_dir / 'dev_databases'}:")
    for db_id in ids:
        print(f"  {db_id}")
    return 0


def cmd_subset(args: argparse.Namespace) -> int:
    settings = load_settings()
    subset = load_subset(settings.subset_path)
    questions = load_questions(settings.data_dir, subset)
    print(json.dumps({"subset": subset.describe(), "loaded": len(questions)}, indent=2))
    print("\nFirst question rendered as the generator will see it:\n")
    print("-" * 70)
    print(questions[0].schema.to_prompt_block())
    print("-" * 70)
    print(f"Q: {questions[0].question}")
    if questions[0].evidence:
        print(f"evidence: {questions[0].evidence}")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    settings = load_settings()
    subset = load_subset(settings.subset_path)
    questions = load_questions(settings.data_dir, subset)
    report = run_baseline(
        questions,
        timeout_seconds=settings.sql_timeout_seconds,
        max_rows=settings.max_rows,
    )
    print(json.dumps(report.to_dict(), indent=2))
    if args.strict:
        gate(report)
        print(f"baseline gate passed: {report.passed}/{report.total}")
    else:
        print(f"baseline (advisory): {report.passed}/{report.total}")
    return 0 if report.passed == report.total else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("dbs", help="list BIRD databases available on disk").set_defaults(func=cmd_dbs)

    subset_parser = sub.add_parser(
        "subset", help="load the subset and preview the generator's prompt input"
    )
    subset_parser.set_defaults(func=cmd_subset)

    baseline_parser = sub.add_parser(
        "baseline", help="M1 gate: every gold query must execute and self-match"
    )
    baseline_parser.add_argument(
        "--strict", action="store_true", help="exit non-zero unless the gate passes"
    )
    baseline_parser.set_defaults(func=cmd_baseline)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (DataMissingError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
