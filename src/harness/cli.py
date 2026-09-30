"""CLI. Every stage of the pipeline is a subcommand so a run is reproducible by hand.

Commands land in order of the milestones: `dbs` / `baseline` are M1, the rest arrive
with their milestones. Kept dependency-free so M1 runs with only pyyaml installed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .config import REPO_ROOT, load_settings
from .dataset.loader import DataMissingError, available_db_ids, load_questions
from .dataset.subset import load_subset
from .generation.cache import GenerationCache
from .generation.standard import StandardGenerator
from .metrics.report import build_report, render_markdown
from .metrics.verify import compare_to_report, recompute_from_matches
from .pipeline import persist, score_generations
from .runs import open_run
from .scoring.baseline import gate, run_baseline
from .tracing import langsmith as tracing


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


def cmd_generate(args: argparse.Namespace) -> int:
    settings = load_settings()
    if not settings.generator_model:
        print("error: GENERATOR_MODEL is not set (see .env.example)", file=sys.stderr)
        return 2
    if not settings.openai_api_key:
        print("error: OPENAI_API_KEY is not set (see .env.example)", file=sys.stderr)
        return 2
    if settings.generator_mode != "standard":
        print(
            f"error: GENERATOR_MODE={settings.generator_mode!r} is not implemented yet; "
            "this command runs standard mode",
            file=sys.stderr,
        )
        return 2

    subset = load_subset(settings.subset_path)
    questions = load_questions(settings.data_dir, subset)
    if args.limit:
        questions = questions[: args.limit]

    run = open_run(settings, "generate")
    run.write_json("subset.json", {"subset": subset.describe(), "loaded": len(questions)})
    cache = GenerationCache(run.path / "generations.jsonl")
    cached_before = len(cache)

    generator = StandardGenerator(settings, cache)
    tracing.configure(settings.langsmith_tracing)

    print(
        f"generating for {len(questions)} questions "
        f"({len(cache)} cached, {len(questions) - cached_before} to request) "
        f"with {settings.generator_model}"
    )
    generations = generator.generate(questions)

    scored = score_generations(questions, generations, settings)
    persist(run, scored)

    report = build_report(scored)
    meta = run.read_json("meta.json") or {}
    meta["subset"] = subset.name
    run.write_json("meta.json", meta)
    run.write_json("report.json", report.to_dict())
    (run.path / "report.md").write_text(render_markdown(report, meta), encoding="utf-8")

    print(f"\nrun: {run.run_id}")
    if not report.valid:
        # Exit non-zero: a caller scripting this must not treat a transport failure as
        # a measurement of the model.
        print(f"INVALID RUN: {report.invalid_reason()}", file=sys.stderr)
        print(f"artifacts: {run.path}", file=sys.stderr)
        return 3
    if report.accuracy is None:
        print("no scored questions")
    else:
        print(f"execution accuracy: {report.accuracy.as_percent()} on n={report.n_scored}")
    print(f"report: {run.path / 'report.md'}")
    return 0


RESULTS_DIRNAME = "results"


def _verify_dir(target) -> int:
    report = json.loads((target / "report.json").read_text(encoding="utf-8"))
    recomputed = recompute_from_matches(target / "match.jsonl")
    problems = compare_to_report(recomputed, report)
    print(f"recomputed from {target / 'match.jsonl'}: {json.dumps(recomputed.to_dict())}")
    if problems:
        print("MISMATCH between report.json and match.jsonl:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("report.json agrees with match.jsonl")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    return _verify_dir(Path(args.run_dir))


def cmd_publish(args: argparse.Namespace) -> int:
    """Promote a run to committed evidence, then verify it before it lands.

    Publishing is a deliberate step rather than a side effect of running, because these
    artifacts become the numbers a reader trusts.
    """
    run_dir = Path(args.run_dir)
    if not (run_dir / "report.json").exists():
        print(f"error: {run_dir} has no report.json", file=sys.stderr)
        return 2
    destination = REPO_ROOT / RESULTS_DIRNAME / run_dir.name
    if destination.exists() and not args.force:
        print(f"error: {destination} exists (use --force to overwrite)", file=sys.stderr)
        return 2
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(run_dir, destination)
    print(f"published to {destination}")
    return _verify_dir(destination)


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

    generate_parser = sub.add_parser(
        "generate", help="M2: generate SQL, execute it, and write the report"
    )
    generate_parser.add_argument(
        "--limit", type=int, default=0, help="only run the first N questions (smoke tests)"
    )
    generate_parser.set_defaults(func=cmd_generate)

    verify_parser = sub.add_parser(
        "verify", help="recompute a run's accuracy from match.jsonl and check report.json"
    )
    verify_parser.add_argument("run_dir", help="a run or published results directory")
    verify_parser.set_defaults(func=cmd_verify)

    publish_parser = sub.add_parser(
        "publish", help="copy a run into results/ so its numbers become committed evidence"
    )
    publish_parser.add_argument("run_dir")
    publish_parser.add_argument("--force", action="store_true")
    publish_parser.set_defaults(func=cmd_publish)

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
