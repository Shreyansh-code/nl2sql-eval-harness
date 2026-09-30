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
from .generation.port import Generation
from .generation.standard import StandardGenerator
from .judge.client import StandardJudgeClient
from .judge.graph import JudgeGraph
from .judge.pipeline import persist as persist_judgements
from .judge.pipeline import reload_scored, run_judge
from .judge.port import Judgement
from .metrics.judge_report import build_judge_report, render_judge_markdown
from .metrics.report import build_report, render_markdown
from .metrics.verify import compare_to_report, recompute_from_matches
from .pipeline import persist, score_generations
from .runs import iter_rows, open_run
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
    if args.seed_cache:
        seed_path = Path(args.seed_cache) / "generations.jsonl"
        if not seed_path.exists():
            seed_path = Path(args.seed_cache)
        if not seed_path.exists():
            print(f"error: no generations.jsonl at {args.seed_cache}", file=sys.stderr)
            return 2
        # Re-scoring a previous run's generations. Because the cache key covers model,
        # prompt, schema, question and evidence, a hit means the SQL is unchanged, so any
        # difference in the report is attributable to the scoring code and nothing else.
        (run.path / "generations.jsonl").write_text(
            seed_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
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
    """Re-derive a run's headline numbers from its raw artifacts.

    Generation runs are checked against `match.jsonl`; judge runs against
    `judgements.jsonl`. Either way the stored report is treated as a claim to be
    re-derived, not as a fact.
    """
    if (target / "report.json").exists():
        return _verify_generation(target)
    if (target / "judge_report.json").exists():
        return _verify_judge(target)
    print(f"error: {target} has neither report.json nor judge_report.json", file=sys.stderr)
    return 2


def _verify_generation(target: Path) -> int:
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


def _verify_judge(target: Path) -> int:
    report = json.loads((target / "judge_report.json").read_text(encoding="utf-8"))
    rows = list(iter_rows(target / "judgements.jsonl"))
    usable = [r for r in rows if r.get("status") == "ok"]
    histogram: dict[str, int] = {}
    for row in usable:
        if row.get("failure_mode"):
            histogram[row["failure_mode"]] = histogram.get(row["failure_mode"], 0) + 1

    problems: list[str] = []
    if report.get("n_judged") != len(rows):
        problems.append(f"n_judged: report {report.get('n_judged')} vs {len(rows)} rows")
    if report.get("n_usable") != len(usable):
        problems.append(f"n_usable: report {report.get('n_usable')} vs {len(usable)} usable")
    stored_histogram = {k: v for k, v in (report.get("failure_histogram") or {}).items() if v}
    if stored_histogram != histogram:
        problems.append(f"failure_histogram: report {stored_histogram} vs {histogram}")

    print(
        f"recomputed from {target / 'judgements.jsonl'}: {len(rows)} judged, {len(usable)} usable"
    )
    if problems:
        print("MISMATCH between judge_report.json and judgements.jsonl:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("judge_report.json agrees with judgements.jsonl")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    return _verify_dir(Path(args.run_dir))


def cmd_publish(args: argparse.Namespace) -> int:
    """Promote a run to committed evidence, then verify it before it lands.

    Publishing is a deliberate step rather than a side effect of running, because these
    artifacts become the numbers a reader trusts.
    """
    run_dir = Path(args.run_dir)
    if not ((run_dir / "report.json").exists() or (run_dir / "judge_report.json").exists()):
        print(f"error: {run_dir} has no report.json or judge_report.json", file=sys.stderr)
        return 2
    destination = REPO_ROOT / RESULTS_DIRNAME / run_dir.name
    if destination.exists() and not args.force:
        print(f"error: {destination} exists (use --force to overwrite)", file=sys.stderr)
        return 2
    # Verify the source *before* copying. Publishing first and checking afterwards would
    # leave a failing artifact sitting in results/ as evidence, which is worse than not
    # publishing: it looks checked when it is not.
    status = _verify_dir(run_dir)
    if status != 0:
        print(f"error: {run_dir} failed verification; nothing published", file=sys.stderr)
        return status

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(run_dir, destination)
    print(f"published to {destination}")
    return _verify_dir(destination)


def _load_human_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    labels: dict[str, str] = {}
    for row in iter_rows(path):
        mode = row.get("failure_mode")
        if row.get("question_id") and mode:
            labels[str(row["question_id"])] = str(mode)
    return labels


def cmd_judge(args: argparse.Namespace) -> int:
    """Judge a finished generation run.

    Reads the source run's artifacts and writes to a *sibling* run directory, so a
    published `results/` directory is never mutated by a later judging pass.
    """
    settings = load_settings()
    if not settings.judge_model:
        print("error: JUDGE_MODEL is not set (see .env.example)", file=sys.stderr)
        return 2
    if not settings.openai_api_key:
        print("error: OPENAI_API_KEY is not set (see .env.example)", file=sys.stderr)
        return 2

    source = Path(args.run_dir)
    if not (source / "match.jsonl").exists():
        print(f"error: {source} has no match.jsonl; run `harness generate` first", file=sys.stderr)
        return 2

    subset = load_subset(settings.subset_path)
    questions = load_questions(settings.data_dir, subset)
    generations = [Generation.from_row(row) for row in iter_rows(source / "generations.jsonl")]
    matches = list(iter_rows(source / "match.jsonl"))

    try:
        scored = reload_scored(questions, generations, matches, settings)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    if args.limit:
        keep = {q.question_id for q in questions[: args.limit]}
        scored = [item for item in scored if item.question.question_id in keep]

    client = StandardJudgeClient(
        model=settings.judge_model,
        api_key=settings.openai_api_key or "",
        base_url=settings.openai_base_url,
    )
    graph = JudgeGraph(client=client, model=settings.judge_model)
    if args.reuse:
        reuse_path = Path(args.reuse)
        if reuse_path.is_dir():
            reuse_path = reuse_path / "judgements.jsonl"
        if not reuse_path.exists():
            print(f"error: no judgements to reuse at {args.reuse}", file=sys.stderr)
            return 2
        # Judgements are cached on (model, prompt, question, sql, gold, verdict), so an
        # unchanged item is free and only genuinely changed questions are re-judged.
        reused = 0
        for row in iter_rows(reuse_path):
            previous = Judgement.from_row(row)
            graph.judge_cache[previous.cache_key] = previous
            reused += 1
        print(f"reusing {reused} prior judgements")
    tracing.configure(settings.langsmith_tracing)

    run = open_run(settings, "judge", run_id=f"{source.name}-judge")
    run.write_json("source_run.json", {"source_run": str(source), "subset": subset.name})

    audit_fraction = 1.0 if args.all else args.audit_fraction
    outcome = run_judge(scored, graph, audit_fraction=audit_fraction, seed=args.seed)
    persist_judgements(run, outcome)

    human = _load_human_labels(Path(args.labels)) if args.labels else {}
    report = build_judge_report(outcome, scored, human_labels=human)
    meta = run.read_json("meta.json") or {}
    meta["judge_model"] = settings.judge_model
    meta["subset"] = subset.name
    run.write_json("meta.json", meta)
    run.write_json("judge_report.json", report.to_dict())
    (run.path / "judge_report.md").write_text(
        render_judge_markdown(report, outcome.plan.describe(), meta), encoding="utf-8"
    )

    print(
        f"judged {report.n_judged} items ({report.n_usable} usable, {report.n_unscored} excluded)"
    )
    if report.failure_histogram:
        print("\nfailure modes:")
        for mode, count in report.failure_histogram.items():
            print(f"  {mode}: {count} ({report.failure_share[mode]:.0%})")
    print()
    if report.verdict_kappa is not None:
        print(f"judge vs execution: {report.verdict_kappa.summary()}")
    else:
        print("judge vs execution: not measurable (no usable verdicts)")
    if report.mode_kappa is not None:
        print(f"judge vs human:     {report.mode_kappa.summary()}")
    else:
        print("judge vs human:     not yet measured (needs M4 human labels)")
    print(f"\nreport: {run.path / 'judge_report.md'}")
    return 0


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
    generate_parser.add_argument(
        "--seed-cache",
        default="",
        help="reuse generations.jsonl from a previous run: re-score without re-calling the model",
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

    judge_parser = sub.add_parser(
        "judge", help="M3: classify failures and measure the judge against execution"
    )
    judge_parser.add_argument("run_dir", help="a generation run directory to judge")
    judge_parser.add_argument("--limit", type=int, default=0, help="judge only the first N")
    judge_parser.add_argument("--seed", type=int, default=20260930)
    judge_parser.add_argument("--audit-fraction", type=float, default=0.20)
    judge_parser.add_argument("--labels", default="", help="path to human_labels.jsonl (M4)")
    judge_parser.add_argument(
        "--all", action="store_true", help="judge every question, not just failures + audit"
    )
    judge_parser.add_argument(
        "--reuse", default="", help="reuse prior judgements.jsonl where inputs are unchanged"
    )
    judge_parser.set_defaults(func=cmd_judge)

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
