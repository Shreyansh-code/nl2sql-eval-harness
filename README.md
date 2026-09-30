# nl2sql-eval-harness

An NL-to-SQL evaluation harness that scores SQL by **executing** it, explains failures
with an LLM judge, and then measures **how reliable that judge is** against human labels.

The third part is the point. Most projects in this space report an LLM's self-assessment
and call it a metric. Here the judge is never allowed to decide correctness — execution
does that — and the judge's failure-mode labels are scored against humans with Cohen's
kappa, which is what makes the diagnostic layer worth anything.

Design rationale, taxonomy, and milestones: [`docs/HLD.md`](docs/HLD.md).

## Status

| Milestone | State |
|---|---|
| M1 dataset + executor + ExecMatch, gated on gold SQL scoring 1.0 | done — 90/90 on BIRD dev |
| M2 generator (standard mode) + report | done — **65.6% EX** (95% CI 56.7–74.4), n=90 |
| M3 judge + structured-output validation | not started |
| M4 50 human labels + kappa | not started |
| M5 report, README with real numbers, second prompt version | not started |
| M6 resume integration | blocked on M4 |

No agreement number is published yet, deliberately. Every number here comes from
`results/*/report.json`, never from prose — and `harness verify` recomputes it from
`match.jsonl` by an independent path, so the table below is checkable rather than
asserted. The M2 run is committed at
[`results/20260930T155111Z-generate`](results/20260930T155111Z-generate).

## Result so far

`gpt-6-luna`, standard mode, BIRD dev mini subset, 90 questions, ~45 seconds wall clock.

| Slice | Execution accuracy | n |
|---|---|---|
| **overall** | **65.6% (95% CI 56.7–74.4)** | 90 |
| superhero | 76.7% (63.3–90.0) | 30 |
| thrombosis_prediction | 66.7% (50.0–83.3) | 30 |
| formula_1 | 53.3% (33.3–70.0) | 30 |
| simple | 66.7% (50.0–83.3) | 30 |
| moderate | 73.3% (56.7–90.0) | 30 |
| challenging | 56.7% (40.0–73.3) | 30 |

`gpt-6-luna` rejects `temperature=0`, so this harness cannot pin sampling to a seed and
**the run is not bit-reproducible**: an earlier identical run scored 66.7% (60/90)
against this one's 65.6% (59/90). A one-question swing is well inside the confidence
interval, but it is the reason the README quotes an interval rather than a point, and
the reason the second prompt version in M5 will be compared with the cache invalidated
on one side only.

The schema-richest database is the weakest, which is the expected shape and a useful
sanity check on the harness. 90/90 generations parsed, 0 SQL errors, 0 questions excluded
from the denominator, and 0 of the 60 correct verdicts were vacuous (both sides empty) —
checked explicitly, because that is the easiest way to inflate this number.

**BIRD's `evidence` field is given to the generator.** Published BIRD numbers are
therefore not directly comparable to this one, and the README says so rather than
letting a reader assume it.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,data]"
cp .env.example .env
cp subset.example.yaml subset.yaml
```

Then fetch BIRD dev. The official bundle is ~4GB; `fetch_bird.py` pulls the questions
table plus only the databases in `subset.yaml` (~30MB), laid out exactly as BIRD ships
them. Nothing is bundled into this repo.

```bash
python scripts/fetch_bird.py --dbs superhero formula_1 thrombosis_prediction
```

## Usage

```bash
harness dbs                      # what is on disk
harness subset                   # load the manifest, preview the generator's input
harness baseline --strict        # M1 gate: gold SQL must execute and self-match
harness generate                 # M2: generate, execute, score, write report.md
harness generate --limit 3       # smoke test before spending anything
harness publish runs/<run_id>    # promote a run to committed, checkable evidence
harness verify results/<run_id>  # recompute accuracy from match.jsonl, check report.json
pytest                           # 139 tests
```

`harness generate` resumes: generations are cached by
`(model, prompt, schema, question, evidence)`, so a second run costs nothing and a
changed prompt is a different key rather than a silent overwrite.

## Two invariants the code enforces

- **The judge cannot produce the correctness metric** (M3). `scoring/exec_match.py` and
  `judge/` have no path between them; correctness comes from running the query.
- **A broken run cannot be reported as a measurement.** If any request fails at the API,
  the report is stamped `INVALID RUN` and the command exits non-zero. This exists because
  the first run of this harness did exactly that — a `temperature` rejection surfaced as
  "0% accuracy", which would have been a fabricated result.

## Layout

```
src/harness/
  config.py            environment settings, redacted snapshots for run metadata
  dataset/             BIRD loader, subset manifest, SQLite schema introspection
  execution/           SQL guard, read-only executor with deadline and row cap
  generation/          SqlGenerator port, standard adapter, prompts, parsing, cache
  scoring/             ExecMatch, the M1 baseline gate
  metrics/             bootstrap intervals, report rendering
  tracing/             opt-in LangSmith wrapper
  pipeline.py          generate -> execute -> ExecMatch, per question
  runs.py              append-only JSONL run artifacts
  cli.py               one subcommand per pipeline stage
docs/HLD.md            the design this implements
```

## The subset

90 questions from BIRD dev (`dev_20240627`), 30 per database, stratified by difficulty
so each database contributes exactly 30 simple / 30 moderate / 30 challenging. Databases
were chosen for schema-shape diversity **before** any query was generated:

| Database | Shape | Why |
|---|---|---|
| `superhero` | 10 tables / 31 cols | join-heavy, narrow |
| `formula_1` | 13 tables / 94 cols | largest schema in the release |
| `thrombosis_prediction` | 3 tables / 64 cols | few tables, many columns |

A contiguous prefix of BIRD's dev.json would have taken only whichever difficulties sit
first in each database's group, which would have made the three databases
incomparable. The sampling method is named in `subset.yaml` and enforced by tests.


## How the pieces fit

```
BIRD dev subset ──> schema introspection ──> generator prompt
                                              │
                        execution (read-only, timed, row-capped)
                                              │
                    ExecMatch: result set vs gold result set
                                              │
                     judge: WHY, on failures + a 20% audit sample
                                              │
                 50 blinded human labels ──> kappa ──> report.md / report.json
```

Two properties are enforced in code rather than convention:

- **The judge cannot produce the correctness metric.** `scoring/exec_match.py` and
  `judge/` have no path between them; correctness comes from running the query.
- **Model-authored SQL is not trusted.** `execution/guard.py` admits only a single
  read-only statement, and the connection itself is opened read-only, so a rejected
  query is a data point (`SQL_SYNTAX`) rather than a corrupted database.
