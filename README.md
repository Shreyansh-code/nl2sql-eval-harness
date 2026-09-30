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
| M2 generator (standard mode) + report | done — **64.4% EX** (95% CI 57.8–71.1), n=180 |
| M3 judge + structured-output validation | not started |
| M4 50 human labels + kappa | not started |
| M5 report, README with real numbers, second prompt version | not started |
| M6 resume integration | blocked on M4 |

No agreement number is published yet, deliberately. Every number here comes from
`results/*/report.json`, never from prose — and `harness verify` recomputes it from
`match.jsonl` by an independent path, so the table below is checkable rather than
asserted. Both M2 runs are committed under
[`results/`](results) and can be re-checked with `harness verify <dir>`.

## Result so far

`gpt-6-luna`, standard mode, BIRD dev mini subset, 180 questions, ~95 seconds wall clock.

| Slice | Execution accuracy | n |
|---|---|---|
| **overall** | **64.4% (95% CI 57.8–71.1)** | 180 |
| superhero | 76.7% (65.0–88.3) | 60 |
| formula_1 | 63.3% (51.7–75.0) | 60 |
| thrombosis_prediction | 53.3% (41.7–66.7) | 60 |
| simple | 72.7% (60.6–81.8) | 66 |
| moderate | 67.7% (55.4–78.5) | 65 |
| challenging | 49.0% (34.7–63.3) | 49 |

Difficulty separates cleanly and monotonically (72.7 / 67.7 / 49.0), while the three
per-database intervals overlap heavily — so the difficulty split is a real finding and
the per-database ranking is not. Reporting it the other way round would be reading noise. Both runs are committed and independently checkable with `harness verify`:

| Run | n | Accuracy | Correct |
|---|---|---|---|
| [`20260930T155111Z-generate`](results/20260930T155111Z-generate) | 90 | 65.6% (56.7–74.4) | 59 |
| [`20260930T155619Z-generate`](results/20260930T155619Z-generate) | 180 | 64.4% (57.8–71.1) | 116 |

**On reproducibility, measured rather than asserted.** `gpt-6-luna` rejects
`temperature=0`, so this harness cannot pin sampling to a seed and a fresh run is a fresh
sample. The 90-question run is contained in the 180-question one (stratified sampling is
prefix-stable within each database — a test enforces it), and across those two runs the
generator **agreed with itself on 84 of the same 90 questions, flipping 6**. So the
run-to-run standard deviation on a single question is roughly 2-3%, well inside the
confidence interval. That is why the table quotes intervals, and why M5 will compare
prompt versions with the cache invalidated on exactly one side.

Of 180 generations: 180 parsed, 0 SQL errors, 0 excluded from the denominator, and 0 of
the 116 correct verdicts were vacuous (both result sets empty) — checked explicitly,
because that is the cheapest way to inflate this number. The 64 failures are 38 row-content
mismatches and 26 column-count mismatches.

**BIRD's `evidence` field is given to the generator**, so published BIRD numbers are not
directly comparable to this one, and the README says so rather than letting a reader
assume it.

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
