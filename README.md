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
| M1 dataset + executor + ExecMatch, gated on gold SQL scoring 1.0 | done — 90/90 on BIRD dev, 66 tests |
| M2 generator (standard mode) | not started |
| M3 judge + structured-output validation | not started |
| M4 50 human labels + kappa | not started |
| M5 report, README with real numbers, second prompt version | not started |
| M6 resume integration | blocked on M4 |

No accuracy or agreement number is published yet, deliberately. The README will be
filled from `report.json`, never by hand.

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
pytest                           # 66 tests
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

## Layout

```
src/harness/
  config.py            environment settings, redacted snapshots for run metadata
  dataset/             BIRD loader, subset manifest, SQLite schema introspection
  execution/           SQL guard, read-only executor with deadline and row cap
  scoring/             ExecMatch, the M1 baseline gate
  cli.py               one subcommand per pipeline stage
docs/HLD.md            the design this implements
```
