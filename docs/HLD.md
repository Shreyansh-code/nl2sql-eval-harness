# HLD — NL-to-SQL Evaluation Harness with an LLM Judge

**Status:** Draft for review
**Date:** 2026-09-30
**Owner:** Shreyansh

---

## 1. Purpose

A reproducible harness that measures how well an LLM writes SQL, using an LLM as a judge to explain *why* queries fail, and publishing the judge's own reliability against human labels.

The design question this project answers:

> Does an LLM judge add real diagnostic signal over execution accuracy alone, and is that judge trustworthy enough to be the headline metric?

## 2. Goals / Non-Goals

### Goals
- **G1.** Compute objective execution accuracy over a BIRD dev subset, with correct, defensible result-set matching.
- **G2.** Classify each failure into a fixed taxonomy using an LLM judge, with a machine-parseable structured output.
- **G3.** Quantify judge reliability: Cohen's kappa between judge labels and a 50-query human-labelled sample.
- **G4.** Support standard (sync/async) and batch generation against a hosted OpenAI-compatible API, chosen per role by config.
- **G5.** Emit a committed report (markdown + JSON) containing every number quoted in the resume/README, so no number is hand-copied.

### Non-Goals
- Not a text-to-SQL leaderboard attempt. No new training, no fine-tuning.
- Not a general-purpose SQL sandbox. Execution is deliberately narrow and read-only.
- Not a hosted service, dashboard, or CI-gated benchmark. One local run produces one report.
- Not a claim of SOTA. A mini-subset result is a measurement, not a competition entry.

## 3. Design Questions (must be answered by the run)

| # | Question | Answered by |
|---|---|---|
| Q1 | What fraction of generated queries are execution-correct? | `metrics.accuracy.exec_accuracy` |
| Q2 | Among failures, what is the dominant failure mode? | `metrics.failure_modes` histogram |
| Q3 | Does the judge pick the same failure mode a human would? | `metrics.agreement.mode_kappa` |
| Q4 | Does the judge agree with objective execution correctness? | `metrics.agreement.judge_vs_exec_kappa` |
| Q5 | On which schema features does the generator break (evidence strings, joins, aggregates)? | per-question breakdown joined with BIRD `evidence` |

Q3/Q4 are the credibility layer. Without them the project is a dataset wrapper.

## 4. High-Level Architecture

```
                        ┌──────────────────────────────┐
                        │  BIRD dev subset (mini)      │
                        │  questions + evidence +      │
                        │  gold SQL + sqlite dbs       │
                        └──────────────┬───────────────┘
                                       │
                        ┌──────────────▼───────────────┐
   config/env ────────▶│  Dataset Loader              │
                        │  (normalises rows to        │
                        │   HarnessQuestion)           │
                        └──────────────┬───────────────┘
                                       │
              ┌────────────────────────▼────────────────────────┐
              │  GENERATION STAGE                               │
              │  port: SqlGenerator                              │
              │  ├─ StandardGenerator  (async, concurrency cap)  │
              │  └─ BatchGenerator      (JSONL submit/poll)      │
              │  prompt: schema-aware, few-shot, evidence-on/off │
              └────────────────────────┬────────────────────────┘
                                       │ generations.jsonl
              ┌────────────────────────▼────────────────────────┐
              │  EXECUTION STAGE                                │
              │  guard: read-only, single stmt, timeout         │
              │  → exec_result rows | timeout | error          │
              └────────────────────────┬────────────────────────┘
                                       │
              ┌────────────────────────▼────────────────────────┐
              │  SCORING STAGE                                  │
              │  ├─ ExecMatch: result set vs gold result set   │
              │  └─ Judge (only on failures + audit sample)     │
              │       → FailureMode + rationale + confidence   │
              └────────────────────────┬────────────────────────┘
                                       │ judgements.jsonl
              ┌────────────────────────▼────────────────────────┐
              │  HUMAN LABEL SET (N=50, blinded from judge)     │
              │  human_labels.jsonl                             │
              └────────────────────────┬────────────────────────┘
                                       │
              ┌────────────────────────▼────────────────────────┐
              │  METRICS + REPORT                              │
              │  accuracy, failure histogram, kappa, per-db     │
              │  → report.md, report.json                      │
              └────────────────────────────────────────────────┘
                                       │
                              LangSmith tracing (optional)
```

**Key architectural decision — the judge never decides correctness.**
Correctness comes from executing SQL. The judge only explains failures and is itself measured. This keeps the "LLM judge" from being a self-graded metric, which is the usual failure mode of this genre of project.

## 5. Components

### 5.1 Dataset Loader
- Source: BIRD dev, restricted to a committed `subset.yaml` (2–3 databases, ~50–100 questions).
- `subset.yaml` is the reproducibility contract: db ids, question ids, count, and a note on how the sample was drawn (contiguous slice, not cherry-picked).
- Loader emits `HarnessQuestion`: `question_id, db_id, question, evidence, schema_ddl, schema_evidence_rows, gold_sql`.
- Schema payload is built by an introspector, not hand-written: `sqlite_master` DDL plus a per-column sample of distinct values for low-cardinality columns. This is what makes the generator prompt schema-aware rather than schema-blind.
- Loader is pure and offline; it raises if the BIRD directory is missing, with the expected path printed.

### 5.2 SQL Generator (port + two adapters)
```
SqlGenerator.generate(questions) -> Iterable[Generation]
```
- **StandardGenerator** — `asyncio` + semaphore concurrency cap; one request per question; retries with exponential backoff on 429/5xx. Used for interactive runs and small subsets.
- **BatchGenerator** — writes a JSONL manifest, submits a batch job, records the job id, then polls and writes `generations.jsonl`. Used for the full subset when cost/time matter.
- Selection is by config, not by code path in callers. The batch adapter is resumable: on re-run it skips questions already present in `generations.jsonl`.
- Both adapters are wrapped by a content-addressed cache keyed on `(model, prompt_version, question_id, schema_hash)`. Re-runs are free and the cache is what makes the human-labeling loop cheap.
- Output is parsed strictly: extract the first SQL block, strip markdown fences and trailing prose, reject anything that is not a single statement.

### 5.3 SQL Executor
Safety is a hard requirement, not a nicety — we execute model output.
- Connection opened **read-only** (`file:...?mode=ro`).
- Single statement enforced by parsing, not by trusting the prompt.
- Writes rejected by an allowlist check on leading keyword (`SELECT` / `WITH` only).
- A wall-clock timeout per query, plus a row cap, so a runaway `CROSS JOIN` cannot hang the run.
- Every failure is captured as a structured outcome (`ok`, `timeout`, `error`, `empty`) rather than an exception that aborts the run.

### 5.4 ExecMatch
BIRD's metric is result-set equality. Matching rules, stated once and tested:
- Compare as **multisets of tuples** (order-insensitive unless the gold query has `ORDER BY`).
- Normalize `NULL`, numeric types, and `1` vs `1.0` before comparison; strip column-name differences.
- If gold fails to execute, the question is marked `skipped` and excluded from the denominator — never counted as correct.
- Match is also recorded per-question as a boolean plus a diff summary (row counts, first differing row) for the report.

### 5.5 LLM Judge
- Runs on **all failures** (to build the failure histogram) and on a **random 20% audit sample of successes** (to catch false failures / unjustified praise). Reporting a judge pass rate without an audit is the standard sin; the audit sample is the fix.
- Input: question, evidence, schema excerpt, generated SQL, gold SQL, the observed execution outcome, and the observed-vs-expected row diff.
- **Decided 2026-09-30: the judge sees the gold SQL.** It cannot classify *why* a query is wrong without knowing what right looks like, and the claim being made is diagnostic accuracy, not blind error detection. Two consequences are accepted deliberately:
  - The judge must never be used to compute correctness. Section 4's separation is what keeps this honest — gold SQL is present, but correctness is still execution-derived.
  - Kappa against humans is the guard against a generous judge. Humans see the same inputs (section 5.7), so any leniency shows up as disagreement.
- Output is **structured JSON** against a fixed schema, not prose:
  ```
  { verdict: "generator_error" | "sql_error",
    failure_mode: <taxonomy enum>,
    confidence: 0..1,
    rationale: "<= 3 sentences",
    evidence_span: "<the schema/question text the judge relied on>" }
  ```
- `evidence_span` is required and must be a verbatim substring of the supplied context. A judge that cannot cite gets rejected. This is the cheapest available guard against a judge that is pattern-matching rather than reasoning, and it is quotable in the README.
- Temperature 0, fixed prompt version, cached by `(judge_model, prompt_version, question_id, sql_hash)`.

### 5.6 Failure Taxonomy (fixed, closed set)
Exactly one label per failure. The set is fixed **before** labelling, so kappa is meaningful.

| Code | Meaning |
|---|---|
| `SCHEMA_MISREAD` | Used a column/table that does not exist, or exists but means something else |
| `WRONG_JOIN` | Correct tables, incorrect join predicate or join cardinality |
| `WRONG_FILTER` | Correct target rows, incorrect WHERE predicate (comparison, date, unit) |
| `MISSING_CONDITION` | Omitted a filter implied by the question or the `evidence` field |
| `AGGREGATION_ERROR` | Wrong group-by, wrong aggregate, or aggregation applied at the wrong level |
| `SELECTION_ERROR` | Right rows filtered, wrong columns projected |
| `UNIT_CONVERSION` | Value scale/format error (percent vs fraction, epoch vs string dates) |
| `PRECISION` | Escaped only on ordering/limit/ties |
| `SQL_SYNTAX` | Did not execute |
| `TIMEOUT` | Executed too slowly to finish within budget |
| `EMPTY_WRONG` | Executed, returned no rows, gold returned rows |

`EMPTY_WRONG` is split out because on BIRD it is disproportionately a schema misunderstanding, and lumping it into `WRONG_FILTER` hides the finding.

### 5.7 Human Labeling Protocol
- **N = 50**, sampled with a fixed seed from the failing + audited set.
- Labels are produced from the same information the judge sees, in a `label` CLI that **does not show judge output** — the labelling UI must not reveal the judge verdict, or the kappa measures anchoring rather than agreement.
- Each label is `{question_id, failure_mode, notes}` and is committed to `labels/human_labels.jsonl`.
- Double-labelling is not required at N=50, but the sampling seed and the exact label instructions are committed so the set is reproducible.
- The instruction sheet is a committed file, not a README paragraph. The number is only defensible if someone else could relabel from the same sheet.

### 5.8 Metrics & Report
- `exec_accuracy`, `n_evaluated`, `n_skipped`, and per-database accuracy.
- `failure_mode` histogram + share of failures.
- **Cohen's kappa** on the failure-mode taxonomy, judge vs human, over the intersection of labelled items. Report the observed agreement alongside kappa — kappa alone is unreadable and is depressed by prevalence, which we expect given the taxonomy's skew.
- Kappa on the binary `sql_correct` judgment, judge vs execution truth, over the audit sample. This is the number that says whether the judge can be trusted at all.
- Confidence intervals on the headline accuracies (bootstrap, 1000 resamples) — a 50-query accuracy without an interval is a marketing claim.
- Outputs: `report.md` (human-readable, this is what the README quotes) and `report.json` (machine-readable, for regression comparison between prompt versions).

## 6. Data Layout & Artifacts

```
runs/<run_id>/
  meta.json            # git sha, config snapshot, model ids, prompt versions, subset hash
  generations.jsonl    # one row per question: raw response, parsed SQL, latency, cache hit
  executions.jsonl     # one row per question: outcome, rows, truncated sample, error
  match.jsonl          # one row per question: match bool, diff summary
  judgements.jsonl     # one row per judged question: verdict, mode, confidence, rationale, span
  report.md / report.json
labels/human_labels.jsonl
subset.yaml
```
Runs are immutable and self-describing: `meta.json` alone is enough to reproduce the report. Report numbers are always derived from these files, never hand-entered — the README cites `report.json`.

`runs/` is working state and is gitignored. A run becomes public evidence only via
`harness publish`, which copies it to `results/<run_id>/` **and recomputes the headline
accuracy from `match.jsonl` by an independent path**, refusing to publish if the
recomputed number disagrees with `report.json`. A published number is therefore
checkable by a reviewer without re-running a model, and a doctored `report.json` is
detectable (`metrics/verify.py`).

## 7. Configuration

All config via environment variables, one `Settings` object, no values in code.

| Var | Purpose | Default |
|---|---|---|
| `GENERATOR_MODEL` | model id for SQL generation | (must be set) |
| `JUDGE_MODEL` | model id for the judge | (must be set) |
| `GENERATOR_MODE` | `standard` \| `batch` | `standard` |
| `JUDGE_MODE` | `standard` \| `batch` | `batch` |
| `OPENAI_BASE_URL` | API base, for gateways | provider default |
| `OPENAI_API_KEY` | credential | (required, never committed) |
| `DATA_DIR` | BIRD download location | `./data/bird` |
| `SUBSET_PATH` | subset manifest | `subset.yaml` |
| `MAX_CONCURRENCY` | standard-mode request cap | `8` |
| `SQL_TIMEOUT_SECONDS` | per-query execution budget | `5` |
| `MAX_ROWS` | row cap per query | `1000` |
| `LABEL_SAMPLE_SIZE` | human-label sample | `50` |
| `LANGSMITH_TRACING` | tracing on/off | `false` |

`.env.example` documents every var with placeholder values. `.env` is gitignored. No key is ever written to an artifact; `meta.json` records only the model ids.

> **Decided 2026-09-30:** `JUDGE_MODE=batch` targets the **standard OpenAI Batch API** (`/v1/batches`, JSONL upload of `/v1/chat/completions` requests, `/v1/batches/{id}` polling, `/v1/batches/{id}/output` retrieval). No gateway-specific behaviour is assumed; `OPENAI_BASE_URL` stays configurable only so the same code runs against a proxy if one is ever needed.
>
> **Still open:** the exact model id strings for `GENERATOR_MODEL` and `JUDGE_MODEL` (referred to as "gpt 6 luna"). Config-only surface, so this blocks M2 and nothing earlier.

## 8. LangGraph / LangSmith

- **LangGraph** is used only where it earns its place: a small judge graph — `prepare_context → judge → validate_span → persist`, with a conditional edge that routes unciteable judge output to one constrained retry before falling back to `unscored`. A linear function would not express the retry-and-degrade behaviour. No other stage uses a graph; generation is a fan-out, not a workflow.
- **LangSmith** tracing is opt-in and wraps the judge node and the generation calls, so the report can link a number to the trace that produced it. Tracing is off by default so a run is reproducible without an account.

## 9. Testing Strategy

Ground truth exists for the deterministic layers, so they are tested properly:

- **ExecMatch** — the highest-value unit tests. Multiset ordering, `NULL` handling, `1` vs `1.0`, `ORDER BY` sensitivity, truncated-row cases. Table-driven.
- **SQL guard** — rejects `DROP`, multi-statement injection, and a `CROSS JOIN` blowup; proves the read-only connection rejects writes.
- **Taxonomy mapping** — every judge emission maps to exactly one enum member, or is rejected; no silent fallthrough.
- **Kappa** — verified against a hand-computed 2x2 example, plus a degenerate (all-one-label) case so we know what it prints when kappa is undefined.
- **Judge output validation** — malformed JSON, out-of-enum mode, and a non-substring `evidence_span` are all rejected in tests.
- The judge's *scoring quality* is not unit-tested. It is measured by the kappa number, which is the honest position to be in.

## 10. Module Layout

```
src/harness/
  config.py            settings
  dataset/             loader, subset, schema introspector
  generation/          port.py, standard.py, batch.py, cache.py, parsing.py
  execution/           executor.py, guard.py
  scoring/             exec_match.py, taxonomy.py
  judge/               port.py, openai.py, graph.py, prompts.py, validate.py
  labeling/            cli, sampling
  metrics/             agreement.py (kappa), report.py
  tracing/             langsmith.py
  cli.py               run | batch-submit | batch-collect | judge | label | report
```

## 11. Milestones

| # | Deliverable | Done when | State |
|---|---|---|---|
| M1 | Dataset + executor + ExecMatch | all tests green; baseline of the *gold* SQL scores 1.0 on the subset (this is the harness's own sanity check) | **done** — 90/90 |
| M2 | Generator, standard mode | full subset runs end to end, `report.md` with a real accuracy number | **done** — 65.0% EX (95% CI 57.8–71.7), n=180 |
| M3 | Judge + validation | all failures carry a taxonomy label with a cited span | **done** — 180/180 usable, kappa 0.779 vs execution |
| M4 | 50 human labels + kappa | judge-vs-human and judge-vs-execution kappas in the report | not started |
| M5 | Report polish, README, second prompt version | a prompt change is measurable as a delta on the same subset | not started |
| M6 | Resume integration | only after M4 has real numbers | blocked on M4 |

M1's gold-SQL baseline is a deliberate first milestone: if the harness cannot score a known-correct query at 100%, every later number is untrustworthy, and that is cheaper to discover on day one.

### Decisions forced by the M2 run (2026-09-30)

1. **`temperature` is not sent unless configured.** `gpt-6-luna` rejects `temperature=0`
   and accepts only its default. Reproducibility therefore comes from the
   content-addressed cache and the recorded `prompt_version`, not from sampling
   determinism. The consequence must be stated in the report: two fresh runs of the same
   prompt are *not* guaranteed identical, so a prompt-version comparison needs the cache
   invalidated on one side only.
2. **A run with any API error is invalid, and says so.** The first run of this harness
   rejected all 90 requests on `temperature` and reported "0% accuracy" — indistinguishable
   from a model that fails everything. `Report.valid` is now false whenever any generation
   errored, `report.md` is stamped `INVALID RUN`, and `harness generate` exits 3. A
   transport failure is not a measurement.
3. **`evidence` is given to the generator**, so this accuracy is not comparable to
   published BIRD numbers. Stated in the README rather than left for a reader to guess.

### Decisions forced by the M3 run (2026-09-30)

1. **The judge needed a third verdict.** Section 5.5 originally gave it `generator_error`
   and `sql_error` only — with no way to say "correct". That made the 20% audit sample of
   successes incoherent: every audited query would have been forced into a "wrong" label.
   `sql_ok` was added, and a `sql_ok` verdict paired with a real failure mode is now
   *rejected* as hedging rather than accepted.
2. **A number and its string form are the same value.** Question `931` was scored wrong
   because gold returned TEXT `'202.484'` and the generated query returned REAL `202.484`.
   `ExecMatch` now canonicalises a numeric string against a number, guarded by a
   round-trip check so a zero-padded identifier ('007') stays distinct from 7. Re-scoring
   the identical generations flipped exactly that one verdict: 64.4% → 65.0%. The fix was
   made despite moving a published number, because a known defect in the foundation is
   worse than an inconvenient number.
3. **Judge runs are snapshots, not logs.** Re-judging appended a second copy of every row
   and doubled every count. `persist` now overwrites, deduplicated by cache key.
   `harness verify` is what caught this.
4. **`publish` verifies before copying.** It had published a run that then failed
   verification, leaving a failing artifact in `results/` as if it were checked evidence.
5. **The judge-standard accuracy is quoted only on a full-run pass.** On a
   failures-plus-audit sample the population is deliberately failure-weighted, so an
   "accuracy" on it is not comparable to the run's accuracy and would invite the wrong
   reading.

### Observed on the M2 run (n=180, generation)

| Slice | Accuracy |
|---|---|
| overall | 65.0% (95% CI 57.8–71.7) |
| superhero (10 tables) | 76.7% |
| formula_1 (13 tables) | 63.3% |
| thrombosis_prediction (3 tables) | 53.3% |
| simple / moderate / challenging | 72.7% / 67.7% / 49.0% |

**Run-to-run variance is real and measured.** Two identical runs of the same prompt and
model scored 66.7% (60/90) and 65.6% (59/90). With `temperature` unavailable, the only
reproducibility levers are the content-addressed cache and the recorded
`prompt_version`; a *fresh* run is a fresh sample. Consequences, both adopted: quote
intervals rather than points, and in M5 compare prompt versions with the cache
invalidated on exactly one side so the delta is attributable to the prompt.

Difficulty separates cleanly and monotonically; the three per-database intervals overlap,
so the per-database ranking is reported as not established.

### Observed on the M3 run (n=180, judge)

| Measure | Value |
|---|---|
| judge outputs usable | 180 / 180 |
| fabricated citations | 0 |
| invented categories | 0 |
| Cohen's kappa vs execution ground truth | **0.779** (observed 90.6%, chance 57.3%) |
| false accusations (correct query called wrong) | **0** / 117 |
| missed failures (wrong query called correct) | **17** / 63 |
| strict execution accuracy | 65.0% |
| accuracy under the judge's semantic standard | 74.4% |

Two findings, both of which are the project rather than a footnote on it:

1. **Judge error is one-directional.** Zero false accusations across 117 correct queries;
   all 17 errors are leniency on wrong queries. This is only visible because the audit
   design put *successes* in front of the judge — a failure-only judging pass would have
   reported a flattering pass rate and hidden the direction entirely.
2. **The correctness definition is worth 9.4 points.** Same 180 questions, same model,
   same gold: 65.0% under strict result-set equality, 74.4% under the judge's semantic
   standard. An eval that does not state its convention is quoting a number with an
   unstated 9-point error bar.

Of the 63 failures, the judge classified 46. The 17 it accepted are the interesting
population for M4: separating "wrong but semantically equivalent" from "wrong and the
judge was lenient" is exactly what the human labels are for, and it is not answerable
without them.

## 12. Risks

| Risk | Mitigation |
|---|---|
| ExecMatch mismatch with BIRD's official scorer | state our rules explicitly and publish the exact-match count alongside the loose count |
| Small N makes the headline accuracy noisy | bootstrap CIs; lead the README with the interval, not the point estimate |
| Judge anchoring itself on the gold SQL and grading generously | Gold SQL is visible **by decision** (section 5.5); the mitigation is structural, not concealment: correctness is execution-derived, and leniency surfaces as kappa-vs-human disagreement. `evidence_span` forces the judge to cite what it relied on. |
| Kappa undefined or depressed by taxonomy skew | report observed agreement alongside kappa; keep the taxonomy coarse enough to have mass |
| Leakage of judge output into human labels | the `label` CLI does not render judge output; enforce by test |
| Model ids / batch endpoint unconfirmed | config-only surface; M1 is unaffected |
| Cost overrun on repeated runs | content-addressed cache plus resumable batch adapter |

## 13. Resume Claims (draft, pending M4)

Only the following are defensible once M4 lands, and each must name its N and its metric:

- Execution accuracy of a hosted LLM on a BIRD dev subset, with a bootstrap CI.
- Dominant failure mode, quantified as a share of failures.
- Judge-vs-human agreement (Cohen's kappa) on the failure taxonomy, and judge-vs-execution agreement on correctness.

No claim of SOTA, no claim of state-of-the-art, and no percentage without its denominator.
