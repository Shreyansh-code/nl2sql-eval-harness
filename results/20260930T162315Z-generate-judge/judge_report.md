# Judge report

- run: `20260930T162315Z-generate-judge`
- judge: `gpt-6-luna`
- judged: 180 (63 failures + 117 audited successes)

## Judge output health

- ok: 180

## Failure modes (generator's failures only)

| Mode | Count | Share | Meaning |
|---|---|---|---|
| `SCHEMA_MISREAD` | 5 | 11% | Used a table or column that does not exist, or exists but means something else than the question implies (e |
| `WRONG_JOIN` | 2 | 4% | Chose the right tables but joined them wrongly: missing join, wrong key, or a join that multiplies rows when it should not |
| `WRONG_FILTER` | 5 | 11% | Filtered on the right column with the wrong comparison, threshold, or literal value (>, >=, <, <=, or an equality against the wrong value) |
| `MISSING_CONDITION` | 2 | 4% | Omitted a condition the question or the evidence field requires, e |
| `AGGREGATION_ERROR` | 12 | 26% | Wrong aggregate, wrong GROUP BY, or aggregation applied at the wrong level (per-row instead of per-group, or counting the wrong thing) |
| `SELECTION_ERROR` | 9 | 20% | Filtered and joined correctly but returned the wrong columns, or the wrong number of columns |
| `UNIT_CONVERSION` | 3 | 7% | Value scale or format error: percent versus fraction, epoch versus date string, unit conversion, or rounding |
| `PRECISION` | 2 | 4% | Right answer to the question, wrong only on ordering, LIMIT, or ties |
| `TIMEOUT` | 1 | 2% | The query was too expensive to finish inside the time budget |
| `EMPTY_WRONG` | 5 | 11% | The query ran and returned no rows, but the gold query returned rows |

## Is the judge trustworthy?

Judge verdict vs execution ground truth, on every failure plus the audit sample of successes. No human labels needed.

- kappa 0.779 (observed agreement 90.6%, expected 57.3%, n=180)

### The disagreement is one-directional

| | execution says correct | execution says wrong |
|---|---|---|
| judge says correct (`sql_ok`) | 117 | 17 |
| judge says wrong | 0 | 46 |

**0 false accusation(s)** — the judge never called a correct query wrong. **17 missed failure(s)** — the judge called a wrong query correct.

Direction is the actionable part. A judge that accused correct queries would be unusable; a judge that is only lenient on wrong ones is a calibration problem with a known direction, and the fix is different. Note also that the failure histogram above is built only from the failures the judge *did* label, so it describes 46 of the failures, not all of them.

Every scored question was judged, so the two standards can be compared directly on the same population: strict execution accuracy is 65.0%, and the judge's own standard would give 74.4%. The gap is the cost of choosing strict result-set equality over semantic equivalence.

**Read this narrowly.** The judge is shown the gold SQL and the gold rows, so recognising a correct query is easier than detecting a wrong one blind. This number says the judge's verdicts are consistent with ground truth when it can see ground truth; it is not a blind error-detection score, and a high value here is not evidence that the judge could diagnose errors unaided.

## Judge vs human (failure mode)

Not yet measured: this needs the 50 blinded human labels from M4. No judge-vs-human agreement is claimed.
