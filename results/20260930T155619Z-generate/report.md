# Run report

- run: `20260930T155619Z-generate`
- git: `d118c75`
- generator: `gpt-6-luna` (standard)
- subset: `bird-dev-mini`

## Execution accuracy

**64.4% (95% CI 57.8-71.1)** on n=180 (0 excluded).

| Slice | Accuracy | n |
|---|---|---|
| database: formula_1 | 63.3% (95% CI 51.7-75.0) | 60 |
| database: superhero | 76.7% (95% CI 65.0-88.3) | 60 |
| database: thrombosis_prediction | 53.3% (95% CI 41.7-66.7) | 60 |
| difficulty: challenging | 49.0% (95% CI 34.7-63.3) | 49 |
| difficulty: moderate | 67.7% (95% CI 55.4-78.5) | 65 |
| difficulty: simple | 72.7% (95% CI 60.6-81.8) | 66 |

## Generation outcomes

- ok: 180

## Execution outcomes

- ok: 171
- empty: 8
- timeout: 1

## Tokens

- input: 217133
- output: 40775

## Not yet measured

Failure-mode classification and judge-vs-human agreement (M3/M4) are absent. No diagnostic claim is made by this report.
