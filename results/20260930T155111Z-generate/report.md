# Run report

- run: `20260930T155111Z-generate`
- git: `c64bebc`
- generator: `gpt-6-luna` (standard)
- subset: `bird-dev-mini`

## Execution accuracy

**65.6% (95% CI 56.7-74.4)** on n=90 (0 excluded).

| Slice | Accuracy | n |
|---|---|---|
| database: formula_1 | 53.3% (95% CI 33.3-70.0) | 30 |
| database: superhero | 76.7% (95% CI 63.3-90.0) | 30 |
| database: thrombosis_prediction | 66.7% (95% CI 50.0-83.3) | 30 |
| difficulty: challenging | 56.7% (95% CI 40.0-73.3) | 30 |
| difficulty: moderate | 73.3% (95% CI 56.7-90.0) | 30 |
| difficulty: simple | 66.7% (95% CI 50.0-83.3) | 30 |

## Generation outcomes

- ok: 90

## Execution outcomes

- ok: 90

## Tokens

- input: 108680
- output: 22979

## Not yet measured

Failure-mode classification and judge-vs-human agreement (M3/M4) are absent. No diagnostic claim is made by this report.
