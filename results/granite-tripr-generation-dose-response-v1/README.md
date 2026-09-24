# Granite generated-answer dose response (experiment 022)

This follow-up tests target-matched learned residuals against 20 random residual seeds on the 63 TripR conflict prompts. It stores answer validity and correctness only, not generated text.

Baseline generated accuracy: 90.5%.
Primary 40% dose: trained-minus-random utility -0.0086 (nested 95% CI [-0.0405, +0.0172]); random-seed rank p=0.9048; behavioral gate FAIL.

| Dose | Trained accuracy | Exact one-word | Label flips | Gains / harms | Trained − random utility | 95% CI | Rank p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | 88.9% | 100.0% | 1 | 0 / 1 | -0.0138 | [-0.0491, +0.0052] | 1.0000 |
| 20% | 88.9% | 100.0% | 1 | 0 / 1 | -0.0112 | [-0.0440, +0.0112] | 0.9048 |
| 40% | 88.9% | 100.0% | 1 | 0 / 1 | -0.0086 | [-0.0405, +0.0172] | 0.9048 |

The 40% dose is the sole primary endpoint; lower doses are descriptive. This reuses the same checkpoint and prompts as 021 and is not an independent replication. See the [frozen protocol](../../docs/experiments/022-granite-generated-dose-response.md), seed-level analysis, provenance, and audit.
