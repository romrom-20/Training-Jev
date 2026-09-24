# Cross-family score/generation alignment (experiment 024)

This observational replication applies the same 233 SemEval prompts to Qwen2.5-1.5B and SmolLM2-1.7B. It stores no review or generated text.

| Model | Exact one-word | Candidate-pair accuracy | Generated accuracy | Pair/generation agreement (95% CI) | All-vocabulary top-1 | Gate |
|---|---:|---:|---:|---:|---:|---|
| qwen-1.5b | 100.0% | 94.8% | 94.8% | 100.0% [100.0%, 100.0%] | 94.8% | PASS |
| smollm2-1.7b | 100.0% | 92.3% | 92.3% | 100.0% [100.0%, 100.0%] | 92.3% | PASS |

SmolLM2 minus Qwen agreement: +0.000% (paired sentence-bootstrap 95% CI [+0.000%, +0.000%]).
Cross-family replication gate: PASS.

This is a noncausal measurement replication on one dataset; it does not establish general calibration or intervention effects. See the [frozen protocol](../../docs/experiments/024-cross-family-score-generation.md), seed, audit, and data attribution.
