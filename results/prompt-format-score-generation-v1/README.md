# Prompt-format effect on score/generation agreement (experiment 025)

This paired ablation compares explicit one-word labels with an open sentiment question on 233 SemEval prompts across Granite, Qwen and SmolLM2. Generated text is not retained.

| Model | Format | Exact one-word | Parseable polarity | Candidate-pair accuracy | Pair/generation agreement |
|---|---|---:|---:|---:|---:|
| granite-3.1-2b | forced_choice | 100.0% | 100.0% | 97.4% | 99.6% |
| granite-3.1-2b | open_question | 0.0% | 12.9% | 93.6% | 100.0% |
| qwen-1.5b | forced_choice | 100.0% | 100.0% | 94.8% | 100.0% |
| qwen-1.5b | open_question | 0.0% | 39.1% | 94.8% | 98.9% |
| smollm2-1.7b | forced_choice | 100.0% | 100.0% | 92.3% | 100.0% |
| smollm2-1.7b | open_question | 0.0% | 65.7% | 87.1% | 91.5% |

Open minus forced-choice agreement: -5.109% (paired sentence-bootstrap 95% CI [-8.268%, -2.500%]).
Open-label extraction coverage >=90% in all families: False. Format effect detected under the frozen rule: False.

This is a prompt-format measurement test, not a causal steering experiment. See the [frozen protocol](../../docs/experiments/025-prompt-format-score-generation.md), per-model manifests, audit, and data attribution.
