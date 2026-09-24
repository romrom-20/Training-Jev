# Granite candidate scores and generated answers (experiment 023)

This observational test compares the sign of positive-versus-negative candidate logits with greedy generated labels on 233 SemEval aspect prompts (112 sentences). It tests score/generation alignment, not steering or causality.

- Exact one-word generation rate: 100.0%.
- Candidate-pair accuracy: 97.4%; generated accuracy: 97.0%; unrestricted next-token top-1 accuracy: 72.1%.
- Candidate-pair/generation agreement: 99.6% (sentence-bootstrap 95% CI [98.7%, 100.0%]).
- Generation errors: 7; exploratory error-detection AUC using negative absolute candidate margin: 0.700 (no interval; seven errors).
- Frozen measurement gate: PASS.

The original protocol's gold-aligned error score used the reference label; the corrected label-free absolute-margin analysis is documented in the [analysis amendment](../../docs/experiments/023-analysis-amendment.md). See also the [frozen protocol](../../docs/experiments/023-granite-score-generation-alignment.md), audit, and data attribution.
