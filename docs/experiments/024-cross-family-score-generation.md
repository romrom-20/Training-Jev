# Experiment 024: does score/generation agreement transfer across model families?

## Motivation

Experiment 023 found 99.6% agreement between positive/negative candidate-pair
decisions and greedy generated labels for Granite on SemEval restaurant prompts.
This follow-up asks whether that alignment also holds for two already cached model
families on the same prompts.

## Frozen protocol

- Evaluate Qwen2.5-1.5B-Instruct and SmolLM2-1.7B-Instruct at their pinned revisions
  from `task_ladder.MODEL_SPECS` on the identical 233 SemEval prompts (112 sentences).
- For every prompt, collect the positive-minus-negative candidate margin and one
  greedy answer (maximum four new tokens). Save only anonymized IDs, scores, token
  IDs, exact-answer validity and correctness; retain no review or generated text.
- Per-model endpoint: agreement between the candidate-pair margin sign and generated
  label among exact one-word outputs, with a sentence-cluster bootstrap (10,000 draws,
  seed 20261025). Report each model's validity and strict gold accuracy.
- Cross-family endpoint: paired sentence-bootstrap difference in agreement rate
  (SmolLM2 minus Qwen). A model's measurement gate passes only if one-word validity is
  at least 95% and the lower 95% agreement interval exceeds chance (0.5). The
  cross-family replication passes only if both model gates pass.

This is a noncausal measurement replication on the same dataset used in 023. It does
not test interventions, calibration under distribution shift, or general task ability.
