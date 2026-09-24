# Experiment 023: does candidate-score ranking match generated answers?

## Motivation

In experiment 021, the positive-versus-negative candidate margin predicted the
generated label on 62/63 TripR conflict prompts, and low gold-aligned margins
identified all six generation errors. This was post hoc and based on a small subset.
Experiment 023 tests score/generation alignment on the separate SemEval-2014
Restaurants benchmark before any intervention.

## Frozen protocol

- Use the same pinned Granite 3.1 2B Instruct checkpoint and the existing frozen
  SemEval-2014 multi-aspect prompt filter (233 prompts across 112 sentences).
- For every prompt, collect the positive-minus-negative next-token candidate margin
  and one greedy answer with a maximum of four new tokens. Save only margins, token
  IDs, exact one-word validity, and correctness; do not save review or generation text.
- Primary endpoint: among exact one-word generations, the fraction where the
  positive/negative candidate-margin sign predicts the generated label. Bootstrap
  sentences (10,000 draws, seed 20261024). The endpoint is considered measurable only
  if at least 95% of generations are exact one-word answers; otherwise report it as
  descriptive and fail the measurement gate.
- Report gold accuracy from the candidate-pair margin and generation, their agreement,
  the all-vocabulary top-1 accuracy, and the AUC of the gold-aligned candidate margin
  for identifying generation errors. The error AUC is secondary because its precision
  depends on the number of errors.

This is an observational measurement study, not a steering or causal test. It checks
whether candidate-pair scores track greedy generated decisions on one new dataset and
does not establish general calibration.
