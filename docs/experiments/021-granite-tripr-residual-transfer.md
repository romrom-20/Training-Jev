# Experiment 021: residual transfer to a third model family

## Question

Does the layer-16 aspect-residual result transfer beyond Qwen2.5 and SmolLM2, and
does it affect generated answers as well as next-token candidate scores? Use the
independent IBM Granite 3.1 2B Instruct family. Its pinned 40-layer model uses layer 23
(57.5% depth), matching Qwen2.5-1.5B layer 16 (57.1% depth).

## Frozen protocol

- Model: `ibm-granite/granite-3.1-2b-instruct`, revision
  `bbc2aed595bd38bd770263dc3ab831db9794441d`; local float32 MPS inference.
- Form three positive-minus-negative food, service, and value directions from only the
  synthetic `mixed_review`, format-0 training split in `task_ladder.make_rows()`. Use
  layer 23, normalize each native direction, decompose it into its shared-sentiment
  projection and orthogonal residual, and scale interventions by 5% of the median
  training activation norm. No TripR text, labels, or activations form directions.
- Evaluate the 385 frozen TripR aspect prompts (187 sentences, 29 cross-aspect polarity
  conflicts) using baseline, native, shared, residual, and norm-matched random-residual
  source-by-target candidate-score interventions. On the conflict subset, add 20
  independent norm-matched random residual controls (seeds 20261101–20261120).
- Primary score endpoint is conflict-only residual diagonal-minus-off-diagonal margin
  shift versus the 20-seed random-control distribution. Use the nested sentence/seed
  bootstrap (10,000 draws, seed 20261003) and one-sided Monte Carlo rank. The score gate
  requires a positive lower 95% interval, Monte Carlo rank <= .05, and conflict-query
  baseline accuracy above chance.
- On the same 63 conflict queries, greedily generate answers with no intervention,
  target-matched learned residual, and target-matched seed-20261001 random residual.
  Apply the intervention at layer 23 on every decoding step, max 4 new tokens. Report
  exact one-word answer validity and strict gold-answer accuracy as a secondary
  behavioral check; do not replace a failed score gate with generation outcomes.

This is a new model-family test, but it reuses TripR-2020Large and small-model-trained
directions, so it does not establish cross-dataset transfer or a universal mechanism.
