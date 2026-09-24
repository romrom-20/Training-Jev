# Experiment 019: shared sentiment versus aspect residual at layer 16

## Question

Experiment 018 found that layer 16 has a larger target-specific fraction than layer
24 on TripR-2020Large reviews. Is that specificity carried by the aspect-specific
residual in each frozen food/service/price direction, or can it be explained by the
shared positive-sentiment component?

Prior work has decomposed steering directions into shared and residual components
(experiment 011), but only on a synthetic review task. Experiment 019 applies that
decomposition to human-annotated, conflicting restaurant reviews. This is a mechanistic
follow-up on the same benchmark as 018, not an independent replication.

## Frozen protocol

- Dataset, prompts, labels, model revisions, and layer-16 training-only directions are
  exactly those in experiment 018. No TripR labels or activations are used to form
  intervention vectors.
- For the three unit directions `d_i`, define the shared axis as the normalized mean
  `u = normalize(mean_i(d_i))`. The shared component is `s_i = dot(d_i,u)u`; the
  residual is `r_i = d_i - s_i`. A random residual is sampled orthogonal to `u` and
  scaled to `||r_i||`, with seed 20261001.
- Apply native, shared, residual, and random-residual vectors from each source aspect
  to every target-aspect prompt. Preserve component norms; use the same 5% median
  layer-16 training-activation norm dose as experiment 018. This yields a paired
  4-condition × 3-source comparison for every query.
- Primary outcome is the within-sentence diagonal-minus-off-diagonal polarity-margin
  shift, with 5,000 sentence-bootstrap draws (seed 20261002). The residual mechanism
  gate passes only if residual selectivity exceeds both shared and random-residual
  selectivity with paired 95% intervals above zero in both models, and each model's
  baseline accuracy exceeds chance. Report every model and all component effects.
- Report conflict-only contrasts, strict accuracy, generic logit shifts, and component
  norms as secondary outcomes. No post-hoc subgroup determines the primary result.

The project has already inspected the 018 outcomes on these sentences. Therefore 019
is a preregistered component comparison, but its benchmark is reused and it must not be
described as independent confirmation.
