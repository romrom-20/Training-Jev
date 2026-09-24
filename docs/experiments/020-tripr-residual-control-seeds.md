# Experiment 020: multi-seed residual control check

## Question

Does the learned layer-16 aspect residual outperform the distribution of equally sized
random residuals on reviews where at least two aspects have conflicting sentiment?
Experiment 019 used one random residual realization per source direction, which leaves
open the possibility that its apparent advantage over that control is seed-specific.

## Frozen protocol

- Use the 29 TripR-2020Large conflict sentences from the frozen 018 category mapping,
  including every mapped aspect query from each sentence. Do not add or remove examples
  based on model answers. Reuse the pinned models, prompt wording, baseline rows, and
  layer-16 training-only directions from 018/019.
- The learned arm is the three orthogonal layer-16 residual directions from 019.
  Generate 20 independent random-residual controls using seeds 20261101–20261120. Each
  control vector is orthogonal to the shared sentiment axis and matches the norm of
  its corresponding learned residual. Apply each source direction to every target
  query at the 019 fixed dose. No TripR labels or activations define a direction.
- For each model and random seed, calculate the mean within-sentence diagonal-minus-
  off-diagonal polarity-margin shift. The primary contrast is the learned residual
  statistic minus the random-seed mean. Also report the one-sided Monte Carlo rank
  `(1 + number of seeds at least as large as learned) / 21`.
- The cross-model control-robustness gate passes only if, in both models, the learned
  residual exceeds the random-seed mean with a nested sentence/seed bootstrap 95%
  interval entirely above zero, the Monte Carlo rank is at most 0.05, and baseline
  strict accuracy on the conflict queries exceeds chance. Bootstrap seed: 20261003;
  10,000 draws.
- Report all 20 per-seed effects for both models. The test focuses on conflict sentences
  because the claim is aspect binding under polarity disagreement. It reuses the 018/019
  benchmark and therefore tests control-seed robustness, not dataset generalization.
