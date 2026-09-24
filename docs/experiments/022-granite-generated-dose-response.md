# Experiment 022: when do aspect residuals change generated answers?

## Motivation

In experiment 021, layer-23 residual steering at 5% of the training activation norm
changed candidate scores but changed none of 63 generated labels. This follow-up asks
whether a stronger intervention crosses the greedy decoding boundary and, if so,
whether learned aspect residuals do that more reliably than norm-matched random ones.

## Frozen protocol

- Reuse the pinned Granite 3.1 2B checkpoint, layer 23, synthetic training-only
  directions, TripR conflict prompts, and baseline generated answers from 021.
- Test doses 10%, 20%, and 40% of the median layer-23 training activation norm.
  Apply each target-matched learned residual and 20 target-matched random residuals
  (seeds 20261101–20261120) on every greedy decoding step; generate at most four new
  tokens. Use 40% as the sole primary dose. The lower doses describe dose response.
- Save only whether output is exactly `positive` or `negative`, and whether it matches
  the gold label. Do not retain generated or review text.
- For each prompt, define utility as generated strict correctness minus baseline
  correctness. Average prompt utility within each sentence. The primary contrast is
  trained-minus-random mean utility at 40%; use a nested sentence/seed bootstrap
  (10,000 draws, seed 20261023) and one-sided Monte Carlo rank over 20 controls.
- The behavioral gate requires a positive lower 95% bootstrap bound, Monte Carlo
  rank p<=0.05, and at least 95% exact one-word validity under trained steering.
  Report accuracy gains and harms separately. No dose is selected after looking at
  results, and lower-dose outcomes cannot rescue a failed primary dose.

This is a targeted follow-up on the same 63 prompts and same checkpoint, not an
independent replication. Its conclusion is limited to greedy answers and these doses.
