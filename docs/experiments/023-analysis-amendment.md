# Experiment 023 analysis amendment

After collection, review found that the frozen protocol's secondary error-detection
endpoint called for a *gold-aligned* margin. That quantity uses the reference label
and therefore encodes candidate-pair correctness; it cannot be presented as a
label-free predictor of generation errors. The primary candidate-pair/generation
agreement endpoint is unaffected.

The packaged analysis replaces the invalid secondary endpoint with negative absolute
candidate margin, which is available before observing the gold label. Its AUC is
reported as exploratory and without a confidence interval because there are only
seven generation errors. The frozen protocol file and its original hash remain
unchanged; this amendment is linked from the result bundle.
