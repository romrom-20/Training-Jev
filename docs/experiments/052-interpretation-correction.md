# Experiment 052 — post-run interpretation correction

The frozen [052 protocol](052-sighan-chinese-decoder-transfer.md) contains one
incorrect sentence in its condition description: it says `aspect_only` shows the
complete review. The executed runner inherited the prompt construction in
`run_laptop_decoder_transfer_049.py`:

- `aspect_only` passes `None` as review text. `build_prompt` renders this as
  `[NOT PROVIDED]`, while still supplying the named target aspect.
- `opinion_masked` supplies the full source review with every annotated non-null
  opinion span replaced by `[MASKED]`, while retaining the target aspect.

Thus `RMSE(aspect_only) - RMSE(opinion_masked)` measures utility (positive) or harm
(negative) from the remaining **non-opinion review context**, relative to an
aspect-only/no-review-text baseline. It does not measure the effect of visible or
unmasked opinion words. The registered estimates, output rows, code execution,
bootstrap, thresholds, hashes, and invalid-output gate are unchanged; only the prose
interpretation was wrong. The aggregate report and research plan use the corrected
estimand. The original protocol file is preserved unchanged so its frozen hash still
matches the run manifest.
