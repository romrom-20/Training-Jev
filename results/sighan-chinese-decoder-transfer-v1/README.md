# Experiment 052: Qwen2.5-3B SIGHAN Chinese transfer

## Result

On the SIGHAN Chinese restaurant test split, finite-grid context gain was 2.407 (95% source-ID interval [2.339, 2.477]) and free-greedy gain was -0.384 (95% interval [-0.426, -0.344]). The registered interaction was 2.791 (95% interval [2.734, 2.849]); its practical gate passed. The descriptive English-minus-Chinese interaction difference was -0.015 (95% independent-release interval [-0.126, 0.100]).

The test is a public, separately curated Chinese restaurant-review release with human continuous aspect-linked VA labels. It is an external release and cross-language transfer; it is not pretraining-blind, changes language and corpus at once, and remains in the restaurant domain. The English-minus-Chinese secondary interval is descriptive and cannot identify language effects separately from release effects.

**Interpretation correction:** the executed `aspect_only` prompt contains the target aspect and `[NOT PROVIDED]` for review text. The `opinion_masked` prompt contains the full review with all annotated opinion spans replaced by `[MASKED]`. Therefore the registered contrast measures utility or harm from the remaining non-opinion review context; it does not measure the value of visible opinion words. The frozen protocol's prose describing `aspect_only` as showing the complete review was incorrect; see `docs/experiments/052-interpretation-correction.md`. The run and estimates are unchanged.

Post-hoc most common numeric outputs by cell are recorded in `summary.json`. Treat them as exploratory mechanism clues.

No review text, target aspects, opinions, IDs, raw generations, or item predictions are published; private outputs remain in ignored `.context/`.

- Protocol: `docs/experiments/052-sighan-chinese-decoder-transfer.md`
- Runner/analyzer: `scripts/run_sighan_chinese_decoder_transfer_052.py`, `scripts/analyze_sighan_chinese_decoder_transfer_052.py`
- Aggregate result and provenance: `summary.json`
