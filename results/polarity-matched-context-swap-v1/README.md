# Experiment 056: polarity-matched context swap

## Result

With donors matched to each recipient's gold-valence polarity, the mean finite-minus-free matched-review interaction was 0.436 VA RMSE points (95% recipient-bootstrap interval [0.272, 0.599]). The three fixed permutation estimates ranged from 0.392 to 0.468; that range is descriptive.

Each donor review shares its recipient's negative/positive gold-valence bucket, but comes from another case. The recipient's aspect and gold stay fixed. This tests whether the decoder-specific exact-review advantage survives controlling that coarse polarity cue.

The test uses the fresh 217-case subset from Experiment 055 (108 negative and 109 positive), one public benchmark, and one model family. Donors still differ in many lexical and semantic properties besides aspect linkage; pretraining exposure is possible. Intervals condition on three fixed mappings. No text, item IDs, donor maps, or item-level outputs are published.

- Protocol: `docs/experiments/056-polarity-matched-context-swap.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_polarity_matched_context_swap_056.py`, `scripts/analyze_polarity_matched_context_swap_056.py`
