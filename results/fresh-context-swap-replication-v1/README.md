# Experiment 055: fresh-sample context-swap replication

## Result

On 217 new IDs disjoint from Experiments 053–054, the mean finite-minus-free matched-review interaction was 0.411 VA RMSE points (95% recipient-bootstrap interval [0.254, 0.563]). The three permutation estimates ranged from 0.259 to 0.506; that range is descriptive.

The 217 recipient cases are disjoint from the first 217-case sample, with 108 negative- and 109 positive-valence cases. All neutral cases were used in the earlier sample, so this replication does not test neutral cases. Each aspect/gold pair stayed fixed while three length-binned donor reviews were assigned in deterministic derangements. The result uses the same public dataset and model family as the earlier tests.

Donor reviews were not matched by polarity, so coarse-polarity effects remain possible. The intervals condition on the three mappings, and the public test may have appeared in pretraining. No text, item IDs, donor mappings, or individual outputs are published.

The bootstrap seed was corrected to the registered `20260955` after an analyzer-wrapper oversight; see `docs/experiments/055-analysis-seed-correction.md`. Model outputs and point estimates did not change.

- Protocol: `docs/experiments/055-fresh-context-swap-replication.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_fresh_context_swap_replication_055.py`, `scripts/analyze_fresh_context_swap_replication_055.py`
