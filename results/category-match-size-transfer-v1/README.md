# Experiment 059: category-matched context control at 1.5B

## Result

At 1.5B, the mean finite-minus-free matched-review interaction was 0.012 VA RMSE points (95% recipient-bootstrap interval [-0.139, 0.157]). All 1104 donor outputs were paired with own-review baselines; invalid free-greedy donor outputs: 0/552.

On the same 184 recipients and the same three donor maps, the 1.5B-minus-3B interaction difference was -0.095 (95% paired recipient-bootstrap interval [-0.324, 0.133]). This is a cross-size diagnostic, not an independent replication.

The study changes model size while holding recipients and donor assignments fixed. Both sizes belong to the Qwen2.5 family and use one public test release; category matching does not control exact aspect identity or specific lexical overlap. No text, case IDs, donor maps, or item predictions are published.

- Protocol: `docs/experiments/059-category-match-size-transfer.md`
- Aggregate estimates and provenance: `summary.json`
- Runner/analyzer: `scripts/run_category_match_size_transfer_059.py`, `scripts/analyze_category_match_size_transfer_059.py`
