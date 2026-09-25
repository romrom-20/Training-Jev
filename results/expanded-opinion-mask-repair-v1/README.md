# Experiment 043: expanded opinion-mask repair

## Result

The primary RMSE contrast (aspect-only minus opinion-masked) was 1.845 (cluster-bootstrap 95% interval [1.649, 2.043]); valence and arousal contrasts were 1.795 and 1.898; the registered support gate passed.

Generation took 43.5 minutes after model load on mps.

This is a disjoint-item rerun on the same DimABSA release, language files and Qwen model. The 0.1-step constrained output grid ensures values stay in range and parse, but can alter model answers. The bootstrap resamples source IDs with all three aligned language rows kept together. This does not establish independent-corpus or human generalization.

No review text, aspects, item IDs or item-level predictions are redistributed. They remain local in ignored `.context/`.

- Protocol: `docs/experiments/043-expanded-opinion-mask-repair.md`
- Runner/analyzer: `scripts/run_expanded_opinion_mask_repair_043.py`, `scripts/analyze_expanded_opinion_mask_repair_043.py`
- Aggregate result and provenance: `summary.json`
