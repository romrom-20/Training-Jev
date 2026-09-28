# Experiment 045: order control by model size

## Result

For Qwen2.5-1.5B, shuffled-minus-natural masked RMSE was -0.010 (95% cluster interval [-0.058, 0.041]); the exploratory natural-order rule did not pass.

Generation took 25.0 minutes after model load.

This adaptive test compares both conditions under Qwen2.5-1.5B and the same finite one-decimal VA grammar. It reuses the same 217 public DimABSA IDs from Experiments 043/044, so it is a model-size diagnostic rather than an independent data replication. The grammar may change answer content, and shuffled whitespace tokens disrupt syntax and discourse together.

No review text, aspects, IDs, or per-item outputs are published; private data remain in ignored `.context/`.

- Protocol: `docs/experiments/045-order-control-model-size.md`
- Runner/analyzer: `scripts/run_opinion_mask_order_model_size_045.py`, `scripts/analyze_opinion_mask_order_model_size_045.py`
- Aggregate result and provenance: `summary.json`
