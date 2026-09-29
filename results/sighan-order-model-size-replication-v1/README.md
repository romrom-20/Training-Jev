# Experiment 065: output-order model-size comparison on SIGHAN

## Result

The 1.5B-minus-3B difference in aggregate order moderation was 0.045 VA-RMSE points (95% paired recipient-bootstrap interval [-0.227, 0.336]) on 173 complete IDs.

This paired difference is the registered primary endpoint. A confidence interval containing zero means the test did not detect a model-size difference; it does not prove transfer, especially if both model-specific estimates are near zero.

The model-specific aggregate order moderations were -0.032 for Qwen2.5-1.5B (95% interval [-0.158, 0.090]) and -0.077 for Qwen2.5-3B (95% interval [-0.335, 0.169]). Both intervals include zero. The study reuses the 180 SIGHAN recipients and exact donor maps from Experiment 064, adding 5,040 1.5B generations. Invalid-output and complete-case counts are in `summary.json`.

This is a same-release, same-family model-size comparison, not an independent corpus replication. Chinese text is scored with English instructions, public-test pretraining exposure cannot be excluded, and donor category changes also alter review content. The experiment cannot explain the English/Chinese difference by itself.

- Frozen protocol: `docs/experiments/065-sighan-order-model-size-replication.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_sighan_output_key_order_065.py`, `scripts/analyze_sighan_order_model_size_065.py`
- Prior Chinese release test: [Experiment 064](../sighan-output-key-order-replication-v1/README.md)
