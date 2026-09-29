# Experiment 067: forced-coordinate prefix coupling

## Result

The average second-coordinate shift after forcing the first value from 2.0 to 8.0 was -0.266 VA points (95% recipient-bootstrap interval [-0.469, -0.070]) on 64 complete recipients.

This is a counterfactual continuation test: the review and user prompt stay fixed, while an assistant output prefix supplies either a low or high first VA score and the model generates the other score. It measures how the continuation responds to a forced prefix, not how accurate a normal score is. The test uses 64 hash-selected SIGHAN reviews and 256 new greedy continuations on Qwen2.5-3B.

The order-specific shifts went in opposite directions: +0.156 when valence was first (95% interval [-0.047, +0.359]) and -0.688 when arousal was first (95% interval [-0.984, -0.391]). The pooled primary is therefore an average of asymmetric, potentially different processes; it should not be described as symmetric coordinate anchoring.

The primary contrast and order-specific results are in `summary.json`. Raw review text, IDs, prefixes and model outputs remain private in `.context/`.

- Frozen protocol: `docs/experiments/067-forced-coordinate-prefix-coupling.md`
- Literature boundary: `docs/experiments/064-literature-boundary-audit.md`
- Runner/analyzer: `scripts/run_forced_coordinate_prefix_067.py`, `scripts/analyze_forced_coordinate_prefix_067.py`
- Related prompt-language experiment: [066 aggregate bundle](../sighan-instruction-language-control-v1/README.md)
