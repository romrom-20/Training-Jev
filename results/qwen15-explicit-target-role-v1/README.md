# Experiment 078: Qwen-1.5B named-target role control

## Result

The final request explicitly names the fixed and target coordinates. Recipient-bootstrap intervals use 24 paired reviews.

| Target order | Validity gate | Assistant-prior shift | User-prior shift | Assistant − user |
|---|---|---:|---:|---:|
| Valence first | True | 0.856 [0.794, 0.918] | 1.332 [1.250, 1.412] | -0.476 [-0.592, -0.364] |
| Arousal first | True | 0.659 [0.615, 0.703] | 0.390 [0.345, 0.434] | 0.269 [0.227, 0.312] |

Greedy shifts, valid canonical probability mass, and per-arm invalid rates are in `summary.json`.

## Interpretation and limits

One 1.5B model on 24 reused public reviews. The within-dialogue source comparison still bundles role with the acknowledgement turn.

Compare with the [Qwen-0.5B Experiment 077 result](../explicit-target-role-control-v1/README.md) and [078 protocol](../../docs/experiments/078-qwen15-explicit-target-role-replication.md). This same-item model-size comparison is descriptive, not independent replication.

Raw prompts and outputs remain in ignored `.context/`.
