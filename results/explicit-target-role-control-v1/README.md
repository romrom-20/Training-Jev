# Experiment 077: name the target axis

## Result

The final request names the fixed and target dimensions and requires the target dimension as the sole JSON key. Intervals resample 24 paired recipients.

| Target order | Validity gate | Assistant-prior shift | User-prior shift | Assistant − user |
|---|---|---:|---:|---:|
| Valence first | True | 0.941 [0.805, 1.077] | 0.668 [0.582, 0.753] | 0.273 [0.178, 0.369] |
| Arousal first | True | 1.202 [1.082, 1.327] | 0.822 [0.705, 0.934] | 0.380 [0.320, 0.444] |

Greedy shifts, valid canonical probability mass, and each arm's invalid rate are in `summary.json`.

## Interpretation and limits

All 192 outputs were valid with the target-axis name made explicit; this resolves the wrong-key failure seen in Experiment 076 on these cases. The assistant-prior shift exceeded the user-prior shift by +0.273 valence-first and +0.380 arousal-first, with recipient-bootstrap intervals excluding zero. This is a within-prompt role/context contrast, not a general role effect: the user-prior condition still includes an acknowledgement turn, the sample is 24 reused public reviews, and only Qwen-0.5B was tested.

Compare with the [Experiment 076 result](../schema-corpus-controls-v1/README.md) and the [077 protocol](../../docs/experiments/077-explicit-target-role-control.md). Explicitly naming the target coordinate changed both compliance and measured conditional shifts, so prompt clarity is part of the phenomenon being measured.

Raw prompts and outputs remain in ignored `.context/`.
