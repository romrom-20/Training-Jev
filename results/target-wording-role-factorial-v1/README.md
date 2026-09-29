# Exploratory target-wording × anchor-source factorial

## Result

This is a post-hoc analysis of all role-control cells in Experiments 076–078. Each shift is the paired 8−2 expected-score change over normalized canonical one-decimal JSON score completions. Intervals bootstrap the 24 recipients.

### Role-source contrast by model and order

| Model | Target order | Generic wording: assistant − user | Explicit wording: assistant − user | Explicit − generic difference-in-differences |
|---|---|---:|---:|---:|
| qwen-0.5b | Valence first | 0.416 [0.333, 0.495] | 0.273 [0.178, 0.371] | -0.143 [-0.209, -0.078] |
| qwen-0.5b | Arousal first | withheld by validity gate | 0.380 [0.319, 0.446] | withheld by validity gate |
| qwen-1.5b | Valence first | -0.306 [-0.376, -0.235] | -0.476 [-0.592, -0.364] | -0.170 [-0.239, -0.101] |
| qwen-1.5b | Arousal first | 0.152 [0.083, 0.225] | 0.269 [0.227, 0.310] | 0.118 [0.027, 0.207] |

Full cell shifts, wording contrasts, and invalid rates are in `summary.json`.

## Interpretation and limits

Same 24 items and two fixed models; post-hoc cell contrasts. Generic-wording Qwen-0.5B prior-user/arousal-first cell fails the preregistered validity gate, so dependent contrasts are withheld.

The experiment components are [076](../../docs/experiments/076-schema-and-corpus-controls.md), [077](../../docs/experiments/077-explicit-target-role-control.md), and [078](../../docs/experiments/078-qwen15-explicit-target-role-replication.md). This exploratory combined analysis cannot establish that the prompt components act independently, and the user-role comparison still includes an acknowledgement turn.

Literature already documents generic score anchoring, output-order effects, and sensitivity to structured formats; this is a narrow audit of how those factors interact in one forced-coordinate affect task. See [Kapetanovic et al. (2026)](https://arxiv.org/abs/2608.25869), [Chen et al. (2024)](https://arxiv.org/abs/2406.02863), and [Parikh (2026)](https://arxiv.org/abs/2607.18476).

Raw model outputs remain in ignored `.context/`.
