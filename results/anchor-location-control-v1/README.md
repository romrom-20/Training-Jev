# Experiment 075: where the numeric anchor appears

## Result

The 1,024-cell MPS run completed on 64 reused English laptop reviews across Qwen2.5-
0.5B and 1.5B. All greedy outputs parsed; the valid canonical-score probability
support was lower in some user-anchor conditions, especially 0.5B.

The 0.5B pattern changed with anchor location. In the assistant-prefix format, the
expected 8−2 shift was +2.542 valence-first and +0.115 arousal-first. When the same
fixed coordinate appeared in the user message and the model returned only the other
coordinate, shifts were +1.031 and +0.654. Paired user-minus-prefix differences were
−1.510 (95% recipient interval [−1.591, −1.425]) and +0.538 [+0.473, +0.604].
The previous weak arousal-first effect therefore became a moderate effect in this
second prompt format, while the strong valence-first effect shrank.

For Qwen-1.5B, assistant-prefix shifts were +1.996 valence-first and +1.061
arousal-first; user-anchor shifts were +0.627 and +0.134. Paired contrasts were
−1.369 [−1.461, −1.274] and −0.927 [−1.098, −0.762]. Adding common parser-accepted
integer and trailing-zero number spellings barely changed any expected shifts.

## Interpretation and limits

The measured response is format-dependent: moving the fixed coordinate from the
active assistant prefix to a user-stated constraint changes the distribution, and
it changes the two models in different ways. This supports a focused next question
about how scoring schema and cue location interact. It does not isolate a pure
assistant-versus-user causal effect because the user arm uses a one-field JSON output
while the assistant arm completes a two-field object. The inputs were already seen
in earlier runs, and two Qwen models cannot support a family-wide claim.

Output-order effects in LLM scoring are already reported in prior work; see
[Chen et al. (2024)](https://arxiv.org/abs/2406.02863). Our result narrows this
particular forced-coordinate setup and identifies a model-dependent interaction,
but it does not establish a new general order-bias phenomenon. More controlled
schema-matched work is needed before deciding whether this merits a LessWrong post.

## Reproduction

- Frozen protocol: [`075`](../../docs/experiments/075-anchor-location-control.md)
- Runner: `scripts/run_anchor_location_control_075.py`
- Analyzer: `scripts/analyze_anchor_location_control_075.py`
- Aggregate: [`summary.json`](summary.json)
- Models and source sample are pinned in the protocol. MPS runtime: 1,596.5 s.
