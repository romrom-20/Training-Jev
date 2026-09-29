# Experiment 073: Qwen2.5-0.5B prefix coupling

## Result

The additional model pass completed locally on MPS: all 256 recipient/prefix cells,
zero invalid greedy continuations, and zero resumed cells. The restricted-grid
expected 8−2 shifts were +2.547 (95% paired-recipient interval [+2.451, +2.634])
when valence was forced first, and +0.138 [+0.083, +0.191] when arousal was forced
first. Greedy shifts were +3.156 [+2.984, +3.336] and +0.336 [+0.273, +0.406].

Compared with the same 64 recipients and outputs from Experiment 071, Qwen-0.5B
minus Qwen-1.5B was +0.551 [+0.426, +0.664] valence-first, but −0.924
[−1.117, −0.729] arousal-first. Relative to Qwen-3B, the contrasts were +1.776
[+1.602, +1.947] and +0.366 [+0.120, +0.612]. Thus the size pattern continues
monotonically from 3B→1.5B→0.5B for valence-first in this sample, while arousal-first
peaks at 1.5B and drops at 0.5B. That order-specific reversal rejects a simple
monotonic “smaller means more sensitive” account on these items.

The 0.5B model's mean probability mass on the 81 valid one-decimal strings was
96.2%/96.4% for valence-first low/high anchors, but 88.1%/90.9% for arousal-first.
Expected-score shifts are normalized conditional on this enumerated valid-score
support. The lower arousal-first support mass could affect comparison with models
whose mass is closer to one, so treat the size interaction as a follow-up signal,
not a final result. The model still produced parseable greedy outputs in every cell.

## Interpretation and limits

This third Qwen size point suggests that field order and model size interact: the
valence-first response rises with smaller Qwen size, but the arousal-first response
does not. This is a three-point within-family result on the same already-seen laptop
cohort, under an artificial continuation prefix. It does not establish a general
scaling law or ordinary rating behavior. The next experiment should check whether the
arousal-first 0.5B anomaly survives a score grammar with fuller probability support
or an independently sourced cohort before drawing a mechanism claim.

## Reproduction

- Frozen protocol: [`073`](../../docs/experiments/073-qwen05-prefix-size-continuation.md)
- Runner: `scripts/run_qwen05_prefix_size_073.py`
- Analyzer: `scripts/analyze_qwen05_prefix_size_073.py`
- Aggregate: [`summary.json`](summary.json)
- Pinned source/model details and the parent cells are in the protocol. Runtime:
  75.6 seconds on the 24-GB MacBook Air.
