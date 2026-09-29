# Experiment 074: Qwen numeric-support audit

## Result

The 24-review audit completed on MPS across 192 model/context cells and 32 direct
sequence-likelihood checks per model. There were no invalid greedy outputs. Cached
trie-scored candidate likelihoods matched independent full forward passes within
`4.2e-5` log probability for Qwen2.5-0.5B and `6.6e-5` for Qwen2.5-1.5B.

For Qwen-0.5B, the conditional 8−2 expected-score shift was +2.577 (95% paired
recipient interval [+2.424, +2.718]) valence-first and +0.114 [+0.038, +0.192]
arousal-first when scoring complete canonical one-decimal JSON numbers including
`}`. Combining those values with parser-accepted integer and trailing-zero spellings
changed the shifts only to +2.573 [+2.420, +2.714] and +0.114 [+0.038, +0.192].
The added spellings violate the prompt's one-decimal formatting rule; their total
probability mass was tiny. The original score-only calculation on the same subset
was +2.582 and +0.138. Adding the closing brace moved the arousal-first estimate by
−0.024, far too little to erase the contrast with 1.5B.

On the same 24 reviews, Qwen-1.5B's complete-canonical expected shifts were +2.008
[+1.829, +2.173] valence-first and +1.012 [+0.741, +1.291] arousal-first. Its
canonical complete-string mass averaged 99.75% and 99.78%; Qwen-0.5B's averaged
95.92% and 88.41%, respectively. Thus 0.5B's arousal-first response remained near
zero despite lower support coverage. Expanded parser-accepted spellings added under
0.2 percentage points of mass.

## Interpretation and limits

This audit makes the Experiment 073 order-by-size interaction harder to explain as
an omitted JSON brace, common alternate numeric spellings, or an error in cached
sequence scoring. It remains a result on 24 reused public laptop reviews, with
partial assistant prefixes and two Qwen sizes. The expanded spelling set is not an
exhaustive grammar, and a three-size trend still cannot support a general scaling
law. The useful next step is to test the 0.5B order asymmetry on the separate English
restaurant recipient set before describing it as a stable phenomenon.

## Reproduction

- Frozen protocol: [`074`](../../docs/experiments/074-qwen05-numeric-support-audit.md)
- Runner: `scripts/run_qwen05_numeric_support_074.py`
- Analyzer: `scripts/analyze_qwen05_numeric_support_074.py`
- Aggregate: [`summary.json`](summary.json)
- Models/source are pinned in the protocol. MPS runtime: 231.9 seconds.
