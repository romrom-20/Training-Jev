# Experiment 070 — prefix-score distribution by model size

Experiment 070 repeated the Experiment 069 token-score audit on Qwen2.5-1.5B
and added one deterministic greedy continuation per prefix. It reused both
disjoint 64-case SIGHAN cohorts, every prompt, and every forced-prefix condition.
The full run had 512 contexts, 512 valid greedy completions, and zero invalid
outputs. The frozen protocol is
[`070`](../../docs/experiments/070-prefix-score-distribution-size-transfer.md);
the paired aggregate is [`summary.json`](summary.json).

## Results

Increasing the forced first value from 2 to 8 shifted the next coordinate
upward in both orders on Qwen2.5-1.5B. Qwen2.5-3B, on the same prompts and
recipients, showed a much smaller positive valence-first distribution shift and
a negative arousal-first shift. The paired 1.5B-minus-3B difference in
conditional expected-score shift was positive in every cohort/order comparison:

| Cohort | First field | 1.5B expected shift (95% CI) | 3B expected shift (95% CI) | Paired 1.5B − 3B (95% CI) |
|---|---|---:|---:|---:|
| Exp067, mixed aspect categories | Valence | +2.826 [+2.685, +2.955] | +0.666 [+0.576, +0.758] | +2.160 [+1.968, +2.337] |
| Exp067, mixed aspect categories | Arousal | +1.786 [+1.637, +1.921] | −0.452 [−0.607, −0.289] | +2.238 [+1.998, +2.454] |
| Exp068, fresh food-quality cohort | Valence | +2.792 [+2.662, +2.917] | +0.621 [+0.530, +0.718] | +2.171 [+2.018, +2.318] |
| Exp068, fresh food-quality cohort | Arousal | +1.817 [+1.684, +1.949] | −0.586 [−0.712, −0.460] | +2.403 [+2.203, +2.591] |

The 1.5B greedy shifts followed the same positive direction: +3.125 and +3.109
in the valence-first condition, and +2.109 and +2.313 in the arousal-first
condition. These differ sharply from the 3B greedy shifts, especially when
arousal was first (−0.688 and −0.809). Thus the 069 discrepancy between
conditional distribution and greedy output was specific to 3B in this test;
the smaller sibling's greedy answers moved strongly in the same direction as
its score distribution. These are paired, adaptive size contrasts on reused
recipients, not an independent dataset replication.

The cached-prefix scorer was checked against full-sequence likelihoods for
1.0, 4.2 and 9.0; the largest absolute difference was `1.9e-5`. The model's
probability mass on the 81 valid numeric continuations ranged from 99.66% to
99.95%. All intervals use 10,000 paired recipient bootstrap draws.

## Interpretation and literature boundary

This is a substantial Qwen-family size difference in the response to a
counterfactual forced assistant prefix. It is interesting because the 1.5B
model follows the supplied first score much more strongly, while 3B reverses
direction when arousal precedes valence. It does not imply that general language
ability suppresses anchoring: only two sizes from one family, one prompt, and
two cohorts in a single Chinese review release were compared, and all prefixes
were artificial.

Numeric anchoring effects across model families and repeated answer distributions
are already established ([Zhang et al., 2025](https://doi.org/10.1007/s42001-025-00435-2));
their results also report model differences in anchoring consistency. Output
order effects in model-based scoring are established too
([Chen et al., 2024](https://arxiv.org/abs/2406.02863)). This experiment adds a
candidate, narrowly scoped observation about scale-dependent next-coordinate
continuation under a forced output prefix. It is not a novelty or priority claim.

This may become a useful short research note if it survives another Qwen scale
or an independent model family. Do not draft a LessWrong post yet; a third size
or family check is the next step.
