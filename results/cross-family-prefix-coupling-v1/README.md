# Experiment 071 — forced-prefix coupling across families

Experiment 071 tested Qwen2.5-1.5B, Qwen2.5-3B, and SmolLM2-1.7B on the same
64 fresh English laptop reviews. The IDs were excluded from Experiment 055 and
balanced 32 negative/32 positive. Each of the 768 model × recipient × order ×
forced-value contexts received both a greedy continuation and scores for the 81
valid one-decimal continuations. All 768 greedy outputs parsed. The frozen
protocol is [`071`](../../docs/experiments/071-cross-family-prefix-coupling.md);
the aggregate is [`summary.json`](summary.json).

## Results

High versus low forced first value (8 minus 2) increased the conditional
expected second score for Qwen2.5-1.5B and SmolLM2 in both field orders. Qwen2.5-3B
showed a smaller positive valence-first shift and an unresolved, slightly
negative arousal-first shift:

| Model | First field | Expected score shift (95% recipient-bootstrap interval) | Greedy score shift (95% interval) |
|---|---|---:|---:|
| Qwen2.5-1.5B | Valence | +1.996 [+1.894, +2.093] | +1.875 [+1.766, +1.969] |
| Qwen2.5-1.5B | Arousal | +1.062 [+0.871, +1.254] | +0.334 [−0.145, +0.842] |
| Qwen2.5-3B | Valence | +0.772 [+0.621, +0.926] | +0.734 [+0.500, +0.953] |
| Qwen2.5-3B | Arousal | −0.227 [−0.470, +0.017] | +0.047 [−0.273, +0.375] |
| SmolLM2-1.7B | Valence | +1.814 [+1.722, +1.904] | +4.016 [+3.500, +4.547] |
| SmolLM2-1.7B | Arousal | +1.338 [+1.268, +1.409] | +2.891 [+2.547, +3.234] |

On these same recipients, the paired Qwen1.5B-minus-3B expected-shift contrast
was +1.225 ([+1.045, +1.400]) valence-first and +1.289 ([+1.042, +1.537])
arousal-first. SmolLM2's expected shifts were close to Qwen1.5B's valence-first
estimate (SmolLM2 minus Qwen1.5B: −0.182 [−0.288, −0.082]) and modestly higher
arousal-first (+0.276 [+0.076, +0.479]). Compared with Qwen3B, SmolLM2's shifts
were larger by +1.043 ([+0.867, +1.213]) and +1.565 ([+1.317, +1.814]). These
are exploratory paired contrasts, not a universal scale law.

The greedy and expected shifts diverged sharply for SmolLM2: greedy movement
exceeded conditional expectation by about 2.20 points valence-first and 1.55
points arousal-first. Its mean valid-number support probability was 97.8%–99.2%,
below the Qwen models' nearly complete support, so the normalized 81-score
distribution describes a slightly narrower portion of SmolLM2's possible score
strings. A cached-prefix/full-sequence likelihood audit matched at 1.0, 4.2 and
9.0 within `2.5e-5` log-probability units.

## Interpretation and literature boundary

This fresh English cohort preserves the broad size pattern seen in the earlier
Chinese SIGHAN runs: Qwen2.5-1.5B shifts upward in both orders; Qwen2.5-3B is
smaller and order-asymmetric. SmolLM2-1.7B resembles the smaller Qwen model in
conditional expectation, which is compatible with a size effect but does not
isolate size from family, tokenizer, training, or chat template. The unexpectedly
large SmolLM2 greedy shifts also show that the restricted distribution mean and
greedy completion are not interchangeable. All interventions still force an
artificial partial JSON answer; this says nothing directly about unconstrained
ratings.

Prior literature already studies numeric anchoring across LLMs and differences
in their repeated answer distributions ([Zhang et al., 2025](https://doi.org/10.1007/s42001-025-00435-2));
model-based scoring order effects are established too
([Chen et al., 2024](https://arxiv.org/abs/2406.02863)). The candidate
contribution remains narrow: model-size/family differences in how a forced VA
coordinate affects the next continuous score on these prompts. This result is
interesting enough to motivate one independent-domain replication, but not a
LessWrong post yet.

SmolLM2 is an Apache-2.0 1.7B model described as suitable for local use in the
[official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct).
