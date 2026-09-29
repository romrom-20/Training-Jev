# Experiment 069 — score distributions after forced prefixes

This post-hoc audit asks whether the Exp067/068 forced-prefix result is visible
across Qwen2.5-3B's conditional score distribution or only in the deterministic
greedy score. It reuses the same 128 recipients, prompts, prefixes and model
revision. It adds no generated completions. The frozen protocol is
[`069`](../../docs/experiments/069-prefix-score-distribution-audit.md); the
machine-readable aggregate is [`summary.json`](summary.json).

For each of 512 prefixes, the runner scored the 81 canonical one-decimal numeric
continuations from 1.0 to 9.0. The normalized distribution is conditional on this
valid-number support. A full-token likelihood check on representative candidates
matched the cached-prefix scorer within `9.3e-5` log-probability units. A
post-hoc support-mass check found that these strings carried 99.77%–99.997% of
model probability across observed contexts, so excluded numeric strings account
for little mass in these cases. Closing-brace probability is not included.

## Main result

Across both disjoint cohorts, the conditional expected second score moved in both
field orders, even though deterministic greedy decoding showed a resolved shift
only when arousal was first:

| Cohort | First field | Expected score shift (8 − 2), 95% recipient-bootstrap interval | Greedy score shift (8 − 2), 95% interval |
|---|---|---:|---:|
| Exp067, mixed aspect categories | Valence | +0.666 [+0.576, +0.758] | +0.156 [−0.047, +0.359] |
| Exp068, fresh food-quality cohort | Valence | +0.621 [+0.530, +0.718] | −0.031 [−0.234, +0.172] |
| Exp067, mixed aspect categories | Arousal | −0.452 [−0.607, −0.289] | −0.688 [−0.984, −0.391] |
| Exp068, fresh food-quality cohort | Arousal | −0.586 [−0.712, −0.460] | −0.809 [−1.108, −0.500] |

For valence-first, the conditional expectation shift exceeded the observed greedy
shift by +0.509 points in Exp067 (95% interval [+0.339, +0.679]) and +0.653 in
Exp068 ([+0.471, +0.827]). This is a post-hoc paired comparison. The normalized
distributions moved substantially in every cohort/order condition (mean total
variation 0.278–0.403; mean Wasserstein-1 distance 0.627–0.772 score points).

## Interpretation

The result suggests that deterministic greedy output concealed a consistent
valence-first conditional-distribution shift in these data. It also shows that
the arousal-first shift is not limited to a single winning continuation. This is
an exploratory mechanism finding from two cohorts of one benchmark release and
one model family; both cohorts use the same artificial assistant-prefix
intervention. It does not show that ordinary ratings are anchored this way.

The literature already measures numeric answer distributions under low/high
anchoring hints across repeated LLM responses ([Zhang et al., 2025](https://doi.org/10.1007/s42001-025-00435-2)). Output-order effects in model-based scoring are also established ([Chen et al., 2024](https://arxiv.org/abs/2406.02863)). The narrower question here is whether directly forcing one emitted affect coordinate changes the conditional distribution of the next coordinate when ordinary greedy outputs appear stable. The contribution is a candidate for follow-up, not a priority claim.

Do not write a LessWrong post yet. The distribution result is interesting, but it
is post-hoc, restricted to one model family and a hand-forced prefix, and closely
adjacent to prior numerical anchoring research. A smaller-model score-distribution
replication is the next useful check if local Laya triage supports it.
