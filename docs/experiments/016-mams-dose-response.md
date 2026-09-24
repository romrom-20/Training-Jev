# 016 — Does aspect selectivity grow faster than generic valence with dose?

**Prospective exploratory follow-up, 24 September 2026.** Experiment 015 found a
consistent, positive target-matched specificity contrast on MAMS conflict reviews in
both small models, but its magnitude was only 0.38% of the generic shift for Qwen and
0.08% for SmolLM2. Both missed the frozen 5% practical-selectivity criterion. The
curiosity now is whether this small aspect-specific component is simply overwhelmed by
the generic valence shift at the 5% dose, or whether its relative size stays negligible
as intervention strength changes.

Use exactly the same 35 MAMS test sentences, 71 mapped-category prompts and 18 conflict
sentences from experiment 015. Do not alter the category crosswalk, direction vectors,
layer, prompt, checkpoint or control vectors. Evaluate dose fractions of the median
training activation norm at `0.0125`, `0.025`, `0.05`, `0.10`, and `0.20`. Run each native
direction, the shared component and the norm-matched random direction at every dose on
both models. Capture each prompt's unmodified logits and every treatment's positive-
minus-negative logit change and strict answer accuracy.

## Analysis and decision rule

For each dose and model, estimate the sentence-clustered native-minus-random
diagonal/off-diagonal specificity contrast, the generic native shift averaged over
directions and prompts, and their ratio. Bootstrap MAMS sentence IDs 5,000 times with
seed `20260924` to form 95% intervals for the specificity and ratio at each dose. Also
fit the prespecified linear trend of specificity/generic-shift ratio against log2 dose;
bootstrap that slope using the same sentence-level resamples. Treat the curve and slope
as exploratory mechanistic evidence; do not select or report only the best dose.

Evidence that higher dose selectively exposes aspect information requires a positive
95% interval for the ratio-vs-log2-dose slope in both models. Call any dose practically
useful only if the specificity point estimate reaches 5% of the generic shift and
unsteered strict accuracy exceeds chance for that model. Report all dose outcomes even
if accuracy collapses, and do not retune the frozen dose grid. This follow-up is
motivated by results from 015; it is not an independent replication of 015.

Run sequentially on the 24 GB MacBook Air in float32 with batches of eight. Store
resumable per-dose checkpoints and manifest hashes; keep the MAMS text in ignored
`.context/datasets/` and omit all text from results.
