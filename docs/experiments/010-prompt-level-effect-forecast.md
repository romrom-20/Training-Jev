# 010 — Can a small readout forecast prompt-specific steering effects?

**Prospective follow-up, frozen before intervention outcomes are collected.** Experiments 008 and 009 showed that exact prompt wording can change both response compliance and candidate-label scores. The field also now includes behavior-level forecasts of steering side effects: Ong et al. (2026) pool outcomes across contexts into behavior-pair cross-effect scores, then predict held-out source/target behaviors from unsteered representations and a propagation model ([paper](https://arxiv.org/html/2608.11227v1)). The untested unit targeted here is the *prompt-specific treatment effect*: within one behavior pair, which held-out prompt will shift more, and in which direction? Prompt-specific forecasting matters because an average cross-effect can hide a subset of prompts with large or reversed responses.

This is an explicit small-model extension/replication, not a novelty claim about activation steering or effect forecasting. We use the completed 008 synthetic aspect task and its saved unsteered activations; no probe or intervention outcomes have been inspected for 010.

## Models, sample and fixed intervention

Run the two complete 008 core captures only: Qwen2.5-1.5B-Instruct and SmolLM2-1.7B-Instruct. Use the `mixed_review`, format-0 rows and the existing group splits, which hold out cue variants 4–7 and counterbalance aspect order in final-test groups. Use the saved 008 activation at transformer block 24 (one-indexed). Selector groups are excluded. Training effects use the 16 `train` groups; early stopping uses six `validation` groups; six `calibration` groups are held aside; the 24 `final_test` groups are the only primary evaluation.

For each of food, service and value, derive a single source direction from the mean block-24 activation difference (positive minus negative) on training groups, using only mixed-review prompts that ask about that source aspect. Normalize each direction to unit L2. At inference, add the positive source direction to the final prompt-token activation at block 24 with a fixed magnitude of 5% of the median training activation norm. There is no dose, layer, or direction search on validation or test outcomes. This matches the dose fraction used in the repository's original pilot.

On each mixed-review prompt, measure the next-token log-odds change for each queried aspect under each of the three source directions:

`effect(prompt, source, target) = logit(positive − negative | +dose) − logit(positive − negative | unmodified)`.

This is a finite intervention effect on the model's candidate-label score. It is not a probability of a real-world behavior or a human-judged outcome. The model's unmodified candidate log-odds are already recorded in 008; the 010 runner performs and records each patched forward pass. Also compute the unsteered first-order gradient forecast `dose × ∇activation log-odds · source_direction` on final-test prompts as a model-specific reference that requires backward access.

## Forecast comparison

The primary forecaster is a rank-four bilinear regression head over the unsteered block-24 activation, target query, and source query. Train three seeds on measured train-group intervention effects; choose checkpoints on validation-group MSE; average seed predictions. The primary baseline is a constant mean for each source/target pair fitted on training effects. Secondary comparisons are a word/character TF-IDF ridge model over the visible review plus source/target names, and the first-order gradient reference. Calibration groups remain unused by the primary estimand and cannot affect test forecasts.

Primary metric: pooled final-test RMSE. Report MAE, signed-effect accuracy, R-squared, and results for all nine ordered source/target pairs. Estimate a paired 95% interval for head-minus-pair-mean RMSE by resampling the 24 final-test scenario groups (5,000 bootstrap draws, fixed seed 20260924). Also report the empirical within-pair effect variance so that a forecast improvement is interpreted against prompt-level heterogeneity rather than the overall average effect.

Evidence for prompt-specific information in the small readout requires at least a 10% relative RMSE reduction against the pair-mean baseline **in both model families**, with the paired group-bootstrap interval below zero for each. Otherwise record a negative or mixed result; do not relax the rule or select favorable pairs after seeing results. The gradient reference is an upper-information comparison, not a deployable baseline. No causal-mechanism claim follows from this one intervention family.

## Compute, artifacts and limitations

Intervene only on train, validation, calibration and final-test rows: 3,744 patched prompt/source pairs per model. Reuse existing full-precision MPS activation captures; load models sequentially, batch at eight, checkpoint each batch, and resume by unique intervention-row ID after interruption. Save full per-row intervention outcomes, predictions, manifests and hashes; keep activation arrays out of Git. Stop if sustained memory paging begins, preserving completed model artifacts and marking the other model incomplete. Do not run the optional 3B checkpoint; its 008 run paged heavily on the 24 GB MacBook Air.
