# 013 — Does the shared shift follow sentiment or the answer identifier?

**Prospective local replication frozen before mapped-answer intervention outcomes are collected.** Experiments 011–012 found that a shared positive-minus-negative logit shift transfers from mixed reviews to three other task structures on two small model families. That still does not establish semantic sentiment control. Gao et al. (2026) show that steering effects can track answer identifiers after remapping while the tested items and intervention remain fixed ([Cross-Encoding Steering Evaluation](https://arxiv.org/html/2608.22985v1)). Experiment 013 applies this control to the local shared aspect direction at a smaller scale.

## Frozen setup

Use the 144 held-out format-0 `isolated_clause` prompts from experiment 008, all three native directions, the shared component, and one norm-matched random direction. Keep the directions, layer 24 and fixed 5% training activation-norm dose unchanged. Render each same question under two answer encodings: (A) `A means positive; B means negative`, and (B) `A means negative; B means positive`. Change only the final answer instruction; do not change review, queried aspect, direction, dose or model. Score one-token A and B at the next generation position and save baseline and steered logits.

This gives 144 prompts × two mappings × five intervention conditions = 1,440 treatment records per model. Also report the unsteered next-token accuracy against the mapped semantic label and valid A/B rate in each mapping. This is a logit-level cross-encoding audit, not an open-ended behavior study.

## Decision rule

Interpret mapped intervention effects as semantic only if the unsteered model has at least 90% mapped next-token accuracy in **both** answer encodings. For each model and each encoding, estimate the shared-component semantic-margin change minus the norm-matched random-control semantic-margin change; bootstrap 24 prompt groups 5,000 times (seed 20260929). Evidence of mapping-robust semantic steering requires the 95% intervals to be above zero for both mappings in both models. Also report identifier-aligned A-minus-B changes: if those stay aligned while semantic-margin changes reverse, the effect follows the fixed answer identifier instead of sentiment.

This is a small-model, single-task replication/control motivated by direct prior work, not a novelty claim. If the competence gate fails, retain the measurements but do not infer whether steering is semantic. No LessWrong post should be drafted from this result alone; the point is to determine whether the positive-score shift survives a necessary attribution control.

## Integrity and compute

Build directions from 008 mixed-review training groups as in 011–012; use no remapped test outcomes for any direction, layer, dose or prompt choice. Run models sequentially on the 24 GB Air, batch at eight, checkpoint by prompt/mapping/condition ID, and hash the source dataset, activation cache, protocol and runner. Preserve all earlier captures and report each model even if the other fails.
