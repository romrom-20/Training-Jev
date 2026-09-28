# Experiment 050 — Does the laptop-domain decoder interaction persist at 3B?

**Status:** adaptive follow-up selected by the pinned local Laya typed-choice checkpoint after Experiment 049. Laya's probabilities are uncalibrated and serve only as agenda triage. Freeze this protocol and runner before generating Qwen2.5-3B outputs.

## Question and motivation

Experiment 049 tested 943 eligible English laptop-review IDs with Qwen2.5-1.5B and identical one-decimal instructions across finite-grid and free-greedy decoding. Its finite-grid context gain was +0.740, free-greedy gain +0.486, and the primary interaction +0.254 (95% source-ID interval [+0.196, +0.313]). This narrowly passed the preregistered +0.25 practical interaction rule. Experiment 048 found a positive but smaller +0.159 interaction on 217 balanced Russian/Ukrainian/Tatar restaurant IDs, below its practical threshold.

Experiment 050 repeats the exact 049 factorial on the same English laptop cases with Qwen2.5-3B-Instruct. It asks whether the 1.5B interaction appears at the larger model size and estimates a paired model-size difference in decoder interactions. Since both runs use the same public benchmark items, this is a paired model-scale diagnostic, not an independent sample replication.

## Data and conditions

- Reconstruct all 943 eligible English laptop cases from the pinned DimABSA commit and source hash in [Experiment 049](049-laptop-domain-decoder-transfer.md). Verify every ID, selected aspect, and two-dimensional gold score against 049's private predictions. Retain at most one eligible target triplet per source ID, choosing the first in source order.
- For each source ID, reuse the 049 `aspect_only` and `opinion_masked` prompt bytes unchanged. Check those prompt bytes against the 049 job construction. Both arms request one-decimal VA scores in [1, 9] at 0.1 increments.
- Cross both conditions with `finite_grid` (the same 6,561 one-decimal JSON candidates as 043/048/049) and `free_greedy` (ordinary greedy JSON generation, maximum 40 new tokens). This yields 3,772 new Qwen2.5-3B outputs.
- Run the pinned `Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, greedy, locally on MPS. All source text, prompts, IDs and raw/item predictions remain in ignored `.context/`.

## Outcomes and decision rules

- For each model and decoder, compute context gain as `RMSE(aspect_only) - RMSE(opinion_masked)` on two-dimensional VA error.
- **Primary 3B contrast:** `context_gain(finite_grid) - context_gain(free_greedy)` on Qwen2.5-3B. Bootstrap source IDs with replacement, retaining all conditions, decoder outputs, both VA dimensions, and the 1.5B paired rows. Use 10,000 draws, seed `20260950`, and percentile 95% intervals.
- Reuse the 048/049 practical rule: a practically meaningful 3B interaction requires estimate ≥ +0.25 and lower interval bound > 0. Failing it is inconclusive, not evidence of equivalence.
- **Secondary scale contrast:** 3B interaction minus 1.5B interaction on source IDs complete across all eight cells. Report its paired interval descriptively; do not make a confirmatory model-size claim from this secondary contrast.
- If more than 2% of 3B free outputs are invalid, withhold all 3B score contrasts. Otherwise calculate on source IDs complete in all 3B cells and report coverage. Calculate scale contrasts only on the subset complete in both model runs.
- Report validity, runtime, hashes, the registered 3B interaction and the paired model-size difference. Interpret any passing result as repeated within-item evidence on one English laptop test split, not cross-corpus generalization.

## Compute and limits

The 3B checkpoint was used for the earlier constrained MPS experiment, and its weights fit the 24-GB Air. Experiment 050 entails 3,772 judgments, with resumable batch checkpoints. The benchmark is public and the test labels may overlap model pretraining; the 1.5B and 3B runs share the same source IDs, domain, language and benchmark release. This experiment addresses a model-size boundary only.

## Provenance

- DimABSA benchmark: [Lee et al., ACL 2026](https://aclanthology.org/2026.acl-long.1881/).
- Prior matched decoder outcomes: [048](../../results/small-model-decoder-factorial-v1/README.md) and [049](../../results/laptop-decoder-transfer-v1/README.md).
- Private Laya agenda trace: `.context/laya-research-triage-050.json` (not committed).
- Runner/analyzer: `scripts/run_laptop_model_size_050.py`, `scripts/analyze_laptop_model_size_050.py`.
