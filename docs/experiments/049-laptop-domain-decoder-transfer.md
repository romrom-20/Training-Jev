# Experiment 049 — Does the decoder interaction transfer to laptop reviews?

**Status:** adaptive transfer test selected by the pinned local Laya typed-choice checkpoint after Experiment 048. Laya probabilities are uncalibrated and serve only as agenda triage. Freeze this protocol and runner before scoring this domain.

## Question and motivation

Experiment 048 held prompt wording fixed on 217 aligned Russian/Ukrainian/Tatar restaurant-review IDs. The finite 0.1-grid decoder's opinion-masked context gain was 0.288 VA RMSE points, versus 0.130 for free greedy decoding. The paired interaction was +0.159 (95% source-ID cluster interval [+0.064, +0.253]), below the registered +0.25 practical threshold. This is a small decoder-dependent difference on the same release and items used by earlier experiments, not an independent replication.

Experiment 049 asks whether this decoder interaction appears on an independently sampled **English laptop-review domain** from the same DimABSA release. The domain shift holds language and model fixed while changing the product domain. Lee et al. (ACL 2026) introduce DimABSA as a multilingual, multi-domain continuous valence-arousal benchmark; Song and Bahri (TMLR 2025) discuss finite numeric tokenizations and their rounding/point-estimation implications in decoding-based regression. These establish adjacent work, but neither tests this paired opinion-masking-by-decoder interaction on the laptop split.

## Data and sample

- Use the official DimABSA `eng_laptop_test_task2.jsonl` at commit `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`, SHA-256 `08a4cc197f068f6fc4c9373f61bfb86a5cd996ef16be3cb516082b434ba92543`. The local cache was byte-verified against that pinned raw GitHub path before this protocol was frozen.
- Evaluate every source row with a non-null target aspect and opinion for which the target aspect and opinion each occur exactly once, every distinct non-null opinion span occurs exactly once, and no annotated opinion overlaps the target aspect. Select the first eligible triplet in source order, at most one target per source ID. Require a valid two-number `V#A` gold annotation. The pre-run source audit found 943 eligible IDs: 224 below valence 4.5, 30 from 4.5 through 5.5 inclusive, and 689 above 5.5.
- This is a full-eligible-split domain transfer, not a fresh human annotation or blind test: test labels are public and this is the same benchmark release. Keep source text, prompts, target aspects, IDs, raw generations, and item predictions in ignored `.context/` files. Publish only aggregate results and source provenance.

## Conditions and model

For every selected source ID, run `aspect_only` and `opinion_masked` prompts with Qwen2.5-1.5B-Instruct, greedy, locally on MPS. The masked input replaces every distinct non-null annotated opinion span with `[MASKED]`; the target aspect remains visible. Both conditions use the same prompt template, which requests one-decimal scores in [1, 9] at 0.1 increments.

Run both decoder arms on both inputs, for four judgments per source ID:

1. `finite_grid`: greedy generation restricted to the same 6,561 one-decimal valence-arousal JSON candidates as Experiment 048.
2. `free_greedy`: ordinary greedy generation with no grammar, maximum 40 new tokens, and strict JSON parsing. No retries or repairs.

The prompt text must be byte-identical between decoder arms within each source ID and input condition. The two arms differ only in whether generation is restricted to the finite grammar. The model revision is pinned to the Qwen2.5-1.5B revision recorded in the repository task-ladder configuration.

## Outcomes and decision rules

- For each decoder, compute `RMSE(aspect_only) - RMSE(opinion_masked)` on the two-dimensional VA prediction. Positive values mean the remaining sentence context reduces prediction error.
- **Primary contrast:** finite-grid context gain minus free-greedy context gain. Positive values mean the finite grammar amplifies the context gain on this domain.
- Bootstrap source IDs with replacement, keeping all four judgments and both VA dimensions together. Use 10,000 draws and seed `20260949`; report percentile 95% intervals.
- If more than 2% of free outputs are invalid, classify execution as failed and withhold every score contrast. Otherwise calculate contrasts on source IDs complete in all four cells and report missing coverage.
- Reuse the Experiment 048 practical rule: evidence for a practically meaningful decoder interaction requires estimate ≥ +0.25 and lower interval bound > 0. Failing the rule is inconclusive; it does not establish equivalence. Separately report whether each decoder's context-gain estimate passes its +0.25/lower-bound rule.
- Descriptive only: report eligible gold-valence bucket counts, invalid counts, context gains by decoder, and the paired contrast. Do not claim a general decoder property from this single product-domain split.

## Compute and limits

There are 943 eligible IDs and 3,772 generations (1,886 per decoder). The 1.5B model and existing 0.1-grid trie fit the 24-GB MacBook Air setup used for Experiment 048. The run may take about an hour on MPS; it does not require remote inference or new training. The one-target-per-row rule, exact-span eligibility filter, positive-heavy natural valence distribution, public labels, single model, and shared benchmark release constrain interpretation. A positive result would support transfer from restaurant to laptop reviews in this one benchmark, not independent corpus replication.

## Provenance

- Dataset paper: [Lee et al., DimABSA, ACL 2026](https://aclanthology.org/2026.acl-long.1881/).
- Adjacent numeric-generation study: [Song & Bahri, Decoding-based Regression, TMLR 2025](https://arxiv.org/abs/2501.19383).
- Prior matched-prompt decoder factorial: [`048 protocol`](048-small-model-decoder-factorial.md) and [`048 aggregate result`](../../results/small-model-decoder-factorial-v1/README.md).
- Private Laya agenda trace: `.context/laya-research-triage-049.json` (not committed).
- Runner/analyzer: `scripts/run_laptop_decoder_transfer_049.py`, `scripts/analyze_laptop_decoder_transfer_049.py`.
