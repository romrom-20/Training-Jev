# Experiment 060 — polarity-matched, cross-category donor control

**Status:** preregistered before new target-model inference. The pinned local
Laya triage selected this direction with 0.349 uncalibrated choice probability.

## Question

Does an official aspect-category match itself explain why the decoder-specific
matched-review interaction was small in Experiment 057, compared with the
polarity-only donor condition in 056? Compare same-category and different-category
donors on the exact same recipients while holding polarity fixed.

## Design

- Use the same 184 fresh laptop recipients from Experiments 057 and 059. Keep
  each recipient's target aspect and gold VA fixed. The sample has 89 negative
  and 95 positive targets; neutral cases are not available.
- Cross-category donors must share the recipient's negative/positive gold-valence
  bucket and must have a different official DimABSA Task 3 aspect category.
  Within each polarity, create a one-to-one recipient-to-donor assignment so each
  donor is used once and no recipient donates to itself.
- Tokenize opinion-masked reviews with the pinned Qwen2.5-3B tokenizer. For each
  of three assignments, find a maximum bipartite matching independently within
  polarity using a deterministic augmenting-path algorithm. Process recipients
  by increasing category-cell size, then masked-review token length, then
  `SHA256(seed:case_id)`, then ID. For each recipient, visit eligible donors
  (same polarity, different category) by absolute masked-review token-length
  difference, then `SHA256(seed:donor_id)`, then ID. Use seeds 20260601,
  20260602, and 20260603. Fail preflight unless all assignments are complete,
  bijective, different-category, same-polarity, non-self, and globally distinct.
- Reuse the exact Qwen2.5-3B own-review baselines from Experiment 050 and the
  same-category donor predictions from 057 after verifying their output hashes
  (`890a16d6363e72549c7ccad9099dece3ce021ab78ab3d8071acde18774500f5f` and
  `21e7fe02b43433fa4b61b5dfe790efef38c57e8a0e240e4443ffe40ffb110e31`) and gold
  labels. Generate only the three new cross-category assignments:
  184 recipients × 3 assignments × 2 decoders = 1,104 outputs.
- Model/prompt/decoding: `Qwen/Qwen2.5-3B-Instruct`, revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`; greedy local MPS, Experiment 050
  prompt wording, 6,561 one-decimal VA grid candidates, free-greedy 40-token cap,
  no retries. Both decoders receive the identical prompt in each condition.
- Per decoder and assignment, compute donor-context RMSE minus own-review RMSE.
  The primary same-recipient contrast is the average across maps of
  `[finite-grid advantage(cross-category) - free-greedy advantage(cross-category)]`
  minus the corresponding Experiment 057 same-category interaction. Positive
  values mean that crossing the category boundary selectively increases the
  finite-grid matched-review disadvantage. Report each condition's interaction,
  all assignment estimates, and their paired difference.
- Use 10,000 recipient-ID bootstrap draws with seed `20260961`, retaining both
  donor conditions, every fixed assignment, both decoders, own-review predictions,
  and VA dimensions for each sampled ID. Intervals condition on these assignments.
  If more than 2% of the 552 new free-greedy outputs are invalid, withhold all score
  contrasts.

## Limits

The same-category arm is already collected, so this is a planned follow-up on an
observed result, not a blind replication. The cross-category donors are algorithmic
matches, not randomized annotations; category differences also change lexical and
semantic content. Official categories do not control exact aspect identity or
review wording. The task data are public, pretraining overlap is possible, and
there is only one model family and split. A result can show whether this benchmark's
decoder interaction varies with broad category match; it cannot establish a general
causal mechanism for language models.

No LessWrong post will be written or published until this comparison and any
necessary follow-up are complete.

- Prior same-category condition: [Experiment 057](../../results/category-matched-context-swap-v1/README.md)
- Runner/analyzer: `scripts/run_cross_category_donor_control_060.py`,
  `scripts/analyze_cross_category_donor_control_060.py`
