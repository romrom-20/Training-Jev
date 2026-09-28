# Experiment 056 — polarity-matched review-context control

**Status:** preregistered before inference. Laya selected this test with very low,
uncalibrated confidence; its choice is agenda triage only.

## Question

Experiments 053–055 repeatedly found a larger finite-grid than free-greedy penalty
when a target's own masked review was replaced by another review. A leading
alternative is that the finite decoder follows the donor review's coarse
negative/positive sentiment, rather than fine-grained context matched to the
target. This experiment matches donor and recipient gold-valence polarity while
still breaking exact review/aspect linkage.

## Sample and donor assignment

- Reuse the exact fresh 217 IDs from Experiment 055: 108 negative and 109
  positive-valence English laptop cases, disjoint from Experiments 053–054. Use
  the pinned source revision `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1` and its
  source SHA-256 `08a4cc197f068f6fc4c9373f61bfb86a5cd996ef16be3cb516082b434ba92543`.
- Reuse each recipient's own-review Qwen2.5-3B `opinion_masked` predictions from
  Experiment 050. Verify parent output hash
  `890a16d6363e72549c7ccad9099dece3ce021ab78ab3d8071acde18774500f5f`, recipient
  IDs, and gold values.
- For each of three independent permutations, separate cases by gold valence
  polarity (<4.5 negative, >5.5 positive). Within each polarity, sort the
  opinion-masked review texts by Qwen2.5-3B tokenizer length and split into 10
  contiguous near-equal-size bins. Within each bin, sort by SHA-256 of
  `exp056-p<permutation>|<case_id>` and assign each recipient the next case's
  masked review cyclically. This yields three no-self derangements in which every
  donor has the same gold-polarity class as its recipient and each donor is used
  once per permutation.
- Keep each recipient's own target aspect and gold unchanged. Generate both
  decoders for all 217 recipients and 3 permutations: 1,302 outputs. Do not pass
  any polarity label or gold score to the model. Keep reviews, IDs, donor maps,
  raw generations, and per-item outputs in ignored `.context/`.

## Model and decoding

- `Qwen/Qwen2.5-3B-Instruct`, revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, greedy local MPS inference.
- Preserve the English prompt and one-decimal [1, 9] numeric grammar from
  Experiment 050. `finite_grid` uses its same 6,561-candidate JSON grammar;
  `free_greedy` uses standard greedy decoding, max 40 new tokens, strict JSON
  parse, no retries or repair. Decoder-arm prompt bytes must match.

## Outcomes and rule

- For each decoder and permutation, calculate matched-review advantage as
  `RMSE(polarity-matched donor context) - RMSE(own opinion-masked review)` over
  both VA dimensions. Positive values mean the model predicts better from the
  exact recipient review than a donor with the same coarse polarity.
- The primary diagnostic is finite-grid matched-review advantage minus
  free-greedy matched-review advantage. Report every permutation, the mean, range,
  and descriptive standard deviation.
- Bootstrap recipient IDs with replacement, retaining all three permutations,
  decoders, matched outputs, and VA dimensions; 10,000 draws with seed `20260956`.
  The 95% interval for the mean interaction is conditional on the fixed mappings.
- If more than 2% of the 651 free outputs are invalid, withhold all score
  contrasts. Otherwise report coverage and invalid counts.

## Interpretation and limits

If the finite-specific matched-review advantage remains positive after this
polarity control, it would rule against a simple negative/positive prior as the
whole explanation. It would not establish which words or relations matter, nor
prove general context understanding. If the effect shrinks, that would support a
coarse-polarity account. This uses a public benchmark, one model family, and a
non-neutral subset; pretraining exposure remains possible. Context masking and
denoising are established ABSA work; the narrower contribution remains how
numeric decoding changes sensitivity to instance-matched evidence.

- Parent data/outputs: [Experiment 055](055-fresh-context-swap-replication.md)
  and [Experiment 050](050-laptop-model-size-factorial.md).
- Runner/analyzer: `scripts/run_polarity_matched_context_swap_056.py`,
  `scripts/analyze_polarity_matched_context_swap_056.py`.
