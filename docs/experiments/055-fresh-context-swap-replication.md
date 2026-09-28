# Experiment 055 — fresh-sample replication of matched-review sensitivity

**Status:** preregistered before inference. Laya selected a fresh disjoint sample
with three donor permutations with low, uncalibrated confidence; this choice is
agenda triage, not evidence.

## Question

Experiments 053–054 found a positive finite-minus-free difference in own-review
versus donor-review RMSE on 217 English laptop cases. Experiment 055 tests whether
that signal repeats on a fresh, non-overlapping subset of the same public split,
rather than depending on the first selected IDs.

## Sample and comparison

- Reconstruct all 943 eligible English laptop cases from the pinned DimABSA
  revision `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`, file SHA-256
  `08a4cc197f068f6fc4c9373f61bfb86a5cd996ef16be3cb516082b434ba92543`.
- Exclude all 217 Experiment 053/054 IDs. That first sample included all 30
  neutral-valence IDs. From the remaining cases choose 108 negative (<4.5) and
  109 positive (>5.5) IDs by sorting SHA-256 of `exp055-sample-v1|<case_id>`
  within each bucket. The fresh replication has no neutral cases and is balanced
  across negative and positive valence; do not generalize to neutral examples.
- Reuse each fresh ID's own-review Qwen2.5-3B `opinion_masked` outputs from
  Experiment 050. Verify the parent file SHA-256
  `890a16d6363e72549c7ccad9099dece3ce021ab78ab3d8071acde18774500f5f`, full
  decoder coverage, and gold labels.
- Construct three donor derangements. For each permutation separately, tokenize
  each opinion-masked review with the pinned Qwen2.5-3B tokenizer, sort by token
  count, and split into 10 contiguous near-equal-size bins. Within each bin, sort
  by SHA-256 of `exp055-p<permutation>|<case_id>` and give each case the next
  case's masked review cyclically. Donors do not self-match and are each used once
  per permutation. Keep recipient target aspect and gold fixed.
- Run both decoders for all 217 recipients in all three permutations: 1,302 new
  generations. Inputs, mappings, raw generations, and item-level predictions stay
  in ignored `.context/`.

## Model, decoding, and outcomes

- `Qwen/Qwen2.5-3B-Instruct`, revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, greedy local MPS inference.
- Keep the English prompt and one-decimal [1, 9] grammar unchanged from
  Experiments 050 and 053. `finite_grid` uses the same 6,561 JSON candidates;
  `free_greedy` uses ordinary greedy generation, maximum 40 new tokens, strict
  JSON parsing, and no retries or repair. Paired decoder prompts are identical.
- For each decoder and permutation, calculate
  `RMSE(swapped_context) - RMSE(own opinion_masked review)` over both VA
  dimensions. Positive values mean the model does better with its own review.
- The per-permutation interaction is finite-grid matched-review advantage minus
  free-greedy matched-review advantage. Report all three, their mean, range and
  standard deviation. Bootstrap recipient IDs with replacement, retaining all
  permutations, both decoders, matched outputs, and VA dimensions; 10,000 draws,
  seed `20260955`. The mean-interaction 95% interval is conditional on the three
  fixed mappings.
- If more than 2% of the 651 free outputs are invalid, withhold all score
  contrasts. Otherwise report invalid counts and complete coverage.

## Limits and next control

This is a fresh recipient sample but the same public dataset, model family, and
prompt. Public test exposure during pretraining is possible. The sample omits
neutral-valence items. Donor mappings are not constrained by recipient polarity,
so this does not resolve the coarse-polarity alternative; the next mechanism
control should use polarity-matched donors if the effect repeats. Existing ABSA
research already studies context masking and denoising; this test is about
decoder-specific sensitivity to whether review context matches its target.

Experiment 054 generated 1,302 outputs in about 1,911 seconds on the 24-GB Air;
055 is the same size. See [053](053-counterfactual-review-context-swap.md),
[054](054-context-swap-donor-robustness.md), and the
[054 aggregate result](../../results/context-swap-donor-robustness-v1/README.md).

- Runner/analyzer: `scripts/run_fresh_context_swap_replication_055.py`,
  `scripts/analyze_fresh_context_swap_replication_055.py`.
