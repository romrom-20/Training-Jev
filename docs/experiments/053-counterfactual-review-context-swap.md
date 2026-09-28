# Experiment 053 — does the right review context matter?

**Status:** preregistered before inference. Laya selected this direction with low,
uncalibrated confidence; that choice is only agenda triage.

## Question

Experiments 050–052 found a large decoder-by-context interaction, but a code audit
corrected the estimand: `aspect_only` supplies an aspect with review text
`[NOT PROVIDED]`; `opinion_masked` supplies that case's review with annotated
opinion spans replaced by `[MASKED]`. The observed contrast is the utility or harm
of residual non-opinion review context. This experiment asks whether the finite
decoder's apparent context advantage requires the *matching review*, rather than
any natural review-like text of similar length.

## Sample and pairing

- Use the pinned English laptop split from DimABSA commit
  `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`, SHA-256
  `08a4cc197f068f6fc4c9373f61bfb86a5cd996ef16be3cb516082b434ba92543`.
- Reconstruct the 943 eligible cases from Experiment 049 exactly. Select 217 by
  sorting SHA-256 of `exp053-sample-v1|<case_id>` within gold-valence strata and
  taking 94 negative (<4.5), all 30 neutral ([4.5, 5.5]), and 93 positive (>5.5).
  This polarity-balanced mechanism sample is not prevalence-representative.
- The matched comparison is each selected case's existing Qwen2.5-3B
  `opinion_masked` prediction from Experiment 050. Verify the parent output hash
  `890a16d6363e72549c7ccad9099dece3ce021ab78ab3d8071acde18774500f5f`, every
  sample ID's full factorial, and the gold labels.
- Construct one counterfactual donor mapping. Tokenize each selected review after
  annotated opinion masking with the pinned model tokenizer. Sort by this token
  length and split into 10 contiguous near-equal-size bins. Within each bin, sort
  by SHA-256 of `exp053-donor-v1|<case_id>`; each case receives the next case's
  masked review cyclically. This is a deterministic derangement: no case donates
  to itself and each donor is used once. The target aspect and gold remain the
  recipient's. Review length is approximately matched; meaning, aspect alignment,
  and wording are intentionally mismatched.
- Generate only the `swapped_context` prompts for both decoders: 217 × 2 = 434
  new outputs. Reuse no raw text in public artifacts.

## Model and decoding

- `Qwen/Qwen2.5-3B-Instruct`, revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, local greedy MPS inference.
- Keep the Experiment 049 English prompt and one-decimal [1, 9] score grammar.
- `finite_grid`: the same finite 6,561-candidate JSON grid.
- `free_greedy`: ordinary greedy decoding, maximum 40 new tokens, strict JSON
  parser, no retries or output repair.
- Prompt bytes for each swapped item are identical across decoder arms. The only
  change from its matched counterpart is the review text; target aspect stays
  fixed.

## Outcomes and decision rules

- For each decoder, calculate the matched-review advantage as
  `RMSE(swapped_context) - RMSE(matched opinion_masked)` on both VA dimensions.
  Positive values mean the model predicts better with the recipient's own review
  than with another length-matched review.
- Primary contrast: matched-review advantage under finite-grid decoding minus
  matched-review advantage under free-greedy decoding. A positive interaction
  means finite decoding is more sensitive to whether the review matches its target.
- Bootstrap the 217 recipient IDs with replacement, retaining matched and swapped
  outputs for both decoders and both VA dimensions, with 10,000 replicates and
  seed `20260953`. Intervals are conditional on the one frozen donor permutation;
  they do not include donor-assignment uncertainty.
- If more than 2% of the 434 new free outputs are invalid, withhold all score
  contrasts. Otherwise analyze complete IDs and publish invalid counts and
  coverage.
- This is a focused mechanism diagnostic on a subset of a public test set. No
  confirmatory population claim, causal claim about language, or evidence of
  pretraining blindness follows from it. The deliberately polarity-balanced
  subset and fixed donor mapping limit generalization.

## Compute and provenance

Experiment 050 used about 6,808 seconds for 3,772 outputs on the 24-GB MacBook Air.
This run adds 434 outputs and is expected to take around 15 minutes. Inputs,
donor mapping, raw outputs, and item-level predictions stay in ignored `.context/`.

- Parent protocol/result: [Experiment 050](050-laptop-model-size-factorial.md)
  and [`results/laptop-model-size-v1`](../../results/laptop-model-size-v1/README.md).
- Earlier context masking and denoising are established ABSA directions; this
  test is a narrow decoder-by-instance-matched-context control, not a claim that
  context interventions themselves are new. See [Aspect-Based Sentiment Analysis
  with Context Denoising](https://aclanthology.org/2024.findings-naacl.194/).
- Runner/analyzer: `scripts/run_counterfactual_context_swap_053.py`,
  `scripts/analyze_counterfactual_context_swap_053.py`.
