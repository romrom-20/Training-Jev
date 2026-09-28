# Experiment 054 — robustness to review-donor assignment

**Status:** preregistered before inference. Laya selected three independently
seeded donor permutations with low, uncalibrated confidence; the engine choice is
agenda triage only.

## Question

Experiment 053 found that replacing a target's opinion-masked laptop review with
one length-binned review from another case increased finite-grid RMSE more than
free-greedy RMSE. That estimate conditions on a single arbitrary donor assignment.
Experiment 054 repeats the same context swaps with three independent deterministic
derangements to measure whether the result is robust to that assignment.

## Fixed sample and inputs

- Reuse the exact 217 Experiment 053 IDs: 94 negative, 30 neutral, and 93
  positive-valence cases from the pinned English laptop test source. Reconstruct
  them with Experiment 053's frozen selector; do not resample.
- Verify Experiment 050 matched predictions using SHA-256
  `890a16d6363e72549c7ccad9099dece3ce021ab78ab3d8071acde18774500f5f`.
- For each permutation, keep each recipient's target aspect and gold fixed. Its
  review text is replaced by another selected case's review after masking every
  annotated opinion span.
- Tokenize masked review text with the pinned Qwen2.5-3B tokenizer, sort by token
  count, split into 10 contiguous near-equal-size length bins, then within each
  bin sort by SHA-256 of `exp054-p<permutation>|<case_id>`. Assign each case the
  next case's text cyclically. This creates three independent, deterministic,
  no-self donor permutations; each selected review donates once in each
  permutation. The same target receives potentially different donors in each.
- Produce both decoder outputs for all 217 recipients and all three permutations:
  1,302 new generations total. Inputs, mappings, raw outputs, and item-level
  predictions stay in ignored `.context/`.

## Model and decoding

- `Qwen/Qwen2.5-3B-Instruct`, revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, local greedy MPS inference.
- Keep Experiment 050's English prompt, one-decimal [1, 9] score instruction,
  and 6,561-candidate finite JSON grammar unchanged.
- Run finite-grid and free-greedy decoding with the Experiment 050 settings:
  maximum 40 new tokens for free output, strict JSON parser, no retries or repair.
- Within each case and permutation, prompt bytes are identical across decoders.

## Outcomes

- Per decoder and permutation, matched-review advantage is
  `RMSE(swapped context) - RMSE(own opinion-masked review)` over both VA
  dimensions. Positive means own review predicts better than a donor review.
- Per-permutation primary diagnostic is finite-grid matched-review advantage
  minus free-greedy matched-review advantage.
- Also report the mean of the three per-permutation primary estimates and their
  range and standard deviation. These describe assignment sensitivity; three
  mappings do not support a precise donor-randomization distribution.
- Bootstrap 217 recipient IDs with replacement, retaining all three permutations,
  both decoders, own-review predictions and both VA dimensions. Use 10,000 draws
  and seed `20260954`. The interval for the mean interaction is conditional on
  the three frozen donor mappings.
- If more than 2% of the 651 new free outputs are invalid, withhold score
  contrasts. Otherwise report complete coverage and invalid counts.

## Interpretation limits

This remains a mechanism diagnostic on a polarity-balanced subset of one public
test release. All three permutations may share coarse polarity mismatches; they
do not control that alternative. They also do not establish pretraining blindness,
population generalization, or a general property of constrained decoding. Context
masking and denoising already have a substantial ABSA literature, including
[Tian et al. (2024)](https://aclanthology.org/2024.findings-naacl.194/); the
narrow question here is interaction between numeric decoding and instance-matched
context for continuous VA estimates.

## Compute and provenance

Experiment 053 completed 434 new outputs in about 614 seconds of generation time.
This 1,302-output follow-up is expected to take roughly 30 minutes on the 24-GB
MacBook Air. See [Experiment 053](053-counterfactual-review-context-swap.md) and
its [aggregate result](../../results/counterfactual-context-swap-v1/README.md).

- Runner/analyzer: `scripts/run_context_swap_donor_robustness_054.py`,
  `scripts/analyze_context_swap_donor_robustness_054.py`.
