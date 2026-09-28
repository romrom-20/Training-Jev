# Experiment 052 — Does the decoder-by-context interaction transfer to Chinese reviews?

**Status:** adaptive follow-up selected by the pinned local Laya typed-choice checkpoint after Experiment 051 and a corpus audit. Freeze this protocol, runner, and analyzer before scoring the Chinese test set.

## Question and motivation

Experiment 050 found a large decoder-by-context interaction on English laptop
reviews at Qwen2.5-3B (+2.666 VA RMSE points). Experiment 051 reproduced it on
English restaurant reviews (+2.776), with an estimated laptop-minus-restaurant
difference of −0.111 (95% independent-source-ID interval [−0.241, +0.023]). Both
domains also exhibited output-mode collapse in aspect-only prompts. These replications
are within one DimABSA release and one language.

Experiment 052 tests the same paired evidence/decoder factorial on the independently
curated SIGHAN 2024 Chinese dimABSA restaurant Task 2/3 test set. This changes both
language and corpus while holding the 3B checkpoint, English prompt, score grid, and
decoding algorithms fixed. It tests transfer of the *interaction* and measured value
of added opinion evidence, not general model quality or the cause of the output modes.

## Data and sample

- Use the public simplified-Chinese files `SIGHAN2024_dimABSA_Testing_Task2+3_Simplified.txt`
  (SHA-256 `5053608647d06ad88ce21a2aecbef0e29dc58a8220f921561d39d4e9cf86fe28`)
  and `SIGHAN2024_dimABSA_Testing_Task2+3_Simplified_truth.txt` (SHA-256
  `6ecc6949d5f77ea34a09826115a0f91327313ae33824a5c373d7322bdbb17108`) from
  `NYCU-NLP/SIGHAN2024-dimABSA` commit
  `d43a482039a41fbc26087ef5b26e94ee772d0fb6`.
- The source audit verified 2,000 unique IDs in both files, identical ordered IDs,
  zero ID overlap between Task 2/3 and Task 1 test sets, and zero ID overlap with
  the 6,050 records in both released training files. The two training files contain
  3,000 and 3,050 unique IDs respectively, with no overlap. Their SHA-256 values are
  `982c3c49084bad508da7e0d954bd92369a164643c3008883b7d88d461749bae1` and
  `7137b8591595cc2860e4c8baa8a9ae574555f67dd41d0f21e9f90702ccfec6f4`.
  The Task 1 input and gold each have 2,000 unique IDs; their hashes are
  `6607e4a2f1e1baa87feea1689d814f082d8e5bf30d6f2661a92f4d6c5e20cb16` and
  `a38dedc7d4e1926eec79087538003974bd7250b7cd55434a1bf44ce8b7b98859`.
- Include every source ID for which at least one gold triplet has a non-null aspect
  and opinion that each occur exactly once in the review, every distinct non-null
  opinion span occurs exactly once and does not overlap the target aspect, and both
  VA gold values parse as finite numbers in [1, 9]. Every distinct non-null opinion
  span must be pairwise non-overlapping, except identical duplicate spans. Select the
  first eligible target in source order, one target per source ID. The stricter
  masking audit yields 1,916 IDs.
- Keep review text, aspects, opinions, IDs, raw generations, and per-item outputs in
  ignored `.context/` files. Publish aggregate results and source provenance only.

## Conditions and model

For each eligible source ID, produce `aspect_only` and `opinion_masked` outputs with
both decoders, for 7,664 generations total. `aspect_only` shows the complete Chinese
review text and named target aspect. `opinion_masked` replaces every annotated
non-null opinion span in that source sentence with `[MASKED]`; the prompt wording,
other context, and target aspect remain the same. Keep prompt bytes identical between
decoder arms within each evidence condition. Use the same English instructions as
Experiments 049–051 to isolate the change in review language.

- Model: `Qwen/Qwen2.5-3B-Instruct`, revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, greedy local MPS inference.
- `finite_grid`: same finite 6,561-candidate JSON grid of one-decimal valence and
  arousal scores in [1, 9].
- `free_greedy`: ordinary greedy generation, maximum 40 new tokens, strict JSON
  parser, no retries or output repair.

## Outcomes and rules

- For each decoder, calculate opinion-masked context gain as
  `RMSE(aspect_only) - RMSE(opinion_masked)` across both VA dimensions. Positive
  values mean masking annotated opinion evidence increases prediction error.
- **Primary contrast:** `context_gain(finite_grid) - context_gain(free_greedy)`.
  Positive values mean the finite output grammar amplifies the measured value of
  unmasked opinion evidence on this Chinese test set.
- Bootstrap source review IDs with replacement, retaining both conditions, both
  decoders, and both VA dimensions; use 10,000 draws, seed `20260952`, and percentile
  95% intervals.
- If free-output invalid rate exceeds 2%, withhold every score contrast. Otherwise,
  analyze complete source IDs and report coverage and all invalid counts.
- The registered practical rule is estimate ≥ +0.25 and lower interval bound > 0.
  Failing it is inconclusive and does not show decoder equivalence.
- **Secondary transfer contrast:** compare the Chinese interaction with Experiment
  051's English restaurant interaction. Estimate `English restaurant − Chinese`
  interaction and bootstrap the source IDs independently within each release. This
  is descriptive, not a second confirmatory claim; language and source release vary
  together.
- Report both decoder-specific context gains, primary interaction, invalid outputs,
  runtime, hashes, output marginals, and the descriptive transfer difference. Do not
  claim language causality or corpus-independent generality from one Chinese release.

## Compute and limits

Experiment 051 took 7,392 seconds for 3,852 outputs on the 24-GB MacBook Air. The
7,664-generation run is expected to take about four hours and writes resumable
checkpoints after each batch. The SIGHAN test labels are public; possible Qwen
pretraining exposure cannot be ruled out. Although SIGHAN is a separate release,
it is also a restaurant-review dataset, and this transfer changes both language and
corpus provenance at once. A positive result would strengthen cross-language/corpus
transfer evidence but would not resolve the output-representation mechanism.

## Provenance

- [SIGHAN 2024 dimABSA task site](https://dimabsa2024.github.io/) and
  [overview paper](https://aclanthology.org/2024.sighan-1.19/).
- [Pinned data repository](https://github.com/NYCU-NLP/SIGHAN2024-dimABSA/tree/d43a482039a41fbc26087ef5b26e94ee772d0fb6).
- Parent outcomes: [050 English laptop](../../results/laptop-model-size-v1/README.md),
  [051 English restaurant](../../results/restaurant-domain-decoder-3b-v1/README.md).
- Private Laya selection: `.context/laya-research-triage-053.json` (not committed).
- Runner/analyzer: `scripts/run_sighan_chinese_decoder_transfer_052.py`,
  `scripts/analyze_sighan_chinese_decoder_transfer_052.py`.
