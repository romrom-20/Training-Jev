# Experiment 071 — forced-prefix score coupling across model families

**Status:** preregister before inference. Laya ranked an independent-family test
highest after Experiment 070 (choice B, 0.328, uncalibrated). Its suggested
Gemma 2 checkpoint is gated by Google usage terms and is not cached, so this
protocol uses the public Apache-2.0 SmolLM2-1.7B-Instruct checkpoint instead.
The model-family objective is unchanged. To avoid testing Chinese ability as a
confound, use English laptop reviews and all three models on the same new cases.

## Question

Does the large positive forced-coordinate response in Qwen2.5-1.5B, versus the
smaller/asymmetric response in Qwen2.5-3B, appear in an independent compact model
family on the same English review inputs? This is a cross-family comparison of
an artificial continuation intervention, not a claim about normal ratings.

## Cohort and prompts

- Start with the 943 eligible English laptop IDs from the pinned DimABSA Task 2
  test file and Experiment 049's frozen one-target-per-ID rule. Exclude all IDs
  used in Experiment 055. From the remaining cases, select 32 negative-valence
  recipients (gold valence <4.5) and 32 positive-valence recipients (gold
  valence >5.5), ranked within each polarity by SHA-256 of
  `exp071-cross-family-prefix-v1|<case_id>`. No model output is used for
  selection; omit neutral-valence cases.
- The sorted selected IDs joined by newline have SHA-256
  `a14b6300c88940b3ccb6ac6b1b78c78b1cb5e202c4ce945ea3fe8220d36c7ddf`. The
  sample is disjoint from all 217 Experiment 055 IDs.
- Keep each case's aspect and gold fixed. Use Experiment 049's exact one-decimal
  numeric VA prompt with the opinion-masked English review as evidence. Add the
  same explicit system instruction to all models: “You are an expert affective
  computing annotator. Follow the user's instructions exactly.” Use each model's
  own chat template; record rendered-prefix hashes. The textual system and user
  message bodies are identical across models.
- For each selected case, generate one greedy continuation after each of four
  assistant prefixes: valence-first with 2.0/8.0 forced, and arousal-first with
  2.0/8.0 forced. Maximum 40 new tokens, strict existing parser, no retries or
  repairs.
- Score the 81 canonical one-decimal second-coordinate strings from 1.0 to 9.0
  in 0.1 increments for every prefix. Verify all three tokenizers represent
  every candidate as digit, decimal point, digit before model inference. Save
  candidate token log probabilities, raw greedy continuations, model outputs,
  and prompts only under ignored `.context/`.
- Models and revisions: Qwen2.5-1.5B-Instruct
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, Qwen2.5-3B-Instruct
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, and
  HuggingFaceTB/SmolLM2-1.7B-Instruct
  `31b70e2e869a7173562077fd711b654946d38674`. Local MPS only.

## Outcomes

For each model and output order, report the paired high-minus-low forced-value
shift in (a) normalized expected second score over the 81-score support and
(b) greedy score. Also report their paired difference, the distribution's
total-variation and Wasserstein-1 distances, invalid-output counts, and
unnormalized probability mass on the valid-score strings.

The main model-family contrast is SmolLM2 minus each Qwen model in expected
score shift, separately by field order. The main within-family size contrast is
Qwen2.5-1.5B minus Qwen2.5-3B. Bootstrap recipient IDs with replacement,
retaining all four forced-value/order cells and all three models; use 10,000
draws and seed `20260971`. If any model × field-order × forced-value arm has
invalid rate above 2%, withhold score contrasts for the affected full comparison.
Otherwise analyze recipients complete in all model cells and report coverage.
Treat all model comparisons as exploratory and report intervals, not binary
significance claims.

## Limits and literature boundary

The sample is a public English laptop test split, selected after earlier work;
public pretraining exposure cannot be ruled out. Model family, tokenizer,
instruction tuning, and chat template vary together in the SmolLM2 comparison.
The common message bodies reduce prompt-content variation but cannot eliminate
template differences. Qwen sizes are compared within family; SmolLM2 is a
different family, not a controlled scaling point. The assistant prefixes are
artificial. Numeric anchoring across LLMs and output-order effects in scoring
are established; this experiment does not claim either broad phenomenon as new.

SmolLM2-1.7B-Instruct is an Apache-2.0, 1.7B-parameter model intended for
lightweight local use; its pinned weights and tokenizer are public. See the
[official model card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct).

- Parent scale/family result: [Experiment 070](070-prefix-score-distribution-size-transfer.md).
- Runner/analyzer: `scripts/run_cross_family_prefix_coupling_071.py`,
  `scripts/analyze_cross_family_prefix_coupling_071.py`.
