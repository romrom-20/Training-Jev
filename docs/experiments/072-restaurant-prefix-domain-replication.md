# Experiment 072 — prefix-score coupling on English restaurant reviews

**Status:** preregistered before prefix scoring, following Laya triage 072 choice A
(0.2842 uncalibrated). Tests whether the prefix-coupling pattern observed on
English laptop inputs in Experiment 071 is also present in restaurant inputs.

## Question

Do forced-prefix shifts in the conditional distribution over valid second-coordinate
scores have the same model-specific shape on English restaurant and laptop reviews?
This is a domain comparison of an artificial assistant-prefix intervention. It is
not a test of ordinary ratings, and no broad anchoring mechanism is presumed.

## Cohort and design

- Source: the pinned DimABSA `eng_restaurant_test_task2.jsonl`, commit
  `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`, SHA-256
  `e90386169422c84ddd42d51029f23934b54724fafe270b65138836067492857d`.
- Use Experiment 051's eligibility rule and one-target-per-ID selection. Select
  32 cases with gold valence <4.5 and 32 with valence >5.5, ranking each group
  by SHA-256 of `exp072-restaurant-prefix-domain-v1|<case_id>`. The sorted IDs
  joined with newline have SHA-256
  `f15f8acb5cf3cc80bf9eea082033fd652b336fcfe3732d8294c831ddd8ae4343`.
- **Prior exposure:** Experiment 051 scored the full 963-ID eligible restaurant
  split under other prompt-evidence/decoder conditions. These 64 recipient IDs
  and source reviews are therefore reused. The 072 prefix-conditioned outputs
  are newly collected, but this is not a fresh-sample or blind replication.
  Report this prominently and interpret only as a new treatment/domain check.
- Use the same user prompt construction and system message as Experiment 071,
  with each model's own chat template. For every recipient, score all 81 canonical
  one-decimal second-coordinate candidates (1.0–9.0) under valence-first or
  arousal-first prefixes forcing 2.0 or 8.0, and record one greedy continuation.
  This is 256 contexts per model, 768 total.
- Models/revisions: Qwen2.5-1.5B-Instruct
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, Qwen2.5-3B-Instruct
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`, and SmolLM2-1.7B-Instruct
  `31b70e2e869a7173562077fd711b654946d38674`; local MPS only.

## Outcomes and analysis

For each model and order, calculate paired high-minus-low shifts in normalized
expected second score and greedy score, total variation and Wasserstein-1 distances,
invalid rates, and the valid-score support mass. Bootstrap recipients with
replacement (10,000 draws, seed 20260972). Also compare restaurant shifts with the
matching Experiment 071 laptop shifts descriptively, bootstrapping recipient IDs
within domain independently. Treat all comparisons as exploratory and report
intervals, not binary significance claims. Withhold score contrasts if any model ×
order × forced-value invalid rate exceeds 2%.

## Limits

The domain comparison reuses the public restaurant source items already scored in
051, while 071's laptop IDs are distinct. Data source, earlier researcher exposure,
model family, tokenizer and chat template can limit interpretation. Model-family
comparisons are not a controlled scaling law. Public benchmark pretraining exposure
cannot be excluded. No LessWrong claim should follow from this one cohort; any
interesting pattern needs genuinely independent source material and controls.
