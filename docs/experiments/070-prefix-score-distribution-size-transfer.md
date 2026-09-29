# Experiment 070 — prefix score distributions across Qwen2.5 sizes

**Status:** preregistered before new 1.5B scoring or generation. Laya selected
this follow-up as its top agenda option (0.418; uncalibrated). The 3B reference
results were already observed in Experiments 067–069; this is a model-size
follow-up, not an independent blind replication.

## Question

Does the distribution-level response to a forced first valence-arousal (VA)
coordinate, and the apparent gap between that response and greedy output,
persist in Qwen2.5-1.5B? Experiment 069 found a positive valence-first shift in
conditional expected arousal (+0.621 to +0.666) in both recipient cohorts even
though 3B greedy shifts were near zero. Arousal-first conditional valence shifts
were negative and accompanied negative greedy shifts.

## Fixed inputs and model

- Reuse the exact 64 Exp067 and 64 disjoint Exp068 recipients, prompt text,
  field orders and forced first values (2.0, 8.0). The sample-ID hashes are
  recorded in the 067/068 protocols. No new recipients are selected.
- Use `Qwen/Qwen2.5-1.5B-Instruct`, pinned revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, local MPS inference only.
- Score the same 81 canonical second-coordinate strings from 1.0 to 9.0 in
  0.1 increments. Verify each tokenizer representation is digit, point, digit.
  Normalize over those 81 strings and separately report unnormalized support
  mass.
- Also generate one deterministic greedy continuation per prefix context
  (512 total), using the same assistant-prefix text and 40-token limit as 067.
  Do not retry or repair invalid outputs. Report invalid counts per cohort, order
  and forced value; withhold score contrasts if any arm exceeds 2% invalid.

## Outcomes

For each cohort and field order, report recipient-bootstrap mean conditional
expected-score shift (first value 8 minus 2), actual greedy-score shift, their
paired difference, total-variation distance, Wasserstein-1 distance and valid
score-string probability mass. Use 10,000 paired recipient bootstrap draws and
seed `20260970`. The two cohorts are reported separately, never pooled as new
independent evidence. Compare 1.5B expected-score shifts against the 3B shifts
from 069 on the same recipient IDs with a paired recipient bootstrap, separately
by cohort/order. This size comparison is secondary and adaptive.

The distribution-vs-greedy contrast is the main diagnostic. A matching positive
valence-first expectation shift with a near-zero greedy shift would show that the
decoding gap persists at the smaller size; a changed sign or resolved greedy
shift would revise that interpretation. Null intervals remain inconclusive.

## Limits

This is one model family at two scales, the same two SIGHAN recipient cohorts,
and an artificial assistant-prefix intervention selected after observing the
3B outputs. It does not establish ordinary rating anchoring, language or domain
generality, or novelty over the numerical anchoring literature. No LessWrong
post should be considered from this size transfer alone.

- 3B comparison: [Experiment 069](069-prefix-score-distribution-audit.md).
- Runner/analyzer: `scripts/run_prefix_score_distribution_size_transfer_070.py`,
  `scripts/analyze_prefix_score_distribution_size_transfer_070.py`.
