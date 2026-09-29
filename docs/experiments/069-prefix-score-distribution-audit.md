# Experiment 069 — conditional score distributions after forced prefixes

**Status:** frozen analysis protocol; post-hoc mechanism audit of Experiments 067
and 068. No new generated answers or recipients are introduced. The question was
selected after the Exp068 result, so inferential intervals describe these fixed
cohorts and should not be treated as an independent confirmatory test.

## Question

Experiments 067 and 068 found that forcing arousal to 8.0 rather than 2.0 before
generating valence lowered the greedy continuation by 0.688 and 0.809 points.
The same intervention before generating arousal did not produce a resolved shift.
Does the arousal-first change affect probability across the valid-score
distribution, or mainly change the one score selected by greedy decoding?

## Frozen inputs and model

- Reconstruct every prompt and assistant prefix from the pinned SIGHAN source,
  using the exact prompt builders and selected IDs from Experiments 067 and 068.
- Frozen selected-ID hashes are `ddd1584d1f5151b4e29cf5de1dfbe5bfc6657f3c90599d494f919193b884a621`
  (067) and `1e738712d6ec47bb90a5bdd47d676f3b79fcb958b9c0d945d3081428924132bd`
  (068). Frozen prediction-file SHA-256 values are
  `0cfaffa2b5c0da553ddb952f6f969d43d966fc404cce4aed35aefe04f2fc0bf6` and
  `873f1ff5b63b5643adb94e480f1e1ee2272d013aea8c991116f4bea26488a2e7`,
  respectively.
- Verify both private prediction files against their run-manifest output hashes,
  verify each case/order/value cell is present once, and confirm the reconstructed
  assistant prefixes match those stored with the generation outputs.
- Use the exact Qwen2.5-3B-Instruct revision and tokenizer recorded by both parent
  runs, on local MPS. No new generation, sampling, training or external compute.
- For each of 128 recipients × 2 orders × 2 forced values (512 contexts), score
  all 81 canonical numeric continuations from 1.0 through 9.0 in steps of 0.1.
  A continuation is its exact numeric token sequence, excluding JSON punctuation.
  Verify the tokenizer represents every candidate as digit, decimal point, digit.

## Outcomes

For each context, add the autoregressive token log probabilities for each of the
81 strings, then normalize those scores over this valid-score set. This is a
restricted distribution; it excludes malformed answers, alternate number
spellings, and the probability of the closing JSON token. It is not the model's
unconditional output distribution.

The primary diagnostic is the recipient-paired change in the normalized expected
second coordinate, high forced first score (8.0) minus low (2.0), in the
arousal-first order, reported separately for Exp067 and Exp068. A broad negative
distribution shift is supported only if both cohort-specific recipient-bootstrap
95% intervals are below zero. Otherwise the distributional result is unresolved;
failure to resolve is not evidence of no shift. Report the corresponding
valence-first expectations as secondary diagnostics.

Also report, by cohort and order, mean total-variation distance between the low-
and high-prefix distributions, mean one-dimensional Wasserstein-1 distance, and
the restricted-distribution argmax. Compare those estimates with the already
observed greedy outputs. Bootstrap recipients with replacement, keeping paired
forced values together, with 10,000 draws and fixed seed 20260969.

## Interpretation limits

The audit can distinguish a shift in the normalized valid-number distribution
from a shift that appears only in the greedy output, within this scoring setup.
The restricted support and conditioning on hand-supplied assistant prefixes are
material limitations. Results remain specific to Qwen2.5-3B, one prompt template,
deterministic token scoring, two SIGHAN cohorts, and an artificial continuation
intervention. Neither outcome establishes ordinary rating behavior or the cause
of earlier field-order effects. This is a follow-up selected after observing the
two parent results, not independent confirmation.

The adjacent literature already studies numeric anchoring in LLM answers, output
order in dialogue evaluation, and field/schema order. No broad anchoring or
output-order novelty claim follows from this audit. See the literature boundary
in [Experiment 064](064-literature-boundary-audit.md).

- Private inputs: `.context/exp067-private-predictions.jsonl` and
  `.context/exp068-private-predictions.jsonl`.
- Runner/analyzer: `scripts/run_prefix_score_distribution_audit_069.py`.
