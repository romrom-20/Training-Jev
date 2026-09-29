# Experiment 074 — audit numeric answer probability support in Qwen

**Status:** preregistered before inference, following Laya local agenda triage 074
(choice B, probability 0.313, uncalibrated). Audits a possible interpretation
confound in Experiment 073: Qwen2.5-0.5B assigned less probability to the frozen
81-string, one-decimal grid than the larger Qwens, especially arousal-first.

## Question

Does Qwen2.5-0.5B's weak arousal-first expected-score shift survive when score
probabilities include the closing JSON brace and common alternative number spellings
accepted by the strict JSON parser? This distinguishes a distribution over the
registered one-decimal strings from probability over complete parseable score
continuations. Alternative spellings such as `4`, `4.00` are parser-accepted but
violate the instruction to emit exactly one decimal place; report them separately.

## Frozen design

- Reuse Experiment 071's English laptop cohort and prompt construction. Select 12
  negative-valence and 12 positive-valence IDs from its 64 IDs by SHA-256 of
  `exp074-score-form-audit-v1|<case_id>`. The sorted 24 IDs joined by newline have
  SHA-256 `4f03b671acd8aedfce6cef7eefae4c738a80d29fbcd7f30f042d4243fd9e001d`.
  These reviews and prompts have already been observed in prior runs.
- Models: Qwen2.5-0.5B-Instruct `7ae557604adf67be50417f59c2c2f167def9a775`
  and Qwen2.5-1.5B-Instruct `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; local MPS.
- For each recipient, score four existing conditions (both field orders × forced
  first values 2.0/8.0) for three disjoint continuation families, including the
  final JSON brace: (i) all 81 canonical one-decimal scores 1.0–9.0; (ii) all 81
  equivalent two-decimal forms `x.y0`; and (iii) the nine integer forms 1–9.
  Aggregate probability by numeric value across forms. This is 192 contexts and
  32,832 candidate continuations total. Record greedy continuations for the same
  contexts and keep all raw data in ignored `.context/`.
- Independently verify joint sequence log probabilities by full forward pass for
  32 deterministic sampled continuations per model, covering each format family.
  Require tokenization at the open-JSON-prefix boundary to match the frozen
  candidate token sequence; otherwise abort before inference.

## Outcomes

Report each format family's unnormalized probability mass, combined mass, and
conditional expected-score 8−2 shift by order/model. Also report the shift after
combining the three parser-accepted numeric spellings (do not call this instruction
compliant), greedy shift, invalid output count, and sequence-likelihood check error.
Bootstrap 24 recipients with replacement (10,000 draws, seed `20260974`). This
small subset is a mechanism audit; intervals are descriptive. No significance gate.

## Limits

The same public laptop reviews and artificial prefixes are reused. The expanded
candidate family still does not enumerate every possible parseable JSON number or
whitespace path, so combined mass is a lower bound on parser-accepted completions.
Only two Qwen sizes are compared. This tests whether support/serialization can
explain the observed small-model order asymmetry; it cannot establish model-size
causality or human-score calibration.

- Triggering result: [Experiment 073](../../results/qwen05-prefix-size-continuation-v1/README.md)
- Common parent prompt/sample: [Experiment 071](071-cross-family-prefix-coupling.md)
