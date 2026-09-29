# Experiment 078 — replicate the named-target role control at 1.5B

**Status:** registered before inference after Laya local typed-choice triage.
Laya selected the larger-model check (B, probability 0.329, uncalibrated):
repeat Experiment 077's clear one-key role-control prompt with Qwen2.5-1.5B.
The choice is agenda guidance only.

## Question

With the target coordinate named explicitly, does the assistant-prior versus
user-prior contrast observed on Qwen2.5-0.5B also appear on Qwen2.5-1.5B, and
does the format-validity improvement persist?

## Frozen design

- Reuse the same 24 polarity-balanced DimABSA laptop reviews from Experiments
  076A and 077. The sorted recipient IDs have SHA-256
  `92f9debeb17ba951aea6840b34d8f24992f3558358f98fb0083bc91ea487e983`.
- Use Qwen2.5-1.5B-Instruct revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` on local MPS.
- Repeat Experiment 077 exactly: two target axes, fixed values 2.0/8.0,
  prior assistant versus prior user location, identical explicit final
  request and one-key JSON output schema.
- Score the 81 canonical one-decimal target values and record one greedy
  continuation. No retries, constrained decoding, or output repair.
- Total: 24 × 2 axes × 2 values × 2 locations = 192 model-context cells.

## Outcomes and gate

Report per-arm invalid-output rates. The primary validity requirement is at
most 2% invalid outputs in every target-axis × location × fixed-value arm.
If it passes, report the paired 8−2 conditional expected-score shift by target
axis and prior location, followed by assistant-prior minus user-prior with
recipient-bootstrap 95% intervals (10,000 draws, seed `20260978`). If the gate
fails, withhold the affected role contrast. Greedy shifts and valid canonical
probability mass are secondary. Compare with Experiment 077 descriptively;
that same-item comparison is not independent evidence.

## Limits

This is one model on 24 reused public reviews. The prior-user condition includes
a neutral assistant acknowledgement, while the prior-assistant condition
contains the score as the preceding assistant response. The manipulation still
bundles anchor source with conversation context. Agreement with Experiment 077
would be a second model-size point, not a family-wide claim or independent
recipient replication.

## Reproduction

Runner, analyzer, and tests are added after this protocol is frozen. Per-context
prompts and raw outputs remain under ignored `.context/`.
