# Experiment 077 — explicit target naming in the role-location control

**Status:** register before inference after Laya local typed-choice triage. Laya
selected the prompt-clarity follow-up (B, probability 0.306, uncalibrated):
Experiment 076's generic “other coordinate” wording produced many wrong-key
outputs for Qwen-0.5B in one user-anchor condition. The choice is agenda
guidance only.

## Question

Does explicitly naming the target coordinate restore valid output in the
Qwen-0.5B prior-user/arousal-first arm, and what same-schema role contrast
remains when both role conditions receive the exact same clear final request?

## Frozen design

- Reuse the 24 polarity-balanced DimABSA laptop cases frozen for Experiment
  076A. Sorted IDs joined by newline have SHA-256
  `92f9debeb17ba951aea6840b34d8f24992f3558358f98fb0083bc91ea487e983`.
- Use Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775` on local MPS.
- Cross two target axes, two fixed values (2.0, 8.0), and two prior-message
  locations. Keep the user/assistant message patterns from Experiment 076A.
- The final user request explicitly names the fixed axis and the target axis,
  instructs the model to preserve the earlier fixed score, and requires the
  one-key JSON output key to equal the named target axis. It omits the numeric
  anchor value so its content is identical between role locations for each
  axis/anchor condition.
- Score all 81 canonical one-decimal target values and record one greedy
  continuation. No retries, constrained decoding, or output repair.
- Total: 24 × 2 axes × 2 values × 2 locations = 192 model-context cells.

## Outcomes and gate

Report invalid-output rates and exact raw failure categories by arm. The
primary validity requirement is at most 2% invalid outputs in every
model × location × target-axis × fixed-value arm. If it passes, report the
paired 8−2 conditional expected-score shift by target axis and role location,
then assistant-prior minus user-prior with recipient-bootstrap 95% intervals
(10,000 draws, seed `20260977`). If it fails, withhold that arm's paired role
contrast. Greedy shifts and valid canonical probability mass are secondary
diagnostics.

## Limits

The assistant-prior condition contains a previous assistant score; the
user-prior condition contains a user-provided score followed by a neutral
assistant acknowledgement. Both end with the same final request and output
schema, but the message-role manipulation necessarily changes this prior-turn
context. The items are reused public reviews and the test has one model and
24 cases. This diagnoses one prompt failure; it cannot establish a general
role effect or explain the new-corpus result.

This follows the Experiment 076 result bundle and sits within existing
literature on anchoring, output order, and structured formats. No publication
decision is part of the protocol.

## Reproduction

Runner, analyzer and tests are added after this protocol is frozen. Per-context
prompts and model outputs remain under ignored `.context/`.
