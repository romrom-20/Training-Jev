# Experiment 075 — does the rating anchor depend on where it appears?

**Status:** preregistered before inference, after Laya local agenda triage 075
(choice C, probability 0.342, uncalibrated). Tests a concrete mechanism suggested
by Experiments 073–074: the same fixed VA coordinate may affect the other coordinate
differently when it appears in an open assistant JSON prefix versus as explicit
user-provided information.

## Question and hypothesis

For Qwen2.5-0.5B and 1.5B, does the 8−2 forced-coordinate shift in the conditional
distribution over the other score differ between (a) the existing partial assistant
continuation and (b) a user message that states the same coordinate as fixed and
requests only the other score? The primary estimand is user-anchor minus
assistant-prefix expected-score shift, separately by model and target coordinate.
No direction is predicted. A large difference would show the prior result depends
on the location/format of the anchor; similar shifts would be consistent with
semantic conditioning across these two prompt formats.

## Frozen design

- Reuse the 64 polarity-balanced English laptop IDs from Experiment 071, disjoint
  from Experiment 055. The sorted ID hash remains
  `a14b6300c88940b3ccb6ac6b1b78c78b1cb5e202c4ce945ea3fe8220d36c7ddf`.
- Models: Qwen2.5-0.5B-Instruct
  `7ae557604adf67be50417f59c2c2f167def9a775` and Qwen2.5-1.5B-Instruct
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; local MPS.
- Cross two anchor locations (partial assistant prefix; explicit user-provided
  fixed coordinate), both target axes, and anchor values 2.0/8.0 for every case.
  This yields 512 contexts per model, 1,024 total.
- Assistant-prefix arm uses the exact Experiment 071 system/user bodies and open
  JSON prefix, with the first coordinate embedded in that assistant prefix.
- User-anchor arm retains the review, aspect and scoring definitions, states that
  one named coordinate is fixed at 2.0 or 8.0, and requests a one-field JSON object
  for the other coordinate. Its assistant prefix opens only that target field.
  Thus the semantic information is explicit in user text while the current answer
  begins at the target field. The user-anchor arm is a new task wording and output
  schema; this is a mechanism contrast between these registered formats, not a
  perfectly token-matched causal intervention.
- Score all 81 canonical 1.0–9.0 one-decimal numeric completions including the
  closing brace, normalize over this score support, and record one greedy completion
  per cell. Also retain the parser-accepted integer/trailing-zero likelihoods as a
  secondary support audit, without including those forms in the primary outcome.
  Keep raw prompts and outputs in ignored `.context/`; no retries or repair.

## Outcomes and decision rule

For each model and target coordinate, report the 8−2 expected-score shift and greedy
shift by anchor location, their paired difference (user minus prefix) and a 95%
recipient-bootstrap interval (10,000 draws, seed `20260975`). Also report valid-score
completion mass and invalid-output rate. If any model × location × target × value
invalid rate exceeds 2%, withhold the affected contrast. Describe an interval that
excludes zero without treating it as evidence of a general mechanism; the formats
and schemas differ as described above.

## Literature boundary and limits

LLM-as-scorer research has studied field order in score/reason outputs, and work on
numeric anchoring and expected-versus-greedy scoring already exists. This test does
not claim either broad phenomenon as new. It asks whether the specific size/order
interaction from this small-model series survives moving the same numeric cue into
user-provided information. The recipient set was already observed, only two Qwen
sizes are used, and the prompt/output formats differ. See the field-order scoring
study by [Chen et al. (AIA 2024)](https://aair-lab.github.io/aia2024/papers/chen_aia24.pdf),
[Experiment 073](073-qwen05-prefix-size-continuation.md), and
[Experiment 074](074-qwen05-numeric-support-audit.md).
