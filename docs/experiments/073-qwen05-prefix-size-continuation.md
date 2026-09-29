# Experiment 073 — does Qwen prefix coupling continue at 0.5B?

**Status:** preregistered before inference, following Laya local agenda triage 073
(choice B, probability 0.354, uncalibrated). This is a single additional model
point on the exact Experiment 071 laptop cohort and prompt set.

## Question

Does the expected second-score response to a forced first coordinate continue to
increase as Qwen2.5 size decreases from 3B to 1.5B to 0.5B? This targets the
within-family size pattern from Experiments 070–072. It is exploratory: only three
model sizes, one model family, one public benchmark cohort, and artificial assistant
prefixes cannot establish a scaling law.

## Frozen design

- Reuse Experiment 071's 64 polarity-balanced English laptop recipients, same
  one-target selection, prompts, system message, field orders, forced values 2.0
  and 8.0, scoring grid, and greedy parse. The selected ID hash is
  `a14b6300c88940b3ccb6ac6b1b78c78b1cb5e202c4ce945ea3fe8220d36c7ddf`.
  This is an additional model test on previously observed items, not an independent
  recipient replication.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, pinned cached revision
  `7ae557604adf67be50417f59c2c2f167def9a775`. Run local MPS only.
- Run 256 contexts (64 recipients × two field orders × two forced values), score
  the same 81 canonical 1.0–9.0 one-decimal second-coordinate strings, and record
  one greedy continuation per cell. No retries or output repair.
- The paired references are the identical recipient-level cells for Qwen2.5-1.5B
  and Qwen2.5-3B in Experiment 071. Keep their already frozen versions and outputs.

## Outcomes

For each field order, report the paired 8−2 change in normalized expected second
score and greedy score for 0.5B; compare each recipient's 0.5B expected-score
shift with its 1.5B and 3B shifts. Bootstrap recipients with replacement, retaining
all cells; 10,000 draws, seed `20260973`. Report invalid rates and valid-score mass.
If any 0.5B model × order × forced-value invalid rate exceeds 2%, withhold score
contrasts. The directional continuation criterion is a positive 0.5B-minus-1.5B
expected-shift interval in both orders; failing it means the observed size pattern
does not continue monotonically at this third point, not that model size has no effect.

## Limits and provenance

All three sizes share the Qwen2.5 family but still differ in capacity and possibly
training details. This design adds no fresh prompts, source split, or model family.
Public benchmark pretraining exposure cannot be excluded. Numeric anchoring and
output-order effects are established; the experiment targets only this narrow model
size comparison. Local resources: one 24-GB MacBook Air.

- Parent model-family/dataset run: [Experiment 071](071-cross-family-prefix-coupling.md)
- Parent restaurant-domain check: [Experiment 072](072-restaurant-prefix-domain-replication.md)
