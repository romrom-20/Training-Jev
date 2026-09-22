# Experiment 002: does a steering probe survive answer remapping?

**Prospective local protocol, 22 September 2026.** Commit this protocol and its
configuration before collecting outcomes. This is not a third-party preregistration.
No LessWrong draft is planned unless the completed results justify one.

## Motivation and prior art

The first pilot's strongest effect was a color readout shifting color-token log odds.
One explanation is semantic control; another is that the readout mostly found an
answer-writing direction. The old task cannot distinguish these explanations because
color value and output token are aligned.

[Canby et al., How Reliable are Causal Probing Interventions?](https://arxiv.org/html/2408.15510v3)
evaluates completeness and selectivity and shows intervention reliability depends on
layer and method. [Makelov et al., An Interpretability Illusion for Subspace Activation
Patching](https://arxiv.org/abs/2311.17030) shows that behavioral control need not identify
a faithful feature representation. Those are direct precedents for caution; we do
not claim to discover the general probing/causality gap.

The bounded empirical question here is whether an ordinary small-probe pipeline can
pass prediction, calibration and steering checks yet fail a simple answer-remapping
control, and whether counterbalanced supervision changes that failure. It is a new
experiment in this repository, not an established novel method.

## Task and factorial design

Every record has three independent binary settings: color (blue/red), shape
(square/circle), animal (dog/cat). A task selects one field. The model returns A when
that field has its positive value and B otherwise, or the reverse code mapping.
All 8 setting combinations × 3 active fields × 2 code mappings are crossed within
each record group. This produces 48 prompts per group. Prompts differ only in the
specified factors within a group. The probe's three semantic questions stay fixed.

Use 16 training groups, 4 validation, 4 calibration, 12 test and 12 out-of-template
groups: 2,304 target prompts per model. Two templates appear in the first four splits;
a third appears only in the out-of-template split. Splits have different record IDs.
The dataset is shared by both models; groups are the unit of uncertainty estimation.

## Two matched-budget training regimes

Train an independent linear bank and a rank-four question-conditioned head. In both
regimes, the probe is supervised on the **currently requested field only**: this
matches training a probe on a task where the queried property controls the answer.

- `fixed_code`: all training groups, positive value always maps to A.
- `balanced_code`: first half of training groups, both mappings.

Each sees 384 activation/question/label examples (128 per property). Apply the same
matching to validation and calibration groups. This does not match scenario diversity:
fixed-code sees twice as many distinct record IDs. Explicitly report that limitation.
An unused-property test is a relevance shift from this training distribution, not an
in-distribution semantic readout evaluation.

Use the original BCE+Brier objective and optimizer. Fit activation scaling using only
the regime's training activations, choose checkpoints using its validation split,
and fit one temperature using its calibration split. Three fixed seeds; rank and
hyperparameters are not chosen on these test results. Evaluate every sampled layer.
The probe never receives the code mapping or active-field ID except through the target
activation. Query embeddings contain only the semantic question.

Primary descriptive scores: Brier, NLL, accuracy and calibration on relevant fields
under original versus reversed mapping, separately in known and held-out templates.
Also report unused-field readout. Save per-prompt probabilities. A single pooled
score must not conceal the mapping-specific results.

## Capability gate

Measure target greedy A/B accuracy and A+B probability mass for every active-field ×
mapping cell. If any ordinary-test cell has <90% greedy accuracy, label that model's
semantic causal conclusions as failing the capability gate. Still release all results.
Do not repair the prompt after looking at outcomes. A failing task is itself a result.
The 1.5B model is a same-family scale check, not an independent-family replication.

## Interventions

Preselect the middle sampled layer (12 for 0.5B, 16 for 1.5B), seed 0 and the first
eight ordinary-test groups: 384 prompts. No selection based on steering outcomes.
For each learned bank, compute the three raw-activation score directions, including
the standardization chain rule. Use each on every active-field/mapping condition.
Add/subtract 5% of the median training activation norm, with a shared norm scale
across regimes. This is one fixed dose, not a complete dose-response experiment.

Comparators: four seeded, equal-norm random directions; a separately trained linear
A/B-output probe direction; a zero-vector sham; and full last-token activation patches
from a matched donor differing in exactly one underlying setting. Donor patches keep
record, active field, mapping and the other two settings fixed. Earlier-token states
remain from the recipient, so even a real donor vector creates a hybrid computation.

Let L = log P(A)/P(B), and c=+1 when positive maps to A, c=-1 otherwise.
For a direction intended to increase property j, measure

    E = c × [L(h + δ_j) − L(h − δ_j)].

A semantic increase should yield E>0 when j is the requested field under **both**
code mappings. An A-token direction can pass the original-code test and reverse sign
under the other mapping. On nonrequested fields report |E| as collateral influence.
Save both signed effects and all mapping-specific results; averaging them can cancel
an answer-writing effect and hide a failure. Compare paired scenario means; bootstrap
12 test groups for prediction and eight for intervention estimates. Interventions
use one training seed, so intervals do not include training-seed variability.

For donor patches, orient the log-odds change by the flipped property's direction
and c. Compare to the target's unmodified matched donor as a behavioral reference.
Do not divide by near-zero effects or present a restoration ratio on irrelevant fields.
Record conditional A probability, A+B mass and top output token; absence of complete
vocabulary KL is a limitation of this study's compact intervention records.

## Decision rule and scope

A compelling local finding would be a replicated, mapping-dependent failure of a
probe that otherwise passes ordinary prediction/calibration/steering checks, or a
consistent separation between learned steering and matched donor patching. A remedial
claim additionally requires counterbalanced supervision to improve the reversed-code
result at matched label budget. Report failures and contradictory layers/models.

This is a constructed task with explicit prompt labels, one model family and a modest
number of templates. It cannot establish general safety-monitor reliability, detect
intent, identify a circuit, or prove novelty. A known caution illustrated once is not
by itself enough to justify a substantive LessWrong research post. Decide whether to
write only after the full experiment and correctness checks.
