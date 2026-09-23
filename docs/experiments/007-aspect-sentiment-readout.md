# Experiment 007: aspect sentiment with a target that passes task checks

**Prospective local protocol, 23 September 2026.** The prior synthetic rule task
failed the target competence gate. The standard sentiment diagnostic then showed
that the cached Qwen2.5-1.5B-Instruct follows ordinary sentiment prompts, while
the 0.5B checkpoint does not do so reliably across wording. This study uses only
the 1.5B checkpoint and asks whether a frozen, query-conditioned activation readout
keeps predicting target behavior when review and question wording changes.

## Task and data

Construct short reviews with three independent aspects: food, service and value.
Each aspect is clearly positive or negative. Cross all eight combinations within
each scenario group and ask about each aspect, yielding 24 prompts per group.
Use disjoint scenario group IDs in 32 train, 8 validation, 8 calibration, 24 source
test and 24 shifted test groups. Each shifted test group has four fixed prompt
formats. Use the 1.5B revision already pinned in `configs/relevance.toml`; do not
run the 0.5B model in this study.

The four target formats preserve the aspect labels but change review clause order,
paraphrase the question and vary surface wording. They are fixed in the script
before data generation. Include all labels; do not remove failures. The first eight
shifted test groups are unlabeled adaptation anchors; the remaining sixteen are
held-out evaluation. Groups, not prompts, are the unit for uncertainty intervals.

## Capability gate

Before fitting any probe, measure target greedy generation and conditional
positive/negative choice on the source test prompts. The study's behavior gate is
at least 90% exact generated-label accuracy for each of the three aspects. If the
gate fails, release the target measurements and stop semantic readout interpretation.

## Readouts and comparisons

Capture final-token activation at blocks 8, 16 and 24. Train one rank-four bilinear
shared probe and one independent linear bank on the 32 training groups; select
checkpoints on validation and fit temperature only on calibration prompts. Use
seeds 0, 1 and 2. The primary comparison is middle block 16, query-conditioned
bilinear readout versus independent linear probes, and each frozen source-format
readout with versus without an unlabeled, per-aspect anchor offset. This offset is
the difference between target-anchor and source-anchor mean logits; do not use
anchor labels. Also report a word/character TF-IDF logistic-regression baseline
trained on the same source prompts, plus the target's own next-token probabilities.

Primary outcome is paired Brier difference on the held-out 16 shifted groups.
Report NLL, accuracy, AUROC and calibration by format and aspect, with group
bootstrap intervals. The text baseline is a confound check; its performance does
not count as evidence for activation-based interpretability. No intervention or
causal claim is in scope.

Treat this as a successful continuation only if the behavior gate passes and the
shared probe has lower held-out Brier than its uncorrected version on at least
three of the three shifted formats, pooled across aspects, without a deterioration
over 0.05 on any aspect-format cell. A result meeting this rule is still a single
family, synthetic-data result. Novelty and any LessWrong post decision require a
separate review after the complete run.
