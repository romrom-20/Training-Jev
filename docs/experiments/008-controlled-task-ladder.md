# Experiment 008: controlled task ladder and gated readout replication

**Prospective local protocol, 23 September 2026.** Experiment 007 stopped before
probe fitting because the target handled food and value above 90% but service at
81.8%, despite passing simple whole-review sentiment. This experiment separates
the possible sources of difficulty in a balanced factorial task ladder, and then
replicates readouts only if target behavior passes on a separate confirmation split.

## Prior work and scope

Aspect-based sentiment analysis already has extensive benchmark and model work,
including broad LLM evaluations and 2026 reasoning-infused ABSA models
([Zhou et al., 2024](https://arxiv.org/abs/2412.02279);
[Liskowski & Jankowski, 2026](https://arxiv.org/abs/2601.03940)). Compositional
generalization across attribute combinations is itself an established evaluation
problem ([Zhong et al., ACL 2024](https://aclanthology.org/2024.acl-long.351/)).
This experiment is not a new ABSA task or a novelty claim. Its practical question
is which simple, controlled task structure lets small local targets reliably bind
a question to one of several attributes, and whether a small activation readout
tracks that behavior across held-out cue wording and prompt form.

## Models, data and controlled task factor

Run three cached checkpoints sequentially and offline: Qwen2.5-1.5B-Instruct,
Qwen2.5-3B-Instruct (same-family scale comparison), and SmolLM2-1.7B-Instruct
(second family). Use float32 on the laptop's MPS device. Do not load Qwen2.5-7B.
The data are deterministic synthetic reviews with three queried attributes:
food, service and value. Positive and negative cue wording has eight fixed
variants per attribute; variants 0–3 appear before the final held-out groups and
variants 4–7 are reserved for final evaluation. Every content group crosses
positive/negative labels and clause order is counterbalanced.

The primary controlled factor is how the target property is represented:

1. `isolated_clause`: one opinion clause about the queried property.
2. `neutral_distractors`: the same clause plus two unrelated neutral facts.
3. `mixed_review`: three independent, potentially conflicting opinion clauses.
4. `keyed_record`: explicit `food=...; service=...; value=...` fields.

All arms ask for the same one-word `positive` or `negative` answer. Two fixed
question formats are crossed with every arm. They change instruction wording but
preserve the task. There are 64 content groups split by ID before collection:
12 task-selection, 16 probe-training, 6 validation, 6 calibration and 24 final
test groups. In each group, isolated and neutral arms have 2 labels × 3 queried
attributes; mixed and keyed arms have all eight label combinations × 3 attributes.
Final test uses cue variants unseen in selector, training, validation and
calibration groups. Its first eight groups are unlabeled format-adaptation anchors;
last sixteen are evaluation. The full matrix is 7,680 prompts per model.

## Stage A: choose a task without test leakage

Measure greedy first-token accuracy, conditional positive/negative accuracy and
positive/negative probability mass for each model, task arm, queried attribute,
and question format. The capability gate is at least 90% greedy accuracy in every
attribute × format cell on the 12 task-selection groups. A task arm qualifies for
probe training only if it passes in **both** Qwen2.5-1.5B and SmolLM2-1.7B. Qwen2.5-3B
is a scale check and cannot substitute for the second family.

If multiple arms qualify, select by this fixed preference order: mixed review,
neutral distractors, isolated clause, keyed record. This chooses the hardest
natural-language task that the independent-family gate supports. If no arm passes,
stop without probe training. Report every arm and failure; do not rewrite prompts.

For the mixed-review and keyed-record arms, also compute paired counterfactual
specificity: flip the queried field while holding group, other fields and format
fixed; then flip each unqueried field while holding the requested answer fixed.
Report the rate the target changes to the new correct label on queried flips and
the rate its answer stays fixed on irrelevant flips. In mixed reviews, additionally
compare requested-aspect accuracy with overall-majority-sentiment accuracy when the
two disagree. These are prespecified target-behavior endpoints; no causal claim is
made about the hidden representation.

## Stage B: confirm behavior and fit probes

For a selected arm, first require at least 90% greedy accuracy in every attribute
× format cell on the separate final-test groups for each model. If a model fails,
do not interpret its probe scores causally or as readouts of successful behavior.
For models that pass, fit on format-0 training groups only. Use final-token
activations at blocks 8/16/24 for Qwen2.5-1.5B and SmolLM2, and blocks 12/24/36
for Qwen2.5-3B; train a rank-four shared bilinear probe and independent linear
probes with seeds 0/1/2, validation checkpoint selection and per-attribute
temperature fitted on calibration groups. The primary layer is the middle layer
for each model. Include a word/character TF-IDF text baseline on the exact same
training prompts and the target's own conditional next-token probabilities.

Compare held-out Brier/NLL, accuracy and AUROC by attribute and question format.
For format 1, compare the frozen source readout with an unlabeled per-attribute
offset from the first eight final-test groups against evaluation on the remaining
sixteen disjoint groups. Report scenario-group bootstrap intervals. Text-only
performance does not count as evidence for activation-based interpretation.

This staged design is deliberately allowed to return a target-only negative result.
Any positive probe result would still require another task family and independent
replication before a publication or LessWrong decision. A/B remapping, intervention
and steering claims are out of scope.
