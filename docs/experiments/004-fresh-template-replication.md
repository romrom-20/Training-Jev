# Experiment 004: fresh-template replication of score transport

**23 September 2026, prospective follow-up.** Experiment 003 found that unlabeled
query-specific centering substantially improved shifted-template accuracy for the
balanced shared readout at the predefined middle layer. This experiment tests whether
that finding survives fresh formats. No parameters are selected from its outcomes.

Freeze the existing seed-0, middle-layer readouts for both models, both supervision
regimes and both probe architectures. Use the already defined first-six source test
groups as unlabeled source anchors. Generate four new target formats: labeled prose,
JSON settings, a sentence-based description, and explicit multiple-choice options.
All encode the same three fields, requested property and A/B response mapping.
Use new record IDs with seed 20260923. For each format, create six adaptation groups
and six disjoint evaluation groups, fully crossed over settings × task × mapping.
That is 2,304 new target prompts per model. Each format is analyzed separately.

Estimate a shared offset and three property-specific offsets from unlabeled adaptation
scores. Compare against the frozen, temperature-scaled source probe. Fit neither
weights nor temperature on the new formats. Report per-format accuracy, Brier, NLL
and within-question/code rank AUROC, plus paired evaluation-group intervals. Also
repeat the 10/50/90% artificial adaptation-prevalence stress from Experiment 003.

The primary result is the balanced shared head, middle layer, seed 0. Other banks are
controls, not alternative winners to choose afterward. A stronger continuation signal
would require lower Brier on at least three of four fresh formats **for both models**,
without a deterioration exceeding 0.05 absolute Brier on the remaining format.
This is a prospective continuation criterion, not a claim of statistical significance.

Record target A/B behavior but do not make semantic causal claims: these are readout
experiments on explicit labels, separate from the failed competence gate in Experiment
002. The anchor distributions are balanced by design; improvements do not establish
robustness to unknown label priors. Both targets belong to one model family, and only
one seed is used here. If the result fails this check, do not promote the original
single-template improvement into a general method claim or a LessWrong research post.
