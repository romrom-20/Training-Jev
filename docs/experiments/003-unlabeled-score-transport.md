# Experiment 003: can unlabeled score centering recover template transfer?

**Prospective follow-up analysis, 23 September 2026.** This follows the observed
aggregate template failures in Experiment 002. It is not an independent replication
or a claim of unsupervised calibration under arbitrary shift. Freeze this description
before running the new analysis. No new model inference is needed.

For each existing model, layer, training regime and seed, use only the **first six**
ordinary-test groups and first six shifted-template groups to estimate adaptation
statistics. No adaptation labels are supplied to the estimation function. Evaluate
only the **last six** shifted-template groups, never the adaptation records.

Compare three frozen corrections to the existing logits:

1. No correction, using the original calibration temperature.
2. One shared offset: subtract shifted-minus-source mean active-query logit.
3. Three query-specific offsets: subtract shifted-minus-source mean logit separately
   for each requested property, pooling both answer mappings.

Apply the original source calibration temperature after subtraction. Do not refit
slopes, temperatures, layers, seeds or any hyperparameter on shifted labels. Report
accuracy, Brier, NLL and within-query AUROC separately for both code mappings, all
sampled layers, both probes and both supervision regimes. The primary descriptive
comparison is query-specific centering versus no correction at the preselected middle
layer. Use paired scenario-bootstrap intervals, with six evaluation clusters; these
are conditional, descriptive intervals rather than broad generalization evidence.

The data are fully balanced by construction. Mean alignment assumes that a difference
in average scores reflects nuisance shift rather than changed semantic prevalence.
Stress this assumption by changing only the adaptation-set label mixture to 10%, 50%
and 90% positive **through benchmark weights**; leave the balanced evaluation set fixed.
Ground-truth labels are used to construct these three simulated populations, not to
fit a label-aware correction. This is an oracle-designed label-shift stress test, not
a deployable way of choosing weights. State that distinction in every interpretation.

Audit expectation: subtracting a constant per query leaves AUROC within a fixed
query/code slice unchanged. Improved threshold accuracy and Brier with unchanged
AUROC would support an offset explanation on this controlled dataset, not recovery of
newly decodable information. Poor AUROC after shift would rule out an offset-only account.

Success here would motivate a fresh dataset with independently varied prevalence,
multiple new templates, unlabeled calibration examples collected before evaluation,
and a comparison against established contextual/domain calibration methods. It would
not justify calling centering a new method or writing a research post before those
limits and the literature overlap are assessed.
