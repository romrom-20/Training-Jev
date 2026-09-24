# Research plan: earn the generalization claim

## Working hypothesis

A small shared activation-to-query subspace may reduce supervision cost when there
are many related semantic properties. It may also make failures easier to measure
than a generative decoder. **Neither advantage follows from the architecture.**
Independent linear probes are the main baseline, and a cheap text classifier is a
necessary confound check whenever labels are recoverable from input text.

The most interesting next step is a **readout/influence dissociation benchmark**:
find properties that are similarly decodable but differ in their effect on target
behavior. Track whether query-conditioned probabilities remain trustworthy across
that distinction and across distribution shifts. This narrows the project from a
universal activation oracle to a falsifiable small-compute investigation.

## Stage 0 — executable pilot (implemented)

- Frozen Qwen2.5-0.5B; balanced instruction settings; three extraction layers.
- Query-conditioned head, independent probes, text and corruption controls.
- Disjoint calibration split, raw/scaled metrics, per-example outputs.
- Fixed-dose interventions and equal-norm random directions on the actual target.
- Offline report and measured local resource use.

Gate: capture and intervention hooks pass tests, labels are balanced, leakage controls
are plausible, and the target follows the intended task. This is a pipeline test,
not an acceptance threshold for the scientific hypothesis.

## Stage 1 — transfer and sample efficiency (weeks 1–3)

Build at least 12 operational properties in four families, such as output format,
lexical selection, role instructions and answerability. Give every property a
programmatic or independently adjudicated labeling rule. Separate **instruction
labels** from **observed behavior labels**; report target noncompliance instead of
silently treating instructions as behavior.

Predeclare a split at the property-family level. Include unseen scenario content,
new templates, paraphrases and changed label prevalence. Train only on training-family
labels; freeze hyperparameters on development families. Repeat leave-one-family-out
evaluation. A held-out paraphrase of a known task is not a substitute.

Compare rank 1/2/4/8/16 heads, separate linear probes, parameter-matched MLP probes,
query-only, prompt-text and no-query baselines. Sweep 16/32/64/128 examples per property.
Where feasible, benchmark frozen oracle checkpoints through their supported interfaces;
otherwise explicitly defer the oracle comparison. Measure query encoding and activation
extraction as well as head latency; do not quote head latency as end-to-end latency.

Primary outcomes: paired held-out Brier and NLL versus independent probes, with
scenario-cluster intervals. Secondary: per-property AUROC, calibration diagrams and
risk/coverage curves using an abstention threshold set on calibration data. Under
shift, report actual errors at that threshold; do not promise a distribution-free
guarantee. Evaluate post-hoc calibration against BCE-only and BCE+Brier training.

**Proposed continuation gate, to freeze before collecting the next benchmark:**
at least a 10% relative Brier improvement at a matched label/parameter budget on
three of four held-out-family evaluations, with a pooled paired 95% interval excluding
zero; no material loss on the fourth. These are proposed decision criteria, not
results. If the baseline Brier is near zero, use NLL and absolute Brier differences
instead and predeclare that case.

## Stage 2 — specificity and intervention forecasts (weeks 4–5)

Use counterfactual prompt pairs that vary one factor, natural activation patching,
matched random directions, unrelated-property directions, sign reversals and a fixed
dose sweep. Include ablations if making necessity claims. Measure intended behavior,
output quality and unrelated behaviors separately.

An ambitious extension is to predict a **behavioral change under a specified
intervention**, rather than treating a calibrated observational classifier as a
causal oracle. Specify the intervention distribution and target outcome first.
Train a separate effect readout only on training interventions; test on held-out
prompts and intervention directions. Binary event probabilities and continuous
log-odds changes require different scoring objectives.

Gate: a reproducible target-specific effect beyond random directions that survives
natural patching and acceptable collateral behavior checks. Failure means retracting
the causal interpretation, even if observational prediction remains good.

## Stage 3 — replication and release (weeks 6–8)

Replicate the best frozen protocol on a second small model/family and independently
generated scenarios. Release failures, complete configs, per-example predictions,
model revisions, runtime measurements and an experiment register. Have an external
researcher reproduce one run. Use AObench-compatible tasks where possible.

## Stop or pivot conditions

If query sharing only matches independent probes while using more parameters, pivot
to a careful negative result or an evaluation toolkit. If text baselines explain the
whole task, move to behavior labels or model-difference settings before interpretability
claims. If calibration fails under mild shifts, study failure detection before safety
applications. If 0.5B cannot perform a task, move to a capable small target before
interpreting probe failure. Larger compute should answer a concrete unresolved
question, not substitute for a better experimental design.

## Experiment register

The included pilot is exploratory and was run while this repository was built.
Its protocol and results are not preregistered. Freeze the next experiment's hypothesis,
splits, endpoints, seeds, comparisons and stop rules in a dated commit **before**
observing its test outcomes. Keep any follow-up prompted by those outcomes separate.

## Execution update — 23 September 2026

The answer-remapping and four-format score-transport follow-ups are documented in
[`results/followup/README.md`](../results/followup/README.md). The answer-remapping
task failed the model behavior gate on both sizes. The fresh-format check failed its
continuation rule on the 1.5B model; no broad calibration claim follows.

Two fixed capability diagnostics then tested ordinary sentiment classification.
The 1.5B checkpoint generated the expected one-word label on all 64 prompts across
four phrasings; the 0.5B checkpoint reached 87.5% overall and failed one phrasing.
This was only a task-selection check. The follow-on three-aspect study crossed food,
service and value labels over 4,032 prompts. It failed its prespecified capability
gate on service (81.8%), so **no activation probes were trained**. Prompt wording
also reduced target accuracy sharply on the third format. The target-only run and
portable audit are in [`results/aspect-sentiment-v2/README.md`](../results/aspect-sentiment-v2/README.md).

The next useful design question is why simple sentiment classification passed while
aspect-specific judgments did not. A new protocol should directly establish target
competence for every property and format before probe training; it should not treat
an incomplete capability task as a probe failure.

Experiment 008 then ran a 7,680-prompt target-only controlled task ladder on
Qwen2.5-1.5B and SmolLM2-1.7B. Neither family cleared the strict-generation gate
on any task across both prompt formats, so the probe stage correctly did not run.
A useful follow-up signal is that under the less constrained wording, models often
ranked the positive/negative token pair in line with labels while failing to generate
either label. Experiment 009, frozen in a separate commit before collection, tests
whether adding an explicit one-word constraint rescues exact responses on simple
and compositional tasks. See the [008 results bundle](../results/task-ladder-v1/README.md).
Experiment 009 completed this follow-up; its full outcome and publication decision
are recorded below and in the result bundle.

Experiment 009 completed the paired output-constraint control. An explicit one-word
constraint moved SmolLM2's final-test label-token compliance from 0% to 100%, but its
strict accuracy did not reach 90% on any task; on the two simple tasks it averaged
83.3%. The predeclared composite rescue criterion failed. Format-following and task accuracy separation is
already established in prior benchmarks, so these narrow synthetic results do not
merit a LessWrong post. See the [009 results bundle](../results/response-policy-control-v1/README.md).

Experiment 010 is the next field-facing test: with an activation direction fixed on
training groups, can a small question-conditioned head predict which held-out prompts
will shift most under that intervention? It compares prompt-conditioned forecasts
against a pair-average baseline, a text baseline and an unsteered first-order gradient
reference. It extends the behavior-pair side-effect forecasting work in the
literature to within-pair, prompt-level effect heterogeneity; a null result is useful
because it bounds the case for deploying a tiny readout instead of measuring the
intervention directly. The prospective protocol is
[`docs/experiments/010-prompt-level-effect-forecast.md`](experiments/010-prompt-level-effect-forecast.md).

Experiment 010 completed on both local model families. The small readout missed its
10% RMSE-reduction criterion: it was 24.3% worse than the source/target mean on Qwen,
and its 5.7% improvement on SmolLM2 had a paired interval crossing zero. The
first-order gradient reference reached RMSE 0.1309 on Qwen and 0.0279 on SmolLM2,
substantially lower than the readout. Every observed final-test score shift was
positive, so sign accuracy was uninformative. See the
[`010 results`](../results/prompt-effect-forecast-v1/README.md).

Experiment 011 decomposed the three training-only aspect directions into their shared
projection and task residuals, then tested these components on the same held-out
mixed-review groups. Pairwise direction cosines were 0.85–0.96. The shared component
reproduced mean shifts of +2.458 log-odds (Qwen) and +1.469 (SmolLM2); residual mean
effects were near zero. The residual-specificity gate failed because the residual did
not beat the norm-matched random residual control on SmolLM2. This is a bounded result
about candidate-label scores on the synthetic task, not a semantic or behavioral
mechanism. See the [`011 results`](../results/shared-residual-steering-v1/README.md).

Experiment 012 passed its frozen shared-versus-random rule for all three non-mixed
task structures on both models. The shared component raised the positive-minus-negative
candidate margin by about +2.29 to +2.32 on Qwen and +1.38 to +1.46 on SmolLM2; native
directions had nearly identical mean effects. Source-to-target selectivity remained
small. The result is consistent across these synthetic task structures but does not
establish generative control. See the [`012 result bundle`](../results/cross-task-shared-shift-v1/README.md).

The strongest unresolved confound is answer encoding. Gao et al. (2026) directly show
that steering can favor extraction-time answer identifiers after semantic labels are
remapped. Experiment 013 therefore freezes each local intervention and swaps positive
and negative meanings assigned to A/B on the same isolated-clause prompts. Only if
unsteered accuracy passes the 90% gate in both encodings and the random-adjusted semantic
effect stays positive in both models and mappings will we treat this as evidence for
semantic score control. The preregistered protocol is
[`013-answer-encoding-control.md`](experiments/013-answer-encoding-control.md).

Experiment 013 passed the 90% unsteered competence gate for Qwen2.5-1.5B under both
answer mappings (100% each). With the intervention fixed, its A-minus-B score change
stayed positive while the semantic positive-minus-negative effect flipped from +0.206
to −0.290. The strongest interpretation is that the intervention followed A, not the
current positive-sentiment meaning. SmolLM2 showed a fixed preference for B, but its
reversed-map accuracy was 87.5%, below the frozen gate; treat that as descriptive. The
[013 results bundle](../results/answer-encoding-control-v1/README.md) includes the
random controls and all per-example scores.

## Natural-review selectivity: experiment 014

Experiment 014 takes the 008 directions, layer and dose unchanged onto the
SemEval-2014 Restaurants gold test set. It uses 233 target-aspect prompts from 112
sentences with multiple labeled categories. Both models pass the baseline task gate
(Qwen 94.8%, SmolLM2 92.3%) and receive large positive-minus-negative margin increases
(+2.28 and +1.38 logits). Yet the preregistered paired specificity estimate is −0.00114
([−0.00323, +0.00136]) for Qwen and +0.000698 ([+0.000515, +0.000872]) for SmolLM2.
The small positive SmolLM2 estimate is about 0.05% of its generic shift, and only eight
examples in the focal categories have opposing polarities. The positive-heavy sample
(178 positive, 55 negative labels) also means small accuracy gains can come from a
general positive bias. Full rows, manifests and checksums are in the
[`014 result bundle`](../results/natural-aspect-selectivity-v1/README.md).

The curiosity-driven question therefore got a useful but not exciting answer: the
synthetic directions transfer as a sentiment-score bias, while practically meaningful
aspect selectivity remains very small. Experiment 015 then moved the frozen directions
to MAMS-ACSA, designed to include reviews with different polarities across aspects.
Across 35 eligible sentences and 18 conflict sentences, both models showed a positive
native-minus-random aspect-specific score residual (+0.00903 logits for Qwen, +0.00113
for SmolLM2), but these were only 0.38% and 0.08% of the generic shifts. Both missed
the 5% practical gate. See the [015 bundle](../results/mams-aspect-selectivity-v1/README.md).

Experiment 016 tested whether dose could reveal a stronger aspect-specific component.
Across 1.25%–20% of the training activation norm, the specificity fraction decreased
with dose in both models; both sentence-bootstrap slope intervals were below zero. No
dose crossed the 5% threshold. At 20%, SmolLM2 accuracy dropped from 73.2% to 54.9%.
This exploratory follow-up reuses the same MAMS sentences and is not an independent
replication. See the [016 bundle](../results/mams-dose-response-v1/README.md).

The result is an empirical asymmetry worth probing: target-matched residuals are
detectable on a conflict-rich benchmark, but dose amplifies generic valence faster than
target specificity. Experiment 017 compared layers 8, 16 and 24 using cached training
activations, per-layer norm-matched doses and the same MAMS prompts and controls. No
layer passed the 5% practical gate. An explicitly post-hoc paired comparison found that
SmolLM2's layer-16 residual exceeded layer 24 by +0.00903 logits (95% sentence-bootstrap
CI [+0.00722, +0.01094]), including +0.01013 on the 18 conflict reviews. For the
specificity fraction, the paired layer-16 minus layer-24 difference was +1.30 percentage
points for SmolLM2 (95% CI [+1.07, +1.52]) and +1.33 points for Qwen ([-1.85, +5.31]).
Layer 8 had near-zero or negative generic sentiment shift, making the fraction undefined
or uninformative there. The possible localization is model-specific: only SmolLM2's
post-hoc interval excluded zero. Pairwise comparisons were not preregistered or
multiplicity-adjusted. See the [017 bundle](../results/mams-layer-selectivity-v1/README.md).

Experiment 018 preregistered the layer-16 versus layer-24 comparison on independent
TripR-2020Large reviews, using 187 sentences, 385 mapped aspect queries and 29
cross-aspect polarity conflicts. The layer-16 specificity fraction exceeded layer 24
in both models: +12.22 percentage points for Qwen (95% paired sentence-bootstrap CI
[+7.94, +19.37]) and +0.84 points for SmolLM2 ([+0.69, +0.98]). At layer 16, both
native directions also beat their random controls. However, the absolute practical
gate passed only for Qwen: specificity accounted for 12.36% of its generic shift,
versus 0.91% for SmolLM2. Layer 24 created especially large generic score shifts,
which shrink the fraction even where absolute residuals remain positive. This confirms
a directional localization pattern, but not practically useful cross-model selectivity
or a change in generated answers. See the
[018 protocol](experiments/018-independent-tripadvisor-layer16.md) and
[audited bundle](../results/tripr-layer-confirmation-v1/README.md).

This cross-benchmark confirmation makes a narrowly framed LessWrong research post
worth drafting: the result is surprising enough to discuss as a replicated layer-
localization pattern, especially its model asymmetry, while the failed SmolLM2 practical
gate and score-only endpoint must remain central. It does not support a general
mechanistic claim.

Experiment 019 decomposes each layer-16 direction into a shared sentiment projection
and an orthogonal residual, then compares both with norm-matched random residuals on
the same 187 TripR sentences. The residual-specificity contrast passed its frozen rule
in both models: residual minus shared was +0.02797 logits for Qwen (95% paired CI
[+0.02105, +0.03547]) and +0.00655 for SmolLM2 ([+0.00565, +0.00746]); residual minus
random residual was +0.01472 ([+0.00468, +0.02514]) and +0.00449 ([+0.00319, +0.00580]),
respectively. The shared component generated the broad polarity shift but essentially
no aspect selectivity. The residual had near-zero generic shift. The pattern is
mechanistically suggestive, but the study reuses the 018 benchmark, and each model has
only one norm-matched random-residual draw; its sentence-bootstrap intervals do not
capture variation across random controls. See the
[019 preregistration](experiments/019-tripr-shared-residual-layer16.md) and
[audited bundle](../results/tripr-component-decomposition-v1/README.md).

Experiment 020 tested the residual on the 29 polarity-conflict TripR sentences against
20 independent, norm-matched random residual seeds. The trained residual exceeded the
random-seed mean in Qwen by +0.05685 logits (nested bootstrap 95% CI [+0.01848,
+0.09852]), but one random direction exceeded it, giving a one-sided Monte Carlo rank
of 0.0952. In SmolLM2, the difference was +0.00920 ([+0.00638, +0.01214]); the trained
residual exceeded all 20 controls (rank p=0.0476, the smallest possible with 20 seeds).
The **cross-model control-robustness gate failed**. Sentence-level uncertainty alone
made Qwen's component advantage look more decisive than its control-seed variability
supports. SmolLM2's conflict-query baseline accuracy was only 58.7%, so its passing
gate is still a narrow score-level result. Full controls are in the
[020 bundle](../results/tripr-residual-control-seeds-v1/README.md).

**LessWrong decision: wait.** The independent layer-localization result is worth a
careful negative/mixed-results post, but the component mechanism now holds robustly
against the seed ensemble in only one of two models, on a reused benchmark, with no
generated-response outcome. The next informative study is a third architecture family
with its own frozen training-only directions and a generated-answer endpoint, while
retaining norm-matched random-seed controls. That can distinguish a portable mechanism
from model-specific geometry. Keep all seeds and per-model outcomes.

Experiment 021 carried that test to Granite 3.1 2B using synthetic-training-only
directions, a depth-matched layer, 20 random residual controls, and generated answers.
On 29 TripR conflict sentences, trained-residual specificity exceeded the random-seed
mean by +0.13629 logits (nested 95% CI [+0.02338, +0.25195]), but its random-seed rank
was p=0.1429, so the preregistered control gate failed. At the 5% dose, no generated
label changed in either the trained or random condition; generated strict accuracy
was 90.5% throughout. A post hoc diagnostic found 88.9% positive-vs-negative
candidate-pair accuracy on conflicts, while the preregistered all-vocabulary top-1
accuracy was 44.4%. The measurement distinction is material: the former matches the
score endpoint, but it was not the frozen gate. This is a score-level lead with wide
seed variation and no observed behavioral change, not evidence of robust transfer.
See the [`021 audited bundle`](../results/granite-tripr-residual-transfer-v1/README.md).

**LessWrong decision: wait.** The useful next experiment is a preregistered generated-
decision dose response with matched random residuals, reporting candidate-pair and
all-vocabulary accuracy separately. Do not write up this result as behavioral control.
