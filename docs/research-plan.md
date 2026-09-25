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

Experiment 022 ran that dose response on the same 63 conflict prompts with 20
random-residual seeds at 10%, 20%, and 40% of the training activation norm. At the
primary 40% dose, trained steering flipped one answer and harmed it; it corrected
none. Strict accuracy fell from 90.5% to 88.9%, while the random-control mean utility
was also negative. Trained-minus-random utility was -0.00862 (nested 95% CI
[-0.04052, +0.01724], rank p=0.9048). The lower doses produced the same harmful flip.
The behavioral gate failed, with perfect one-word validity but no evidence that
learned residuals beat random ones. This bounds the tested steering range; it does
not show that the representation has no behavioral role. See the
[`022 audited bundle`](../results/granite-tripr-generation-dose-response-v1/README.md).

**LessWrong decision: wait.** Neither the 5% test nor this 10–40% escalation found a
beneficial generated-answer effect. A more useful pivot is to identify prompts where
the model itself is uncertain about the target aspect, then preregister a test with
adequate baseline headroom and a behavior-first outcome. Avoid escalating dose alone:
the current dose range only produced a harmful answer change.

Experiment 023 checked whether candidate-pair scores track generated labels on a
different benchmark before intervention. On 233 SemEval aspect prompts, the pairwise
margin selected the same label as greedy generation on 232/233 prompts (99.6%,
sentence-bootstrap 95% CI [98.7%, 100%]); candidate-pair and generated accuracies
were 97.4% and 97.0%. All-vocabulary top-1 accuracy was 72.1%, showing why it should
not be conflated with forced-choice pair accuracy. The label-free absolute-margin
error AUC was 0.70 across seven errors; the initially specified gold-aligned AUC was
invalid because it used the reference label and has been replaced via the documented
[analysis amendment](experiments/023-analysis-amendment.md). This is an observational
result for one model and one constrained sentiment task. Rank-calibration and
verbalized-probability research already study relationships between confidence
signals and generated answer quality, so the result does not establish a new
measurement principle. See the [`023 audited bundle`](../results/granite-semeval-margin-generation-v1/README.md).

**LessWrong decision: wait.** The finding is a useful measurement check, but near-
perfect agreement is expected for explicit two-label prompts, and the relevant
confidence/generation relationship has prior work. Next, test the same frozen prompt
set on the already cached Qwen and SmolLM2 families. If candidate-pair/generation
agreement varies across families, investigate answer formatting and label-token
calibration before making any broader claim.

Experiment 024 ran the same SemEval prompts through Qwen2.5-1.5B and SmolLM2-1.7B.
Both produced exact one-word labels on all 233 prompts; the candidate-pair choice
matched every generated label in both models (100%, sentence-bootstrap interval
[100%, 100%]). Their candidate-pair, generated and all-vocabulary accuracies were
identical within each model: 94.8% for Qwen and 92.3% for SmolLM2. This strongly
suggests the agreement follows from this explicit forced-choice prompt and greedy
decoding, rather than revealing a new internal mechanism. It is a useful replication
of the measurement sanity check, not a novel result. See the
[`024 audited bundle`](../results/cross-family-score-generation-v1/README.md).

**LessWrong decision: wait.** The next discriminating experiment is a paired prompt-
format ablation on the same review sentences: compare explicit one-word choices with
a natural open question across the three local model families. This will test whether
the score/generation match depends on answer-format constraints. Avoid presenting the
current agreement as surprising until that condition is checked.

Experiment 025 paired the explicit one-word prompt with an open sentiment question
on the same 233 SemEval items across Granite, Qwen, and SmolLM2. Exact one-word
compliance fell from 100% to 0% in all three models. The open responses contained a
unique positive/negative word on only 12.9% of Granite outputs, 39.1% of Qwen, and
65.7% of SmolLM2. On that selected parseable subset, open-minus-forced candidate/
generation agreement was -5.1 percentage points (sentence-bootstrap 95% CI
[-8.3, -2.5]), but the frozen coverage rule failed, so this contrast is descriptive.
This shows how strongly output-format instructions govern the measurement; it does
not establish a cross-family latent-score relationship. The
[`025 audited bundle`](../results/prompt-format-score-generation-v1/README.md)
contains outcomes and audit.

**LessWrong decision: wait.** The format effect is expected, and open-answer polarity
was mostly not extractable using the predeclared rule. A next study would need a
behavior-first outcome for natural completions, with an independently validated
polarity evaluator and enough baseline errors to test whether score shifts predict
behavioral changes. The current line does not support a novelty claim.

An exploratory feasibility check trained a small aspect-aware TF-IDF classifier on
MAMS-ATSA train, then evaluated it on MAMS validation/test and SemEval restaurant
aspects. It reached 0.793 macro-F1 on MAMS test and 0.786 on SemEval, with 0.709
negative recall on SemEval. That is too weak to label open generations reliably, and
source-review transfer does not validate generated-answer judging anyway. No outputs
from 025 were scored and no substantive generation claim follows. See the
[`evaluator-feasibility pilot`](../results/freeform-evaluator-feasibility-v1/README.md).
The next step needs either human-labeled short aspect summaries or a local judge that
passes a frozen, held-out semantic validation before its outputs are used.

Experiment 027 tested that next route. A cached Qwen2.5-3B judge passed the source
SemEval screen (96.1% accuracy, 98.2% negative recall), so the frozen protocol's
conditional phase ran on the same 233 prompts and three model families. Against
review-level gold labels, judge-estimated open-answer accuracy was 53.6% for Granite,
81.1% for Qwen2.5-1.5B, and 87.1% for SmolLM2. Candidate-pair accuracy was 93.6%,
94.8%, and 87.1%. All 699 candidate margins exactly reproduced experiment 025; the
score/judge accuracy gap was +32.6 to +47.2 points for Granite and +8.6 to +19.0 for
Qwen under exploratory sentence-cluster bootstrap intervals, while SmolLM2's interval
crossed zero. The key caveat is that source-text judge validation does not validate
judging model-written answers. This could be judge distribution shift, genuine answer
errors, or both. The full audit and limitations are in the
[`027 result bundle`](../results/local-open-judge-v1/README.md). The next discriminating
step is a second, independently validated judge or blinded human coding on a small
stratified sample. No publication decision follows from 027.

Experiment 028 ran the preregistered cross-family check. A cached Phi-3 Mini judge
passed the same source-label gate (94.4% accuracy, 98.2% negative recall). Its
agreement with the Qwen judge on open answers was 59.3% for Granite, 94.1% for Qwen,
and 72.7% for SmolLM (sentence-cluster intervals are in the
[`028 bundle`](../results/independent-judge-check-v1/README.md)). Both judges put
Granite answers near chance against review-level gold; for Qwen their labels were
highly consistent but only 78–81% accurate; SmolLM agreement was intermediate.
Candidate-pair scores still matched experiment 025 exactly. This is evidence that
semantic evaluation of open answers is target-dependent, and that judge agreement
can be strong without high correctness. It is not yet evidence that answer text is
wrong, because neither judge saw a human-labeled answer benchmark. LLM-as-a-judge
biases are prior art; we should not frame the evaluation problem itself as new.
Experiment 030 used the local open-weights Laya decision engine on all 233 source
reviews and 60 paired generations. Source-review clear-polarity coverage was 88.0%;
conditional accuracy was 94.6%, but strict all-item accuracy was 83.3% and negative
recall was 76.4%, equal to the always-positive accuracy baseline. Performance also
varied by aspect: price coverage was 71.2% and negative recall 57.9%, compared with
97.1% coverage and 93.3% negative recall for service. This cautions against treating
the high clear-only accuracy as a general validation gate.

On the 20-item answer sample, Laya marked 10/20 Qwen answers mixed and only gave
clear binary polarity for 8/20; clear-answer accuracy against review gold was 75.0%
(8 items). It gave binary polarity for 15/20 Granite answers (53.3% clear-only
accuracy) and 17/20 SmolLM answers (88.2%). On rows where both gave binary polarity,
Laya agreed with Qwen's judge on 39/40 cases across targets, but coverage ranged
from 40% to 85% and agreement with Phi was lower. This points to binary judges
collapsing some answers Laya regards as mixed, but it does not show which reading is
right. The result is especially tentative because the generated-answer set is
stratified and SmolLM repeated identical aspect-answer inputs six times. Full metrics
and an explicitly post-hoc duplicate-input sensitivity check are in the
[`030 result bundle`](../results/laya-decision-audit-v1/README.md).

The lead is now a concrete annotation question: do humans also mark a substantial
share of Qwen completions as mixed, and are the binary judges wrong on those cases?
Laya is an automated judge with uneven source performance, so the answer needs
human-coded outputs or a larger independent validation set. LessWrong remains a
wait.

Experiment 031 tested whether the 8-token cap itself made open answers look mixed or
incomplete to Laya. Every one of the 699 short answers hit the cap. On the same
prompts, the mixed/unclear fraction fell from 45.5% at 8 tokens to 12.7% at 32
(paired difference −32.8 points; sentence-cluster interval −36.9 to −28.6), with
the same direction on Granite, Qwen and SmolLM2. The eight-token Laya results
reproduced experiment 030 on all 60 overlap rows. This makes generation censoring a
plausible contributor to 030's ambiguous labels, but Laya is still only one
automated evaluator and 560/699 of the 32-token answers hit that cap too. The result
is therefore a strong lead, not a validated interpretation or novelty claim. See the
[`031 result bundle`](../results/answer-length-censoring-v1/README.md).

The next discriminating test runs Qwen2.5-3B and Phi-3 Mini on the exact same 8- and
32-token texts. It asks whether their paired polarity judgments also move with the
cap and whether any movement aligns with the review-level proxy. This is a
post-result robustness follow-up, not a second confirmatory test. LLM-judge length
bias is already studied in preference comparisons; the narrower potential
contribution here is a within-prompt generation-cap intervention on aspect-sentiment
extraction, if independent evaluation survives. LessWrong remains a wait pending
cross-engine and ideally human answer-level validation.

Experiment 032 supplied that first cross-engine check on the same answer pairs.
Qwen2.5-3B and Phi-3 Mini both reproduced their earlier 8-token labels exactly, then
their review-label-proxy accuracy rose from 74.0% to 96.7% and 69.8% to 95.9%,
respectively, at 32 tokens. Their mutual agreement rose from 75.8% to 99.1%.
Every target family moved in the same direction. This strengthens the evaluator
sensitivity finding, while leaving answer-level correctness unresolved: the review
label is not gold for the generated text, and both judges are language models. See
the [`032 result bundle`](../results/cross-judge-length-robustness-v1/README.md).

Experiment 033 preregistered a dose-response curve over nested prefixes of the same
32-token continuations, avoiding another target-generation run. The protocol tested
4/8/12/16/24/32-token checkpoints for Laya ambiguity and Qwen/Phi agreement and
label stability.

Experiment 033 found a sharp transition between 8 and 12 tokens: Qwen–Phi agreement
rose from 75.8% to 92.6%, and both judges' accuracy against the review-level proxy
reached about 89%. Agreement was 97.5% at 16 tokens and about 99% by 24; Laya's
mixed/unclear rate fell throughout. At four tokens, Qwen/Phi agreed on every jointly
parseable row while Laya called 99.6% of prefixes unclear/mixed and proxy accuracy
was only 61.5%/44.2%. This illustrates why agreement alone cannot validate a judge.
By 12 tokens roughly 90% of each judge's labels matched its own 32-token label at
every later tested checkpoint. These are post-result repeated-prefix measurements
on the same 233 items, not independent replication or answer-level ground truth.
See the [`033 result bundle`](../results/prefix-dose-response-v1/README.md).

**LessWrong decision: wait.** The cap effect is now a compelling local lead, but the
answer labels are automated and the source-review polarity is not gold for the
generated answer. Before writing a field-facing post, the next experiment should
repeat the frozen 8/12/16/32 comparison on an independent aspect-sentiment domain
with an answer-level validation set or blinded human coding. The relevant literature
already covers preference-evaluation length bias; a potential contribution would
need to stay narrow to generation-cap sensitivity in short semantic extraction and
must show that the resolution point transfers.

Experiment 034 freezes the first transfer test on TripR-2020Large: 385 filtered
aspect queries across 187 annotated review sentences, using the same three target
families and the same Laya, Qwen and Phi evaluators at nested 8/12/32-token prefixes.
This is an independent restaurant-review corpus, not an unrelated domain or
answer-level annotation of the generations. The primary comparison is the paired
sentence-clustered 12-minus-8 change in Qwen/Phi agreement. The source filter follows
experiment 018 and has positive-skewed class balance, so label accuracy remains
secondary to agreement and Laya-ambiguity outcomes.

**Experiment 034 result.** The pooled Qwen/Phi agreement gain from 8 to 12 tokens
was +4.2 percentage points (sentence-cluster 95% interval +2.5 to +5.7), while
Laya mixed/unclear labels fell by 13.9 points. However, the frozen descriptive
transfer rule requiring a positive change for each target family did not pass:
the effect was +1.7 points for Granite (interval crosses zero), +10.7 for
Qwen-1.5B, and 0.0 for SmolLM2. This is partial transfer on another restaurant
corpus, driven primarily by the Qwen target. A post-hoc class split found that
the 8→12 binary-judge accuracy gain was much larger on negative source labels
(Qwen +32.3 points; Phi +28.1) than positive labels (Qwen +6.7; Phi +0.5).
These negative-source cases came from 64 independent sentences; TripR gold still
labels the source review, not the generated answer. Treat the class split as a
lead, not confirmation. See the
[`034 result bundle`](../results/tripr-prefix-threshold-v1/README.md).

The literature review narrows the question further. The 2025 length-bias paper
studies pairwise preference and information mass
([Hu et al.](https://aclanthology.org/2025.findings-emnlp.358/)); a 2026 judge
bias preprint directly contrasts filler expansions with genuinely more-complete
truncation pairs ([Soumik](https://arxiv.org/abs/2604.23178)). DABS, an ACL 2026
aspect-sentiment model, reports that negation and contrast benefit from deeper
aspect-conditioned reading ([Xia et al.](https://aclanthology.org/2026.acl-long.667/)).
Thus, neither generic length sensitivity nor negation difficulty is a novelty
claim. The useful next test is a controlled factorial that moves a known
polarity clause across the 8/12-word boundary while balancing positive/negative
labels and direct/negated forms. That can separate simple evidence availability
from a polarity- or composition-specific judge error without generating another
large set of target-model answers. Experiment 035 should use Laya, Qwen-3B and
Phi on identical minimal-contrast texts, with deterministic label gold, the
template/aspect scaffold as the bootstrap cluster, and a capability gate at
12 words. Freeze the stimuli and code before judging. LessWrong remains a wait:
there is a promising asymmetric error pattern but no human-validated natural
answer result or independent-domain replication.

**Experiment 035 outcome.** The preregistered late-minus-early accuracy
difference-in-differences was +61.2 points (scaffold-cluster 95% interval
+58.6 to +63.5), but the frozen full-answer capability gate failed: pooled
12-word binary accuracy was 87.8%, below 90%. Do not treat the large contrast as
confirmatory. The failure exposed two distinct problems. First, Qwen labeled
“not bad” positive in only 4/96 cases; Laya was mixed on many of those cases,
while Phi labeled all positive, so this phrase was not a defensible binary gold
condition. Second, when the late eight-word input had no polarity clue, Qwen
defaulted negative on 44/48 identical prefixes, and Phi failed the one-word
format on 41/48. Laya gave different decoded labels on repeated identical text
because the randomized key map changed by stimulus ID, a confound for its
no-cue breakdown. These are useful evaluator-behavior diagnostics but not a
validated natural-answer finding. Full analysis and limitations are in the
[`035 result bundle`](../results/cue-position-polarity-v1/README.md).

The next preregistered question is whether forced binary judges can abstain
appropriately when a short prefix contains no aspect-polarity evidence. Reuse
only the known-clear 035 direct good/bad and negative “not good” sentences,
exclude the pragmatic “not bad” construction, and add an explicit
`insufficient evidence` option for Qwen/Phi. Score a decision as correct when
the cue is absent and the judge abstains, or when the cue is visible and it
returns the deterministic polarity. Compare against the exact same forced
binary judgments already collected and Laya's four-way outputs, with Laya's key
mapping held fixed for identical prefixes. This tests a practical response to
the observed failure mode; it does not generalize beyond short controlled text.

**Experiment 036 outcome.** Adding an explicit `insufficient` decision to Qwen2.5-3B
and Phi-3 Mini improved appropriate decisions by 23.6 percentage points on the
paired controlled prefixes (scaffold-cluster 95% CI +22.8 to +24.3); both abstained
on all 96 no-cue prefixes. Cue-visible polarity accuracy fell by 2.8 points for Qwen
and 1.0 for Phi. The preregistered operational rule passed. Laya returned its mapped
abstention label on only 37.5% of the same no-cue cases, despite consistent behavior
on identical text. This wrapper result remains synthetic: the response set and
instruction both changed, and the evidence-free case was engineered by truncation.
General evidence sufficiency and abstention are already active topics, including
RAG/unanswerable-QA settings. The next useful test is natural text with explicit
human annotations for both aspect and opinion-evidence spans, to distinguish actual
evidence availability from reactions to a constructed missing-cue template. The
[`036 result bundle`](../results/evidence-aware-abstention-v1/README.md) records the
full paired analysis and limits. LessWrong remains a wait pending a natural-data
result that survives benchmark and judge differences.

## Natural opinion-span test: experiment 037

The ASTE literature already annotates target/opinion/sentiment triplets, while
general abstention benchmarks cover unanswerable questions. A targeted search did
not locate a paired test asking aspect judges to classify natural excerpts just
before versus after their human-annotated target-linked opinion phrase. That narrow
gap is provisional, not a novelty claim. Experiment 037 freezes 234 balanced
single-triplet SemEval test sentences across restaurant and laptop splits. Each
aspect remains visible while its opinion span is withheld or revealed. Two local
generative judges receive matched forced-binary and abstention wrappers; Laya's
typed four-way decision is the independent engine comparison. Raw review text stays
private because the upstream repository does not state a license. The frozen
[protocol](experiments/037-natural-opinion-span-abstention.md) and
[identifier/span manifest](experiments/037-stimuli.json) must be committed before
running judgments. The opinion span is an annotation-based evidence marker, not
proof that preceding prose is impossible to interpret or that humans agree it is
insufficient.

**Experiment 037 outcome.** Forced-binary Qwen/Phi predicted the full-review
polarity at 73.5% and 77.8%, respectively, even before the sole ASTE-annotated
opinion phrase was visible. Thus the registered “insufficient” label was an
annotation-visibility proxy, not validated evidence of unanswerability. The
explicit-abstention wrapper's operational score rose by 26.8 points, but it should
not be read as an epistemic-quality win. More striking, Qwen abstained on 78.2% of
these prefixes while Phi did so on only 30.8% (paired gap 47.4 points); Laya's
four-way engine returned `unclear` on 8.1%. The pre-span polarity accuracy could
come from genuine context, aspect priors, or contamination from these old reviews.
ABSA already has research on spurious correlations and broad abstention work is
established, so this is a lead rather than a novelty claim. See the
[`037 result bundle`](../results/natural-opinion-span-abstention-v1/README.md).

**Experiment 038** freezes an aspect-only control: remove every review word while
keeping the same aspect query, balanced full-review labels, and wrapper prompts.
The primary paired comparison reuses 037's natural-prefix predictions and tests
whether their above-chance polarity signal exceeds the aspect-name baseline. This
exploratory, post-result follow-up separates visible context from target priors;
it cannot rule out training-data exposure. Its protocol is committed before new
judgments in [`038-aspect-only-prior-control.md`](experiments/038-aspect-only-prior-control.md).

**Experiment 038 outcome.** Removing all review words reduced forced-binary
accuracy to 50.0% for Qwen and 39.3% for Phi, compared with 73.5% and 77.8% on the
natural prefix. The paired pooled context gain was +31.0 points (cluster-bootstrap
95% CI +22.2 to +39.5); it was positive in each of the four SemEval subsets.
Both generative judges abstained on every no-text prompt, and Laya returned
`unclear` on every one. This suggests pre-opinion words carry polarity-predictive
signal beyond aspect identity, while leaving open context interpretation versus
memorized completion. The samples are old and benchmark contamination remains
possible; these repeated restaurant/laptop sets do not establish independent
generalization. Full results are in the
[`038 bundle`](../results/aspect-only-prior-control-v1/README.md).

Experiment 039 is the next diagnostic: train a fixed TF-IDF/logistic model on
three SemEval subsets and evaluate on the fourth, comparing aspect-masked prefix
features with aspect-only features. This small non-pretrained baseline tests
whether context signal transfers across the available subsets without relying on
LLM pretraining, although the subsets are not fully independent domains. The
[frozen protocol](experiments/039-cross-subset-context-lexical-baseline.md) requires
a 5-point gain and a positive cluster interval; LessWrong remains on hold until
this and stronger independent validation clarify the interpretation.

**Experiment 039 outcome.** The non-pretrained cross-subset model reached 52.6%
with target-masked context and 49.1% with aspect-only features; the paired gain
was 3.4 points (95% sentence-cluster interval −5.6 to +12.4), so its registered
gate failed. The low-training lexical baseline does not explain the 73–78% local
LLM result, but the small, related folds leave training volume, domain shift and
pretraining exposure unresolved. Full details are in the
[`039 bundle`](../results/lexical-context-generalization-v1/README.md).

For the next choice, Laya's pinned local typed-choice engine selected independent
corpus replication over an in-domain lexical baseline, token-order intervention,
and human sufficiency calibration. A source audit found a useful, more specific
test in TRABL, a 2026 travel-review dataset with two annotators and aspect,
opinion-span, polarity and evidence annotations. Existing work already studies
mixed-aspect sentiment and target-specific context, so the idea is not framed as
a new task: among 48 test reviews with exact annotator agreement on one positive
and one negative target, Experiment 040 tests whether the earlier target's
opinion cue is copied onto the later target before that later opinion span is
visible, and whether deleting the earlier cue changes that behavior. This
sequential intervention uses a second dataset family and 48 review clusters; it
is still an exploratory benchmark test, not an independent human replication.
The [frozen protocol](experiments/040-trabl-prior-opinion-interference.md),
[selection metadata](experiments/040-stimuli.json), runner and tests are committed
before any new target-model judgments. LessWrong remains deferred until the
experiment runs and a result survives suitable follow-up.

**Experiment 040 outcome.** The preregistered Qwen later-target opposite-prior
copy-rate contrast was +2.1 points (95% review-cluster interval 0.0 to +6.3),
below the 5-point gate and without a strictly positive interval. The targeted
sentiment-copy hypothesis did not pass. A predeclared secondary did show that
Phi-3 Mini's polarity accuracy on natural prefixes was 83.3% versus 57.3% on
aspect-only input (paired gain +26.0 points, 95% interval +16.7 to +35.4); Qwen's
gain was smaller and uncertain. This strengthens the independent-corpus
pre-opinion signal for one model, but the small, selected TRABL test set, possible
source-review exposure, and lack of human sufficiency labels keep the result
short of a field-level claim. The complete
[`040 result bundle`](../results/trabl-prior-opinion-interference-v1/README.md)
records all judges and limits. LessWrong remains deferred pending a stronger
follow-up.
