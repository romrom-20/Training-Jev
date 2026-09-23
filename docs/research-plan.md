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
