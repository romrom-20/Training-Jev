# Experimental protocol

## Estimands and labels

The pilot predicts **three binary settings explicitly supplied in a target model's
system prompt**: red/blue, circle/square, cat/dog. These are instruction-reading labels,
not beliefs, plans, safety judgments, or deception labels. The target is told to output
only its color setting. Shape and animal are irrelevant to that requested output.
Whether an irrelevant setting is decodable and whether its direction changes behavior
are separate empirical questions.

Each scenario contains all eight combinations of the three settings. Setting order and
a record identifier vary by scenario. Within each scenario, only the factor values
change. The user message is constant. The target never sees the probe question.
All query rows derived from a scenario remain in the same split.

The default configuration contains 24 training scenarios, 6 validation, 6 calibration,
8 test and 8 out-of-template scenarios: **416 model prompts, 1,248 property labels**.
Training uses two instruction templates. Ordinary test uses new record IDs with those
templates; the shifted split uses one entirely different template. This is weak
linguistic shift, not unseen-task generalization. There are only eight independent
scenario clusters in each test condition, and they share templates.

## Frozen target and readout

Target: Qwen/Qwen2.5-0.5B-Instruct, pinned revision in `configs/macbook.toml`.
Capture transformer **block outputs** at blocks 6, 12 and 18, one-indexed, at the
last prompt token before generation. Hooks avoid confusing embedding states, block
outputs and final RMSNorm states. Left padding keeps the final real token at index −1.
Forward computation disables the KV cache. Only the final hidden state is projected
through the vocabulary head, reducing memory without changing next-token logits.

No target weights are updated. Query embeddings are the mean final-normalized hidden
states of the same frozen model on the question alone, then RMS-normalized. They contain
no scenario text or labels. There are three canonical questions and one held-out
paraphrase per property. **The paraphrase check is not a held-out property test.**

For standardized activation x and fixed query embedding e:

```
s(x,e) = <U x, V e> / sqrt(r) + b(e)
p(y=1 | x,e) = sigmoid(s(x,e) / T)
```

At a fixed question the head remains affine in activations. U and V share a learned
rank-r subspace across questions. Activation means and scales come only from training
examples. Training minimizes BCE + 0.1 × binary Brier, with AdamW weight decay. Checkpoints
are selected on validation NLL. T is fitted on the separate calibration split using a
fixed log-spaced grid from 0.1 to 10; a boundary solution is not a calibration guarantee.

The pilot rank is four: 8,065 parameters. With only three properties, separate linear
probes have **fewer parameters (2,691)**. This pilot demonstrates the architecture and
controls, not parameter savings. To test the sharing hypothesis, increase the number
of properties and compare matched parameter budgets, ranks and label budgets.

## Baselines and controls

- **Independent linear:** separate supervised linear readout per known property,
  same activation normalization, loss, validation procedure and calibration split.
- **Query-only:** no activation dependence. Balanced labels make 50% expected accuracy
  and 0.25 binary Brier the uninformative reference.
- **Shuffled training labels:** fixed per-property training-label permutations;
  validation and calibration labels stay real. This is an optimistic null because
  validation can select the least-bad checkpoint. It is not a formal permutation test.
- **Shuffled test activations:** keep query and label, permute activation rows.
- **Swapped test questions:** cyclically rotate questions while keeping original labels.
- **Mean activation:** zero after training-only standardization.
- **Full-prompt TF-IDF + logistic regression:** privileged text baseline, fitted only
  on training prompts. It sees explicit labels and should perform well.
- **User-only text:** same classifier given only the visible user message.

All methods and all sampled layers are reported, including failures. Binary Brier is
mean `(p-y)^2` (maximum 1), not the two-class sum (maximum 2). ECE uses ten fixed-width
bins of positive-class probability; it is a noisy diagnostic, not a primary endpoint.
Also report NLL, accuracy, AUROC, and per-property scores. Pooled AUROC should not
replace the per-property values.

Three seeds vary probe initialization, not data generation. Paired Brier comparisons
average losses over seeds and resample complete test scenarios. These intervals are
conditional on this dataset design and do not include cross-template or cross-model
uncertainty. They are not simultaneous multiple-comparison intervals.

## Steering diagnostic

Choose one layer by mean validation NLL across seeds. Use the first configured seed
for interventions; do not select a seed on test performance. Compute each question's
raw-activation score gradient, including the chain rule through standardization. Add
and subtract a vector with length 5% of the median training activation norm at the
same captured block and token. Use four preselected test scenarios (32 prompts).

Primary descriptive estimand:

```
mean over fixed test prompts of
  log[P_target(red)/P_target(blue)] at +dose
  − log[P_target(red)/P_target(blue)] at −dose
```

Report color, shape and animal directions, eight seeded equal-norm random directions,
and a zero-vector sham. Save per-prompt full-vocabulary KL from the unmodified model,
red+blue probability mass, probe red probability, and output log odds. This separates
changes to the probe's own prediction from changes to the actual target model.
Intervals resample scenario clusters; four clusters are too few for strong inference.
Random-direction variation is displayed, not treated as a comprehensive null distribution.

**This is a sufficiency diagnostic.** Natural-direction patching, negative-dose curves,
ablation/necessity checks, intervention-distribution sensitivity, and behavior quality
on unrelated tasks are required before claiming mechanism identification. Low
vocabulary KL is not by itself proof that behavior is unaffected elsewhere.

## What the pilot does not test

Unseen property families; open-set queries or an `unknown` option; multiclass choice;
model-family transfer; real safety monitoring; long-context scaling; comparison with
an actual trained activation oracle; SAE features; causal training losses; calibrated
forecasts of intervention effects. These are research milestones, not implemented features.

A positive pilot is an engineering gate. A fundable research result would require the
stronger preregistered comparisons in [the research plan](research-plan.md).
