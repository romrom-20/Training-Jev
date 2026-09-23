# 014 — Does synthetic aspect steering select a real review aspect?

**Prospective local protocol, 24 September 2026.** Experiments 011–012 found
that directions learned from synthetic aspect reviews create large, transferable
positive-vs-negative logit shifts, while target selectivity was weak. Experiment
013 found answer-identifier following in one competent local model, consistent
with the broader attribution problem studied by Gao et al. (2026). The unresolved
question I care about is more basic: when one real review mentions multiple
aspects, does a direction learned for one aspect selectively move the model's
sentiment score when asked about that aspect, or does it act like a generic
positive-answer bias?

This is a small, falsifiable generalization test, not a claim that aspect-based
sentiment or steering is new. Use the SemEval-2014 Restaurant gold test set,
which was annotated with aspect-category polarity by human annotators. The
official task description explicitly includes reviews where categories disagree
(for example, negative price and positive food). The local evaluation source is
`Restaurants_Test_Gold.xml`; it is obtained from a public mirror of the official
gold file linked by the task organizers. Keep raw review text in ignored
`.context/datasets/` and do not redistribute it in the result bundle.

## Frozen sample and prompts

Use all gold-test sentences with at least two distinct annotated categories from
`food`, `service`, and `price`, where each included label is positive or
negative. For each included category, ask a fixed question about that category
using the complete sentence as the review. Map the frozen synthetic directions
`food`, `service`, and `value` to SemEval categories `food`, `service`, and
`price`, respectively. Do not filter on the model's answers or on steering
outcomes. The unit for uncertainty estimates is the original sentence ID.

For each sentence/category prompt, record the unmodified next-token logits for
`positive` and `negative`. Then apply each of the three frozen native directions
from experiment 008, their shared component, and one norm-matched random control
at block 24 and the fixed 5% training-activation-norm dose. Score the change in
`logit(positive) - logit(negative)` relative to that same prompt's baseline.
Directions, layer, dose, prompt template, category mapping, and sample filter
remain frozen. Run Qwen2.5-1.5B and SmolLM2-1.7B sequentially and offline.

## Primary endpoint and interpretation

Within each multi-category sentence, compare the native direction's effect on
its matching queried category (diagonal) with its mean effect on the other
annotated queried categories (off-diagonal). The primary statistic is the
sentence-clustered mean of `native diagonal - native off-diagonal`, minus the
same diagonal/off-diagonal contrast for the norm-matched random direction. A
positive contrast means frozen synthetic directions retain some aspect
selectivity on real review language beyond a random perturbation. Report the
shared component with the same contrast to distinguish a generic valence shift
from aspect-selective movement. Bootstrap original sentence IDs 5,000 times
(seed 20260924) for 95% intervals. Report each model separately; do not pool
models to turn one model's result into a pass.

Also report baseline accuracy and valid positive/negative argmax rate for each
model, all intervention mean shifts, the contrast on the subset whose annotated
categories disagree in polarity, and per-category direction effects. The
polarity-disagreement subset is prespecified as descriptive because it may be
small. Logit shifts remain the primary endpoint even if baseline task accuracy
is low; make behavioral claims only if unmodified strict accuracy exceeds 50%
for that model. No prompt tuning, target selection, new direction fitting,
hidden-state cache adaptation, or dose adjustment is allowed after outcomes are
seen.

## Integrity and compute

The source dataset and 008 data/activation hashes, protocol hash, runner hash,
device, dose, and outcomes are recorded in per-model manifests. Keep resumable
prompt-level checkpoints. On the 24 GB MacBook Air, load only one model at a
time and use batches of eight. Result bundles contain row IDs and aggregate
statistics but no copied review text. This experiment can reveal a local,
small-model generalization result; the benchmark is old and one task/domain, so
even a positive result needs independent datasets and model families.
