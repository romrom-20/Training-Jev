# 018 — Independent test of the layer-16 specificity lead

**Prospective confirmatory follow-up, 24 September 2026.** Experiment 017 found a
post-hoc layer-16 versus layer-24 specificity-fraction difference in SmolLM2 on 35 MAMS
reviews (+1.30 percentage points, sentence-bootstrap 95% CI [+1.07, +1.52]); Qwen's
interval crossed zero. Layer 16 itself remained below the 5% practical threshold in
both models. Experiment 018 tests whether the *direction* of that relative layer effect
replicates on independent TripR-2020Large TripAdvisor reviews.

## Data source and eligibility

Use the authors' public
[TripR-2020Large release](https://github.com/ari-dasci/OD-TripR-2020Large), pinned at
Git revision `2e3f56f8f3691019139bdec36db0b6115ed7b191`. It contains 474 reviews, 2,522
sentences, and manually annotated aspect-polarity opinions. The source release is
CC BY-SA 4.0; raw reviews stay under ignored `.context/datasets/` and are excluded from
the results package. Attribute Zuheros et al. (2022) and preserve the source license in
the results notice.

Freeze this category crosswalk: `FOOD#QUALITY` and `FOOD#STYLE_OPTIONS` map to food;
`SERVICE#GENERAL` maps to service; any category ending in `#PRICES` maps to price/value.
Ignore all other categories, neutral opinions and opinions without positive/negative
polarity. If the mapped categories within a sentence give one group both positive and
negative labels, drop that group from that sentence; retain other unambiguous groups.
Keep a sentence only if at least two distinct mapped groups remain. A sentence is in
the conflict slice if two or more retained groups have opposite labels. This frozen
filter yields 187 sentences, 385 queries and 29 conflict sentences. Keep only the
sentence identifier, mapped group, label and scores in outcomes; never save review text
in the published result bundle.

## Frozen intervention and prompts

Use the same wording and token-scoring endpoint as experiments 015–017:
`Review: {sentence}` followed by the mapped question `What is the sentiment about
{food/menu, service/staff, or price}? Reply with exactly one word: positive or negative.`
Treat each eligible sentence separately, without adding surrounding review sentences. Use the
cached training-only mixed-review format-0 directions for Qwen2.5-1.5B and
SmolLM2-1.7B. Recompute those same three directions independently at layers 16 and 24;
do not use TripR text, labels or outcomes to fit or select directions. Normalize and
construct shared/random controls exactly as experiment 017, using seed `20260930`.
Dose is 5% of each layer's own median training activation norm. Test all native, shared
and norm-matched random treatments on every eligible query, with three source labels
for each treatment. Use float32, sequential MPS inference and batches of eight.

## Outcomes and decision rules

For each model and layer, report baseline strict accuracy, valid polarity-token rate,
steered strict accuracy by condition, generic native positive-minus-negative logit
shift, native-minus-random diagonal/off-diagonal specificity, its sentence-bootstrap
95% interval, specificity divided by generic shift, and the 29-sentence conflict-slice
specificity. Bootstrap sentence IDs 5,000 times with seed `20260924`.

The primary test is the paired sentence-bootstrap difference in specificity fraction
`(layer 16) − (layer 24)`, computed on the same resampled sentence IDs within each
model. The layer-16 localization hypothesis replicates only if this difference's 95%
interval is above zero in both models, layer-16 native-minus-random specificity also
has a positive 95% interval in both models, and both unsteered baseline accuracies are
above chance. If either layer has a nonpositive or bootstrap-unstable generic shift,
mark its ratio undefined and the primary comparison inconclusive for that model. Report
all outcomes regardless.

Separately call layer 16 practically useful only if its specificity fraction is at
least 5%, its specificity interval is positive, and baseline accuracy is above chance.
Do not lower this threshold after seeing results. This experiment evaluates one
preselected cross-model layer contrast on one independent dataset; it does not establish
general layer locality without replication on a third model family or another dataset.

## Reproduction

```bash
PYTHONPATH=scripts:src .venv/bin/python scripts/tripr_layer_confirmation.py all --offline --resume
PYTHONPATH=scripts:src .venv/bin/python scripts/analyze_tripr_layer_confirmation.py
PYTHONPATH=scripts:src .venv/bin/python scripts/package_tripr_layer_confirmation.py \
  runs/tripr-layer-confirmation-v1 results/tripr-layer-confirmation-v1
```
