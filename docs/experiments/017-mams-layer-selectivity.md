# 017 — Does layer choice change aspect selectivity on conflicting reviews?

**Prospective exploratory experiment, 24 September 2026.** Experiments 015–016 found
small, consistently positive target-matched score residuals on the MAMS conflict-rich
test slice, while generic positive shifts were hundreds to thousands of times larger.
Experiment 016 found that the specificity/generic-shift ratio decreases with dose at
layer 24. The new question is whether that ratio depends on the intervention layer.

## Frozen design

Use only the MAMS-ACSA test filter frozen in experiment 015: 35 sentences, 71 mapped
category queries and 18 sentences with different mapped polarities. Keep its category
crosswalk, prompts, candidate tokens, controls, outcome definitions and sentence
bootstrap unchanged. No MAMS text or labels may be used to fit directions.

Compare layers 8, 16 and 24 in Qwen2.5-1.5B and SmolLM2-1.7B. At each layer, independently
derive the three positive-minus-negative directions from the existing mixed-review,
format-0 training split only. Normalize each native direction to unit length, construct
the shared and random controls with the same procedure and seed as experiment 013, and
set dose to 5% of the median training activation norm at that layer. Run all three
conditions (native, shared, norm-matched random) on all 71 MAMS prompts, with three
source directions for diagonal/off-diagonal contrasts. Use float32, sequential MPS
inference and batches of eight. Checkpoint each layer/model independently.

## Outcomes and decision rules

Primary endpoint at each model/layer: sentence-bootstrap native-minus-random
diagonal/off-diagonal specificity in logit margin. Also report the generic native
positive-minus-negative shift, specificity divided by generic shift, unsteered and
steered strict accuracy by condition, and the 18-sentence conflict-only estimate.
Bootstrap sentence IDs 5,000 times with seed `20260924`.

This is exploratory; report all six model/layer results and do not call the largest one
a discovered optimum. A layer is practically promising only if its specificity is at
least 5% of its generic shift, its 95% interval excludes zero, and unsteered accuracy
exceeds chance. A layer-dependent mechanism is not supported merely because one of six
point estimates is positive or largest. Any promising layer must be locked before an
independent benchmark and model-family replication.

This comparison is not a test of whether any layer is globally optimal. Layer choice
is established as an important steering design choice; the narrow endpoint here is
whether target-aspect specificity, as a fraction of broad sentiment movement, varies
across layers on natural conflicting reviews.

## Reproduction

```bash
PYTHONPATH=scripts:src .venv/bin/python scripts/mams_layer_selectivity.py all --offline --resume
PYTHONPATH=scripts:src .venv/bin/python scripts/analyze_mams_layer_selectivity.py
PYTHONPATH=scripts:src .venv/bin/python scripts/package_mams_layer_selectivity.py \
  runs/mams-layer-selectivity-v1 results/mams-layer-selectivity-v1
```
