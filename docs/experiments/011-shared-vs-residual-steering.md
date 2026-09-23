# 011 — Do aspect-specific residuals carry selective steering effects?

**Prospective follow-up frozen after 010, before these component-control outcomes are collected.** Experiment 010 found that food, service, and value directions are strongly aligned and that every tested direction raised every queried positive-vs-negative logit score. It also found that prompt-specific effects were not forecast by the small readout. This follow-up asks whether an aspect direction's shared component explains the broad positive shift, while its aspect-specific residual produces target selectivity.

This is a local diagnostic on the 008 synthetic mixed-review task, not a claim about steering generally. Related literature already studies steering-vector geometry and steering reliability (e.g. [Braun 2025](https://arxiv.org/abs/2505.22637); [Imdad 2026](https://zenodo.org/records/20710572)); this experiment adds a paired causal decomposition in our exact setting and should be interpreted as such.

## Frozen setup

Use the same two fully captured models, block 24, 008 format-0 mixed-review final-test prompts, and 5% median training activation-norm scale as experiment 010. Directions are computed from training groups only. No final-test activation labels or outcomes enter construction. The 24 held-out scenario groups provide the resampling unit.

Let `d_s` be the unit direction for source aspect `s`, and let `u` be the unit-normalized mean of the three `d_s`. For each prompt, query each source `s` under four interventions, all at the same scalar dose used in 010:

1. **Original:** `d_s`.
2. **Shared component:** `(d_s · u) u`.
3. **Aspect residual:** `d_s − (d_s · u) u`.
4. **Random residual control:** a deterministic random unit vector orthogonal to `u`, scaled to the residual's norm for that source.

The components retain their natural norms: their sum exactly reconstructs the original direction, so this tests the decomposition at its actual contribution magnitudes rather than amplifying the smaller residual. Random residual vectors are generated once per model from seed 20260925 and are held fixed across prompts. Record the finite change in the queried aspect's positive-minus-negative next-token log-odds. Each of the 576 held-out prompts receives all 12 source-by-condition interventions per model (6,912 rows).

## Estimands and decision rule

For each condition, define target selectivity as the equally weighted mean over source aspects of `(effect when queried target equals source) − (mean effect for the other two queried targets)`. Estimate paired 95% intervals by resampling the 24 scenario groups 5,000 times with seed 20260926.

The primary contrast is residual selectivity minus shared-component selectivity. The residual-specificity interpretation is supported only if its 95% interval is entirely above zero **and** residual selectivity exceeds random-residual selectivity with an interval entirely above zero in both models. Otherwise report the measured decomposition without that interpretation. Original-versus-component additivity is descriptive because finite interventions can be nonlinear. Report overall shifts and all source-by-target cell means so generic positive-logit shifts remain visible. Do not treat sign accuracy as evidence.

No LessWrong draft or novelty claim follows automatically. Any field-level claim needs a larger, naturalistic outcome and independent models/tasks.

## Compute and integrity

Reuse the local 008 activation cache and model weights. Capture outcomes sequentially on MPS/CPU with resumable JSONL checkpoints; do not load the optional 3B model. Save source-data/protocol/code hashes and per-row outcomes. Preserve the existing 010 run. Raw activation arrays remain ignored and outside the result bundle.
