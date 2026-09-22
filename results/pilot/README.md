# Pilot 001 — what the evidence supports

**Completed 2026-09-22 on a 24 GB Apple M5 MacBook Air.**
Frozen Qwen2.5-0.5B-Instruct; 416 prompts, three probe seeds, three extraction layers.
[Interactive local report](report.html) · [Metrics](metrics.json) · [Audit](audit.json)

## Outcome

At validation-selected block **18**, mean calibrated results across three seeds:

- Shared bilinear probe: **94.97% accuracy**,
  **0.0299 binary Brier** on held-out scenarios.
- Independent linear probes: **95.49% accuracy**,
  **0.0280 Brier**. The simpler baseline is slightly better.
- Paired Brier difference (shared minus independent): **+0.0019**,
  scenario-bootstrap 95% interval **[-0.0028, +0.0077]**.
  This interval does not support an advantage for either method on this small test.
- New prompt template: shared-head accuracy **78.65%**,
  Brier **0.1706**, ECE **0.1642**.
  Calibration does not transfer cleanly even to this mild shift.
- Held-out question paraphrases: **93.92% accuracy**.
  These are known properties, not unseen semantic questions.
- Shuffled activations: **47.05% accuracy**.
  Swapped questions: **49.48%**.
  The result depends on both the activation and question inputs.
- Full-prompt text baseline: **100.0% accuracy**.
  The prompt explicitly contains the settings. User-only and query-only baselines are at chance.

The shared head has 8,065 parameters versus 2,691 for separate linear probes. With
only three properties this experiment cannot establish a parameter-efficiency gain.
The target's greedy output matches the requested color on
100.0% of prompts; mean red+blue next-token
probability mass is 0.983.

## The promising observation

On 32 held-out prompts from four scenarios, at a fixed 5% activation-norm dose,
adding versus subtracting the **color** readout direction changes target red/blue
log odds by **+0.983** on average. Its scenario-bootstrap
95% interval is [+0.918, +1.085].
The largest absolute mean effect among eight equal-norm random directions is
**0.179**. Shape and animal direction effects are respectively
+0.124 and -0.088; the zero sham is
0.000.

This is an exploratory specificity signal, not a significance test against all possible
random directions, a calibrated prediction of intervention effects, or evidence of
natural causal necessity. Four scenario clusters and one seed are insufficient for
strong causal claims. Raw per-prompt outputs, probe probabilities, vocabulary KL and
color-token mass are in [interventions.json](interventions.json).

![Measured pilot plots](overview.png)

## Local resources

Capture: **12.5 seconds** including model load and query encoding.
CPU head fitting and evaluation: **5.9 seconds**.
Interventions: **19.6 seconds** including model reload.
Capture-process peak RSS: **2.77 GiB**; this excludes some
Metal/driver allocations and is **not total unified-memory use**. Compressed activation
cache: **3.99 MiB**. Short-run timings do not predict
sustained fanless-laptop throughput.

## Integrity and limitations

The audit recomputed 360 aggregate metric values from
4608 per-example prediction rows, checked scenario isolation,
validation-only layer selection, intervention arithmetic and the zero sham.
This validates artifact consistency, not scientific generalization.

All labels refer to explicit prompt settings. Only three known properties and one
held-out template are tested; text inversion is a confound. No real safety task,
unseen property family, second model, actual activation oracle or SAE comparison
was run. Training seeds share one fixed dataset. Bootstrap intervals are conditional
on these few scenario groups and are not corrected for multiple comparisons.
The experiment is exploratory, not preregistered.

## Reproduce

```bash
uv sync --frozen --extra dev --python 3.12
uv run latent-decisions all --offline --run runs/reproduction-001
uv run python scripts/audit_run.py runs/reproduction-001
```

Omit `--offline` if weights are not cached. The model revision, dependency versions,
source fingerprint and dataset/cache hashes are recorded in [manifest.json](manifest.json).
The manifest honestly records a dirty development worktree; use the source fingerprint
rather than treating its initial git commit as a released implementation.
Binary activations and weights are intentionally excluded. The dataset and predictions
are included, so `uv run python scripts/audit_run.py results/pilot` works without them.
`SHA256SUMS` records hashes of this bundle's measured artifacts.
