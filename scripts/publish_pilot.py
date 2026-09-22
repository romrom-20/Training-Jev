"""Copy a completed, audited run into the repository's reviewable pilot bundle."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def publish(source, dest):
    audit = json.loads((source / "audit.json").read_text())
    if audit["status"] != "passed":
        raise ValueError("Audit must pass before publishing a result bundle")
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("Destination must be empty to preserve existing evidence")
    dest.mkdir(parents=True, exist_ok=True)
    names = (
        "dataset.json",
        "manifest.json",
        "metrics.json",
        "predictions.json",
        "interventions.json",
        "audit.json",
        "overview.png",
        "report.html",
    )
    for name in names:
        shutil.copy2(source / name, dest / name)
    m = json.loads((dest / "manifest.json").read_text())
    r = json.loads((dest / "metrics.json").read_text())
    c = json.loads((dest / "interventions.json").read_text())
    layer = r["selected_layer"]

    def mean(method, split, metric):
        return np.mean(
            [
                x["evaluations"][split]["calibrated"][metric]
                for x in r["records"]
                if x["layer"] == layer and x["method"] == method
            ]
        )

    ci = r["paired_brier_difference"]
    random_max = max(abs(v["mean"]) for k, v in c["summary"].items() if k.startswith("random_"))
    collection = m["collection"]
    text = f"""# Pilot 001 — what the evidence supports

**Completed {m["collected_at"][:10]} on a 24 GB Apple M5 MacBook Air.**
Frozen Qwen2.5-0.5B-Instruct; 416 prompts, three probe seeds, three extraction layers.
[Interactive local report](report.html) · [Metrics](metrics.json) · [Audit](audit.json)

## Outcome

At validation-selected block **{layer}**, mean calibrated results across three seeds:

- Shared bilinear probe: **{mean("bilinear", "test", "accuracy"):.2%} accuracy**,
  **{mean("bilinear", "test", "brier"):.4f} binary Brier** on held-out scenarios.
- Independent linear probes: **{mean("independent_linear", "test", "accuracy"):.2%} accuracy**,
  **{mean("independent_linear", "test", "brier"):.4f} Brier**. The simpler baseline is slightly better.
- Paired Brier difference (shared minus independent): **{ci["mean"]:+.4f}**,
  scenario-bootstrap 95% interval **[{ci["low"]:+.4f}, {ci["high"]:+.4f}]**.
  This interval does not support an advantage for either method on this small test.
- New prompt template: shared-head accuracy **{mean("bilinear", "ood", "accuracy"):.2%}**,
  Brier **{mean("bilinear", "ood", "brier"):.4f}**, ECE **{mean("bilinear", "ood", "ece"):.4f}**.
  Calibration does not transfer cleanly even to this mild shift.
- Held-out question paraphrases: **{mean("bilinear", "paraphrase", "accuracy"):.2%} accuracy**.
  These are known properties, not unseen semantic questions.
- Shuffled activations: **{mean("bilinear", "shuffled_activations", "accuracy"):.2%} accuracy**.
  Swapped questions: **{mean("bilinear", "swapped_queries", "accuracy"):.2%}**.
  The result depends on both the activation and question inputs.
- Full-prompt text baseline: **{r["text_baselines"]["full_prompt_text"]["test"]["calibrated"]["accuracy"]:.1%} accuracy**.
  The prompt explicitly contains the settings. User-only and query-only baselines are at chance.

The shared head has 8,065 parameters versus 2,691 for separate linear probes. With
only three properties this experiment cannot establish a parameter-efficiency gain.
The target's greedy output matches the requested color on
{collection["target_greedy_color_accuracy"]:.1%} of prompts; mean red+blue next-token
probability mass is {collection["mean_color_token_mass"]:.3f}.

## The promising observation

On {c["prompts"]} held-out prompts from four scenarios, at a fixed 5% activation-norm dose,
adding versus subtracting the **color** readout direction changes target red/blue
log odds by **{c["summary"]["color"]["mean"]:+.3f}** on average. Its scenario-bootstrap
95% interval is [{c["summary"]["color"]["low"]:+.3f}, {c["summary"]["color"]["high"]:+.3f}].
The largest absolute mean effect among eight equal-norm random directions is
**{random_max:.3f}**. Shape and animal direction effects are respectively
{c["summary"]["shape"]["mean"]:+.3f} and {c["summary"]["animal"]["mean"]:+.3f}; the zero sham is
{c["summary"]["zero"]["mean"]:.3f}.

This is an exploratory specificity signal, not a significance test against all possible
random directions, a calibrated prediction of intervention effects, or evidence of
natural causal necessity. Four scenario clusters and one seed are insufficient for
strong causal claims. Raw per-prompt outputs, probe probabilities, vocabulary KL and
color-token mass are in [interventions.json](interventions.json).

![Measured pilot plots](overview.png)

## Local resources

Capture: **{collection["seconds"]:.1f} seconds** including model load and query encoding.
CPU head fitting and evaluation: **{r["training_seconds"]:.1f} seconds**.
Interventions: **{c["seconds"]:.1f} seconds** including model reload.
Capture-process peak RSS: **{collection["peak_process_rss_gib"]:.2f} GiB**; this excludes some
Metal/driver allocations and is **not total unified-memory use**. Compressed activation
cache: **{collection["activation_cache_mib"]:.2f} MiB**. Short-run timings do not predict
sustained fanless-laptop throughput.

## Integrity and limitations

The audit recomputed {audit["recomputed_metric_checks"]} aggregate metric values from
{audit["prediction_rows"]} per-example prediction rows, checked scenario isolation,
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
"""
    (dest / "README.md").write_text(text)
    (dest / "SHA256SUMS").write_text(
        "".join(f"{hashlib.sha256((dest / n).read_bytes()).hexdigest()}  {n}\n" for n in names)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    publish(args.source, args.destination)
