"""Audit and package experiment 012's cross-task shared-direction outcomes."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from prompt_effect_forecast import sha
from shared_residual_steering import MODELS

TASKS = ("mixed_review", "isolated_clause", "neutral_distractors", "keyed_record")
PRIMARY_TASKS = ("isolated_clause", "neutral_distractors", "keyed_record")
PROMPT_COUNTS = {"mixed_review": 576, "isolated_clause": 144, "neutral_distractors": 144, "keyed_record": 576}


def validate(root, source_root):
    analysis = json.loads((root / "analysis.json").read_text())
    audit = {"checks_passed": True, "all_six_primary_comparisons_pass": True, "models": {}}
    for model in MODELS:
        folder = root / model
        manifest = json.loads((folder / "manifest.json").read_text())
        source_manifest = json.loads((source_root / model / "manifest.json").read_text())
        path = folder / "outcomes.json"
        outcomes = json.loads(path.read_text())
        task_counts = {task: sum(x["task"] == task for x in outcomes) // 5 for task in TASKS}
        checks = {
            "outcome_hash_matches": manifest["outcomes_sha256"] == sha(path),
            "row_count_matches": len(outcomes) == manifest["n_effects"] == 7200,
            "ids_unique": len({x["effect_id"] for x in outcomes}) == len(outcomes),
            "all_task_prompt_counts_match": task_counts == PROMPT_COUNTS,
            "five_conditions_per_prompt": all(sum(x["base_id"] == base for x in outcomes) == 5 for base in {x["base_id"] for x in outcomes}),
            "24_groups_per_task": all(len({x["group_no"] for x in outcomes if x["task"] == task}) == 24 for task in TASKS),
            "protocol_hash_matches": manifest["protocol_sha256"] == sha(Path("docs/experiments/012-cross-task-shared-shift.md")),
            "code_hash_matches": manifest["code_sha256"] == sha(Path("scripts/cross_task_shared_shift.py")),
            "source_dataset_hash_matches": manifest["source_dataset_sha256"] == sha(source_root / model / "dataset.json"),
            "source_activation_hash_matches": manifest["source_activation_sha256"] == source_manifest["activation_sha256"],
            "random_seed_matches": manifest["random_seed"] == 20260927,
            "frozen_commit_matches": manifest["provenance"]["git_commit"].startswith("5c95e56"),
        }
        if not all(checks.values()):
            audit["checks_passed"] = False
        pass_by_task = {
            task: analysis[model]["tasks"][task]["shared_minus_random_group_bootstrap_95_ci"][0] > 0
            for task in PRIMARY_TASKS
        }
        if not all(pass_by_task.values()):
            audit["all_six_primary_comparisons_pass"] = False
        audit["models"][model] = {
            "checks": checks,
            "primary_task_rules": pass_by_task,
            "manifest": manifest,
        }
    if not audit["checks_passed"]:
        raise ValueError("012 integrity audit failed")
    return analysis, audit


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def make_plot(analysis, path):
    xlabels = ["Isolated", "Neutral", "Keyed", "Mixed"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)
    for ax, model, title in zip(axes, MODELS, ("Qwen2.5-1.5B", "SmolLM2-1.7B")):
        order = ["isolated_clause", "neutral_distractors", "keyed_record", "mixed_review"]
        x = np.arange(len(order))
        conditions = (
            ("shared", "Shared component", "#38598c"),
            ("random", "Random control", "#c44e52"),
            ("native_mean_over_sources", "Native direction mean", "#55a868"),
        )
        width = 0.24
        for i, (key, label, color) in enumerate(conditions):
            values = [analysis[model]["tasks"][task]["mean_effect"][key] for task in order]
            ax.bar(x + (i - 1) * width, values, width, label=label, color=color)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_title(title)
        ax.set_xticks(x, xlabels, rotation=20, ha="right")
        ax.set_ylabel("Positive-minus-negative log-odds change")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Shared-direction effect by task structure")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path("runs/task-ladder-v1"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    analysis, audit = validate(args.run_root, args.source_root)
    write_json(args.output / "analysis.json", analysis)
    write_json(args.output / "audit.json", audit)
    for model in MODELS:
        with (args.output / f"{model}-outcomes.json.gz").open("wb") as stream:
            with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as archive:
                archive.write(json.dumps(json.loads((args.run_root / model / "outcomes.json").read_text()), separators=(",", ":")).encode())
        (args.output / f"{model}-manifest.json").write_bytes((args.run_root / model / "manifest.json").read_bytes())
    make_plot(analysis, args.output / "cross-task-effects.png")
    lines = []
    for model in MODELS:
        lines.append(f"**{model}**")
        for task in TASKS:
            result = analysis[model]["tasks"][task]
            ci = result["shared_minus_random_group_bootstrap_95_ci"]
            lines.append(
                f"- {task}: shared {result['mean_effect']['shared']:+.4f}, random {result['mean_effect']['random']:+.4f}, "
                f"native mean {result['mean_effect']['native_mean_over_sources']:+.4f}; "
                f"shared−random 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]."
            )
    readme = """# Cross-task shared-score shift (experiment 012)

## Frozen decision rule

Transfer beyond mixed reviews requires the shared component to exceed a norm-matched random direction with paired 24-group bootstrap 95% intervals entirely above zero in all three other task structures for both model families. All six comparisons are required. The mixed-review result is descriptive because it was already examined in 010–011.

## Results

""" + "\n".join(lines) + """

""" + ("**The frozen all-six rule passed.**" if audit["all_six_primary_comparisons_pass"] else "**The frozen all-six rule failed.**") + """

![Shared, random and native intervention shifts across task structures](cross-task-effects.png)

This measures finite positive-minus-negative next-token log-odds changes on synthetic prompts. It does not measure generated behavior or prove that the model uses a general sentiment mechanism. The study reuses the 008 task ladder and its final-test group construction; it is a held-out-structure extension, not an independent replication. A pass warrants follow-up on natural ABSA examples and remapped output tokens, not a broad control claim.

The closest literature includes behavior-level side-effect prediction and intervention-encoding sensitivity ([Ong et al. 2026](https://arxiv.org/html/2608.11227v1); [Gao et al. 2026](https://arxiv.org/html/2608.22985v1)). This local experiment evaluates a small-model task-structure transfer question. Protocol: [`012-cross-task-shared-shift.md`](../../docs/experiments/012-cross-task-shared-shift.md).

Reproduce analysis and packaging from local captures:

```bash
uv run python scripts/cross_task_shared_shift.py analyze --root runs/cross-task-shared-shift-v1
uv run python scripts/package_cross_task_shared_shift.py \\
  runs/cross-task-shared-shift-v1 results/cross-task-shared-shift-v1
```

The gzip archives preserve per-prompt outcomes; activations are excluded. The audit checks hashes, row counts, task balance and frozen protocol provenance. `SHA256SUMS` covers the bundle.
"""
    (args.output / "README.md").write_text(readme)
    checksums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (args.output / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    print(f"Packaged audited 012 results in {args.output}")


if __name__ == "__main__":
    main()
