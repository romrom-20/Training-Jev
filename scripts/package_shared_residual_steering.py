"""Validate and package experiment 011's shared/residual intervention results."""

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
from shared_residual_steering import MODELS, load_directions

ARMS = ("original", "shared", "residual", "random_residual")


def validate(root, source_root):
    analysis = json.loads((root / "analysis.json").read_text())
    audit = {"checks_passed": True, "models": {}}
    geometry = {}
    for model in MODELS:
        folder = root / model
        manifest = json.loads((folder / "manifest.json").read_text())
        path = folder / "outcomes.json"
        outcomes = json.loads(path.read_text())
        source_manifest = json.loads((source_root / model / "manifest.json").read_text())
        checks = {
            "outcome_hash_matches_manifest": sha(path) == manifest["outcomes_sha256"],
            "row_count_matches_manifest": len(outcomes) == manifest["n_effects"] == 6912,
            "row_ids_unique": len({x["effect_id"] for x in outcomes}) == len(outcomes),
            "all_four_arms_balanced": all(sum(x["arm"] == arm for x in outcomes) == 1728 for arm in ARMS),
            "all_groups_present": len({x["group_no"] for x in outcomes}) == 24,
            "all_source_target_cells_balanced": all(
                sum(x["arm"] == arm and x["source"] == source and x["target"] == target for x in outcomes) == 192
                for arm in ARMS for source in range(3) for target in range(3)
            ),
            "protocol_hash_matches": manifest["protocol_sha256"] == sha(Path("docs/experiments/011-shared-vs-residual-steering.md")),
            "code_hash_matches": manifest["code_sha256"] == sha(Path("scripts/shared_residual_steering.py")),
            "source_dataset_hash_matches": manifest["source_dataset_sha256"] == sha(source_root / model / "dataset.json"),
            "source_activation_hash_matches": manifest["source_activation_sha256"] == source_manifest["activation_sha256"],
            "direction_seed_matches_protocol": manifest["direction_seed"] == 20260925,
            "dose_fraction_matches_protocol": manifest["dose_fraction"] == 0.05,
            "all_original_effects_positive": analysis[model]["all_original_effects_positive"],
            "run_commit_is_frozen_commit": manifest["provenance"]["git_commit"].startswith("6510853"),
            "analysis_model_matches": analysis[model]["model"] == model,
        }
        if not all(checks.values()):
            audit["checks_passed"] = False
        audit["models"][model] = {"checks": checks, "manifest": manifest}
        _, _, arms = load_directions(model, source_root)
        original = arms["original"].numpy()
        shared = arms["shared"].numpy()
        residual = arms["residual"].numpy()
        random_residual = arms["random_residual"].numpy()
        geometry[model] = {
            "direction_cosine_matrix": (original @ original.T).tolist(),
            "shared_projection_norm_fraction": (np.linalg.norm(shared, axis=1) / np.linalg.norm(original, axis=1)).tolist(),
            "residual_norm_fraction": (np.linalg.norm(residual, axis=1) / np.linalg.norm(original, axis=1)).tolist(),
            "random_residual_orthogonal_to_shared_max_abs_cosine": float(
                np.max(np.abs(random_residual @ arms["shared"][0].numpy() / (np.linalg.norm(random_residual, axis=1) * np.linalg.norm(arms["shared"][0].numpy()))))
            ),
        }
    if not audit["checks_passed"]:
        raise ValueError("011 integrity audit failed")
    return analysis, audit, geometry


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def plot_results(analysis, path):
    arms = ("original", "shared", "residual", "random_residual")
    labels = ["Original", "Shared component", "Task residual", "Random residual"]
    models = (("qwen-1.5b", "Qwen2.5-1.5B"), ("smollm2-1.7b", "SmolLM2-1.7B"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    for column, (model, title) in enumerate(models):
        overall = [analysis[model]["overall_mean_effect_by_arm"][arm] for arm in arms]
        selectivity = [analysis[model]["target_selectivity_by_arm"][arm] for arm in arms]
        x = np.arange(len(arms))
        for row, values, ylabel in ((0, overall, "Mean log-odds shift"), (1, selectivity, "Target selectivity")):
            ax = axes[row, column]
            bars = ax.bar(x, values, color=("#38598c", "#55a868", "#c44e52", "#999999"))
            ax.axhline(0, color="black", linewidth=.8)
            ax.set_xticks(x, labels if row == 1 else [""] * len(labels), rotation=25, ha="right")
            ax.set_ylabel(ylabel)
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(title)
    fig.suptitle("Overall shifts and target selectivity by intervention component")
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
    analysis, audit, geometry = validate(args.run_root, args.source_root)
    dump_json(args.output / "analysis.json", analysis)
    dump_json(args.output / "audit.json", audit)
    dump_json(args.output / "training-direction-geometry.json", geometry)
    for model in MODELS:
        with (args.output / f"{model}-outcomes.json.gz").open("wb") as stream:
            with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as zipped:
                zipped.write(json.dumps(json.loads((args.run_root / model / "outcomes.json").read_text()), separators=(",", ":")).encode())
        (args.output / f"{model}-manifest.json").write_bytes((args.run_root / model / "manifest.json").read_bytes())
    plot_results(analysis, args.output / "component-effects.png")
    lines = []
    for model in MODELS:
        result = analysis[model]
        rule_a = result["paired_group_bootstrap_95_ci"]["residual_minus_shared"]
        rule_b = result["paired_group_bootstrap_95_ci"]["residual_minus_random_residual"]
        lines.append(
            f"- **{model}:** selectivity original {result['target_selectivity_by_arm']['original']:+.4f}, "
            f"shared {result['target_selectivity_by_arm']['shared']:+.4f}, "
            f"residual {result['target_selectivity_by_arm']['residual']:+.4f}, "
            f"random residual {result['target_selectivity_by_arm']['random_residual']:+.4f}; "
            f"residual-minus-shared 95% CI [{rule_a[0]:+.4f}, {rule_a[1]:+.4f}], "
            f"residual-minus-random 95% CI [{rule_b[0]:+.4f}, {rule_b[1]:+.4f}]."
        )
    overall_lines = [
        f"- **{model}:** overall mean log-odds change was "
        + ", ".join(
            f"{arm.replace('_', ' ')} {analysis[model]['overall_mean_effect_by_arm'][arm]:+.4f}"
            for arm in ARMS
        )
        + "."
        for model in MODELS
    ]
    readme = """# Shared-versus-residual steering control (experiment 011)

This paired experiment decomposes three training-only task directions into a shared component and orthogonal residuals, then measures each on the same held-out prompts. Components retain their natural norms; random residual controls are norm-matched.

## Frozen decision rule

Residual-specific steering is supported only if residual target selectivity exceeds both the shared component and the matched random residual with paired 24-group bootstrap 95% intervals entirely above zero in both models. The analysis preserves all source-by-target cells and overall shifts. A failed criterion is reported as a negative or mixed local result, not repaired by selecting cells after inspection.

## Results

""" + "\n".join(lines) + """

The frozen rule **failed**: the residual-minus-random interval crosses zero for SmolLM2. The residual exceeded the shared component in both models, but the protocol requires both comparisons in both models.

Overall intervention shifts:

""" + "\n".join(overall_lines) + """

![Overall log-odds shifts and target selectivity by intervention component](component-effects.png)

""" + "\n".join(
        f"Training-direction geometry for **{model}** has pairwise cosine matrix `{np.array(geometry[model]['direction_cosine_matrix']).round(3).tolist()}`."
        for model in MODELS
    ) + """

These finite log-odds interventions concern a synthetic candidate-label task, not real-world behavior. The unit of uncertainty is 24 constructed scenario groups. This study is small, runs on two sub-2B models, and tests one layer and one dose. Even a positive result needs independent tasks and naturalistic outcomes before making a field-level claim.

Relevant prior work already studies intervention-side-effect prediction and geometric decompositions ([Ong et al. 2026](https://arxiv.org/html/2608.11227v1); [Aparin & Gaintseva 2026](https://arxiv.org/abs/2606.06735); [Shen et al. 2026](https://arxiv.org/abs/2606.08365)). The current result is a local component control within this benchmark. It does not establish a new general steering method. Protocol: [`011-shared-vs-residual-steering.md`](../../docs/experiments/011-shared-vs-residual-steering.md).

Reproduce analysis and package from the ignored local capture:

```bash
uv run python scripts/shared_residual_steering.py analyze --root runs/shared-residual-steering-v1
uv run python scripts/package_shared_residual_steering.py \\
  runs/shared-residual-steering-v1 results/shared-residual-steering-v1
```

The gzip archives retain every per-prompt effect; activations are excluded. `audit.json` checks integrity, row balance, group coverage and protocol provenance. `SHA256SUMS` covers the bundle.
"""
    (args.output / "README.md").write_text(readme)
    checksums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (args.output / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    print(f"Packaged validated 011 results in {args.output}")


if __name__ == "__main__":
    main()
