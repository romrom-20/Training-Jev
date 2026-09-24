"""Audit and package experiment 019 while excluding TripR review text."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_tripr_component_decomposition import analyze
from mams_layer_selectivity import directions_at_layer
from prompt_effect_forecast import sha
from tripr_component_decomposition import (
    CONDITIONS,
    LAYER,
    MODELS,
    PROTOCOL,
    RANDOM_SEED,
    tripr,
)

from latent_decisions.steering import decompose_shared_residual

RUNNER = Path("scripts/tripr_component_decomposition.py")
ANALYZER = Path("scripts/analyze_tripr_component_decomposition.py")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def audit(root, data_path):
    stimuli, labels, conflicts = tripr.load_tripr(data_path)
    prompt_ids = {stim["id"] for stim in stimuli}
    report = {"experiment": 19, "models": {}, "checks_passed": True}
    commits = set()
    for model in MODELS:
        folder = root / model
        baseline = json.loads((folder / "baseline.json").read_text())
        effects = json.loads((folder / "effects.json").read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        expected = {
            f"{stim_id}|{condition}|source={source}"
            for stim_id in prompt_ids
            for condition in CONDITIONS
            for source in range(3)
        }
        source_baseline = (
            Path("runs/tripr-layer-confirmation-v1") / "layer-16" / model / "baseline.json"
        )
        vectors, _dose, _activation_hash, _dataset_hash = directions_at_layer(
            model, LAYER, Path("runs/task-ladder-v1")
        )
        component_norms = {
            condition: [float(v.norm()) for v in matrix]
            for condition, matrix in decompose_shared_residual(
                vectors["native"], RANDOM_SEED
            ).items()
        }
        expected_norms = manifest["component_l2_norms"]
        norms_match = all(
            np.allclose(component_norms[condition], expected_norms[condition], rtol=0, atol=1e-7)
            for condition in CONDITIONS
        )
        checks = {
            "baseline_matches_experiment_018": sha(folder / "baseline.json") == sha(source_baseline),
            "baseline_prompt_ids_match": {row["id"] for row in baseline} == prompt_ids,
            "dataset_hash_matches": manifest["dataset_xml_sha256"] == sha(data_path),
            "protocol_hash_matches": manifest["protocol_sha256"] == sha(PROTOCOL),
            "runner_hash_matches": manifest["runner_sha256"] == sha(RUNNER),
            "direction_helper_hash_matches": manifest["direction_helper_sha256"]
            == sha(Path("scripts/mams_layer_selectivity.py")),
            "dataset_helper_hash_matches": manifest["dataset_helper_sha256"]
            == sha(Path("scripts/tripr_layer_confirmation.py")),
            "decomposition_code_hash_matches": manifest["decomposition_code_sha256"]
            == sha(Path("src/latent_decisions/steering.py")),
            "baseline_hash_matches": manifest["baseline_sha256"] == sha(folder / "baseline.json"),
            "effects_hash_matches": manifest["effects_sha256"] == sha(folder / "effects.json"),
            "source_activation_hash_matches": manifest["source_activation_sha256"]
            == json.loads((Path("runs/task-ladder-v1") / model / "manifest.json").read_text())["activation_sha256"],
            "prompt_count_matches": manifest["n_prompts"] == len(prompt_ids),
            "sentence_count_matches": manifest["n_sentences"] == len(labels),
            "conflict_count_matches": manifest["n_conflict_sentences"] == conflicts,
            "effect_ids_unique": len({row["effect_id"] for row in effects}) == len(effects),
            "factorial_complete": {row["effect_id"] for row in effects} == expected,
            "review_text_absent": all("user" not in row and "text" not in row for row in baseline + effects),
            "clean_capture_commit": not manifest["provenance"]["dirty_worktree"],
            "local_device": manifest["device"] in ("mps", "cpu"),
            "layer_and_dose_match_protocol": manifest["layer"] == 16
            and manifest["dose_fraction_of_training_activation_norm"] == 0.05,
            "component_norms_match_definition": norms_match,
        }
        commits.add(manifest["provenance"]["git_commit"])
        report["models"][model] = {"checks": checks, "manifest": manifest}
        report["checks_passed"] &= all(checks.values())
    report["same_capture_commit"] = len(commits) == 1
    report["checks_passed"] &= len(commits) == 1
    if not report["checks_passed"]:
        raise ValueError("019 capture audit failed")
    return report


def plot(analysis, path):
    names = {"qwen-1.5b": "Qwen2.5-1.5B", "smollm2-1.7b": "SmolLM2-1.7B"}
    colors = {"qwen-1.5b": "#38598c", "smollm2-1.7b": "#55a868"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = list(range(len(CONDITIONS)))
    for model in MODELS:
        row = analysis[model]
        means = [row["component_specificity"][c]["specificity_mean_logits"] for c in CONDITIONS]
        intervals = [row["component_specificity"][c]["specificity_95_ci"] for c in CONDITIONS]
        err = [[means[i] - intervals[i][0] for i in range(len(x))], [intervals[i][1] - means[i] for i in range(len(x))]]
        axes[0].errorbar(x, means, yerr=err, marker="o", capsize=4, label=names[model], color=colors[model])
        keys = ("residual_minus_shared", "residual_minus_random_residual")
        contrasts = [row["paired_component_comparisons"][k]["mean_difference"] for k in keys]
        cis = [row["paired_component_comparisons"][k]["paired_sentence_bootstrap_95_ci"] for k in keys]
        offset = -0.08 if model == "qwen-1.5b" else 0.08
        for j, (mean, ci) in enumerate(zip(contrasts, cis)):
            axes[1].errorbar(j + offset, mean, yerr=[[mean-ci[0]], [ci[1]-mean]], marker="o", capsize=4, color=colors[model], label=names[model] if j == 0 else None)
    axes[0].set_xticks(x, [c.replace("_", "\n") for c in CONDITIONS])
    axes[0].set_ylabel("Diagonal − off-diagonal shift (logits)")
    axes[0].set_title("Specificity by vector component")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks([0, 1], ["Residual − shared", "Residual − random\nresidual"])
    axes[1].set_ylabel("Paired specificity difference (logits)")
    axes[1].set_title("Preregistered component contrasts")
    axes[1].axhline(0, color="black", linewidth=0.8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("TripR-2020Large: where does layer-16 specificity live?")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/tripr-component-decomposition-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/tripr-component-decomposition-v1"))
    parser.add_argument("--data", type=Path, default=tripr.DATA)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    results = analyze(args.root)
    audit_report = audit(args.root, args.data)
    args.output.mkdir(parents=True)
    for model in MODELS:
        for name in ("baseline.json", "effects.json"):
            with (args.root / model / name).open("rb") as source, gzip.open(args.output / f"{model}-{name}.gz", "wb") as target:
                shutil.copyfileobj(source, target)
    write_json(args.output / "analysis.json", results)
    write_json(args.output / "audit.json", audit_report)
    plot(results, args.output / "component-specificity.png")
    attribution = (Path("results/tripr-layer-confirmation-v1") / "DATA_ATTRIBUTION.md").read_text()
    (args.output / "DATA_ATTRIBUTION.md").write_text(attribution)
    lines = [
        "# Where does layer-16 aspect specificity live? (experiment 019)",
        "",
        "This preregistered decomposition reuses the TripR-2020Large prompts from 018. It is a mechanism follow-up, not an independent replication. Frozen layer-16 training-only food/service/price directions were split into their shared-sentiment projections and orthogonal aspect residuals. Each natural component was applied at the same 5% training-activation-norm dose, with norm-matched random residual controls.",
        "",
    ]
    for model in MODELS:
        row = results[model]
        lines.append(f"## {model}")
        lines.append("")
        lines.append(f"Baseline strict accuracy: {row['baseline_strict_accuracy']:.1%}; {row['n_sentences']} sentences, {row['n_conflict_sentences']} polarity-conflict sentences.")
        for condition in CONDITIONS:
            item = row["component_specificity"][condition]
            ci = item["specificity_95_ci"]
            lines.append(f"- **{condition}:** specificity {item['specificity_mean_logits']:+.6f} logits (95% CI [{ci[0]:+.6f}, {ci[1]:+.6f}]); generic shift {item['generic_margin_shift_mean_logits']:+.4f}; strict accuracy {item['steered_strict_accuracy']:.1%}.")
        for key, label in (("residual_minus_shared", "residual − shared"), ("residual_minus_random_residual", "residual − random residual")):
            item = row["paired_component_comparisons"][key]
            ci = item["paired_sentence_bootstrap_95_ci"]
            lines.append(f"- **{label}:** {item['mean_difference']:+.6f} logits (95% paired sentence-bootstrap CI [{ci[0]:+.6f}, {ci[1]:+.6f}]).")
        lines.append("")
    lines += [
        f"**Residual-mechanism gate:** {'PASS' if results['cross_model_residual_mechanism_gate'] else 'FAIL'}.",
        "",
        "![Component specificity estimates](component-specificity.png)",
        "",
        "Raw TripR review text is excluded. The local dataset is attributed under CC BY-SA 4.0 in `DATA_ATTRIBUTION.md`. See the [preregistered protocol](../../docs/experiments/019-tripr-shared-residual-layer16.md).",
        "",
    ]
    (args.output / "README.md").write_text("\n".join(lines))
    sums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (args.output / "SHA256SUMS").write_text(sums)
    print(f"Packaged experiment 019 in {args.output}; audit passed={audit_report['checks_passed']}")


if __name__ == "__main__":
    main()
