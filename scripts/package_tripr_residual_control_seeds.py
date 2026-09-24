"""Audit and package the multi-seed TripR residual-control experiment."""

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
from analyze_tripr_residual_control_seeds import analyze
from mams_layer_selectivity import directions_at_layer
from prompt_effect_forecast import sha
from tripr_layer_confirmation import load_tripr
from tripr_residual_control_seeds import MODELS, PROTOCOL, SEEDS

from latent_decisions.steering import decompose_shared_residual

RUNNER = Path("scripts/tripr_residual_control_seeds.py")
ANALYZER = Path("scripts/analyze_tripr_residual_control_seeds.py")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def audit(root, data_path, analysis):
    stimuli, labels, _ = load_tripr(data_path)
    conflict_ids = {
        sid for sid, groups in labels.items() if len(set(groups.values())) > 1
    }
    prompts = [row for row in stimuli if row["sentence_id"] in conflict_ids]
    prompt_ids = {row["id"] for row in prompts}
    report = {
        "experiment": 20,
        "analysis_code_sha256": sha(ANALYZER),
        "analysis_recomputes": analyze(root) == analysis,
        "checks_passed": True,
        "models": {},
    }
    commits = set()
    for model in MODELS:
        folder = root / model
        manifest = json.loads((folder / "manifest.json").read_text())
        baseline = json.loads((folder / "baseline.json").read_text())
        effects = json.loads((folder / "effects.json").read_text())
        expected_ids = {
            f"{prompt_id}|seed={seed}|source={source}"
            for prompt_id in prompt_ids
            for seed in ("trained", *SEEDS)
            for source in range(3)
        }
        source_vectors, _dose, _activation_hash, _dataset_hash = directions_at_layer(
            model, 16, Path("runs/task-ladder-v1")
        )
        source_baseline = json.loads(
            (Path("runs/tripr-layer-confirmation-v1") / "layer-16" / model / "baseline.json").read_text()
        )
        expected_baseline = [row for row in source_baseline if row["id"] in prompt_ids]
        residual_norms = decompose_shared_residual(
            source_vectors["native"], seed=20261001
        )["residual"].norm(dim=1).numpy()
        checks = {
            "dataset_hash_matches": manifest["dataset_xml_sha256"] == sha(data_path),
            "protocol_hash_matches": manifest["protocol_sha256"] == sha(PROTOCOL),
            "runner_hash_matches": manifest["runner_sha256"] == sha(RUNNER),
            "baseline_hash_matches": manifest["baseline_sha256"] == sha(folder / "baseline.json"),
            "effects_hash_matches": manifest["effects_sha256"] == sha(folder / "effects.json"),
            "source_activation_hash_matches": manifest["source_activation_sha256"]
            == json.loads((Path("runs/task-ladder-v1") / model / "manifest.json").read_text())["activation_sha256"],
            "prompt_ids_match_conflict_filter": {row["id"] for row in baseline} == prompt_ids,
            "baseline_matches_018_conflict_slice": baseline == expected_baseline,
            "sentence_count_matches": len({row["sentence_id"] for row in baseline}) == len(conflict_ids),
            "seed_set_matches_protocol": tuple(manifest["random_control_seeds"]) == SEEDS,
            "effect_ids_unique": len({row["effect_id"] for row in effects}) == len(effects),
            "factorial_complete": {row["effect_id"] for row in effects} == expected_ids,
            "review_text_absent": all("user" not in row and "text" not in row for row in baseline + effects),
            "clean_capture_commit": not manifest["provenance"]["dirty_worktree"],
            "local_device": manifest["device"] in ("mps", "cpu"),
            "layer_and_dose_match_protocol": manifest["layer"] == 16
            and manifest["dose_fraction_of_training_activation_norm"] == 0.05,
            "matched_residual_norms": bool(
                np.allclose(residual_norms, np.asarray(manifest["residual_l2_norms"]), rtol=0, atol=1e-7)
            ),
        }
        commits.add(manifest["provenance"]["git_commit"])
        report["models"][model] = {"checks": checks, "manifest": manifest}
        report["checks_passed"] &= all(checks.values())
    report["same_capture_commit"] = len(commits) == 1
    report["checks_passed"] &= len(commits) == 1 and report["analysis_recomputes"]
    if not report["checks_passed"]:
        raise ValueError("020 capture audit failed")
    return report


def plot(analysis, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    positions = {"qwen-1.5b": 0, "smollm2-1.7b": 1}
    for model, color, name in (
        ("qwen-1.5b", "#38598c", "Qwen2.5-1.5B"),
        ("smollm2-1.7b", "#55a868", "SmolLM2-1.7B"),
    ):
        row = analysis[model]
        values = list(row["random_seed_specificity_by_seed"].values())
        x = np.full(len(values), positions[model], dtype=float)
        ax.scatter(x, values, color=color, alpha=0.65, s=24, label=f"{name} random controls")
        y = row["trained_residual_specificity_mean_logits"]
        ax.scatter([positions[model]], [y], marker="D", color="#c44e52", s=68, zorder=3, label=f"{name} trained residual")
        ax.hlines(np.mean(values), positions[model]-0.18, positions[model]+0.18, color=color, linewidth=2)
    ax.set_xticks([0, 1], ["Qwen2.5-1.5B", "SmolLM2-1.7B"])
    ax.set_ylabel("Conflict-only diagonal − off-diagonal shift (logits)")
    ax.set_title("Trained residual versus 20 norm-matched random controls")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/tripr-residual-control-seeds-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/tripr-residual-control-seeds-v1"))
    parser.add_argument("--data", type=Path, default=Path(".context/datasets/OD-TripR-2020Large/TripR-2020Large_AnnotatedReviews.xml"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    analysis = analyze(args.root)
    audit_report = audit(args.root, args.data, analysis)
    args.output.mkdir(parents=True)
    for model in MODELS:
        for name in ("baseline.json", "effects.json"):
            with (args.root / model / name).open("rb") as src, gzip.open(args.output / f"{model}-{name}.gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
    write_json(args.output / "analysis.json", analysis)
    write_json(args.output / "audit.json", audit_report)
    plot(analysis, args.output / "random-control-seeds.png")
    attribution = Path("results/tripr-layer-confirmation-v1/DATA_ATTRIBUTION.md").read_text()
    (args.output / "DATA_ATTRIBUTION.md").write_text(attribution)
    lines = [
        "# Layer-16 residual against random controls (experiment 020)",
        "",
        "Experiment 020 tests whether the layer-16 learned aspect residual exceeds the distribution of 20 independent, norm-matched random residual controls on the 29 polarity-conflict TripR sentences (all 63 mapped queries). This is a seed-robustness check on the 018/019 benchmark, not an independent dataset replication.",
        "",
    ]
    for model in MODELS:
        row = analysis[model]
        ci = row["nested_sentence_seed_bootstrap_95_ci"]
        lines += [
            f"## {model}",
            "",
            f"Baseline strict accuracy on conflict queries: {row['baseline_strict_accuracy']:.1%}.",
            f"- Learned residual specificity: {row['trained_residual_specificity_mean_logits']:+.6f} logits.",
            f"- Random-seed mean: {row['random_seed_specificity_mean_logits']:+.6f} logits; seed range [{row['random_seed_specificity_range'][0]:+.6f}, {row['random_seed_specificity_range'][1]:+.6f}].",
            f"- Learned minus random mean: {row['trained_minus_random_seed_mean']:+.6f} (nested sentence/seed bootstrap 95% CI [{ci[0]:+.6f}, {ci[1]:+.6f}]).",
            f"- One-sided Monte Carlo rank p: {row['monte_carlo_one_sided_p']:.4f}.",
            "",
        ]
    lines += [
        f"**Cross-model control-robustness gate:** {'PASS' if analysis['cross_model_control_robustness_gate'] else 'FAIL'}.",
        "",
        "![Seed-ensemble comparison](random-control-seeds.png)",
        "",
        "All 20 control seeds and per-example outcomes are retained in the compressed files. Review text is omitted. Attribution and license are in `DATA_ATTRIBUTION.md`; checks are in `audit.json` and `SHA256SUMS`.",
        "",
    ]
    (args.output / "README.md").write_text("\n".join(lines))
    checksums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (args.output / "SHA256SUMS").write_text(checksums)
    print(f"Packaged experiment 020 in {args.output}; audit passed={audit_report['checks_passed']}")


if __name__ == "__main__":
    main()
