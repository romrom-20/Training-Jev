"""Audit and package experiment 021 without review text or generated text."""

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
from analyze_granite_tripr_residual_transfer import analyze
from granite_tripr_residual_transfer import (
    CONDITIONS,
    CONTROL_SEEDS,
    LAYER,
    MODEL,
    PROTOCOL,
    training_rows,
)
from prompt_effect_forecast import sha
from tripr_layer_confirmation import load_tripr

RUNNER = Path("scripts/granite_tripr_residual_transfer.py")
ANALYZER = Path("scripts/analyze_granite_tripr_residual_transfer.py")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def audit(root, data_path, analysis):
    folder = root / "evaluation"
    baseline = json.loads((folder / "baseline.json").read_text())
    effects = json.loads((folder / "effects.json").read_text())
    random_effects = json.loads((folder / "random_seed_effects.json").read_text())
    generations = json.loads((folder / "generation.json").read_text())
    manifest = json.loads((folder / "manifest.json").read_text())
    stimuli, labels, _ = load_tripr(data_path)
    conflict_ids = {
        sid for sid, groups in labels.items() if len(set(groups.values())) > 1
    }
    conflict_stimuli = [row for row in stimuli if row["sentence_id"] in conflict_ids]
    prompt_ids = {row["id"] for row in stimuli}
    train_rows = json.loads((root / "training" / "dataset.json").read_text())
    source_training_rows = training_rows()
    expected_effects = {
        f"{stim['id']}|{condition}|source={source}"
        for stim in stimuli
        for condition in CONDITIONS
        for source in range(3)
    }
    expected_random = {
        f"{stim['id']}|seed={seed}|source={source}"
        for stim in conflict_stimuli
        for seed in CONTROL_SEEDS
        for source in range(3)
    }
    generation_ids = {row["effect_id"] for row in generations}
    expected_generation_ids = {
        f"{stim['id']}|{condition}"
        for stim in conflict_stimuli
        for condition in ("baseline", "trained_residual", "random_residual")
    }
    training_manifest = json.loads((root / "training" / "manifest.json").read_text())
    with np.load(root / "training" / "activations.npz", allow_pickle=False) as cache:
        activation_shape = list(cache["h"].shape)
    checks = {
        "pinned_third_family_model": manifest["model"] == MODEL,
        "granite_40_layer_depth_and_layer_match_protocol": manifest["model_layers"] == 40
        and manifest["layer"] == LAYER
        and abs(manifest["relative_depth"] - 0.575) < 1e-9,
        "tripr_data_hash_matches": manifest["dataset_xml_sha256"] == sha(data_path),
        "protocol_hash_matches": manifest["protocol_sha256"] == sha(PROTOCOL),
        "runner_hash_matches": manifest["runner_sha256"] == sha(RUNNER),
        "tripr_prompt_filter_matches": {row["id"] for row in baseline} == prompt_ids,
        "baseline_count_matches": len(baseline) == 385,
        "training_rows_match_frozen_filter": train_rows == source_training_rows,
        "training_data_hash_matches": training_manifest["dataset_sha256"]
        == sha(root / "training" / "dataset.json"),
        "training_activation_hash_matches": training_manifest["activation_sha256"]
        == sha(root / "training" / "activations.npz")
        and manifest["training_activation_sha256"] == training_manifest["activation_sha256"],
        "training_activations_shape_matches": activation_shape[0] == len(train_rows)
        and activation_shape[1] == 1,
        "all_score_conditions_present": {row["effect_id"] for row in effects}
        == expected_effects,
        "all_random_control_seeds_present": {row["effect_id"] for row in random_effects}
        == expected_random,
        "generated_output_factorial_complete": generation_ids == expected_generation_ids,
        "expected_conflict_counts": len(conflict_ids) == 29
        and len(conflict_stimuli) == 63
        and manifest["n_conflict_sentences"] == 29
        and manifest["n_conflict_queries"] == 63,
        "raw_review_and_generation_text_absent": all(
            "user" not in row and "text" not in row and "answer" not in row
            for row in baseline + effects + random_effects + generations
        ),
        "clean_capture_commit": not manifest["provenance"]["dirty_worktree"],
        "local_device": manifest["device"] in ("mps", "cpu"),
        "dose_matches_protocol": manifest["dose_fraction_of_training_activation_norm"] == 0.05,
        "analysis_recomputes": analyze(root) == analysis,
    }
    report = {
        "experiment": 21,
        "checks": checks,
        "checks_passed": all(checks.values()),
        "analysis_code_sha256": sha(ANALYZER),
        "model_manifest": manifest,
    }
    if not report["checks_passed"]:
        raise ValueError("021 integrity audit failed")
    return report


def plot(analysis, path):
    conditions = ("native", "shared", "residual", "random_residual")
    means = [analysis["score_specificity_by_condition"][c]["mean_logits"] for c in conditions]
    intervals = [analysis["score_specificity_by_condition"][c]["sentence_bootstrap_95_ci"] for c in conditions]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    x = np.arange(len(conditions))
    axes[0].errorbar(
        x,
        means,
        yerr=[[means[i] - intervals[i][0] for i in range(len(x))], [intervals[i][1] - means[i] for i in range(len(x))]],
        marker="o",
        capsize=4,
        color="#38598c",
    )
    axes[0].set_xticks(x, [c.replace("_", "\n") for c in conditions])
    axes[0].set_ylabel("Diagonal − off-diagonal shift (logits)")
    axes[0].set_title("Candidate-score specificity")
    axes[0].axhline(0, color="black", linewidth=0.8)

    behavior = analysis["generated_answer_behavior"]
    names = ("baseline", "trained_residual", "random_residual")
    values = [behavior[name]["strict_gold_accuracy"] for name in names]
    axes[1].bar(np.arange(3), values, color=("#999999", "#55a868", "#8172b2"))
    axes[1].set_xticks(np.arange(3), ["baseline", "trained\nresidual", "random\nresidual"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Exact generated-answer accuracy")
    axes[1].set_title("Conflict-query generation check")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Granite 3.1 2B: layer-23 residual transfer")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/granite-tripr-residual-transfer-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/granite-tripr-residual-transfer-v1"))
    parser.add_argument("--data", type=Path, default=Path(".context/datasets/OD-TripR-2020Large/TripR-2020Large_AnnotatedReviews.xml"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    result = analyze(args.root)
    audit_report = audit(args.root, args.data, result)
    args.output.mkdir(parents=True)
    for source_name in (
        "baseline.json",
        "effects.json",
        "random_seed_effects.json",
        "generation.json",
    ):
        with (args.root / "evaluation" / source_name).open("rb") as source, gzip.open(
            args.output / f"{source_name}.gz", "wb"
        ) as target:
            shutil.copyfileobj(source, target)
    write_json(args.output / "analysis.json", result)
    write_json(args.output / "audit.json", audit_report)
    plot(result, args.output / "granite-residual-transfer.png")
    (args.output / "DATA_ATTRIBUTION.md").write_text(
        Path("results/tripr-layer-confirmation-v1/DATA_ATTRIBUTION.md").read_text()
    )
    gen = result["generated_answer_behavior"]
    ci = result["conflict_nested_sentence_seed_bootstrap_95_ci"]
    lines = [
        "# Granite 3.1 2B residual transfer (experiment 021)",
        "",
        "This third-family study trains food/service/value directions on synthetic training rows, then applies their layer-23 shared and residual components to TripR-2020Large. It uses 20 norm-matched random residual seeds on the 29 polarity-conflict sentences and a greedy exact-answer check. Review and generated text are omitted from this package.",
        "",
        f"- Full-set baseline strict accuracy: {result['baseline_strict_accuracy_all']:.1%}.",
        f"- Conflict-set baseline strict accuracy: {result['baseline_strict_accuracy_conflicts']:.1%}.",
        f"- Conflict trained residual specificity: {result['conflict_trained_residual_specificity_mean']:+.6f} logits.",
        f"- Conflict random-seed mean: {result['conflict_random_seed_specificity_mean']:+.6f}; trained-minus-control {result['conflict_trained_minus_random_mean']:+.6f} (95% nested CI [{ci[0]:+.6f}, {ci[1]:+.6f}]).",
        f"- Monte Carlo rank p: {result['conflict_monte_carlo_one_sided_p']:.4f}; score gate: {'PASS' if result['score_gate'] else 'FAIL'}.",
        "",
        "Generated answer behavior on the conflict queries:",
    ]
    for condition in ("baseline", "trained_residual", "random_residual"):
        row = gen[condition]
        lines.append(
            f"- **{condition}:** exact one-word rate {row['exact_one_word_rate']:.1%}; strict gold accuracy {row['strict_gold_accuracy']:.1%} (n={row['n']})."
        )
    lines += [
        "",
        "![Granite score and generation results](granite-residual-transfer.png)",
        "",
        "This tests one checkpoint from a third family, at one layer and dose, on the reused TripR benchmark. It is not by itself a general cross-model result. The [protocol](../../docs/experiments/021-granite-tripr-residual-transfer.md), per-seed outcomes, provenance, and integrity checks are included.",
        "",
    ]
    (args.output / "README.md").write_text("\n".join(lines))
    checksums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (args.output / "SHA256SUMS").write_text(checksums)
    print(f"Packaged experiment 021 in {args.output}; audit passed={audit_report['checks_passed']}")


if __name__ == "__main__":
    main()
