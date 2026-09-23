"""Integrity-check and package experiment 014 without review text."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_natural_aspect_selectivity import augment_per_category
from natural_aspect_selectivity import CONDITIONS, MODELS, analyze_model, load_stimuli
from prompt_effect_forecast import sha

PROTOCOL = Path("docs/experiments/014-natural-aspect-selectivity.md")
CODE = Path("scripts/natural_aspect_selectivity.py")
ANALYSIS_CODE = Path("scripts/analyze_natural_aspect_selectivity.py")


def audit(root, source_root, data_path):
    results = {"checks_passed": True, "models": {}, "dataset_sha256": sha(data_path)}
    results["analysis_code_sha256"] = sha(ANALYSIS_CODE)
    expected_stimuli = load_stimuli(data_path)
    expected_ids = {row["id"] for row in expected_stimuli}
    expected_effects = {
        f"{stimulus_id}|{condition}|{source}"
        for stimulus_id in expected_ids
        for condition in CONDITIONS
        for source in range(3)
    }
    results["n_stimuli"] = len(expected_stimuli)
    results["n_sentences"] = len({row["sentence_id"] for row in expected_stimuli})
    results["n_effects_per_model"] = len(expected_effects)
    for model in MODELS:
        folder = root / model
        manifest = json.loads((folder / "manifest.json").read_text())
        baseline_path, effects_path = folder / "baseline.json", folder / "effects.json"
        baseline = json.loads(baseline_path.read_text())
        effects = json.loads(effects_path.read_text())
        source_manifest = json.loads((source_root / model / "manifest.json").read_text())
        effect_ids = {row["effect_id"] for row in effects}
        baseline_ids = {row["id"] for row in baseline}
        checks = {
            "baseline_hash_matches": manifest["baseline_sha256"] == sha(baseline_path),
            "effects_hash_matches": manifest["effects_sha256"] == sha(effects_path),
            "protocol_hash_matches": manifest["protocol_sha256"] == sha(PROTOCOL),
            "runner_hash_matches": manifest["code_sha256"] == sha(CODE),
            "gold_dataset_hash_matches": manifest["gold_xml_sha256"] == sha(data_path),
            "source_dataset_hash_matches": manifest["source_dataset_sha256"]
            == sha(source_root / model / "dataset.json"),
            "source_activation_hash_matches": manifest["source_activation_sha256"]
            == source_manifest["activation_sha256"],
            "prompt_ids_match_frozen_filter": baseline_ids == expected_ids,
            "baseline_ids_unique": len(baseline) == len(baseline_ids),
            "all_effect_ids_unique": len(effects) == len(effect_ids),
            "effect_ids_match_frozen_factorial": effect_ids == expected_effects,
            "counts_match_manifest": len(baseline) == manifest["n_gold_prompts"]
            and len(effects) == manifest["n_gold_prompts"] * 9,
            "raw_text_absent_from_outcomes": all(
                "user" not in row and "text" not in row for row in baseline + effects
            ),
            "manifest_has_git_commit": bool(manifest["provenance"]["git_commit"]),
            "device_is_bounded_local_target": manifest["device"] in ("mps", "cpu"),
        }
        recomputed = augment_per_category(root, model, analyze_model(root, model))
        saved = json.loads((root / "analysis.json").read_text())[model]
        checks["analysis_matches_recomputation"] = recomputed == saved
        if not all(checks.values()):
            results["checks_passed"] = False
        results["models"][model] = {"checks": checks, "manifest": manifest}
    commits = {
        info["manifest"]["provenance"]["git_commit"] for info in results["models"].values()
    }
    results["same_repository_commit_for_both_captures"] = len(commits) == 1
    results["checks_passed"] &= len(commits) == 1
    if not results["checks_passed"]:
        raise ValueError("014 integrity audit failed; see failing checks")
    return results


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def plot(analysis, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    models = ("qwen-1.5b", "smollm2-1.7b")
    labels = ("Qwen2.5-1.5B", "SmolLM2-1.7B")
    colors = {"native": "#38598c", "shared": "#55a868", "random": "#c44e52"}
    x = np.arange(len(models))
    width = 0.22
    for i, condition in enumerate(("native", "shared", "random")):
        values = []
        for model in models:
            source_rows = [analysis[model]["effects"][f"{condition}_{k}"]["mean_margin_delta"] for k in range(3)]
            values.append(float(np.mean(source_rows)))
        bars = axes[0].bar(
            x + (i - 1) * width,
            values,
            width,
            color=colors[condition],
            label=condition.title(),
        )
        axes[0].bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Change in positive − negative logit margin")
    axes[0].set_title("Steering shifts the general sentiment score")
    axes[0].legend(frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)

    for pos, model in enumerate(models):
        stat = analysis[model]["primary_native_minus_random_specificity"]
        low, high = stat["sentence_bootstrap_95_ci"]
        mean = stat["mean_native_minus_random_specificity"]
        axes[1].errorbar(
            mean,
            pos,
            xerr=[[mean - low], [high - mean]],
            marker="o",
            capsize=4,
            color=colors["native"],
        )
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks([0, 1], labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Native minus random aspect-specificity contrast (logits)")
    axes[1].set_title("Aspect-selective signal is near zero")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Synthetic aspect directions on natural SemEval reviews")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path("runs/task-ladder-v1"))
    parser.add_argument(
        "--data", type=Path, default=Path(".context/datasets/semeval2014/Restaurants_Test_Gold.xml")
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    analysis = json.loads((args.run_root / "analysis.json").read_text())
    audit_info = audit(args.run_root, args.source_root, args.data)
    args.output.mkdir(parents=True)
    write_json(args.output / "analysis.json", analysis)
    write_json(args.output / "audit.json", audit_info)
    for model in MODELS:
        folder = args.run_root / model
        for name in ("baseline", "effects"):
            with (args.output / f"{model}-{name}.json.gz").open("wb") as stream:
                with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as archive:
                    archive.write(
                        json.dumps(
                            json.loads((folder / f"{name}.json").read_text()),
                            separators=(",", ":"),
                        ).encode()
                    )
        (args.output / f"{model}-manifest.json").write_bytes(
            (folder / "manifest.json").read_bytes()
        )
    plot(analysis, args.output / "natural-aspect-selectivity.png")
    lines = []
    for model in MODELS:
        row = analysis[model]
        spec = row["primary_native_minus_random_specificity"]
        low, high = spec["sentence_bootstrap_95_ci"]
        class_stats = row["label_balance_and_class_accuracy"]
        native_mean = np.mean(
            [row["effects"][f"native_{source}"]["mean_margin_delta"] for source in range(3)]
        )
        lines.append(
            f"- **{model}:** baseline accuracy {row['baseline_strict_accuracy']:.1%}; "
            f"mean native shift {native_mean:+.4f} logits; native-minus-random specificity "
            f"{spec['mean_native_minus_random_specificity']:+.6f} "
            f"(sentence bootstrap 95% CI [{low:+.6f}, {high:+.6f}]); "
            f"{spec['n_sentences']} sentences; baseline accuracy on negative labels "
            f"{class_stats['baseline_accuracy_by_label']['0']:.1%} and positive labels "
            f"{class_stats['baseline_accuracy_by_label']['1']:.1%}."
        )
    qwen_steered_accuracy = np.mean(
        [analysis["qwen-1.5b"]["effects"][f"native_{source}"]["mean_strict_accuracy_under_steering"] for source in range(3)]
    )
    smol_steered_accuracy = np.mean(
        [analysis["smollm2-1.7b"]["effects"][f"native_{source}"]["mean_strict_accuracy_under_steering"] for source in range(3)]
    )
    readme = f"""# Natural aspect selectivity (experiment 014)

## Question

Do frozen aspect-specific directions learned from synthetic reviews selectively move the sentiment score for the matching aspect when applied to human-annotated, multi-aspect restaurant reviews?

## Results

The unsteered models perform well on the included aspect queries, and the native directions cause a large positive-minus-negative logit shift. The primary within-sentence contrast is much smaller:

{chr(10).join(lines)}

This supports a generic valence-shift interpretation more than a useful aspect-selective-control interpretation. SmolLM2 has a positive interval in the frozen contrast, but the estimate is roughly 0.0007 logits against a general shift above 1.3 logits; Qwen is centered near zero. The polarity-disagreement slice contains only eight sentences and is descriptive. Also, 178 of 233 prompts (76.4%) are positive-labeled, so the small overall accuracy gain under positive steering is sensitive to class balance: it rises from {analysis['qwen-1.5b']['baseline_strict_accuracy']:.1%} to {qwen_steered_accuracy:.1%} on Qwen and from {analysis['smollm2-1.7b']['baseline_strict_accuracy']:.1%} to {smol_steered_accuracy:.1%} on SmolLM2. Looking by class makes the tradeoff visible: negative-label accuracy falls from 98.2% to 94.5% on Qwen and 89.1% on SmolLM2, while positive-label accuracy rises. The intervention is making the model more positive overall; it is not reliably correcting the requested aspect. This is another reason not to read the logit movement as target-specific behavioral control. The run does not establish useful naturalistic aspect-specific steering.

![Generic shift and aspect selectivity](natural-aspect-selectivity.png)

The official [SemEval-2014 Task 4 description](https://alt.qcri.org/semeval2014/task4/index.php) defines aspect-category polarity and documents human annotations. The experiment used the public gold XML mirrored at [HSLCY/ABSA-BERT-pair](https://github.com/HSLCY/ABSA-BERT-pair/blob/master/data/semeval2014/Restaurants_Test_Gold.xml), SHA-256 `{audit_info['dataset_sha256']}`. The raw file is not redistributed here. Steering answer-encoding confounds have also been directly examined by [Gao et al. (2026)](https://arxiv.org/html/2608.22985v1); this experiment asks a different, narrow generalization question about aspect selectivity on natural reviews.

Protocol: [`014-natural-aspect-selectivity.md`](../../docs/experiments/014-natural-aspect-selectivity.md). The bundle includes every anonymized row-level logit outcome, model manifests, the frozen analysis, and integrity audit. It contains no review text.

Reproduce the local run and package from the frozen protocol:

```bash
PYTHONPATH=scripts:src .venv/bin/python scripts/natural_aspect_selectivity.py all --offline --resume
PYTHONPATH=scripts:src .venv/bin/python scripts/analyze_natural_aspect_selectivity.py
PYTHONPATH=scripts:src .venv/bin/python scripts/package_natural_aspect_selectivity.py \\
  runs/natural-aspect-selectivity-v1 results/natural-aspect-selectivity-v1
```

`audit.json` checks source, model-capture, code and protocol hashes; factorial balance; outcome counts; text exclusion; and deterministic analysis. `SHA256SUMS` covers this bundle.
"""
    (args.output / "README.md").write_text(readme)
    checksums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (args.output / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    print(f"Packaged audited 014 results in {args.output}")


if __name__ == "__main__":
    main()
