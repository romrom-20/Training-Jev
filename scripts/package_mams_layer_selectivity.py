"""Audit and package experiment 017 without MAMS review text."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_mams_layer_selectivity import analyze, analyze_cell
from mams_aspect_selectivity import DATA, MODELS, REPOSITORY_REVISION, load_mams
from mams_layer_selectivity import LAYERS, PROTOCOL
from prompt_effect_forecast import sha

RUNNER = Path("scripts/mams_layer_selectivity.py")
ANALYZER = Path("scripts/analyze_mams_layer_selectivity.py")
HELPER = Path("scripts/natural_aspect_selectivity.py")


def audit(root, source_root, data_path, analysis):
    stimuli, sentences, conflicts = load_mams(data_path)
    prompt_ids = {row["id"] for row in stimuli}
    effect_ids = {
        f"{prompt_id}|{condition}|{source}"
        for prompt_id in prompt_ids
        for condition in ("native", "shared", "random")
        for source in range(3)
    }
    commits = set()
    full_analysis_matches = analyze(root) == analysis
    report = {
        "checks_passed": True,
        "dataset_sha256": sha(data_path),
        "dataset_repository_revision": REPOSITORY_REVISION,
        "n_prompts": len(stimuli),
        "n_sentences": len(sentences),
        "n_polarity_conflict_sentences": conflicts,
        "analysis_code_sha256": sha(ANALYZER),
        "runner_sha256": sha(RUNNER),
        "full_analysis_recomputes": full_analysis_matches,
        "layers": {},
    }
    for layer in LAYERS:
        key = str(layer)
        report["layers"][key] = {}
        for model in MODELS:
            folder = root / f"layer-{layer:02d}" / model
            manifest = json.loads((folder / "manifest.json").read_text())
            baseline_path, effects_path = folder / "baseline.json", folder / "effects.json"
            baseline = json.loads(baseline_path.read_text())
            effects = json.loads(effects_path.read_text())
            source = source_root / model
            source_manifest = json.loads((source / "manifest.json").read_text())
            checks = {
                "layer_matches_directory": manifest["layer"] == layer,
                "dose_fraction_is_frozen": manifest[
                    "dose_fraction_of_training_activation_norm"
                ]
                == 0.05,
                "dataset_hash_matches": manifest["mams_xml_sha256"] == sha(data_path),
                "dataset_revision_matches": manifest[
                    "dataset_repository_revision"
                ]
                == REPOSITORY_REVISION,
                "protocol_hash_matches": manifest["protocol_sha256"] == sha(PROTOCOL),
                "runner_hash_matches": manifest["runner_sha256"] == sha(RUNNER),
                "helper_hash_matches": manifest["capture_helper_sha256"] == sha(HELPER),
                "baseline_hash_matches": manifest["baseline_sha256"] == sha(baseline_path),
                "effects_hash_matches": manifest["effects_sha256"] == sha(effects_path),
                "source_dataset_hash_matches": manifest["source_dataset_sha256"]
                == sha(source / "dataset.json"),
                "source_activation_hash_matches": manifest["source_activation_sha256"]
                == source_manifest["activation_sha256"],
                "prompts_match_frozen_filter": {row["id"] for row in baseline}
                == prompt_ids,
                "baseline_ids_unique": len(baseline) == len(prompt_ids),
                "effect_ids_unique": len({row["effect_id"] for row in effects})
                == len(effects),
                "factorial_matches": {row["effect_id"] for row in effects} == effect_ids,
                "review_text_absent": all(
                    "user" not in row and "text" not in row for row in baseline + effects
                ),
                "capture_commit_clean": not manifest["provenance"]["dirty_worktree"],
                "local_device": manifest["device"] in ("mps", "cpu"),
                "analysis_recomputes": analyze_cell(root, model, layer)
                == analysis[model][key],
            }
            if not all(checks.values()):
                report["checks_passed"] = False
            commits.add(manifest["provenance"]["git_commit"])
            report["layers"][key][model] = {"checks": checks, "manifest": manifest}
    report["same_capture_commit"] = len(commits) == 1
    report["checks_passed"] &= len(commits) == 1 and full_analysis_matches
    if not report["checks_passed"]:
        raise ValueError("017 integrity audit failed")
    return report


def plot(analysis, path):
    names = {"qwen-1.5b": "Qwen2.5-1.5B", "smollm2-1.7b": "SmolLM2-1.7B"}
    colors = {"qwen-1.5b": "#38598c", "smollm2-1.7b": "#55a868"}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    for model in MODELS:
        rows = [analysis[model][str(layer)] for layer in LAYERS]
        ratio = np.asarray(
            [
                np.nan
                if row["specificity_fraction_of_generic_shift"] is None
                else row["specificity_fraction_of_generic_shift"]
                for row in rows
            ]
        ) * 100
        ratio_ci = np.asarray(
            [
                [np.nan, np.nan] if row["ratio_95_ci"] is None else row["ratio_95_ci"]
                for row in rows
            ]
        ).T * 100
        axes[0].errorbar(
            LAYERS,
            ratio,
            yerr=np.vstack((ratio - ratio_ci[0], ratio_ci[1] - ratio)),
            marker="o",
            capsize=4,
            color=colors[model],
            label=names[model],
        )
        specificity = np.asarray(
            [row["native_minus_random_specificity_mean_logits"] for row in rows]
        )
        ci = np.asarray([row["specificity_95_ci"] for row in rows]).T
        axes[1].errorbar(
            LAYERS,
            specificity,
            yerr=np.vstack((specificity - ci[0], ci[1] - specificity)),
            marker="o",
            capsize=4,
            color=colors[model],
            label=names[model],
        )
    axes[0].axhline(5, color="#c44e52", linestyle="--", linewidth=1, label="5% practical threshold")
    axes[0].set_ylabel("Specificity / generic shift (%)")
    axes[0].set_title("Target selectivity by layer")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Native − random specificity (logits)")
    axes[1].set_title("Absolute target-matched effect")
    for ax in axes:
        ax.set_xlabel("Intervention layer")
        ax.set_xticks(LAYERS)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("MAMS target selectivity at matched training-norm dose")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path("runs/task-ladder-v1"))
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    analysis = json.loads((args.run_root / "analysis.json").read_text())
    audit_report = audit(args.run_root, args.source_root, args.data, analysis)
    args.output.mkdir(parents=True)
    write_json(args.output / "analysis.json", analysis)
    write_json(args.output / "audit.json", audit_report)
    for layer in LAYERS:
        for model in MODELS:
            folder = args.run_root / f"layer-{layer:02d}" / model
            for name in ("baseline", "effects"):
                with (args.output / f"layer-{layer:02d}-{model}-{name}.json.gz").open(
                    "wb"
                ) as stream:
                    with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as archive:
                        archive.write(
                            json.dumps(
                                json.loads((folder / f"{name}.json").read_text()),
                                separators=(",", ":"),
                            ).encode()
                        )
            (args.output / f"layer-{layer:02d}-{model}-manifest.json").write_bytes(
                (folder / "manifest.json").read_bytes()
            )
    plot(analysis, args.output / "layer-selectivity.png")
    lines = []
    for model in MODELS:
        for layer in LAYERS:
            row = analysis[model][str(layer)]
            low, high = row["specificity_95_ci"]
            ratio = row["specificity_fraction_of_generic_shift"]
            ratio_text = (
                f"{ratio:.3%} of generic shift"
                if ratio is not None
                else "ratio undefined (no reliable positive generic shift)"
            )
            lines.append(
                f"- **{model}, layer {layer}:** baseline {row['baseline_accuracy']:.1%}; "
                f"generic shift {row['generic_native_shift_mean_logits']:+.4f}; "
                f"specificity {row['native_minus_random_specificity_mean_logits']:+.6f} "
                f"logits (95% CI [{low:+.6f}, {high:+.6f}]); "
                f"{ratio_text}; "
                f"practical gate {row['practical_layer_gate']}."
            )
    smol_difference = analysis["post_hoc_pairwise_layer_differences"][
        "smollm2-1.7b"
    ]["16_minus_24"]
    qwen_difference = analysis["post_hoc_pairwise_layer_differences"]["qwen-1.5b"]
    smol_low, smol_high = smol_difference["sentence_bootstrap_95_ci"]
    conflict_low, conflict_high = smol_difference[
        "conflict_sentence_bootstrap_95_ci"
    ]
    qwen_low, qwen_high = qwen_difference["16_minus_24"][
        "sentence_bootstrap_95_ci"
    ]
    smol_ratio_difference = smol_difference["specificity_fraction_difference"]
    smol_ratio_low, smol_ratio_high = smol_difference[
        "specificity_fraction_difference_95_ci"
    ]
    qwen_ratio_difference = qwen_difference["16_minus_24"][
        "specificity_fraction_difference"
    ]
    qwen_ratio_low, qwen_ratio_high = qwen_difference["16_minus_24"][
        "specificity_fraction_difference_95_ci"
    ]
    readme = f"""# MAMS layer selectivity (experiment 017)

## Result

This exploratory sweep applies training-only directions derived separately at layers 8, 16 and 24 to the same 35-sentence, 71-query MAMS-ACSA test filter used in experiment 015. Each layer uses a 5% dose of its own median training activation norm and native, shared and norm-matched random controls. The 18-sentence conflict slice is included in `analysis.json`.

{chr(10).join(lines)}

No layer was selected after seeing results. The practical gate is frozen at at least 5% specificity relative to generic shift, positive specificity interval and above-chance baseline accuracy. This is one benchmark and two model families; any layer-dependent signal needs independent confirmation.

An explicitly **post-hoc, unadjusted** paired comparison found a layer-16 minus layer-24 specificity difference of {smol_difference['mean_specificity_difference_logits']:+.6f} logits for SmolLM2 (sentence-bootstrap 95% CI [{smol_low:+.6f}, {smol_high:+.6f}]); on the 18 conflict sentences the difference was {smol_difference['conflict_mean_difference_logits']:+.6f} (95% CI [{conflict_low:+.6f}, {conflict_high:+.6f}]). Qwen's same all-sentence contrast was {qwen_difference['16_minus_24']['mean_specificity_difference_logits']:+.6f} (95% CI [{qwen_low:+.6f}, {qwen_high:+.6f}]). These post-hoc comparisons were not preregistered and do not establish a cross-model layer effect; the SmolLM2 pattern needs a locked independent replication.

For the preregistered relative endpoint, the paired layer-16 minus layer-24 specificity-fraction difference was {smol_ratio_difference * 100:+.2f} percentage points in SmolLM2 (95% CI [{smol_ratio_low * 100:+.2f}, {smol_ratio_high * 100:+.2f}]) and {qwen_ratio_difference * 100:+.2f} points in Qwen (95% CI [{qwen_ratio_low * 100:+.2f}, {qwen_ratio_high * 100:+.2f}]). This ratio comparison is also post-hoc and unadjusted; only the SmolLM2 interval excludes zero.

![Specificity fraction and absolute target-matched effect by layer](layer-selectivity.png)

Protocol: [`017-mams-layer-selectivity.md`](../../docs/experiments/017-mams-layer-selectivity.md). `audit.json` checks all six capture manifests, data and code hashes, factorial balance, analysis recomputation and absence of review text. `SHA256SUMS` covers the bundle.
"""
    (args.output / "README.md").write_text(readme)
    sums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"Packaged audited layer-selectivity results in {args.output}")


if __name__ == "__main__":
    main()
