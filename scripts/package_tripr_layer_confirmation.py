"""Audit and package experiment 018 without TripR review text."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_tripr_layer_confirmation import analyze, load_cell
from prompt_effect_forecast import sha
from tripr_layer_confirmation import (
    DATA,
    LAYERS,
    MODELS,
    PROTOCOL,
    SOURCE_REPOSITORY,
    SOURCE_REPOSITORY_REVISION,
    load_tripr,
)

RUNNER = Path("scripts/tripr_layer_confirmation.py")
ANALYZER = Path("scripts/analyze_tripr_layer_confirmation.py")
HELPER = Path("scripts/natural_aspect_selectivity.py")


def audit(root, source_root, data_path, analysis):
    stimuli, sentence_labels, conflicts = load_tripr(data_path)
    prompt_ids = {row["id"] for row in stimuli}
    expected_effect_ids = {
        f"{prompt_id}|{condition}|{source}"
        for prompt_id in prompt_ids
        for condition in ("native", "shared", "random")
        for source in range(3)
    }
    report = {
        "checks_passed": True,
        "dataset_repository": SOURCE_REPOSITORY,
        "dataset_repository_revision": SOURCE_REPOSITORY_REVISION,
        "dataset_license": "CC BY-SA 4.0",
        "dataset_xml_sha256": sha(data_path),
        "protocol_sha256": sha(PROTOCOL),
        "runner_sha256": sha(RUNNER),
        "analysis_code_sha256": sha(ANALYZER),
        "n_prompts": len(stimuli),
        "n_sentences": len(sentence_labels),
        "n_conflict_sentences": conflicts,
        "cells": {},
    }
    commits = set()
    full_analysis_matches = analyze(root) == analysis
    for layer in LAYERS:
        report["cells"][str(layer)] = {}
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
                "dose_fraction_matches_protocol": manifest[
                    "dose_fraction_of_training_activation_norm"
                ]
                == 0.05,
                "dataset_hash_matches": manifest["dataset_xml_sha256"] == sha(data_path),
                "source_repository_matches": manifest["dataset_repository"]
                == SOURCE_REPOSITORY,
                "source_revision_matches": manifest["dataset_repository_revision"]
                == SOURCE_REPOSITORY_REVISION,
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
                "factorial_matches": {row["effect_id"] for row in effects}
                == expected_effect_ids,
                "review_text_absent": all(
                    "user" not in row and "text" not in row for row in baseline + effects
                ),
                "capture_commit_clean": not manifest["provenance"]["dirty_worktree"],
                "local_device": manifest["device"] in ("mps", "cpu"),
                "analysis_recomputes": load_cell(root, model, layer)["n_sentences"]
                == analysis[model][str(layer)]["n_sentences"],
            }
            if not all(checks.values()):
                report["checks_passed"] = False
            commits.add(manifest["provenance"]["git_commit"])
            report["cells"][str(layer)][model] = {
                "checks": checks,
                "manifest": manifest,
            }
    report["same_capture_commit"] = len(commits) == 1
    report["full_analysis_recomputes"] = full_analysis_matches
    report["checks_passed"] &= len(commits) == 1 and full_analysis_matches
    if not report["checks_passed"]:
        raise ValueError("018 integrity audit failed")
    return report


def plot(analysis, path):
    names = {"qwen-1.5b": "Qwen2.5-1.5B", "smollm2-1.7b": "SmolLM2-1.7B"}
    colors = {"qwen-1.5b": "#38598c", "smollm2-1.7b": "#55a868"}
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6))
    for model in MODELS:
        rows = [analysis[model][str(layer)] for layer in LAYERS]
        ratios = np.asarray(
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
            ratios,
            yerr=np.vstack((ratios - ratio_ci[0], ratio_ci[1] - ratios)),
            marker="o",
            capsize=4,
            color=colors[model],
            label=names[model],
        )
        effects = np.asarray(
            [row["native_minus_random_specificity_mean_logits"] for row in rows]
        )
        intervals = np.asarray([row["specificity_95_ci"] for row in rows]).T
        axes[1].errorbar(
            LAYERS,
            effects,
            yerr=np.vstack((effects - intervals[0], intervals[1] - effects)),
            marker="o",
            capsize=4,
            color=colors[model],
            label=names[model],
        )
    axes[0].axhline(5, color="#c44e52", linestyle="--", linewidth=1, label="5% practical threshold")
    axes[0].set_ylabel("Specificity / generic shift (%)")
    axes[0].set_title("Paired layer test on independent reviews")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Native − random specificity (logits)")
    axes[1].set_title("Absolute target-matched effect")
    for axis in axes:
        axis.set_xlabel("Intervention layer")
        axis.set_xticks(LAYERS)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("TripR-2020Large: locked layer-16 specificity test")
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
    plot(analysis, args.output / "layer-confirmation.png")
    lines = []
    for model in MODELS:
        row = analysis[model]
        for layer in LAYERS:
            cell = row[str(layer)]
            low, high = cell["specificity_95_ci"]
            ratio = cell["specificity_fraction_of_generic_shift"]
            ratio_text = f"{ratio:.3%}" if ratio is not None else "undefined"
            lines.append(
                f"- **{model}, layer {layer}:** baseline {cell['baseline_accuracy']:.1%}; "
                f"generic shift {cell['generic_native_shift_mean_logits']:+.4f}; "
                f"specificity {cell['native_minus_random_specificity_mean_logits']:+.6f} "
                f"logits (95% CI [{low:+.6f}, {high:+.6f}]); specificity fraction "
                f"{ratio_text}; conflict-only specificity "
                f"{cell['conflict_only_specificity_mean_logits']:+.6f}."
            )
    primary_lines = []
    for model in MODELS:
        paired = analysis[model]["primary_layer16_minus_layer24_ratio_difference"]
        ci = paired.get("sentence_bootstrap_95_ci")
        if ci is None:
            primary_lines.append(f"- **{model}:** paired ratio comparison undefined.")
        else:
            primary_lines.append(
                f"- **{model}:** layer-16 minus layer-24 ratio difference "
                f"{paired['layer16_minus_layer24_specificity_fraction'] * 100:+.2f} "
                f"percentage points (95% CI [{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}])."
            )
    readme = f"""# Independent layer-16 test (experiment 018)

## Result

This preregistered comparison evaluates frozen training-only directions at layers 16 and 24 on TripR-2020Large, a separate TripAdvisor review collection from the MAMS and SemEval restaurant sets. The frozen label filter retained 187 sentences, 385 mapped aspect queries and 29 cross-aspect polarity conflicts. Each layer uses a 5% dose of its own median training activation norm, with native, shared and norm-matched random controls.

{chr(10).join(lines)}

Primary paired layer-16 minus layer-24 specificity-fraction differences:

{chr(10).join(primary_lines)}

**Cross-model localization replication gate: {'PASS' if analysis['cross_model_localization_replication_gate'] else 'FAIL'}.** **Cross-model practical layer-16 gate (5% specificity fraction): {'PASS' if analysis['cross_model_layer16_practical_gate'] else 'FAIL'}.** The paired comparison and practical threshold were frozen before model outcomes. See `analysis.json` for controls, strict accuracies and all sentence-bootstrap intervals.

![Layer-16 confirmation estimates](layer-confirmation.png)

## Data attribution and license

TripR-2020Large, by Zuheros et al., is distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The raw reviews remain local and are omitted here. This bundle contains only review identifiers, mapped labels, model scores and aggregate statistics. Cite: C. Zuheros et al., “Crowd Decision Making: Sparse Representation Guided by Sentiment Analysis for Leveraging the Wisdom of the Crowd,” IEEE TSMC: Systems (2022), [doi:10.1109/TSMC.2022.3180938](https://doi.org/10.1109/TSMC.2022.3180938). `DATA_ATTRIBUTION.md` records the source revision and license.

Protocol: [`018-independent-tripadvisor-layer16.md`](../../docs/experiments/018-independent-tripadvisor-layer16.md). `audit.json` checks the four captures, hashes, factorial balance, recomputed analysis and absence of review text. `SHA256SUMS` covers the bundle.
"""
    (args.output / "README.md").write_text(readme)
    attribution = f"""# Data attribution

This result uses TripR-2020Large by C. Zuheros, E. Martínez-Cámara, E. Herrera-Viedma and F. Herrera, pinned at source revision `{SOURCE_REPOSITORY_REVISION}` from {SOURCE_REPOSITORY}.

The source dataset is licensed CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/. Raw review text is excluded from this results package. The derived query labels and model outcome records are shared under the same license. Please retain this attribution and license notice if redistributing those files.
"""
    (args.output / "DATA_ATTRIBUTION.md").write_text(attribution)
    sums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"Packaged audited TripR layer confirmation in {args.output}")


if __name__ == "__main__":
    main()
