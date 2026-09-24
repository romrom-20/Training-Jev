"""Integrity-check and package experiment 016 dose-response outcomes."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_mams_dose_response import analyze_model
from mams_aspect_selectivity import DATA, MODELS, REPOSITORY_REVISION, load_mams
from mams_dose_response import DOSES, PROTOCOL, dose_name
from prompt_effect_forecast import sha

RUNNER = Path("scripts/mams_dose_response.py")
HELPER = Path("scripts/natural_aspect_selectivity.py")
ANALYZER = Path("scripts/analyze_mams_dose_response.py")


def audit(root, source_root, data_path):
    stimuli, sentences, conflicts = load_mams(data_path)
    prompt_ids = {row["id"] for row in stimuli}
    effect_ids = {
        f"{prompt_id}|{condition}|{source}"
        for prompt_id in prompt_ids
        for condition in ("native", "shared", "random")
        for source in range(3)
    }
    info = {
        "checks_passed": True,
        "dataset_sha256": sha(data_path),
        "dataset_repository_revision": REPOSITORY_REVISION,
        "analysis_code_sha256": sha(ANALYZER),
        "n_prompts": len(stimuli),
        "n_sentences": len(sentences),
        "n_polarity_conflict_sentences": conflicts,
        "doses": {},
    }
    commits = set()
    for dose in DOSES:
        dose_key = f"{dose:.4f}"
        info["doses"][dose_key] = {}
        analysis_root = root / dose_name(dose)
        for model in MODELS:
            folder = analysis_root / model
            manifest = json.loads((folder / "manifest.json").read_text())
            baseline_path, effects_path = folder / "baseline.json", folder / "effects.json"
            baseline = json.loads(baseline_path.read_text())
            effects = json.loads(effects_path.read_text())
            source_manifest = json.loads((source_root / model / "manifest.json").read_text())
            commit = manifest["provenance"]["git_commit"]
            commits.add(commit)
            checks = {
                "dose_matches_directory": manifest[
                    "dose_fraction_of_training_activation_norm"
                ] == dose,
                "dataset_hash_matches": manifest["mams_xml_sha256"] == sha(data_path),
                "dataset_revision_matches": manifest["dataset_repository_revision"]
                == REPOSITORY_REVISION,
                "protocol_hash_matches": manifest["protocol_sha256"] == sha(PROTOCOL),
                "runner_hash_matches": manifest["runner_sha256"] == sha(RUNNER),
                "capture_helper_hash_matches": manifest["capture_helper_sha256"]
                == sha(HELPER),
                "baseline_hash_matches": manifest["baseline_sha256"] == sha(baseline_path),
                "effects_hash_matches": manifest["effects_sha256"] == sha(effects_path),
                "source_dataset_hash_matches": manifest["source_dataset_sha256"]
                == sha(source_root / model / "dataset.json"),
                "source_activation_hash_matches": manifest["source_activation_sha256"]
                == source_manifest["activation_sha256"],
                "prompt_ids_match_frozen_filter": {row["id"] for row in baseline}
                == prompt_ids,
                "baseline_ids_unique": len(baseline) == len(prompt_ids),
                "effect_ids_unique": len({row["effect_id"] for row in effects})
                == len(effects),
                "effect_ids_match_factorial": {row["effect_id"] for row in effects}
                == effect_ids,
                "review_text_absent": all(
                    "user" not in row and "text" not in row for row in baseline + effects
                ),
                "capture_commit_clean": not manifest["provenance"]["dirty_worktree"],
                "local_device": manifest["device"] in ("mps", "cpu"),
            }
            if not all(checks.values()):
                info["checks_passed"] = False
            info["doses"][dose_key][model] = {"checks": checks, "manifest": manifest}
    saved = json.loads((root / "analysis.json").read_text())
    for model in MODELS:
        recomputed = analyze_model(root, model)
        if recomputed != saved[model]:
            info["checks_passed"] = False
            info.setdefault("analysis_checks", {})[model] = False
        else:
            info.setdefault("analysis_checks", {})[model] = True
    info["same_repository_commit_for_all_captures"] = len(commits) == 1
    info["checks_passed"] &= len(commits) == 1
    if not info["checks_passed"]:
        raise ValueError("016 integrity audit failed")
    return info


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def plot(analysis, path):
    models = ("qwen-1.5b", "smollm2-1.7b")
    labels = ("Qwen2.5-1.5B", "SmolLM2-1.7B")
    colors = ("#38598c", "#55a868")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    doses = np.asarray(DOSES)
    for model, label, color in zip(models, labels, colors):
        rows = [analysis[model]["doses"][f"{dose:.4f}"] for dose in DOSES]
        ratio = np.asarray([row["specificity_fraction_of_generic_shift"] for row in rows]) * 100
        ratio_ci = np.asarray([row["ratio_95_ci"] for row in rows]).T * 100
        axes[0].errorbar(
            doses * 100,
            ratio,
            yerr=np.vstack((ratio - ratio_ci[0], ratio_ci[1] - ratio)),
            marker="o",
            capsize=3,
            label=label,
            color=color,
        )
        mean = [row["native_minus_random_specificity_mean_logits"] for row in rows]
        ci = np.asarray([row["specificity_95_ci"] for row in rows]).T
        axes[1].errorbar(
            doses * 100,
            mean,
            yerr=np.vstack((np.asarray(mean) - ci[0], ci[1] - np.asarray(mean))),
            marker="o",
            capsize=3,
            label=label,
            color=color,
        )
    axes[0].axhline(5, color="#c44e52", linestyle="--", linewidth=1, label="5% practical threshold")
    axes[0].set_ylabel("Specificity / generic shift (%)")
    axes[0].set_title("Does target information scale up with dose?")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Native − random specificity (logits)")
    axes[1].set_title("Absolute target-matched effect")
    for ax in axes:
        ax.set_xlabel("Dose (% median training activation norm)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(doses * 100, [f"{x:g}%" for x in doses * 100])
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("MAMS target selectivity across intervention strengths")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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
    audit_info = audit(args.run_root, args.source_root, args.data)
    args.output.mkdir(parents=True)
    write_json(args.output / "analysis.json", analysis)
    write_json(args.output / "audit.json", audit_info)
    for dose in DOSES:
        for model in MODELS:
            folder = args.run_root / dose_name(dose) / model
            for name in ("baseline", "effects"):
                archive_path = args.output / f"{dose_name(dose)}-{model}-{name}.json.gz"
                with archive_path.open("wb") as stream:
                    with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as archive:
                        archive.write(
                            json.dumps(
                                json.loads((folder / f"{name}.json").read_text()),
                                separators=(",", ":"),
                            ).encode()
                        )
            (args.output / f"{dose_name(dose)}-{model}-manifest.json").write_bytes(
                (folder / "manifest.json").read_bytes()
            )
    plot(analysis, args.output / "dose-response.png")
    lines = []
    for model in MODELS:
        row = analysis[model]
        slope = row["ratio_vs_log2_dose_slope"]
        lines.append(
            f"**{model}:** ratio slope {slope['point_estimate']:+.5f} per log2 dose "
            f"(95% CI [{slope['sentence_bootstrap_95_ci'][0]:+.5f}, "
            f"{slope['sentence_bootstrap_95_ci'][1]:+.5f}]); "
            f"any practical dose passed: {row['any_practical_dose_pass']}."
        )
    readme = f"""# MAMS dose response (experiment 016)

## Question

Does aspect selectivity grow faster than the generic positive sentiment shift as the fixed synthetic directions are applied more strongly?

## Result

The run uses the same frozen MAMS-ACSA test prompts and 008 direction vectors as experiments 015; only the dose changes. Each of five doses was tested on both models, with native, shared and norm-matched random controls. The sentence-level specificity-to-generic-shift ratio was bootstrapped over the 35 held-out MAMS sentences.

{chr(10).join(lines)}

The cross-model positive-ratio-trend gate {'passed' if analysis['cross_model_positive_ratio_trend_pass'] else 'failed'}, and the cross-model practical-dose gate {'passed' if analysis['cross_model_any_practical_dose_pass'] else 'failed'}. At the smallest dose, specificity was 0.445% of the generic shift for Qwen and 0.086% for SmolLM2; at 20%, it was 0.057% and 0.059%. At that highest dose, strict accuracy changed from 78.9% to 76.5% for Qwen and from 73.2% to 54.9% for SmolLM2. `analysis.json` reports baseline, native, shared and random-control accuracy at every dose; no dose was selected after seeing outcomes.

![Dose response for specificity relative to general valence shift](dose-response.png)

The protocol was written after experiment 015 and before these dose outcomes, so this is a planned exploratory follow-up to the MAMS result, not independent validation. Raw review text is excluded. `audit.json` checks all ten capture manifests, balance, exact code/data/protocol hashes and recomputed analyses; `SHA256SUMS` covers this bundle.

Protocol: [`016-mams-dose-response.md`](../../docs/experiments/016-mams-dose-response.md).

```bash
PYTHONPATH=scripts:src .venv/bin/python scripts/mams_dose_response.py all --offline --resume
PYTHONPATH=scripts:src .venv/bin/python scripts/analyze_mams_dose_response.py
PYTHONPATH=scripts:src .venv/bin/python scripts/package_mams_dose_response.py \\
  runs/mams-dose-response-v1 results/mams-dose-response-v1
```
"""
    (args.output / "README.md").write_text(readme)
    sums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"Packaged audited dose-response results in {args.output}")


if __name__ == "__main__":
    main()
