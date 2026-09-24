"""Audit and package experiment 015 results without review text."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_mams_aspect_selectivity import analyze_one
from mams_aspect_selectivity import DATA, MODELS, PROTOCOL, REPOSITORY_REVISION, load_mams
from prompt_effect_forecast import sha

RUNNER = Path("scripts/mams_aspect_selectivity.py")
ANALYZER = Path("scripts/analyze_mams_aspect_selectivity.py")


def audit(root, source_root, data_path):
    stimuli, sentence_labels, conflicts = load_mams(data_path)
    ids = {row["id"] for row in stimuli}
    expected_effect_ids = {
        f"{stimulus_id}|{condition}|{source}"
        for stimulus_id in ids
        for condition in ("native", "shared", "random")
        for source in range(3)
    }
    audit_info = {
        "checks_passed": True,
        "dataset_sha256": sha(data_path),
        "dataset_repository_revision": REPOSITORY_REVISION,
        "analysis_code_sha256": sha(ANALYZER),
        "n_prompts": len(stimuli),
        "n_sentences": len(sentence_labels),
        "n_conflict_sentences": conflicts,
        "models": {},
    }
    for model in MODELS:
        folder = root / model
        manifest = json.loads((folder / "manifest.json").read_text())
        baseline_path, effects_path = folder / "baseline.json", folder / "effects.json"
        baseline, effects = json.loads(baseline_path.read_text()), json.loads(effects_path.read_text())
        source_manifest = json.loads((source_root / model / "manifest.json").read_text())
        checks = {
            "dataset_hash_matches": manifest["mams_xml_sha256"] == sha(data_path),
            "dataset_revision_matches": manifest["dataset_repository_revision"]
            == REPOSITORY_REVISION,
            "protocol_hash_matches": manifest["protocol_sha256"] == sha(PROTOCOL),
            "runner_hash_matches": manifest["code_sha256"] == sha(RUNNER),
            "baseline_hash_matches": manifest["baseline_sha256"] == sha(baseline_path),
            "effects_hash_matches": manifest["effects_sha256"] == sha(effects_path),
            "source_dataset_hash_matches": manifest["source_dataset_sha256"]
            == sha(source_root / model / "dataset.json"),
            "source_activation_hash_matches": manifest["source_activation_sha256"]
            == source_manifest["activation_sha256"],
            "prompt_ids_match_frozen_filter": {row["id"] for row in baseline} == ids,
            "baseline_ids_unique": len(baseline) == len(ids),
            "effect_ids_unique": len({row["effect_id"] for row in effects}) == len(effects),
            "effect_ids_match_frozen_factorial": {
                row["effect_id"] for row in effects
            }
            == expected_effect_ids,
            "no_review_text_in_outcomes": all(
                "user" not in row and "text" not in row for row in baseline + effects
            ),
            "manifest_capture_commit_clean": not manifest["provenance"]["dirty_worktree"]
            and bool(manifest["provenance"]["git_commit"]),
            "local_device": manifest["device"] in ("mps", "cpu"),
            "analysis_recomputes": analyze_one(root, model)
            == json.loads((root / "analysis.json").read_text())[model],
        }
        if not all(checks.values()):
            audit_info["checks_passed"] = False
        audit_info["models"][model] = {"checks": checks, "manifest": manifest}
    commits = {
        item["manifest"]["provenance"]["git_commit"]
        for item in audit_info["models"].values()
    }
    audit_info["same_commit_for_both_models"] = len(commits) == 1
    audit_info["checks_passed"] &= len(commits) == 1
    if not audit_info["checks_passed"]:
        raise ValueError("015 audit failed")
    return audit_info


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def plot(analysis, output):
    models = ("qwen-1.5b", "smollm2-1.7b")
    names = ("Qwen2.5-1.5B", "SmolLM2-1.7B")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    x = np.arange(2)
    width = 0.22
    colors = {"native": "#38598c", "shared": "#55a868", "random": "#c44e52"}
    for i, condition in enumerate(("native", "shared", "random")):
        values = [
            np.mean(
                [analysis[m]["effects"][f"{condition}_{k}"]["mean_margin_delta"] for k in range(3)]
            )
            for m in models
        ]
        bars = axes[0].bar(
            x + (i - 1) * width, values, width, color=colors[condition], label=condition.title()
        )
        axes[0].bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("Positive − negative logit margin change")
    axes[0].set_title("Broad sentiment shift")
    axes[0].legend(frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)
    for i, model in enumerate(models):
        stat = analysis[model]["primary_native_minus_random_specificity"]
        mean = stat["mean_native_minus_random_specificity"]
        low, high = stat["sentence_bootstrap_95_ci"]
        axes[1].errorbar(
            mean,
            i,
            xerr=[[mean - low], [high - mean]],
            marker="o",
            capsize=4,
            color=colors["native"],
        )
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks([0, 1], names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Native − random specificity contrast (logits)")
    axes[1].set_title("Specificity is positive but very small")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Frozen synthetic directions on MAMS mixed-polarity reviews")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
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
    plot(analysis, args.output / "mams-specificity.png")
    lines = []
    for model in MODELS:
        row = analysis[model]
        specificity = row["primary_native_minus_random_specificity"]
        low, high = specificity["sentence_bootstrap_95_ci"]
        lines.append(
            f"- **{model}:** baseline accuracy {row['baseline_strict_accuracy']:.1%}; "
            f"generic shift {row['generic_shift_mean_logits']:+.4f}; target-specific contrast "
            f"{specificity['mean_native_minus_random_specificity']:+.6f} logits "
            f"(95% CI [{low:+.6f}, {high:+.6f}]), "
            f"{row['specificity_fraction_of_generic_shift']:.2%} of generic shift; "
            f"conflict-only contrast "
            f"{row['polarity_conflict_primary_slice']['native_minus_random_specificity']:+.6f} "
            f"(95% CI [{row['polarity_conflict_primary_slice']['sentence_bootstrap_95_ci'][0]:+.6f}, "
            f"{row['polarity_conflict_primary_slice']['sentence_bootstrap_95_ci'][1]:+.6f}])."
        )
    gate = analysis["cross_model_practical_selectivity_gate_pass"]
    readme = f"""# MAMS conflict selectivity (experiment 015)

## Result

This is a held-out transfer test on the MAMS-ACSA benchmark, whose source paper constructed reviews with multiple aspects and differing polarities ([Jiang et al., EMNLP-IJCNLP 2019](https://aclanthology.org/D19-1654/)). The preregistered filter retained 35 test sentences and 71 queries across food/menu, service/staff and price/value, including 18 mapped-polarity conflict sentences. The MAMS authors' [repository](https://github.com/siat-nlp/MAMS-for-ABSA) provides the dataset. Raw text is excluded from this bundle.

The frozen native directions produce a positive target-matched specificity contrast in both local models, and the same sign appears in the 18 conflict-only cases:

{chr(10).join(lines)}

The effect is consistent, but it fails the preregistered 5%-of-generic-shift practical threshold in both models. The main interpretation remains a strong general positive sentiment bias with a much smaller target-aspect component. The small test set and category crosswalk limit generalization; this does not show an effect on every MAMS aspect.

**Preregistered cross-model practical-selectivity gate: {'PASS' if gate else 'FAIL'}.**

![Generic shifts and specificity estimates](mams-specificity.png)

Protocol: [`015-mams-conflict-selectivity.md`](../../docs/experiments/015-mams-conflict-selectivity.md). `audit.json` verifies source/protocol/code/outcome hashes, exact factorial balance, analysis recomputation, and absence of review text. `SHA256SUMS` covers this bundle.

Reproduce with the local cached models:

```bash
PYTHONPATH=scripts:src .venv/bin/python scripts/mams_aspect_selectivity.py all --offline --resume
PYTHONPATH=scripts:src .venv/bin/python scripts/analyze_mams_aspect_selectivity.py
PYTHONPATH=scripts:src .venv/bin/python scripts/package_mams_aspect_selectivity.py \\
  runs/mams-aspect-selectivity-v1 results/mams-aspect-selectivity-v1
```
"""
    (args.output / "README.md").write_text(readme)
    sums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"Packaged audited MAMS results in {args.output}")


if __name__ == "__main__":
    main()
