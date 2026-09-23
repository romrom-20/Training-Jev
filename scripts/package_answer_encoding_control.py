"""Audit and package experiment 013's cross-encoding intervention results."""

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

MAPPINGS = ("positive_is_A", "positive_is_B")
CONDITIONS = ("native", "shared", "random")


def audit(root, source_root):
    analysis = json.loads((root / "analysis.json").read_text())
    results = {"checks_passed": True, "models": {}}
    for model in MODELS:
        folder = root / model
        manifest = json.loads((folder / "manifest.json").read_text())
        effects_path = folder / "effects.json"
        baseline_path = folder / "baseline.json"
        effects = json.loads(effects_path.read_text())
        baseline = json.loads(baseline_path.read_text())
        source_manifest = json.loads((source_root / model / "manifest.json").read_text())
        effects_by_prompt = {}
        for row in effects:
            effects_by_prompt.setdefault((row["base_id"], row["mapping"]), set()).add(
                (row["condition"], row["source"])
            )
        expected_conditions = {
            ("native", 0), ("native", 1), ("native", 2), ("shared", None), ("random", None)
        }
        checks = {
            "effect_hash_matches": manifest["effects_sha256"] == sha(effects_path),
            "baseline_hash_matches": manifest["baseline_sha256"] == sha(baseline_path),
            "effect_count_matches": len(effects) == manifest["n_effects"] == 1440,
            "baseline_count_matches": len(baseline) == manifest["n_baseline_rows"] == 288,
            "effect_ids_unique": len({x["effect_id"] for x in effects}) == len(effects),
            "baseline_ids_unique": len({(x["base_id"], x["mapping"]) for x in baseline}) == len(baseline),
            "all_prompts_have_all_conditions_under_both_maps": len(effects_by_prompt) == 288
            and all(v == expected_conditions for v in effects_by_prompt.values()),
            "24_groups_per_mapping": all(
                len({x["group_no"] for x in baseline if x["mapping"] == mapping}) == 24
                for mapping in MAPPINGS
            ),
            "protocol_hash_matches": manifest["protocol_sha256"] == sha(Path("docs/experiments/013-answer-encoding-control.md")),
            "code_hash_matches": manifest["code_sha256"] == sha(Path("scripts/answer_encoding_control.py")),
            "source_dataset_hash_matches": manifest["source_dataset_sha256"] == sha(source_root / model / "dataset.json"),
            "source_activation_hash_matches": manifest["source_activation_sha256"] == source_manifest["activation_sha256"],
            "commit_recorded": bool(manifest["provenance"]["git_commit"]),
        }
        if not all(checks.values()):
            results["checks_passed"] = False
        results["models"][model] = {"checks": checks, "manifest": manifest}
    if not results["checks_passed"]:
        raise ValueError("013 integrity audit failed")
    commits = {info["manifest"]["provenance"]["git_commit"] for info in results["models"].values()}
    results["same_repository_commit_for_both_captures"] = len(commits) == 1
    if len(commits) != 1:
        raise ValueError("013 model captures used different repository commits")
    return analysis, results


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def plot(analysis, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), sharey=True)
    conditions = ("shared", "random", "native")
    colors = ("#38598c", "#c44e52", "#55a868")
    labels = ("Shared", "Random", "Native mean")
    for ax, model, title in zip(axes, MODELS, ("Qwen2.5-1.5B", "SmolLM2-1.7B")):
        x = np.arange(2)
        width = 0.22
        for i, (condition, color, label) in enumerate(zip(conditions, colors, labels)):
            values = [
                analysis[model]["mappings"][mapping]["condition_effects"][condition]["mean_semantic_margin_delta"]
                for mapping in MAPPINGS
            ]
            bars = ax.bar(x + (i - 1) * width, values, width, label=label, color=color)
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(x, ("Positive = A", "Positive = B"))
        ax.set_title(title)
        ax.set_ylabel("Semantic positive-minus-negative margin change")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Does the shared steering effect follow sentiment after answer remapping?")
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
    analysis, results = audit(args.run_root, args.source_root)
    args.output.mkdir(parents=True)
    dump_json(args.output / "analysis.json", analysis)
    dump_json(args.output / "audit.json", results)
    for model in MODELS:
        folder = args.run_root / model
        for name in ("baseline", "effects"):
            with (args.output / f"{model}-{name}.json.gz").open("wb") as stream:
                with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as archive:
                    archive.write(json.dumps(json.loads((folder / f"{name}.json").read_text()), separators=(",", ":")).encode())
        (args.output / f"{model}-manifest.json").write_bytes((folder / "manifest.json").read_bytes())
    plot(analysis, args.output / "answer-encoding-effects.png")
    lines = []
    for model in MODELS:
        lines.append(f"**{model}**")
        for mapping in MAPPINGS:
            result = analysis[model]["mappings"][mapping]
            shared = result["condition_effects"]["shared_minus_random"]
            ci = shared["group_bootstrap_95_ci"]
            lines.append(
                f"- {mapping}: unsteered next-token accuracy {result['baseline_mapped_next_token_accuracy']:.1%}; "
                f"A/B validity {result['baseline_valid_a_or_b_rate']:.1%}; shared semantic-margin change "
                f"{result['condition_effects']['shared']['mean_semantic_margin_delta']:+.4f}, "
                f"random-adjusted {shared['mean_semantic_margin_delta']:+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]; "
                f"shared A−B identifier-margin change {result['condition_effects']['shared']['mean_identifier_ab_delta']:+.4f}."
            )
    readme = """# Answer-encoding control (experiment 013)

## Frozen rule

Interpret mapped changes semantically only if unsteered next-token accuracy reaches 90% in both encodings. Mapping-robust semantic steering requires the shared component to beat a norm-matched random direction in both answer mappings for both models, with group-bootstrap 95% intervals above zero.

## Results

""" + "\n".join(lines) + """

""" + ("**The full semantic-mapping rule passed.**" if all(x["semantic_mapping_robustness_gate_pass"] for x in analysis.values()) else "**The full semantic-mapping rule did not pass.**") + """

Qwen is the clean result: unsteered next-token accuracy is 100% under both mappings, but the shared intervention moves the fixed A-minus-B margin toward A in both. Once the meaning of A flips, the semantic positive-minus-negative effect changes from +0.206 to −0.290. That is identifier following in this task, despite perfect baseline task accuracy. SmolLM2's A-minus-B effect consistently favors B, but its reversed-mapping baseline accuracy is 87.5%, below the frozen 90% competence gate, so its semantic interpretation remains unresolved.

![Shared, native and random intervention effect under both answer mappings](answer-encoding-effects.png)

Semantic margins are oriented positive-minus-negative under the current mapping. A semantic effect should remain positive when A and B swap meanings. If the semantic effect reverses while the A-minus-B effect retains its sign, the intervention follows an answer identifier. The full per-mapping results retain both orientations.

This is a small-model replication/control of an issue already studied directly by Gao et al. ([Cross-Encoding Steering Evaluation](https://arxiv.org/html/2608.22985v1)). It uses 144 simple synthetic prompts and a next-token score; it does not establish generative behavior or a new method. It is a useful warning against reading a positive-label logit change as semantic sentiment control. Protocol: [`013-answer-encoding-control.md`](../../docs/experiments/013-answer-encoding-control.md).

Reproduce the analysis and package from local captures:

```bash
uv run python scripts/answer_encoding_control.py analyze --root runs/answer-encoding-control-v1
uv run python scripts/package_answer_encoding_control.py \\
  runs/answer-encoding-control-v1 results/answer-encoding-control-v1
```

The gzip files contain all baseline and intervention rows. The audit checks outcome, code, protocol and source hashes plus mapping/condition balance. `SHA256SUMS` covers the bundle.
"""
    (args.output / "README.md").write_text(readme)
    checksums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (args.output / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    print(f"Packaged audited 013 results in {args.output}")


if __name__ == "__main__":
    main()
