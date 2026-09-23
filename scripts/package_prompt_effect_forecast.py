"""Validate and package experiment 010's prompt-effect forecast results."""

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

MODELS = ("qwen-1.5b", "smollm2-1.7b")
SPLIT_COUNTS = {"train": 1152, "validation": 432, "calibration": 432, "final_test": 1728}


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def gzip_json(source, target):
    raw = json.dumps(json.loads(source.read_text()), separators=(",", ":")).encode()
    with target.open("wb") as out:
        with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as archive:
            archive.write(raw)


def audit_run(run_root):
    analysis = json.loads((run_root / "analysis.json").read_text())
    audit = {"models": {}, "checks_passed": True}
    for name in MODELS:
        folder = run_root / name
        manifest = json.loads((folder / "manifest.json").read_text())
        effects_path = folder / "effects.json"
        gradient_path = folder / "gradient-final-test.json"
        effects = json.loads(effects_path.read_text())
        gradients = json.loads(gradient_path.read_text())
        counts = {split: sum(x["split"] == split for x in effects) for split in SPLIT_COUNTS}
        ids = [x["effect_id"] for x in effects]
        gradient_ids = [x["base_id"] for x in gradients]
        final = [x for x in effects if x["split"] == "final_test"]
        all_positive = all(x["observed_delta"] > 0 for x in final)
        checks = {
            "manifest_effect_hash_matches": manifest["effects_sha256"] == sha(effects_path),
            "manifest_gradient_hash_matches": manifest["gradient_sha256"] == sha(gradient_path),
            "effect_count_matches_manifest": len(effects) == manifest["n_effects"] == 3744,
            "split_counts_match_protocol": counts == SPLIT_COUNTS,
            "effect_ids_unique": len(ids) == len(set(ids)),
            "gradient_prompt_count_matches_manifest": len(gradients) == manifest["n_final_prompts_gradient"] == 576,
            "gradient_prompt_ids_unique": len(gradient_ids) == len(set(gradient_ids)),
            "final_test_group_count_is_24": len({x["group_no"] for x in final}) == 24,
            "all_final_test_effects_positive": all_positive,
            "analysis_test_count_matches": analysis[name]["n_test"] == len(final),
            "provenance_commit_is_frozen_protocol_commit": manifest["provenance"]["git_commit"].startswith("c312911"),
        }
        if not all(checks.values()):
            audit["checks_passed"] = False
        audit["models"][name] = {
            "checks": checks,
            "split_counts": counts,
            "final_test_groups": len({x["group_no"] for x in final}),
            "final_test_observed_delta_range": [
                min(x["observed_delta"] for x in final),
                max(x["observed_delta"] for x in final),
            ],
            "all_final_test_effects_positive": all_positive,
            "manifest": manifest,
        }
    if not audit["checks_passed"]:
        raise ValueError("010 artifact integrity audit failed; refusing to package")
    return analysis, audit


def make_plot(analysis, path):
    methods = [
        ("shared_bilinear_head", "Small readout"),
        ("pair_mean", "Source/target mean"),
        ("text_ridge", "TF-IDF text"),
        ("first_order_gradient_reference", "Gradient reference"),
    ]
    labels = [label for _, label in methods]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    for i, (model, title) in enumerate((("qwen-1.5b", "Qwen2.5-1.5B"), ("smollm2-1.7b", "SmolLM2-1.7B"))):
        values = [analysis[model]["pooled"][key]["rmse"] for key, _ in methods]
        bars = ax.bar(x + (i - 0.5) * width, values, width, label=title)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_ylabel("Final-test RMSE (lower is better)")
    ax.set_title("Forecasting prompt-level steering effects")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    analysis, audit = audit_run(args.run_root)
    dump_json(args.output / "analysis.json", analysis)
    dump_json(args.output / "audit.json", audit)
    for model in MODELS:
        folder = args.run_root / model
        gzip_json(folder / "effects.json", args.output / f"{model}-effects.json.gz")
        gzip_json(folder / "gradient-final-test.json", args.output / f"{model}-gradient-reference.json.gz")
        (args.output / f"{model}-manifest.json").write_bytes(
            (folder / "manifest.json").read_bytes()
        )
    make_plot(analysis, args.output / "forecast-rmse.png")
    headline = []
    for model in MODELS:
        p = analysis[model]["pooled"]
        ci = p["head_minus_pair_mean_rmse_group_bootstrap_95_ci"]
        headline.append(
            f"- **{model}:** small-readout RMSE {p['shared_bilinear_head']['rmse']:.4f}; "
            f"pair-mean {p['pair_mean']['rmse']:.4f}; gradient reference {p['first_order_gradient_reference']['rmse']:.4f}. "
            f"Readout-vs-mean relative improvement {p['relative_rmse_reduction_vs_pair_mean']:.1%}; "
            f"paired group-bootstrap RMSE-difference 95% interval [{ci[0]:+.4f}, {ci[1]:+.4f}]."
        )
    readme = """# Prompt-level steering-effect forecast (experiment 010)

## Finding

The prespecified requirement—at least 10% lower test RMSE than the source/target mean in both models, with group-bootstrap intervals below zero—**was not met**. The small readout was worse than the mean baseline on Qwen and had only a small, uncertain gain on SmolLM2. The gradient reference was substantially more accurate on both models. This is a useful negative result about this low-cost forecaster, not evidence that prompt-specific effects cannot be forecast.

""" + "\n".join(headline) + """

![Test RMSE by forecasting method](forecast-rmse.png)

Every final-test intervention increased the queried positive-minus-negative score. The 100% sign-accuracy values are therefore degenerate and carry no evidence of useful sign prediction. Effects were all positive even for cross-aspect source/target pairs. This pattern, plus strongly aligned training directions, motivates experiment 011's shared-component control.

The target is a finite change in a synthetic candidate-label logit, not a real-world behavioral probability. The small model readout used an unsteered layer-24 activation and source/target query encoding. The gradient reference uses a model-specific derivative and is an upper-information reference, not a cheap deployable forecast. Bootstrap intervals resample 24 scenario groups; conclusions are conditional on these synthetic prompts and two small models.

The nearest direct recent work we found predicts behavior-pair side effects pooled across contexts ([Ong et al. 2026](https://arxiv.org/html/2608.11227v1)). This experiment asked a narrower prompt-specific forecasting question, but its prespecified readout gate failed, so it does not support a LessWrong research claim by itself. Broader geometric reliability work is also relevant ([Braun 2025](https://arxiv.org/abs/2505.22637); [Imdad 2026](https://zenodo.org/records/20710572)). A further geometric decomposition separates steering-induced angular and radial changes ([Aparin & Gaintseva 2026](https://arxiv.org/abs/2606.06735)); the local experiment 011 uses a different shared-task-direction decomposition and will be interpreted narrowly.

Protocol: [`010-prompt-level-effect-forecast.md`](../../docs/experiments/010-prompt-level-effect-forecast.md). Reproduce the analysis from ignored local captures with:

```bash
uv run python scripts/prompt_effect_forecast.py analyze --root runs/prompt-effect-forecast-v1
uv run python scripts/package_prompt_effect_forecast.py \\
  runs/prompt-effect-forecast-v1 results/prompt-effect-forecast-v1
```

Full per-row outcomes and gradient records are gzip-compressed here; activations are excluded. `audit.json` validates row counts, group counts, checkpoint hashes, and protocol-commit provenance. `SHA256SUMS` covers every file in this bundle.
"""
    (args.output / "README.md").write_text(readme)
    lines = []
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (args.output / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"Packaged validated 010 results to {args.output}")


if __name__ == "__main__":
    main()
