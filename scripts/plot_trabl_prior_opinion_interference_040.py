"""Plot preregistered Experiment 040 outcomes."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULT = Path("results/trabl-prior-opinion-interference-v1")
OUTPUT = RESULT / "figures" / "prior-opinion-interference.png"
COLORS = {"qwen2.5-3b": "#087e8b", "phi3-mini": "#d09537", "laya": "#7655a5"}
NAMES = {"qwen2.5-3b": "Qwen 2.5 3B", "phi3-mini": "Phi-3 Mini", "laya": "Laya"}


def main():
    report = json.loads((RESULT / "analysis.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    conditions = ("aspect_only", "natural_prefix", "opinion_visible")
    labels = ("Aspect only", "Before target opinion", "Target opinion visible")
    x = np.arange(len(conditions))
    for judge in ("qwen2.5-3b", "phi3-mini"):
        metrics = [
            report["condition_metrics"][f"{judge}/forced_binary/{condition}"]
            for condition in conditions
        ]
        means = [metric["accuracy_all_trials"] for metric in metrics]
        lows = [metric["accuracy_all_trials_cluster_bootstrap"]["cluster_bootstrap_95_ci"][0] for metric in metrics]
        highs = [metric["accuracy_all_trials_cluster_bootstrap"]["cluster_bootstrap_95_ci"][1] for metric in metrics]
        offset = -0.045 if judge == "qwen2.5-3b" else 0.045
        axes[0].errorbar(
            x + offset,
            means,
            yerr=[np.asarray(means) - lows, np.asarray(highs) - means],
            marker="o",
            capsize=3,
            linewidth=1.8,
            color=COLORS[judge],
            label=NAMES[judge],
        )
    axes[0].axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, labels, rotation=12, ha="right")
    axes[0].set_ylim(0.35, 1.0)
    axes[0].set_ylabel("Forced-binary accuracy")
    axes[0].set_title("Independent 2026 travel-review test")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    entries = (
        ("qwen2.5-3b", "forced_binary"),
        ("phi3-mini", "forced_binary"),
        ("laya", "four_way"),
    )
    y = np.arange(len(entries))
    estimates, lower, upper = [], [], []
    for judge, wrapper in entries:
        metric = report["later_target_copy_rate_contrasts"][f"{judge}/{wrapper}"]
        lo, hi = metric["cluster_bootstrap_95_ci"]
        estimates.append(metric["estimate"])
        lower.append(metric["estimate"] - lo)
        upper.append(hi - metric["estimate"])
    axes[1].errorbar(
        estimates,
        y,
        xerr=[lower, upper],
        fmt="none",
        ecolor="#333333",
        capsize=4,
        linewidth=1.5,
    )
    for index, ((judge, _wrapper), estimate) in enumerate(zip(entries, estimates)):
        axes[1].scatter(estimate, index, s=55, color=COLORS[judge], zorder=3)
    axes[1].axvline(0, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_yticks(y, [NAMES[judge] for judge, _ in entries])
    axes[1].invert_yaxis()
    axes[1].set_xlim(-0.25, 0.2)
    axes[1].set_xlabel("Copy rate: natural − earlier-opinion-deleted")
    axes[1].set_title("No clear earlier-opinion copy effect")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("A second corpus shows pre-opinion signal; targeted copying remains unconfirmed", y=1.02)
    fig.text(
        0.5,
        -0.035,
        "96 target trials in 48 reviews; right panel uses 48 later targets. Bars show 95% review-cluster bootstrap intervals.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    manifest_path = RESULT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["plotter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["figure_sha256"] = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
