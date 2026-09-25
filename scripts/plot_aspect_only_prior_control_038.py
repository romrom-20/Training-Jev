"""Plot natural-prefix gains over an aspect-only baseline."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULT = Path("results/aspect-only-prior-control-v1")
PARENT = Path("results/natural-opinion-span-abstention-v1")
OUTPUT = RESULT / "figures" / "context-versus-aspect-prior.png"
JUDGES = ("qwen2.5-3b", "phi3-mini", "laya")
NAMES = {"qwen2.5-3b": "Qwen2.5-3B", "phi3-mini": "Phi-3 Mini", "laya": "Laya"}
COLORS = {"natural": "#087e8b", "aspect": "#9a9a9a", "after": "#d95d39"}


def uncertainty(metric):
    if metric["estimate"] is None:
        return 0.0, 0.0, 0.0
    value = metric["estimate"]
    low, high = metric["cluster_bootstrap_95_ci"]
    return value, value - low, high - value


def main():
    result = json.loads((RESULT / "analysis.json").read_text())
    parent = json.loads((PARENT / "analysis.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))

    binary = ("qwen2.5-3b", "phi3-mini")
    positions = np.arange(2)
    width = 0.34
    for offset, (condition, label, color) in enumerate(
        (("natural_prefix_accuracy", "Natural pre-opinion prefix", COLORS["natural"]),
         ("aspect_only_accuracy", "Aspect only", COLORS["aspect"]))
    ):
        values, lower, upper = [], [], []
        for judge in binary:
            metric = result["per_judge"][judge][condition if condition != "aspect_only_accuracy" else condition]
            value, lo, hi = uncertainty(metric)
            values.append(value)
            lower.append(lo)
            upper.append(hi)
        axes[0].bar(
            positions + (offset - 0.5) * width,
            values,
            width,
            yerr=[lower, upper],
            capsize=3,
            color=color,
            label=label,
        )
    axes[0].axhline(0.5, color="#444444", linestyle="--", linewidth=1, label="Balanced chance")
    axes[0].set_xticks(positions, [NAMES[judge] for judge in binary])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Forced-binary polarity accuracy")
    axes[0].set_title("Words before the opinion span add signal")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    positions = np.arange(len(JUDGES))
    for offset, (label, color) in enumerate(
        (("Natural pre-opinion prefix", COLORS["natural"]), ("Aspect only", COLORS["aspect"]))
    ):
        values, lower, upper = [], [], []
        for judge in JUDGES:
            if judge == "laya":
                metric = (
                    parent["laya"]["before_opinion"]["abstention_recall"]
                    if offset == 0
                    else result["laya_aspect_only"]["unclear_rate"]
                )
            else:
                metric = (
                    parent["abstention_metrics"][judge]["recall_before_annotated_opinion"]
                    if offset == 0
                    else result["abstention_behavior"][judge]["aspect_only_abstention_rate"]
                )
            value, lo, hi = uncertainty(metric)
            values.append(value)
            lower.append(lo)
            upper.append(hi)
        axes[1].bar(
            positions + (offset - 0.5) * width,
            values,
            width,
            yerr=[lower, upper],
            capsize=3,
            color=color,
            label=label,
        )
    axes[1].set_xticks(positions, [NAMES[judge] for judge in JUDGES])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Abstention / unclear rate")
    axes[1].set_title("Text-free control triggers abstention")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Natural pre-opinion context beats the target-name baseline", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "234 balanced items; aspect-only input is an artificial control. 95% sentence-cluster bootstrap intervals.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
