"""Plot out-of-fold lexical baseline accuracies and fold-wise context gains."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULT = Path("results/lexical-context-generalization-v1")
OUTPUT = RESULT / "figures" / "cross-subset-context-baseline.png"
VARIANTS = ("aspect_only", "masked_context", "context_plus_aspect")
LABELS = {
    "aspect_only": "Aspect only",
    "masked_context": "Masked prefix",
    "context_plus_aspect": "Prefix + aspect",
}
COLORS = {"aspect_only": "#9a9a9a", "masked_context": "#087e8b", "context_plus_aspect": "#d09537"}


def main():
    report = json.loads((RESULT / "analysis.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    values, lower, upper = [], [], []
    for variant in VARIANTS:
        metric = report["accuracy"][variant]
        values.append(metric["estimate"])
        lo, hi = metric["cluster_bootstrap_95_ci"]
        lower.append(metric["estimate"] - lo)
        upper.append(hi - metric["estimate"])
    x = np.arange(len(VARIANTS))
    axes[0].bar(
        x,
        values,
        yerr=[lower, upper],
        capsize=4,
        color=[COLORS[variant] for variant in VARIANTS],
        width=0.62,
    )
    axes[0].axhline(0.5, color="#444444", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, [LABELS[variant] for variant in VARIANTS])
    axes[0].set_ylim(0.25, 0.75)
    axes[0].set_ylabel("Out-of-fold polarity accuracy")
    axes[0].set_title("Leave one SemEval subset out")
    axes[0].grid(axis="y", alpha=0.25)

    datasets = list(report["per_held_out_subset"])
    deltas = [
        report["per_held_out_subset"][dataset]["masked_context_minus_aspect_only"]
        for dataset in datasets
    ]
    axes[1].bar(
        np.arange(len(datasets)),
        deltas,
        color=["#087e8b" if value >= 0 else "#d95d39" for value in deltas],
        width=0.62,
    )
    axes[1].axhline(0, color="#444444", linestyle="--", linewidth=1)
    axes[1].set_xticks(np.arange(len(datasets)), datasets)
    axes[1].set_ylabel("Masked-context minus aspect-only accuracy")
    axes[1].set_title("Paired gain varies by held-out subset")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Small lexical context baseline is near chance out of subset", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "234 items; folds train on the other three subsets. 95% sentence-cluster intervals; domain sets are related.",
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
