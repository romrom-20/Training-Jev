"""Plot pre-opinion polarity prediction and evidence-state abstention rates."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULT = Path("results/natural-opinion-span-abstention-v1")
OUTPUT = RESULT / "figures" / "opinion-span-evidence-mismatch.png"
JUDGES = ("qwen2.5-3b", "phi3-mini", "laya")
NAMES = {"qwen2.5-3b": "Qwen2.5-3B", "phi3-mini": "Phi-3 Mini", "laya": "Laya"}
COLORS = {"guess": "#226f90", "before": "#087e8b", "after": "#d95d39"}


def errorbar(metric):
    value = metric["estimate"]
    low, high = metric["cluster_bootstrap_95_ci"]
    return value, value - low, high - value


def main():
    report = json.loads((RESULT / "analysis.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))

    names = [NAMES[judge] for judge in JUDGES]
    positions = np.arange(len(JUDGES))
    values, lower, upper = [], [], []
    for judge in JUDGES:
        key = (
            "forced_binary_polarity_accuracy_before_opinion"
            if judge != "laya"
            else "polarity_accuracy_when_clear"
        )
        metric = report["pre_opinion_predictive_polarity_diagnostics"].get(judge)
        if metric:
            result = metric[key]
        else:
            result = report["laya"]["before_opinion"][key]
        value, lo, hi = errorbar(result)
        values.append(value)
        lower.append(lo)
        upper.append(hi)
    bars = axes[0].bar(
        positions,
        values,
        yerr=[lower, upper],
        capsize=4,
        color=[COLORS["guess"], COLORS["guess"], "#a276b5"],
        width=0.6,
    )
    bars[2].set_hatch("//")
    axes[0].axhline(0.5, color="#444444", linestyle="--", linewidth=1, label="Balanced chance")
    axes[0].set_xticks(positions, names)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Polarity accuracy")
    axes[0].set_title("Before annotated opinion phrase")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].grid(axis="y", alpha=0.25)
    laya_coverage = report["laya"]["before_opinion"]["polarity_coverage"]["estimate"]
    axes[0].text(
        2,
        values[2] + 0.08,
        f"Laya conditional accuracy\n(clear-label coverage {laya_coverage:.0%})",
        ha="center",
        va="bottom",
        fontsize=8,
    )

    width = 0.34
    for offset, (metric_key, label, color) in enumerate(
        (
            ("recall_before_annotated_opinion", "Abstains before span", COLORS["before"]),
            ("false_abstention_after_opinion", "Abstains after span", COLORS["after"]),
        )
    ):
        means, lows, highs = [], [], []
        for judge in JUDGES:
            metric = report["abstention_metrics"].get(judge)
            if metric is None:
                metric = report["laya"]
                subkey = "abstention_recall" if offset == 0 else "false_abstention_rate"
                value, lo, hi = errorbar(metric["before_opinion" if offset == 0 else "opinion_visible"][subkey])
            else:
                value, lo, hi = errorbar(metric[metric_key])
            means.append(value)
            lows.append(lo)
            highs.append(hi)
        axes[1].bar(
            positions + (offset - 0.5) * width,
            means,
            width,
            yerr=[lows, highs],
            capsize=3,
            color=color,
            label=label,
        )
    axes[1].set_xticks(positions, names)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Abstention tracks the span differently")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Annotated opinion visibility is not the same as evidence sufficiency", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "Natural prefixes stop before the sole annotated opinion phrase; polarity accuracy uses full-review labels. "
        "95% sentence-cluster bootstrap intervals.",
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
