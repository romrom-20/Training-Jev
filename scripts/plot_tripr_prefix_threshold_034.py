"""Render the exploratory polarity-stratified result from Experiment 034."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path("results/tripr-prefix-threshold-v1")
ANALYSIS = RESULTS / "analysis.json"
OUTPUT = RESULTS / "figures" / "gold-stratified-accuracy.png"
JUDGES = ("qwen2.5-3b", "phi3-mini")
COLORS = {"qwen2.5-3b": "#087e8b", "phi3-mini": "#d95d39"}
LABELS = {"qwen2.5-3b": "Qwen 2.5 3B", "phi3-mini": "Phi-3 Mini"}
BUDGETS = (8, 12, 32)


def main():
    data = json.loads(ANALYSIS.read_text())["analysis"][
        "exploratory_gold_stratified_accuracy"
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for axis, polarity in zip(axes, ("negative", "positive")):
        for judge in JUDGES:
            metrics = data[judge][polarity]["accuracy_by_budget"]
            estimates = [metrics[str(budget)]["estimate"] for budget in BUDGETS]
            lower = [
                estimate - metrics[str(budget)]["sentence_cluster_bootstrap_95_ci"][0]
                for budget, estimate in zip(BUDGETS, estimates)
            ]
            upper = [
                metrics[str(budget)]["sentence_cluster_bootstrap_95_ci"][1] - estimate
                for budget, estimate in zip(BUDGETS, estimates)
            ]
            axis.errorbar(
                BUDGETS,
                estimates,
                yerr=[lower, upper],
                color=COLORS[judge],
                marker="o",
                linewidth=2,
                capsize=4,
                label=LABELS[judge],
            )
        axis.set_title(f"{polarity.capitalize()} source polarity")
        axis.set_xlabel("Visible answer prefix (tokens)")
        axis.set_xticks(BUDGETS)
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Strict accuracy vs TripR source label")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Judge accuracy by polarity and visible answer prefix", y=1.02)
    fig.text(
        0.5,
        -0.01,
        "95% source-sentence-cluster bootstrap intervals; TripR source labels are a proxy, not answer-level gold.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
