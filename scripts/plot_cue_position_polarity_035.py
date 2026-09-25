"""Plot the exploratory cue-position responses from Experiment 035."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULT = Path("results/cue-position-polarity-v1")
ANALYSIS = RESULT / "analysis.json"
OUTPUT = RESULT / "figures" / "cue-position-accuracy.png"
COLORS = {"early": "#087e8b", "late": "#d95d39"}
LABELS = {"early": "Cue visible by 8 words", "late": "Cue begins after word 7"}
CHECKPOINTS = (8, 12)


def main():
    analysis = json.loads(ANALYSIS.read_text())["analysis"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    panels = (
        ("qwen2.5-3b", "Qwen2.5-3B accuracy", "binary_accuracy_by_judge_position_and_prefix"),
        ("phi3-mini", "Phi-3 Mini accuracy", "binary_accuracy_by_judge_position_and_prefix"),
        (None, "Laya mixed/unclear", "laya_mixed_unclear_by_position_and_prefix"),
    )
    for axis, (judge, title, key) in zip(axes, panels):
        for position in ("early", "late"):
            values = analysis[key][judge][position] if judge else analysis[key][position]
            estimates = [values[str(checkpoint)]["estimate"] for checkpoint in CHECKPOINTS]
            lower = [
                estimate - values[str(checkpoint)]["cluster_bootstrap_95_ci"][0]
                for checkpoint, estimate in zip(CHECKPOINTS, estimates)
            ]
            upper = [
                values[str(checkpoint)]["cluster_bootstrap_95_ci"][1] - estimate
                for checkpoint, estimate in zip(CHECKPOINTS, estimates)
            ]
            axis.errorbar(
                CHECKPOINTS,
                estimates,
                yerr=[lower, upper],
                marker="o",
                linewidth=2,
                capsize=4,
                color=COLORS[position],
                label=LABELS[position],
            )
        axis.set_title(title)
        axis.set_xlabel("Visible prefix (words)")
        axis.set_xticks(CHECKPOINTS)
        axis.set_ylim(0, 1.02)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Proportion")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Judgments when the same sentiment clause appears early or late", y=1.03)
    fig.text(
        0.5,
        -0.02,
        "384 short synthetic reviews; 95% scaffold-cluster intervals. The 12-word capability gate failed.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
