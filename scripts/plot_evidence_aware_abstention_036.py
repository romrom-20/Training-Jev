"""Plot paired decision utility and abstention behavior for Experiment 036."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULT = Path("results/evidence-aware-abstention-v1")
ANALYSIS = RESULT / "analysis.json"
OUTPUT = RESULT / "figures" / "abstention-policy.png"
JUDGES = ("qwen2.5-3b", "phi3-mini", "laya")
COLORS = {"forced": "#a6a6a6", "abstain": "#087e8b", "no_cue": "#087e8b", "false": "#d95d39"}
LABELS = {"qwen2.5-3b": "Qwen2.5-3B", "phi3-mini": "Phi-3 Mini", "laya": "Laya"}


def err(metric):
    value = metric["estimate"]
    lower, upper = metric["cluster_bootstrap_95_ci"]
    return value, value - lower, upper - value


def main():
    report = json.loads(ANALYSIS.read_text())["analysis"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    binary = ("qwen2.5-3b", "phi3-mini")
    positions = [float(index) for index in range(len(binary))]
    width = 0.33
    for offset, (key, label) in enumerate(
        (("forced_binary_appropriate_accuracy", "Forced binary"),
         ("abstention_enabled_appropriate_accuracy", "With abstention"))
    ):
        estimates, lower, upper = [], [], []
        for judge in binary:
            value, lo, hi = err(report["appropriate_accuracy_by_binary_judge"][judge][key])
            estimates.append(value)
            lower.append(lo)
            upper.append(hi)
        axes[0].bar(
            [position + (offset - 0.5) * width for position in positions],
            estimates,
            width,
            yerr=[lower, upper],
            capsize=3,
            color=COLORS["forced" if offset == 0 else "abstain"],
            label=label,
        )
    axes[0].set_xticks(positions, [LABELS[judge] for judge in binary])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Appropriate-decision accuracy")
    axes[0].set_title("Same prefixes, paired wrapper change")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    positions = [float(index) for index in range(len(JUDGES))]
    for offset, (key, label) in enumerate(
        (("appropriate_insufficient_on_late_8", "Abstains when cue is absent"),
         ("false_insufficient_on_visible_cue", "Abstains despite visible cue"))
    ):
        estimates, lower, upper = [], [], []
        for judge in JUDGES:
            value, lo, hi = err(report["no_cue_abstention_and_visible_false_abstention"][judge][key])
            estimates.append(value)
            lower.append(lo)
            upper.append(hi)
        axes[1].bar(
            [position + (offset - 0.5) * width for position in positions],
            estimates,
            width,
            yerr=[lower, upper],
            capsize=3,
            color=COLORS["no_cue" if offset == 0 else "false"],
            label=label,
        )
    axes[1].set_xticks(positions, [LABELS[judge] for judge in JUDGES])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Abstention behavior")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Explicit abstention improves evidence-aware decisions", y=1.02)
    fig.text(
        0.5,
        -0.015,
        "95% scaffold-cluster bootstrap intervals; 192 balanced synthetic prompts; not natural-answer validation.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
