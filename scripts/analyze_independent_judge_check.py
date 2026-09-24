"""Audit and summarize experiment 028's cross-family judge results."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_prompt_format_score_generation import MODELS

REFERENCE = Path("results/prompt-format-score-generation-v1")
QWEN = Path("results/local-open-judge-v1/analysis.json")
QWEN_SCREEN = Path("results/local-judge-validation-v1/analysis.json")
PROTOCOL = Path("docs/experiments/028-independent-judge-check.md")
RUNNER = Path("scripts/independent_judge_check.py")
DATA = Path(".context/datasets/semeval2014/Restaurants_Test_Gold.xml")
OUT = Path("results/independent-judge-check-v1")
REPS = 10_000
SEED = 20261028
DISPLAY = {
    "granite-3.1-2b": "Granite 3.1 2B",
    "qwen-1.5b": "Qwen2.5 1.5B",
    "smollm2-1.7b": "SmolLM2 1.7B",
}


def bootstrap(rows, metric, seed=SEED):
    groups = {}
    for row in rows:
        groups.setdefault(row["sentence_id"], []).append(row)
    sentence_ids = np.asarray(sorted(groups))
    rng = np.random.default_rng(seed)
    estimates = np.empty(REPS)
    for index in range(REPS):
        chosen = rng.choice(sentence_ids, size=len(sentence_ids), replace=True)
        sample = [row for sentence_id in chosen for row in groups[sentence_id]]
        estimates[index] = metric(sample)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def accuracy(rows, key):
    return float(np.mean([row[key] == row["gold"] for row in rows]))


def parseability(rows, key):
    return float(np.mean([row[key] is not None for row in rows]))


def agreement(rows):
    valid = [row for row in rows if row["phi_label"] is not None and row["qwen_label"] is not None]
    point = float(np.mean([row["phi_label"] == row["qwen_label"] for row in valid]))
    return {
        "n_both_parseable": len(valid),
        "agreement": point,
        "sentence_cluster_bootstrap_95_ci": bootstrap(
            valid, lambda sample: np.mean([row["phi_label"] == row["qwen_label"] for row in sample])
        ),
    }


def load_reference(name):
    with gzip.open(REFERENCE / f"{name}-outcomes.json.gz", "rt") as stream:
        outcomes = json.load(stream)
    return {row["id"]: row for row in outcomes if row["format"] == "open_question"}


def create_plot(summary, out):
    names = list(MODELS)
    labels = [DISPLAY[name] for name in names]
    x = np.arange(len(names))
    width = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    colors = ["#384B76", "#E07A5F", "#3D9970"]
    series = [
        ("Open candidate-pair accuracy", "candidate_accuracy"),
        ("Qwen judge accuracy", "qwen_accuracy"),
        ("Phi-3 judge accuracy", "phi_accuracy"),
    ]
    for index, (label, key) in enumerate(series):
        values = [summary["models"][name][key] for name in names]
        axes[0].bar(x + (index - 1) * width, values, width, label=label, color=colors[index])
    axes[0].set_xticks(x, labels, rotation=14, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy against review aspect label")
    axes[0].set_title("Score versus judged open answer")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[0].grid(axis="y", alpha=0.2)

    values = [summary["models"][name]["judge_agreement"]["agreement"] for name in names]
    lower = [
        value - summary["models"][name]["judge_agreement"]["sentence_cluster_bootstrap_95_ci"][0]
        for name, value in zip(names, values)
    ]
    upper = [
        summary["models"][name]["judge_agreement"]["sentence_cluster_bootstrap_95_ci"][1] - value
        for name, value in zip(names, values)
    ]
    axes[1].bar(x, values, color="#5C677D", width=0.58)
    axes[1].errorbar(x, values, yerr=[lower, upper], fmt="none", ecolor="#222222", capsize=4)
    axes[1].set_xticks(x, labels, rotation=14, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Agreement, both judges parseable")
    axes[1].set_title("Qwen–Phi-3 judge agreement")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Experiment 028: judge labels vary by target family", fontsize=13)
    fig.text(
        0.5,
        0.005,
        "Accuracy bars are descriptive; judges were validated on source reviews, not human-labeled generations.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(root=OUT):
    captured = json.loads((root / "analysis.json").read_text())
    qwen = json.loads(QWEN.read_text())
    qwen_screen = json.loads(QWEN_SCREEN.read_text())
    references = {name: load_reference(name) for name in MODELS}
    screen_rows = captured["source_screen_outcomes"]
    source_accuracy = float(np.mean([row["judge_label"] == row["gold"] for row in screen_rows]))
    negative_rows = [row for row in screen_rows if row["gold"] == 0]
    source_negative_recall = float(np.mean([row["judge_label"] == 0 for row in negative_rows]))
    checks = {
        "phi_source_gate_recomputed": len(screen_rows) == 233
        and abs(source_accuracy - captured["source_label_screen"]["accuracy"]) < 1e-12
        and abs(source_negative_recall - captured["source_label_screen"]["negative_recall"]) < 1e-12
        and captured["source_label_screen"]["gate_passed"]
        == (source_accuracy >= 0.90 and source_negative_recall >= 0.80),
        "same_source_screen_items_as_qwen": {row["id"]: row["gold"] for row in screen_rows}
        == {row["id"]: row["gold"] for row in qwen_screen["outcomes"]},
        "frozen_protocol_hash_matches": captured["protocol_sha256"]
        == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "judge_revision_matches_protocol": captured["judge"]["revision"]
        == "f39ac1d28e925b323eae81227eaba4464caced4e",
        "dataset_and_runner_hashes_match": captured["dataset_sha256"]
        == hashlib.sha256(DATA.read_bytes()).hexdigest()
        and captured["runner_sha256"] == hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
        "qwen_labels_match_experiment_027": all(
            row["qwen_label"]
            == next(
                result["judge_label"]
                for result in qwen["outcomes"]
                if result["model"] == row["model"] and result["id"] == row["id"]
            )
            for row in captured["outcomes"]
        ),
        "margins_and_gold_match_experiment_025": all(
            row["id"] in references[row["model"]]
            and row["gold"] == references[row["model"]][row["id"]]["label"]
            and row["candidate_pair_prediction"]
            == references[row["model"]][row["id"]]["candidate_pair_prediction"]
            and abs(
                row["candidate_margin"] - references[row["model"]][row["id"]]["candidate_margin"]
            )
            <= 1e-5
            for row in captured["outcomes"]
        ),
        "target_revisions_match_experiment_025": all(
            captured["target_models"][name]["revision"]
            == json.loads((REFERENCE / f"{name}-manifest.json").read_text())["model"]["revision"]
            for name in MODELS
        ),
        "699_unique_rows_and_no_raw_text": len(captured["outcomes"]) == 699
        and len({(row["model"], row["id"]) for row in captured["outcomes"]}) == 699
        and all(
            not ({"answer", "text", "user", "review"} & row.keys())
            for row in [*captured["source_screen_outcomes"], *captured["outcomes"]]
        ),
    }
    summary = {
        "experiment": 28,
        "bootstrap": {"repetitions": REPS, "seed": SEED, "cluster": "original sentence_id"},
        "source_screen": captured["source_label_screen"],
        "target_review_label_prevalence": {
            "positive_rate": float(np.mean([row["gold"] for row in captured["outcomes"]])),
            "always_positive_accuracy": float(
                np.mean([row["gold"] == 1 for row in captured["outcomes"]])
            ),
        },
        "models": {},
        "audit": {"checks": checks, "checks_passed": all(checks.values())},
        "interpretation_limit": (
            "Agreement is judge robustness, not answer correctness. Neither judge was human-validated "
            "on generated text; low accuracy can reflect model errors, evaluator shift, or both."
        ),
    }
    for name in MODELS:
        rows = [row for row in captured["outcomes"] if row["model"] == name]
        valid_phi = [row for row in rows if row["phi_label"] is not None]
        summary["models"][name] = {
            "n": len(rows),
            "candidate_accuracy": accuracy(rows, "candidate_pair_prediction"),
            "qwen_accuracy": accuracy(rows, "qwen_label"),
            "phi_accuracy": accuracy(rows, "phi_label"),
            "candidate_minus_qwen_accuracy_sentence_cluster_bootstrap_95_ci": bootstrap(
                rows,
                lambda sample: (
                    accuracy(sample, "candidate_pair_prediction") - accuracy(sample, "qwen_label")
                ),
            ),
            "candidate_minus_phi_accuracy_sentence_cluster_bootstrap_95_ci": bootstrap(
                rows,
                lambda sample: (
                    accuracy(sample, "candidate_pair_prediction") - accuracy(sample, "phi_label")
                ),
            ),
            "qwen_parseability": parseability(rows, "qwen_label"),
            "phi_parseability": parseability(rows, "phi_label"),
            "phi_qwen_accuracy_difference_sentence_cluster_bootstrap_95_ci": bootstrap(
                rows,
                lambda sample: accuracy(sample, "phi_label") - accuracy(sample, "qwen_label"),
            ),
            "judge_agreement": agreement(rows),
            "judge_positive_rates": {
                "qwen": float(
                    np.mean([row["qwen_label"] for row in rows if row["qwen_label"] is not None])
                ),
                "phi": float(np.mean([row["phi_label"] for row in valid_phi]))
                if valid_phi
                else None,
            },
        }
    if not summary["audit"]["checks_passed"]:
        raise ValueError("Experiment 028 audit failed")
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (root / "audit.json").write_text(json.dumps(summary["audit"], indent=2, sort_keys=True) + "\n")
    create_plot(summary, root / "judge-agreement.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=OUT)
    analyze(parser.parse_args().root)
