"""Recompute analysis 027 from captured labels and frozen 025 outcomes."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from analyze_prompt_format_score_generation import MODELS

ROOT = Path("results/local-open-judge-v1")
REFERENCE = Path("results/prompt-format-score-generation-v1")
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261024


def load_reference(name):
    with gzip.open(REFERENCE / f"{name}-outcomes.json.gz", "rt") as stream:
        rows = json.load(stream)
    return {row["id"]: row for row in rows if row["format"] == "open_question"}


def sentence_bootstrap(rows, value_fn, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    groups = {}
    for row in rows:
        groups.setdefault(row["sentence_id"], []).append(row)
    ids = np.asarray(sorted(groups))
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps)
    for index in range(reps):
        chosen = rng.choice(ids, size=len(ids), replace=True)
        sampled = [row for sentence_id in chosen for row in groups[sentence_id]]
        estimates[index] = value_fn(sampled)
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def summarize(rows):
    n = len(rows)
    valid = [row for row in rows if row["judge_label"] is not None]

    def candidate_correct(subset):
        return np.mean([row["candidate_pair_prediction"] == row["gold"] for row in subset])

    def judge_correct(subset):
        return np.mean([row["judge_label"] == row["gold"] for row in subset])

    def joint_difference(subset):
        return candidate_correct(subset) - judge_correct(subset)

    def agreement(subset):
        return np.mean([row["candidate_pair_prediction"] == row["judge_label"] for row in subset])

    matrix = [
        [
            sum(row["gold"] == gold and row["judge_label"] == prediction for row in rows)
            for prediction in (0, 1, None)
        ]
        for gold in (0, 1)
    ]
    categories = {}
    for category in sorted({row["category"] for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        categories[category] = {
            "n": len(subset),
            "gold_positive_rate": float(np.mean([row["gold"] for row in subset])),
            "judge_accuracy_all_items": float(
                np.mean([row["judge_label"] == row["gold"] for row in subset])
            ),
            "judge_parseability": float(
                np.mean([row["judge_label"] is not None for row in subset])
            ),
        }
    return {
        "n": n,
        "judge_parseability": len(valid) / n,
        "judge_accuracy_all_items": float(
            np.mean([row["judge_label"] == row["gold"] for row in rows])
        ),
        "candidate_pair_accuracy": float(candidate_correct(rows)),
        "candidate_minus_judge_accuracy_paired_sentence_bootstrap_95_ci": sentence_bootstrap(
            rows, joint_difference
        ),
        "candidate_judge_agreement_among_parseable": float(agreement(valid)) if valid else None,
        "candidate_judge_agreement_sentence_bootstrap_95_ci": (
            sentence_bootstrap(valid, agreement) if valid else None
        ),
        "gold_positive_rate": float(np.mean([row["gold"] for row in rows])),
        "judge_positive_rate_among_parseable": float(
            np.mean([row["judge_label"] for row in valid])
        ),
        "gold_by_judge_confusion_with_unparsed": matrix,
        "by_category": categories,
    }


def analyze(root=ROOT):
    captured = json.loads((root / "analysis.json").read_text())
    reference = {name: load_reference(name) for name in MODELS}
    checks = {
        "frozen_source_gate_passed": captured["source_label_screen"]["gate_passed"],
        "same_model_item_ids_and_gold_as_025": all(
            len([row for row in captured["outcomes"] if row["model"] == name]) == 233
            and all(
                row["id"] in reference[name] and row["gold"] == reference[name][row["id"]]["label"]
                for row in captured["outcomes"]
                if row["model"] == name
            )
            for name in MODELS
        ),
        "open_candidate_margins_reproduce_025": all(
            abs(row["candidate_margin"] - reference[row["model"]][row["id"]]["candidate_margin"])
            <= 1e-5
            and row["candidate_pair_prediction"]
            == reference[row["model"]][row["id"]]["candidate_pair_prediction"]
            for row in captured["outcomes"]
        ),
        "raw_text_absent": all(
            not ({"answer", "text", "user", "review"} & row.keys()) for row in captured["outcomes"]
        ),
        "expected_factorial_complete": len(captured["outcomes"]) == 699
        and len({(row["model"], row["id"]) for row in captured["outcomes"]}) == 699,
    }
    summary = {
        "experiment": 27,
        "analysis_type": "descriptive; sentence-cluster bootstrap intervals are exploratory",
        "bootstrap": {"repetitions": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED},
        "models": {
            name: summarize([row for row in captured["outcomes"] if row["model"] == name])
            for name in MODELS
        },
        "audit": {"checks": checks, "checks_passed": all(checks.values())},
        "limitation": (
            "A single judge was evaluated on source reviews, not human-labeled generated answers. "
            "Low agreement can reflect judge distribution shift, generation errors, or both."
        ),
    }
    if not summary["audit"]["checks_passed"]:
        raise ValueError("Experiment 027 audit failed")
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (root / "audit.json").write_text(json.dumps(summary["audit"], indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    analyze(parser.parse_args().root)
