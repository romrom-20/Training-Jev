"""Compare natural-prefix sentiment predictions with an aspect-only prior baseline."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RESULT = Path("results/aspect-only-prior-control-v1")
PARENT = Path("results/natural-opinion-span-abstention-v1")
PROTOCOL = Path("docs/experiments/038-aspect-only-prior-control.md")
JUDGES = ("qwen2.5-3b", "phi3-mini")
REPS = 10_000
SEED = 20260938


def bootstrap(rows, statistic, reps=REPS, seed=SEED):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["cluster_id"]].append(row)
    clusters = np.asarray(sorted(grouped))
    if not len(clusters):
        return {"estimate": None, "cluster_bootstrap_95_ci": None, "n": 0}
    estimate = float(statistic(rows))
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for index in range(reps):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        sample = [row for cluster in chosen for row in grouped[cluster]]
        draws[index] = float(statistic(sample))
    return {
        "estimate": estimate,
        "cluster_bootstrap_95_ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "n": len(rows),
        "n_sentence_clusters": len(clusters),
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
    }


def analyze(stimuli, natural_outcomes, control_outcomes):
    natural = {}
    for row in natural_outcomes:
        key = (row["judge"], row["wrapper"], row["id"], row["condition"])
        if key in natural:
            raise ValueError(f"Duplicate parent outcome: {key}")
        natural[key] = row
    control = {}
    for row in control_outcomes:
        key = (row["judge"], row["wrapper"], row["id"], row["condition"])
        if key in control:
            raise ValueError(f"Duplicate control outcome: {key}")
        control[key] = row

    ids = {row["id"] for row in stimuli}
    expected_control = {
        (judge, wrapper, stimulus_id, "aspect_only")
        for judge in JUDGES
        for wrapper in ("forced_binary", "abstention")
        for stimulus_id in ids
    } | {
        ("laya", "four_way", stimulus_id, "aspect_only") for stimulus_id in ids
    }
    if set(control) != expected_control:
        raise ValueError("Aspect-only output join is incomplete or unexpected")

    paired = []
    by_judge = {judge: [] for judge in JUDGES}
    per_judge = {}
    abstention_control = {}
    for judge_index, judge in enumerate(JUDGES):
        judge_rows = []
        for stimulus in stimuli:
            natural_row = natural[(judge, "forced_binary", stimulus["id"], "before_opinion")]
            control_row = control[(judge, "forced_binary", stimulus["id"], "aspect_only")]
            if natural_row["polarity"] != stimulus["polarity"] or control_row["polarity"] != stimulus["polarity"]:
                raise ValueError("Polarity metadata does not match frozen stimulus")
            if natural_row["cluster_id"] != stimulus["cluster_id"] or control_row["cluster_id"] != stimulus["cluster_id"]:
                raise ValueError("Cluster metadata does not match frozen stimulus")
            if natural_row["label"] not in {"positive", "negative"} or control_row["label"] not in {
                None,
                "positive",
                "negative",
            }:
                raise ValueError("Invalid forced-binary label")
            row = {
                "cluster_id": stimulus["cluster_id"],
                "dataset": stimulus["dataset"],
                "id": stimulus["id"],
                "judge": judge,
                "polarity": stimulus["polarity"],
                "natural_label": natural_row["label"],
                "aspect_only_label": control_row["label"],
                "natural_correct": int(natural_row["label"] == stimulus["polarity"]),
                "aspect_only_correct": int(control_row["label"] == stimulus["polarity"]),
                "improvement": int(natural_row["label"] == stimulus["polarity"])
                - int(control_row["label"] == stimulus["polarity"]),
            }
            paired.append(row)
            by_judge[judge].append(row)
            judge_rows.append(row)
        per_judge[judge] = {
            "natural_prefix_accuracy": bootstrap(
                judge_rows,
                lambda sample: np.mean([row["natural_correct"] for row in sample]),
                seed=SEED + 10 + judge_index,
            ),
            "aspect_only_accuracy": bootstrap(
                judge_rows,
                lambda sample: np.mean([row["aspect_only_correct"] for row in sample]),
                seed=SEED + 20 + judge_index,
            ),
            "paired_context_gain": bootstrap(
                judge_rows,
                lambda sample: np.mean([row["improvement"] for row in sample]),
                seed=SEED + 30 + judge_index,
            ),
        }

        abstention_rows = []
        for stimulus in stimuli:
            baseline = natural[(judge, "abstention", stimulus["id"], "before_opinion")]
            aspect = control[(judge, "abstention", stimulus["id"], "aspect_only")]
            abstention_rows.append(
                {
                    "cluster_id": stimulus["cluster_id"],
                    "natural_abstains": int(baseline["label"] == "insufficient"),
                    "aspect_only_abstains": int(aspect["label"] == "insufficient"),
                    "aspect_only_label": aspect["label"],
                }
            )
        abstention_control[judge] = {
            "natural_prefix_abstention_rate": bootstrap(
                abstention_rows,
                lambda sample: np.mean([row["natural_abstains"] for row in sample]),
                seed=SEED + 40 + judge_index,
            ),
            "aspect_only_abstention_rate": bootstrap(
                abstention_rows,
                lambda sample: np.mean([row["aspect_only_abstains"] for row in sample]),
                seed=SEED + 50 + judge_index,
            ),
            "aspect_only_parse_coverage": float(
                np.mean([row["aspect_only_label"] is not None for row in abstention_rows])
            ),
        }

    laya_counts = Counter()
    laya_clear = []
    laya_items = []
    for stimulus in stimuli:
        row = control[("laya", "four_way", stimulus["id"], "aspect_only")]
        label = row["label"]
        laya_counts[label] += 1
        laya_items.append(
            {
                "cluster_id": stimulus["cluster_id"],
                "polarity": stimulus["polarity"],
                "label": label,
            }
        )
        if label in {"positive", "negative"}:
            laya_clear.append(
                {
                    "cluster_id": stimulus["cluster_id"],
                    "correct": int(label == stimulus["polarity"]),
                }
            )
    laya_summary = {
        "label_counts": dict(laya_counts),
        "unclear_rate": bootstrap(
            laya_items,
            lambda sample: np.mean([row["label"] == "unclear" for row in sample]),
            seed=SEED + 60,
        ),
        "clear_polarity_coverage": bootstrap(
            laya_items,
            lambda sample: np.mean([row["label"] in {"positive", "negative"} for row in sample]),
            seed=SEED + 61,
        ),
        "polarity_accuracy_when_clear": bootstrap(
            laya_clear,
            lambda sample: np.mean([row["correct"] for row in sample]),
            seed=SEED + 62,
        ),
    }

    primary = bootstrap(
        paired,
        lambda sample: np.mean([row["improvement"] for row in sample]),
        seed=SEED,
    )
    per_dataset = {}
    for dataset in sorted({row["dataset"] for row in paired}):
        rows = [row for row in paired if row["dataset"] == dataset]
        per_dataset[dataset] = {
            "n": len(rows),
            "natural_prefix_accuracy": float(np.mean([row["natural_correct"] for row in rows])),
            "aspect_only_accuracy": float(np.mean([row["aspect_only_correct"] for row in rows])),
            "paired_context_gain": float(np.mean([row["improvement"] for row in rows])),
        }
    passed = primary["estimate"] >= 0.05 and primary["cluster_bootstrap_95_ci"][0] > 0
    return {
        "experiment": "038",
        "status": "exploratory preregistered follow-up; not independent confirmation",
        "primary_endpoint": "paired Qwen/Phi forced-binary polarity accuracy gain: natural pre-opinion prefix minus aspect-only prompt",
        "primary_pooled_context_gain": primary,
        "per_judge": per_judge,
        "abstention_behavior": abstention_control,
        "laya_aspect_only": laya_summary,
        "per_dataset": per_dataset,
        "preregistered_diagnostic_gate_passed": passed,
        "n_stimuli": len(stimuli),
        "n_sentence_clusters": len({row["cluster_id"] for row in stimuli}),
        "n_aspect_only_outcomes": len(control_outcomes),
        "limitations": [
            "This follow-up was selected after inspecting Experiment 037 outcomes and is exploratory.",
            "Aspect-only text is an artificial baseline; it measures target priors and prompt behavior, not natural review understanding.",
            "The SemEval reviews are old and may have appeared in model pretraining.",
            "The corpus is small and limited to restaurant/laptop reviews and three local judge systems.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(".context/aspect-only-prior-control-038"))
    parser.add_argument("--parent", type=Path, default=PARENT)
    args = parser.parse_args()
    stimuli = json.loads((args.results / "stimuli.json").read_text())
    control_outcomes = json.loads((args.results / "predictions.json").read_text())
    natural_outcomes = json.loads((args.parent / "predictions.json").read_text())
    analysis = analyze(stimuli, natural_outcomes, control_outcomes)
    (args.results / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps(analysis, indent=2), flush=True)


if __name__ == "__main__":
    main()
