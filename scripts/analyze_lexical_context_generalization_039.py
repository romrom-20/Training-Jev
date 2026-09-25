"""Analyze the preregistered leave-one-subset-out lexical baseline."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RESULT = Path("results/lexical-context-generalization-v1")
PARENT_037 = Path("results/natural-opinion-span-abstention-v1")
PARENT_038 = Path("results/aspect-only-prior-control-v1")
PROTOCOL = Path("docs/experiments/039-cross-subset-context-lexical-baseline.md")
REPS = 10_000
SEED = 20260939
VARIANTS = ("aspect_only", "masked_context", "context_plus_aspect")


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


def analyze(stimuli, predictions):
    stimulus_by_id = {row["id"]: row for row in stimuli}
    if len(stimulus_by_id) != len(stimuli):
        raise ValueError("Duplicate frozen stimulus ID")
    rows_by_id = {}
    for row in predictions:
        if row["id"] in rows_by_id:
            raise ValueError(f"Duplicate prediction ID: {row['id']}")
        if row["id"] not in stimulus_by_id:
            raise ValueError(f"Prediction outside the frozen stimulus set: {row['id']}")
        stimulus = stimulus_by_id[row["id"]]
        if row["dataset"] != stimulus["dataset"] or row["gold"] != stimulus["polarity"]:
            raise ValueError("Prediction metadata differs from frozen stimuli")
        rows_by_id[row["id"]] = row
    if set(rows_by_id) != set(stimulus_by_id):
        raise ValueError("Incomplete out-of-fold predictions")

    scored = []
    for stimulus in stimuli:
        prediction = rows_by_id[stimulus["id"]]
        scored.append(
            {
                "id": stimulus["id"],
                "cluster_id": stimulus["cluster_id"],
                "dataset": stimulus["dataset"],
                "gold": stimulus["polarity"],
                **{
                    f"{variant}_correct": int(
                        prediction[f"{variant}_label"] == stimulus["polarity"]
                    )
                    for variant in VARIANTS
                },
            }
        )
    primary_rows = [
        row
        | {
            "primary_difference": row["masked_context_correct"] - row["aspect_only_correct"]
        }
        for row in scored
    ]
    accuracy = {
        variant: bootstrap(
            scored,
            lambda sample, variant=variant: np.mean(
                [row[f"{variant}_correct"] for row in sample]
            ),
            seed=SEED + index,
        )
        for index, variant in enumerate(VARIANTS)
    }
    paired = {
        "masked_context_minus_aspect_only": bootstrap(
            primary_rows,
            lambda sample: np.mean([row["primary_difference"] for row in sample]),
            seed=SEED,
        ),
        "context_plus_aspect_minus_aspect_only": bootstrap(
            [
                row
                | {"difference": row["context_plus_aspect_correct"] - row["aspect_only_correct"]}
                for row in scored
            ],
            lambda sample: np.mean([row["difference"] for row in sample]),
            seed=SEED + 10,
        ),
        "context_plus_aspect_minus_masked_context": bootstrap(
            [
                row
                | {"difference": row["context_plus_aspect_correct"] - row["masked_context_correct"]}
                for row in scored
            ],
            lambda sample: np.mean([row["difference"] for row in sample]),
            seed=SEED + 11,
        ),
    }
    by_dataset = {}
    for dataset in sorted({row["dataset"] for row in scored}):
        rows = [row for row in scored if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "n": len(rows),
            "accuracy": {
                variant: float(np.mean([row[f"{variant}_correct"] for row in rows]))
                for variant in VARIANTS
            },
            "masked_context_minus_aspect_only": float(
                np.mean(
                    [row["masked_context_correct"] - row["aspect_only_correct"] for row in rows]
                )
            ),
        }
    macro = {
        variant: float(np.mean([report["accuracy"][variant] for report in by_dataset.values()]))
        for variant in VARIANTS
    }
    gate = (
        paired["masked_context_minus_aspect_only"]["estimate"] >= 0.05
        and paired["masked_context_minus_aspect_only"]["cluster_bootstrap_95_ci"][0] > 0
    )
    return {
        "experiment": "039",
        "primary_endpoint": "paired out-of-fold accuracy difference: aspect-masked pre-opinion context minus aspect-only TF-IDF/logistic baseline",
        "accuracy": accuracy,
        "paired_differences": paired,
        "per_held_out_subset": by_dataset,
        "macro_subset_accuracy": macro,
        "chance_accuracy": 0.5,
        "preregistered_diagnostic_gate_passed": gate,
        "n_stimuli": len(stimuli),
        "n_sentence_clusters": len({row["cluster_id"] for row in stimuli}),
        "n_out_of_fold_predictions": len(predictions),
        "limitations": [
            "The four SemEval subsets are small and related; three are restaurant sets, and only 14lap is a different product domain.",
            "The held-out fold is a dataset split, not an independently collected corpus.",
            "Cluster intervals are conditional on the four fitted training-fold models and do not estimate training-set variation.",
            "Review text and opinion annotations are not redistributed due missing upstream license metadata.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(".context/lexical-context-generalization-039"))
    args = parser.parse_args()
    stimuli = json.loads((args.results / "stimuli.json").read_text())
    predictions = json.loads((args.results / "predictions.json").read_text())
    manifest = json.loads((args.results / "manifest.json").read_text())
    analysis = analyze(stimuli, predictions)
    (args.results / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    checks = {
        "234_frozen_stimuli": len(stimuli) == 234,
        "234_complete_oof_predictions": len(predictions) == 234,
        "balanced_sentiment": Counter(row["polarity"] for row in stimuli)
        == {"positive": 117, "negative": 117},
        "manifest_stimuli_hash_matches": manifest["stimuli_sha256"]
        == hashlib.sha256((args.results / "stimuli.json").read_bytes()).hexdigest(),
        "manifest_prediction_hash_matches": manifest["predictions_sha256"]
        == hashlib.sha256((args.results / "predictions.json").read_bytes()).hexdigest(),
        "manifest_protocol_hash_matches": manifest["protocol_sha256"]
        == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "source_revision_pinned": manifest["source_revision"]
        == "d0df6600b259b6114de23cc5047c7e776cd89750",
        "no_review_text_in_public_artifacts": all(
            all(key not in row for key in ("text", "sentence", "visible_text", "tokens"))
            for row in stimuli + predictions
        ),
        "all_4_held_out_subsets_present": set(analysis["per_held_out_subset"])
        == {"14res", "14lap", "15res", "16res"},
    }
    (args.results / "audit.json").write_text(
        json.dumps(
            {
                "checks": checks,
                "checks_passed": all(checks.values()),
                "preregistered_diagnostic_gate_passed": analysis[
                    "preregistered_diagnostic_gate_passed"
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(analysis, indent=2), flush=True)


if __name__ == "__main__":
    main()
