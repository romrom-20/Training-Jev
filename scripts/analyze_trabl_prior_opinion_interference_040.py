"""Analyze the frozen local judgments from Experiment 040."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from trabl_prior_opinion_interference_040 import SOURCE_REVISION, STIMULI

RESULT = Path("results/trabl-prior-opinion-interference-v1")
PROTOCOL = Path("docs/experiments/040-trabl-prior-opinion-interference.md")
REPS = 10_000
SEED = 20260940
CONDITIONS = ("aspect_only", "natural_prefix", "prior_opinion_deleted", "opinion_visible")


def cluster_bootstrap(rows, statistic, seed):
    by_cluster = defaultdict(list)
    for row in rows:
        by_cluster[row["cluster_id"]].append(row)
    clusters = np.asarray(sorted(by_cluster))
    if not len(clusters):
        return {"estimate": None, "cluster_bootstrap_95_ci": None, "n": 0}
    estimate = float(statistic(rows))
    rng = np.random.default_rng(seed)
    draws = np.empty(REPS)
    for rep in range(REPS):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        sample = [row for cluster in chosen for row in by_cluster[cluster]]
        draws[rep] = float(statistic(sample))
    return {
        "estimate": estimate,
        "cluster_bootstrap_95_ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "n_trials": len(rows),
        "n_review_clusters": len(clusters),
        "bootstrap_reps": REPS,
        "bootstrap_seed": seed,
    }


def analyze(stimuli, predictions):
    stimuli_by_id = {item["id"]: item for item in stimuli}
    if len(stimuli_by_id) != len(stimuli):
        raise ValueError("Duplicate stimulus ID")
    required = {
        (item["id"], judge, wrapper, condition)
        for item in stimuli
        for judge, wrapper, conditions in (
            ("laya", "four_way", CONDITIONS),
            ("qwen2.5-3b", "forced_binary", CONDITIONS),
            ("qwen2.5-3b", "abstention", CONDITIONS),
            ("phi3-mini", "forced_binary", CONDITIONS),
            ("phi3-mini", "abstention", CONDITIONS),
        )
        for condition in conditions
        if condition != "prior_opinion_deleted" or item["role"] == "later"
    }
    seen = set()
    for row in predictions:
        key = (row["id"], row["judge"], row["wrapper"], row["condition"])
        if key in seen:
            raise ValueError(f"Duplicate prediction key: {key}")
        if key not in required:
            raise ValueError(f"Prediction outside the registered job set: {key}")
        stimulus = stimuli_by_id[row["id"]]
        for field in ("cluster_id", "role", "gold", "prior_gold"):
            if row[field] != stimulus[field]:
                raise ValueError(f"Prediction/stimulus {field} mismatch for {row['id']}")
        seen.add(key)
    if seen != required:
        raise ValueError(f"Incomplete prediction set: missing {len(required - seen)} judgments")

    by_key = {
        (row["id"], row["judge"], row["wrapper"], row["condition"]): row
        for row in predictions
    }
    rows_by_group = defaultdict(list)
    for stimulus in stimuli:
        for judge, wrapper, _conditions in (
            ("laya", "four_way", CONDITIONS),
            ("qwen2.5-3b", "forced_binary", CONDITIONS),
            ("qwen2.5-3b", "abstention", CONDITIONS),
            ("phi3-mini", "forced_binary", CONDITIONS),
            ("phi3-mini", "abstention", CONDITIONS),
        ):
            for condition in CONDITIONS:
                if condition == "prior_opinion_deleted" and stimulus["role"] != "later":
                    continue
                pred = by_key[(stimulus["id"], judge, wrapper, condition)]
                rows_by_group[(judge, wrapper, condition)].append(
                    {
                        "id": stimulus["id"],
                        "cluster_id": stimulus["cluster_id"],
                        "role": stimulus["role"],
                        "gold": stimulus["gold"],
                        "prior_gold": stimulus["prior_gold"],
                        "label": pred["label"],
                    }
                )

    summary = {}
    for group, rows in rows_by_group.items():
        judge, wrapper, condition = group
        covered = [row for row in rows if row["label"] in {"positive", "negative"}]
        correctness = [row | {"correct": int(row["label"] == row["gold"])} for row in rows]
        summary["/".join(group)] = {
            "n": len(rows),
            "coverage": len(covered) / len(rows) if rows else None,
            "abstention_or_nonbinary_rate": 1 - len(covered) / len(rows) if rows else None,
            "accuracy_all_trials": float(
                np.mean([row["label"] == row["gold"] for row in rows])
            ),
            "accuracy_all_trials_cluster_bootstrap": cluster_bootstrap(
                correctness, lambda sample: np.mean([row["correct"] for row in sample]), SEED + 1
            ),
            "accuracy_when_binary": (
                float(np.mean([row["label"] == row["gold"] for row in covered]))
                if covered
                else None
            ),
            "label_counts": dict(Counter(row["label"] or "unparsed" for row in rows)),
        }

    paired_accuracy = {}
    paired_accuracy_differences = {}
    for judge, wrapper in (
        ("laya", "four_way"),
        ("qwen2.5-3b", "forced_binary"),
        ("qwen2.5-3b", "abstention"),
        ("phi3-mini", "forced_binary"),
        ("phi3-mini", "abstention"),
    ):
        for condition in ("aspect_only", "natural_prefix", "opinion_visible"):
            rows = rows_by_group[(judge, wrapper, condition)]
            clusters = defaultdict(list)
            for row in rows:
                clusters[row["cluster_id"]].append(row)
            complete = [
                {
                    "cluster_id": cluster_id,
                    "both_correct": int(
                        len(cluster_rows) == 2
                        and all(row["label"] == row["gold"] for row in cluster_rows)
                    ),
                }
                for cluster_id, cluster_rows in clusters.items()
            ]
            paired_accuracy[f"{judge}/{wrapper}/{condition}"] = {
                "both_opposite_aspects_correct_rate": cluster_bootstrap(
                    complete,
                    lambda sample: np.mean([row["both_correct"] for row in sample]),
                    SEED + 2,
                ),
            }
        for first_condition, second_condition in (
            ("natural_prefix", "aspect_only"),
            ("opinion_visible", "natural_prefix"),
        ):
            first = {
                row["id"]: row
                for row in rows_by_group[(judge, wrapper, first_condition)]
            }
            second = {
                row["id"]: row
                for row in rows_by_group[(judge, wrapper, second_condition)]
            }
            differences = []
            for stimulus in stimuli:
                left, right = first[stimulus["id"]], second[stimulus["id"]]
                differences.append(
                    {
                        "cluster_id": stimulus["cluster_id"],
                        "difference": int(left["label"] == left["gold"])
                        - int(right["label"] == right["gold"]),
                    }
                )
            paired_accuracy_differences[
                f"{judge}/{wrapper}/{first_condition}_minus_{second_condition}"
            ] = cluster_bootstrap(
                differences,
                lambda sample: np.mean([row["difference"] for row in sample]),
                SEED + 3,
            )

    copy_rate = {}
    for judge, wrapper in (
        ("qwen2.5-3b", "forced_binary"),
        ("phi3-mini", "forced_binary"),
        ("laya", "four_way"),
    ):
        natural = rows_by_group[(judge, wrapper, "natural_prefix")]
        deleted = rows_by_group[(judge, wrapper, "prior_opinion_deleted")]
        natural_by_id = {row["id"]: row for row in natural if row["role"] == "later"}
        deleted_by_id = {row["id"]: row for row in deleted}
        paired_rows = []
        for item in stimuli:
            if item["role"] != "later":
                continue
            nrow, drow = natural_by_id[item["id"]], deleted_by_id[item["id"]]
            paired_rows.append(
                {
                    "id": item["id"],
                    "cluster_id": item["cluster_id"],
                    "gold": item["gold"],
                    "prior_gold": item["prior_gold"],
                    "natural_label": nrow["label"],
                    "deleted_label": drow["label"],
                    "natural_copy": int(nrow["label"] == item["prior_gold"]),
                    "deleted_copy": int(drow["label"] == item["prior_gold"]),
                    "natural_covered": int(nrow["label"] in {"positive", "negative"}),
                    "deleted_covered": int(drow["label"] in {"positive", "negative"}),
                    "natural_correct": int(nrow["label"] == item["gold"]),
                    "deleted_correct": int(drow["label"] == item["gold"]),
                }
            )
        differences = [row | {"difference": row["natural_copy"] - row["deleted_copy"]} for row in paired_rows]
        primary = cluster_bootstrap(
            differences,
            lambda sample: np.mean([row["difference"] for row in sample]),
            SEED,
        )
        primary["natural_copy_rate"] = float(np.mean([row["natural_copy"] for row in paired_rows]))
        primary["deleted_copy_rate"] = float(np.mean([row["deleted_copy"] for row in paired_rows]))
        primary["natural_accuracy"] = float(np.mean([row["natural_correct"] for row in paired_rows]))
        primary["deleted_accuracy"] = float(np.mean([row["deleted_correct"] for row in paired_rows]))
        primary["natural_coverage"] = float(np.mean([row["natural_covered"] for row in paired_rows]))
        primary["deleted_coverage"] = float(np.mean([row["deleted_covered"] for row in paired_rows]))
        primary["paired_cluster_rows"] = len(paired_rows)
        primary["natural_accuracy_cluster_bootstrap"] = cluster_bootstrap(
            [row | {"correct": row["natural_correct"]} for row in paired_rows],
            lambda sample: np.mean([row["correct"] for row in sample]),
            SEED + 4,
        )
        primary["deleted_accuracy_cluster_bootstrap"] = cluster_bootstrap(
            [row | {"correct": row["deleted_correct"]} for row in paired_rows],
            lambda sample: np.mean([row["correct"] for row in sample]),
            SEED + 5,
        )
        primary["natural_minus_deleted_accuracy_cluster_bootstrap"] = cluster_bootstrap(
            [row | {"difference": row["natural_correct"] - row["deleted_correct"]} for row in paired_rows],
            lambda sample: np.mean([row["difference"] for row in sample]),
            SEED + 6,
        )
        primary["gate_passed"] = (
            primary["estimate"] >= 0.05 and primary["cluster_bootstrap_95_ci"][0] > 0
        )
        copy_rate[f"{judge}/{wrapper}"] = primary

    return {
        "experiment": "040",
        "primary_endpoint": "Qwen2.5-3B forced-binary later-target opposite-prior copy rate: natural prefix minus prior-opinion-deleted prefix",
        "primary_diagnostic_gate_passed": copy_rate["qwen2.5-3b/forced_binary"]["gate_passed"],
        "later_target_copy_rate_contrasts": copy_rate,
        "condition_metrics": summary,
        "paired_review_metrics": paired_accuracy,
        "paired_accuracy_differences": paired_accuracy_differences,
        "n_target_trials": len(stimuli),
        "n_review_clusters": len({row["cluster_id"] for row in stimuli}),
        "n_later_target_trials": sum(row["role"] == "later" for row in stimuli),
        "n_predictions": len(predictions),
        "limitations": [
            "The 48 clusters are selected from one test split, and the paired set may not represent travel reviews generally.",
            "TRABL annotations began from LLM suggestions followed by human correction; exact annotator agreement reduces but does not remove label noise.",
            "Deleting an opinion phrase changes fluency as well as sentiment-bearing content.",
            "An annotated opinion phrase is not a complete boundary for all evidence supporting a sentiment.",
            "The source reviews may have appeared in model pretraining even though the dataset release is recent.",
            "Laya and Phi are secondary checks; one local model is the registered primary judge.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULT)
    args = parser.parse_args()
    stimuli = json.loads((args.results / "stimuli.json").read_text())
    predictions = json.loads((args.results / "predictions.json").read_text())
    manifest = json.loads((args.results / "manifest.json").read_text())
    analysis = analyze(stimuli, predictions)
    analysis_path = args.results / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")
    manifest["analyzer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["analysis_sha256"] = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    (args.results / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    check_map = {
        "96_frozen_target_trials": len(stimuli) == 96,
        "48_review_clusters": len({row["cluster_id"] for row in stimuli}) == 48,
        "balanced_opposite_labels": Counter(row["gold"] for row in stimuli)
        == {"positive": 48, "negative": 48},
        "48_later_targets": sum(row["role"] == "later" for row in stimuli) == 48,
        "1680_complete_predictions": len(predictions) == 1680,
        "protocol_hash_matches": manifest["protocol_sha256"]
        == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "stimulus_hash_matches": manifest["stimuli_sha256"]
        == hashlib.sha256((args.results / "stimuli.json").read_bytes()).hexdigest(),
        "prediction_hash_matches": manifest["predictions_sha256"]
        == hashlib.sha256((args.results / "predictions.json").read_bytes()).hexdigest(),
        "analysis_hash_matches": manifest["analysis_sha256"]
        == hashlib.sha256(analysis_path.read_bytes()).hexdigest(),
        "laya_triage_hash_recorded": manifest["laya_triage_trace_sha256"]
        == "f6728865fd43d3b55ad3825c1a3b54733093389236d84130bb3401b2d1c9fb51",
        "frozen_selection_hash_matches": hashlib.sha256(STIMULI.read_bytes()).hexdigest()
        == "7abd6a1ab5d4f86190cfd727f452042953d4e030f39d335a88cec17f37db997b",
        "dataset_revision_pinned": manifest["dataset_revision"] == SOURCE_REVISION,
        "no_review_text_in_public_rows": all(
            all(key not in row for key in ("text", "review", "visible_text", "aspect"))
            for row in stimuli + predictions
        ),
    }
    (args.results / "audit.json").write_text(
        json.dumps(
            {
                "checks": check_map,
                "checks_passed": all(check_map.values()),
                "primary_diagnostic_gate_passed": analysis["primary_diagnostic_gate_passed"],
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(analysis, indent=2), flush=True)


if __name__ == "__main__":
    main()
