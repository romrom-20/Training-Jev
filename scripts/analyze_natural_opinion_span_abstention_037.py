"""Analyze the preregistered natural opinion-span visibility comparison."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path("results/natural-opinion-span-abstention-v1")
PROTOCOL = Path("docs/experiments/037-natural-opinion-span-abstention.md")
STIMULI_INDEX = Path("docs/experiments/037-stimuli.json")
JUDGES = ("qwen2.5-3b", "phi3-mini")
REPS = 10_000
SEED = 20260937


def bootstrap(rows, statistic, reps=REPS, seed=SEED):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["cluster_id"]].append(row)
    clusters = np.asarray(sorted(grouped))
    if not len(clusters):
        return {"estimate": None, "cluster_bootstrap_95_ci": None, "n": 0}

    def evaluate(sample):
        return float(statistic(sample))

    estimate = evaluate(rows)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for index in range(reps):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        sample = [row for cluster in chosen for row in grouped[cluster]]
        draws[index] = evaluate(sample)
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


def normalize_laya(label):
    if label == "unclear":
        return "insufficient"
    if label in {"positive", "negative", "mixed"}:
        return label
    return None


def analyze(stimuli, outcomes):
    indexed = {}
    for row in outcomes:
        key = (row["judge"], row.get("wrapper", "four_way"), row["id"], row["condition"])
        if key in indexed:
            raise ValueError(f"Duplicate outcome: {key}")
        indexed[key] = row

    ids = {row["id"] for row in stimuli}
    expected_count = len(stimuli) * 2 * 5
    if len(outcomes) != expected_count:
        raise ValueError(f"Expected {expected_count} outcomes; found {len(outcomes)}")
    expected_keys = {
        (judge, wrapper, stimulus_id, condition)
        for judge in JUDGES
        for wrapper in ("forced_binary", "abstention")
        for stimulus_id in ids
        for condition in ("before_opinion", "opinion_visible")
    } | {
        ("laya", "four_way", stimulus_id, condition)
        for stimulus_id in ids
        for condition in ("before_opinion", "opinion_visible")
    }
    if set(indexed) != expected_keys:
        raise ValueError("Outcome join is incomplete or includes an unexpected row")

    primary_rows = []
    by_judge = {judge: [] for judge in JUDGES}
    condition_metrics = {}
    abstention_metrics = {}
    predictive_prefix_diagnostics = {}
    parse_coverage = {}
    for judge_index, judge in enumerate(JUDGES):
        judge_rows = []
        for stimulus in stimuli:
            for condition in ("before_opinion", "opinion_visible"):
                new = indexed[(judge, "abstention", stimulus["id"], condition)]
                old = indexed[(judge, "forced_binary", stimulus["id"], condition)]
                for row in (new, old):
                    if row["cluster_id"] != stimulus["cluster_id"] or row["polarity"] != stimulus["polarity"]:
                        raise ValueError("Stimulus metadata does not match paired output")
                expected = "insufficient" if condition == "before_opinion" else stimulus["polarity"]
                if new["expected_decision"] != expected or old["expected_decision"] != expected:
                    raise ValueError("Expected decision does not track the frozen span condition")
                new_label, old_label = new["label"], old["label"]
                if old_label not in {None, "positive", "negative"}:
                    raise ValueError("Forced-binary wrapper produced an invalid label")
                if new_label not in {None, "positive", "negative", "insufficient"}:
                    raise ValueError("Abstention wrapper produced an invalid label")
                row = {
                    "cluster_id": stimulus["cluster_id"],
                    "dataset": stimulus["dataset"],
                    "id": stimulus["id"],
                    "judge": judge,
                    "condition": condition,
                    "expected": expected,
                    "polarity": stimulus["polarity"],
                    "new_label": new_label,
                    "old_label": old_label,
                    "new_correct": int(new_label == expected),
                    "old_correct": int(old_label == expected),
                    "improvement": int(new_label == expected) - int(old_label == expected),
                }
                judge_rows.append(row)
                by_judge[judge].append(row)
                primary_rows.append(row)

        condition_metrics[judge] = {}
        for condition_index, condition in enumerate(("before_opinion", "opinion_visible")):
            rows = [row for row in judge_rows if row["condition"] == condition]
            condition_metrics[judge][condition] = {
                "forced_binary_appropriate_accuracy": bootstrap(
                    rows,
                    lambda sample: np.mean([row["old_correct"] for row in sample]),
                    seed=SEED + 20 + 2 * judge_index + condition_index,
                ),
                "abstention_appropriate_accuracy": bootstrap(
                    rows,
                    lambda sample: np.mean([row["new_correct"] for row in sample]),
                    seed=SEED + 30 + 2 * judge_index + condition_index,
                ),
                "paired_change": bootstrap(
                    rows,
                    lambda sample: np.mean([row["improvement"] for row in sample]),
                    seed=SEED + 40 + 2 * judge_index + condition_index,
                ),
            }
        cue_rows = [row for row in judge_rows if row["condition"] == "opinion_visible"]
        absent_rows = [row for row in judge_rows if row["condition"] == "before_opinion"]
        answered_prefix_rows = [
            row for row in absent_rows if row["new_label"] in {"positive", "negative"}
        ]
        predictive_prefix_diagnostics[judge] = {
            "forced_binary_polarity_accuracy_before_opinion": bootstrap(
                absent_rows,
                lambda sample: np.mean([row["old_label"] == row["polarity"] for row in sample]),
                seed=SEED + 400 + judge_index,
            ),
            "abstention_wrapper_coverage_before_opinion": bootstrap(
                absent_rows,
                lambda sample: np.mean([row["new_label"] in {"positive", "negative"} for row in sample]),
                seed=SEED + 410 + judge_index,
            ),
            "abstention_wrapper_polarity_accuracy_when_it_answers": bootstrap(
                answered_prefix_rows,
                lambda sample: np.mean([row["new_label"] == row["polarity"] for row in sample]),
                seed=SEED + 420 + judge_index,
            ),
        }
        abstention_metrics[judge] = {
            "recall_before_annotated_opinion": bootstrap(
                absent_rows,
                lambda sample: np.mean([row["new_label"] == "insufficient" for row in sample]),
                seed=SEED + 50 + judge_index,
            ),
            "false_abstention_after_opinion": bootstrap(
                cue_rows,
                lambda sample: np.mean([row["new_label"] == "insufficient" for row in sample]),
                seed=SEED + 60 + judge_index,
            ),
            "cue_visible_polarity_accuracy_forced": bootstrap(
                cue_rows,
                lambda sample: np.mean([row["old_label"] == row["polarity"] for row in sample]),
                seed=SEED + 70 + judge_index,
            ),
            "cue_visible_polarity_accuracy_abstention": bootstrap(
                cue_rows,
                lambda sample: np.mean([row["new_label"] == row["polarity"] for row in sample]),
                seed=SEED + 80 + judge_index,
            ),
        }
        parse_coverage[judge] = {
            "forced_binary": float(np.mean([row["old_label"] is not None for row in judge_rows])),
            "abstention": float(np.mean([row["new_label"] is not None for row in judge_rows])),
        }

    primary = bootstrap(
        primary_rows,
        lambda sample: np.mean([row["improvement"] for row in sample]),
        seed=SEED,
    )
    primary_by_judge = {
        judge: bootstrap(
            rows,
            lambda sample: np.mean([row["improvement"] for row in sample]),
            seed=SEED + 100 + index,
        )
        for index, (judge, rows) in enumerate(by_judge.items())
    }
    laya_rows = []
    for stimulus in stimuli:
        for condition in ("before_opinion", "opinion_visible"):
            source = indexed[("laya", "four_way", stimulus["id"], condition)]
            expected = "insufficient" if condition == "before_opinion" else stimulus["polarity"]
            laya_rows.append(
                {
                    "cluster_id": stimulus["cluster_id"],
                    "dataset": stimulus["dataset"],
                    "id": stimulus["id"],
                    "condition": condition,
                    "expected": expected,
                    "polarity": stimulus["polarity"],
                    "label": normalize_laya(source["label"]),
                    "raw_label": source["label"],
                }
            )
    laya_metrics = {}
    for condition_index, condition in enumerate(("before_opinion", "opinion_visible")):
        rows = [row for row in laya_rows if row["condition"] == condition]
        expected = "insufficient" if condition == "before_opinion" else None
        laya_metrics[condition] = {
            "appropriate_accuracy": bootstrap(
                rows,
                lambda sample: np.mean([row["label"] == row["expected"] for row in sample]),
                seed=SEED + 200 + condition_index,
            ),
            "label_counts": dict(Counter(row["raw_label"] for row in rows)),
        }
        if expected == "insufficient":
            laya_metrics[condition]["abstention_recall"] = bootstrap(
                rows,
                lambda sample: np.mean([row["label"] == "insufficient" for row in sample]),
                seed=SEED + 210,
            )
        else:
            laya_metrics[condition]["false_abstention_rate"] = bootstrap(
                rows,
                lambda sample: np.mean([row["label"] == "insufficient" for row in sample]),
                seed=SEED + 211,
            )
    laya_before = [row for row in laya_rows if row["condition"] == "before_opinion"]
    laya_clear = [row for row in laya_before if row["raw_label"] in {"positive", "negative"}]
    laya_metrics["before_opinion"]["polarity_coverage"] = bootstrap(
        laya_before,
        lambda sample: np.mean([row["raw_label"] in {"positive", "negative"} for row in sample]),
        seed=SEED + 212,
    )
    laya_metrics["before_opinion"]["polarity_accuracy_when_clear"] = bootstrap(
        laya_clear,
        lambda sample: np.mean([row["label"] == row["polarity"] for row in sample]),
        seed=SEED + 213,
    )

    paired_abstention_gap = []
    for stimulus_id in sorted(ids):
        qwen = indexed[("qwen2.5-3b", "abstention", stimulus_id, "before_opinion")]
        phi = indexed[("phi3-mini", "abstention", stimulus_id, "before_opinion")]
        paired_abstention_gap.append(
            {
                "cluster_id": qwen["cluster_id"],
                "qwen_minus_phi_abstention": int(qwen["label"] == "insufficient")
                - int(phi["label"] == "insufficient"),
            }
        )

    by_dataset = {}
    for dataset in sorted({row["dataset"] for row in primary_rows}):
        rows = [row for row in primary_rows if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "n_stimuli": len({row["id"] for row in rows}),
            "paired_change": bootstrap(
                rows,
                lambda sample: np.mean([row["improvement"] for row in sample]),
                seed=SEED + 300 + len(by_dataset),
            ),
        }

    visible_costs_ok = all(
        abstention_metrics[judge]["cue_visible_polarity_accuracy_abstention"]["estimate"]
        >= abstention_metrics[judge]["cue_visible_polarity_accuracy_forced"]["estimate"] - 0.05
        for judge in JUDGES
    )
    gate = primary["estimate"] >= 0.10 and primary["cluster_bootstrap_95_ci"][0] > 0 and visible_costs_ok
    return {
        "experiment": "037",
        "primary_endpoint": "paired change in appropriate-decision accuracy from forced-binary to abstention-enabled wrapper, averaged across opinion-span visibility states and Qwen/Phi",
        "primary_pooled_change": primary,
        "primary_change_by_judge": primary_by_judge,
        "condition_metrics": condition_metrics,
        "abstention_metrics": abstention_metrics,
        "pre_opinion_predictive_polarity_diagnostics": predictive_prefix_diagnostics,
        "parse_coverage": parse_coverage,
        "paired_qwen_minus_phi_abstention_recall_before_opinion": bootstrap(
            paired_abstention_gap,
            lambda sample: np.mean([row["qwen_minus_phi_abstention"] for row in sample]),
            seed=SEED + 500,
        ),
        "laya": laya_metrics,
        "per_dataset": by_dataset,
        "preregistered_success_rule_passed": gate,
        "n_stimuli": len(stimuli),
        "n_sentence_clusters": len({row["cluster_id"] for row in stimuli}),
        "n_outcomes": len(outcomes),
        "limitations": [
            "The before-span prefix may contain implicit or unannotated evidence; gold only marks visibility of the annotated opinion phrase.",
            "The wrapper changes the allowed response set and instruction together.",
            "ASTE-Data-V2 is a small set of SemEval domain test splits, not a random sample of reviews or judges.",
            "No per-example review text is redistributed because the upstream repository has no explicit license metadata.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=OUT)
    args = parser.parse_args()
    stimuli = json.loads((args.results / "stimuli.json").read_text())
    outcomes = json.loads((args.results / "predictions.json").read_text())
    manifest = json.loads((args.results / "manifest.json").read_text())
    analysis = analyze(stimuli, outcomes)
    (args.results / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    stimulus_ids = {row["id"] for row in stimuli}
    checks = {
        "234_frozen_stimuli": len(stimuli) == 234,
        "balanced_polarity": Counter(row["polarity"] for row in stimuli)
        == {"positive": 117, "negative": 117},
        "balanced_within_dataset": all(
            Counter(row["polarity"] for row in stimuli if row["dataset"] == dataset)
            == counts
            for dataset, counts in {
                "14res": {"positive": 27, "negative": 27},
                "14lap": {"positive": 29, "negative": 29},
                "15res": {"positive": 39, "negative": 39},
                "16res": {"positive": 22, "negative": 22},
            }.items()
        ),
        "unique_sentence_clusters": len({row["cluster_id"] for row in stimuli}) == 234,
        "complete_2340_paired_outcomes": len(outcomes) == 2340,
        "manifest_stimuli_hash_matches": manifest["stimuli_sha256"]
        == hashlib.sha256((args.results / "stimuli.json").read_bytes()).hexdigest(),
        "manifest_prediction_hash_matches": manifest["predictions_sha256"]
        == hashlib.sha256((args.results / "predictions.json").read_bytes()).hexdigest(),
        "manifest_protocol_hash_matches": manifest["protocol_sha256"]
        == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "manifest_source_revision_pinned": manifest["source_revision"]
        == "d0df6600b259b6114de23cc5047c7e776cd89750",
        "source_hashes_cover_four_test_splits": set(manifest["source_files_sha256"])
        == {"14res", "14lap", "15res", "16res"},
        "public_stimuli_match_frozen_identifier_index": stimuli
        == json.loads(STIMULI_INDEX.read_text()),
        "no_review_text_in_public_stimuli": all(
            "text" not in row and "sentence" not in row for row in stimuli
        ),
        "no_review_text_in_public_outcomes": all(
            "visible_text" not in row and "sentence" not in row for row in outcomes
        ),
        "all_outputs_match_frozen_ids": all(row["id"] in stimulus_ids for row in outcomes),
        "preregistered_operational_rule_passed": analysis["preregistered_success_rule_passed"],
    }
    (args.results / "audit.json").write_text(
        json.dumps({"checks": checks, "checks_passed": all(checks.values())}, indent=2) + "\n"
    )
    print(json.dumps(analysis, indent=2), flush=True)


if __name__ == "__main__":
    main()
