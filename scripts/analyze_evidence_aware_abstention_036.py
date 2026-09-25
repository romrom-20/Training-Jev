"""Analyze paired forced-binary and explicit-abstention decisions."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PRIVATE = Path(".context/evidence-aware-abstention-036")
BASE_PRIVATE = Path(".context/cue-position-polarity-035")
OUT = Path("results/evidence-aware-abstention-v1")
PROTOCOL = Path("docs/experiments/036-evidence-aware-abstention.md")
JUDGES = ("laya", "qwen2.5-3b", "phi3-mini")
BINARIES = ("qwen2.5-3b", "phi3-mini")
REPS = 10_000
SEED = 20260936


def cluster_bootstrap(rows, statistic, reps=REPS, seed=SEED):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["cluster_id"]].append(row)
    clusters = np.asarray(sorted(grouped))
    if not len(clusters):
        return {"estimate": None, "cluster_bootstrap_95_ci": None, "n": 0}

    def estimate(sample):
        return float(statistic(sample))

    value = estimate(rows)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for index in range(reps):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        sample = [row for cluster in selected for row in grouped[cluster]]
        draws[index] = estimate(sample)
    return {
        "estimate": value,
        "cluster_bootstrap_95_ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "n": len(rows),
        "n_scaffold_aspect_clusters": len(clusters),
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
    }


def normalize_new(judge, label):
    if judge == "laya":
        if label in {"positive", "negative"}:
            return label
        if label == "unclear":
            return "insufficient"
        return "mixed" if label == "mixed" else None
    return label if label in {"positive", "negative", "insufficient"} else None


def normalize_forced(label):
    if label == 1:
        return "positive"
    if label == 0:
        return "negative"
    return None


def is_appropriate(label, expected):
    return label == expected


def analyze(stimuli, new_outcomes, base_outcomes):
    indexed_new = {}
    for row in new_outcomes:
        key = (row["judge"], row["id"], row["budget"])
        if key in indexed_new:
            raise ValueError(f"Duplicate new outcome: {key}")
        indexed_new[key] = row
    indexed_base = {}
    for row in base_outcomes:
        key = (row["judge"], row["id"], row["budget"])
        if key in indexed_base:
            raise ValueError(f"Duplicate baseline outcome: {key}")
        indexed_base[key] = row

    paired_by_judge = {judge: [] for judge in BINARIES}
    pooled = []
    condition_rates = {}
    abstention = {}
    visible_accuracy = {}
    parse_coverage = {}
    no_cue_laya_consistency = {}
    for judge in JUDGES:
        judged_rows = []
        for stimulus in stimuli:
            for budget in (8, 12):
                key = (judge, stimulus["id"], budget)
                new = indexed_new[key]
                old = indexed_base.get(key)
                if judge in BINARIES and old is None:
                    raise ValueError(f"Missing matched Experiment 035 baseline {key}")
                if (new["cluster_id"], new["gold"], new["expected_decision"]) != (
                    stimulus["cluster_id"],
                    stimulus["gold"],
                    "insufficient"
                    if stimulus["position"] == "late" and budget == 8
                    else stimulus["polarity"],
                ):
                    raise ValueError(f"Mismatched expected label metadata for {key}")
                new_label = normalize_new(judge, new["label"])
                old_label = normalize_forced(old["label"]) if judge in BINARIES else None
                row = {
                    "judge": judge,
                    "id": stimulus["id"],
                    "cluster_id": stimulus["cluster_id"],
                    "polarity": stimulus["polarity"],
                    "position": stimulus["position"],
                    "budget": budget,
                    "cue_visible": new["cue_visible"],
                    "expected": new["expected_decision"],
                    "new_label": new_label,
                    "new_correct": int(is_appropriate(new_label, new["expected_decision"])),
                }
                if judge in BINARIES:
                    row["forced_label"] = old_label
                    row["forced_correct"] = int(
                        is_appropriate(old_label, new["expected_decision"])
                    )
                    row["forced_visible_correct"] = int(
                        old_label == stimulus["polarity"]
                    )
                    row["improvement"] = row["new_correct"] - row["forced_correct"]
                    paired_by_judge[judge].append(row)
                    pooled.append(row)
                judged_rows.append(row)

        no_cue = [
            row
            for row in judged_rows
            if row["position"] == "late" and row["budget"] == 8
        ]
        cue_visible = [row for row in judged_rows if row["cue_visible"]]
        abstention[judge] = {
            "appropriate_insufficient_on_late_8": cluster_bootstrap(
                no_cue,
                lambda rows: float(np.mean([row["new_label"] == "insufficient" for row in rows])),
                seed=SEED + 10 + JUDGES.index(judge),
            ),
            "false_insufficient_on_visible_cue": cluster_bootstrap(
                cue_visible,
                lambda rows: float(np.mean([row["new_label"] == "insufficient" for row in rows])),
                seed=SEED + 20 + JUDGES.index(judge),
            ),
        }
        visible_accuracy[judge] = {
            "strict_polarity_accuracy": cluster_bootstrap(
                cue_visible,
                lambda rows: float(
                    np.mean(
                        [
                            row["new_label"] == row["polarity"]
                            for row in rows
                        ]
                    )
                ),
                seed=SEED + 30 + JUDGES.index(judge),
            ),
            "labeled_coverage": float(
                np.mean([row["new_label"] in {"positive", "negative"} for row in cue_visible])
            ),
        }
        if judge in BINARIES:
            baseline_rows = paired_by_judge[judge]
            baseline_visible = [row for row in baseline_rows if row["cue_visible"]]
            visible_accuracy[judge]["forced_binary_polarity_accuracy"] = cluster_bootstrap(
                baseline_visible,
                lambda rows: float(np.mean([row["forced_visible_correct"] for row in rows])),
                seed=SEED + 35 + JUDGES.index(judge),
            )
        parse_coverage[judge] = {
            "all_decisions": float(np.mean([row["new_label"] is not None for row in judged_rows])),
            "no_cue_late_8": float(np.mean([row["new_label"] is not None for row in no_cue])),
        }
        condition_rates[judge] = {
            "new_appropriate_decision_accuracy": cluster_bootstrap(
                judged_rows,
                lambda rows: float(np.mean([row["new_correct"] for row in rows])),
                seed=SEED + 40 + JUDGES.index(judge),
            )
        }
        if judge == "laya":
            by_cluster = defaultdict(set)
            for row in no_cue:
                source = indexed_new[(judge, row["id"], 8)]
                by_cluster[row["cluster_id"]].add(source["label"])
            no_cue_laya_consistency = {key: len(labels) == 1 for key, labels in by_cluster.items()}

    primary = cluster_bootstrap(
        pooled,
        lambda rows: float(np.mean([row["improvement"] for row in rows])),
        seed=SEED,
    )
    by_judge = {
        judge: cluster_bootstrap(
            rows,
            lambda sample: float(np.mean([row["improvement"] for row in sample])),
            seed=SEED + 100 + index,
        )
        for index, (judge, rows) in enumerate(paired_by_judge.items())
    }
    old_new = {}
    for judge in BINARIES:
        rows = paired_by_judge[judge]
        old_new[judge] = {
            "forced_binary_appropriate_accuracy": cluster_bootstrap(
                rows,
                lambda sample: float(np.mean([row["forced_correct"] for row in sample])),
                seed=SEED + 200 + BINARIES.index(judge),
            ),
            "abstention_enabled_appropriate_accuracy": cluster_bootstrap(
                rows,
                lambda sample: float(np.mean([row["new_correct"] for row in sample])),
                seed=SEED + 210 + BINARIES.index(judge),
            ),
        }
    gate = (
        primary["estimate"] >= 0.05
        and primary["cluster_bootstrap_95_ci"][0] > 0
        and all(
            visible_accuracy[judge]["strict_polarity_accuracy"]["estimate"]
            >= visible_accuracy[judge]["forced_binary_polarity_accuracy"]["estimate"] - 0.05
            for judge in BINARIES
        )
    )
    return {
        "primary_endpoint": "paired change in appropriate-decision accuracy from forced binary to explicit abstention, pooled equally over Qwen/Phi",
        "primary_pooled_change": primary,
        "primary_change_by_binary_judge": by_judge,
        "appropriate_accuracy_by_binary_judge": old_new,
        "appropriate_accuracy_by_judge": condition_rates,
        "no_cue_abstention_and_visible_false_abstention": abstention,
        "visible_cue_polarity_accuracy": visible_accuracy,
        "parse_coverage": parse_coverage,
        "laya_identical_no_cue_prefix_consistency": {
            "consistent_clusters": sum(no_cue_laya_consistency.values()),
            "total_clusters": len(no_cue_laya_consistency),
        },
        "preregistered_success_rule_passed": bool(gate),
    }


def run(private=PRIVATE, output=OUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest_path = private / "manifest.json"
    outcomes_path = private / "outcomes.json"
    stimuli_path = private / "stimuli.json"
    manifest = json.loads(manifest_path.read_text())
    for path, field in (
        (outcomes_path, "outcomes_sha256"),
        (stimuli_path, "stimuli_sha256"),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[field]:
            raise ValueError(f"Experiment 036 private hash mismatch: {path.name}")
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != manifest["protocol_sha256"]:
        raise ValueError("Experiment 036 protocol changed after data collection")
    base_manifest = json.loads((BASE_PRIVATE / "manifest.json").read_text())
    base_path = BASE_PRIVATE / "outcomes.json"
    if hashlib.sha256(base_path.read_bytes()).hexdigest() != base_manifest["outcomes_sha256"]:
        raise ValueError("Experiment 035 matched baseline hash mismatch")
    stimuli = json.loads(stimuli_path.read_text())
    outcomes = json.loads(outcomes_path.read_text())
    base_outcomes = json.loads(base_path.read_text())
    if len(stimuli) != 192 or len({row["cluster_id"] for row in stimuli}) != 48:
        raise ValueError("Experiment 036 stimulus count changed")
    if len(outcomes) != 1152:
        raise ValueError("Experiment 036 outcome count changed")
    analysis = {
        "experiment": "036",
        "stimuli": len(stimuli),
        "clusters": len({row["cluster_id"] for row in stimuli}),
        "new_judgment_outcomes": len(outcomes),
        "analysis": analyze(stimuli, outcomes, base_outcomes),
        "limitations": [
            "The sample is a balanced subset of short deterministic review templates, not natural generated answers.",
            "The expected insufficient label for late-8 prefixes follows the constructed omission of every polarity cue.",
            "The judges are fixed model families; results do not estimate the distribution of evaluator models.",
            "The new prompt changes both allowed labels and abstention instructions relative to the forced-binary baseline.",
        ],
    }
    output.mkdir(parents=True)
    public_stimuli = output / "stimuli.json"
    public_stimuli.write_text(json.dumps(stimuli, indent=2, ensure_ascii=False) + "\n")
    public_rows = [{key: value for key, value in row.items() if key != "visible_text"} for row in outcomes]
    public_predictions = output / "predictions.json"
    public_predictions.write_text(json.dumps(public_rows, indent=2, sort_keys=True) + "\n")
    public_manifest = {key: value for key, value in manifest.items() if key != "outcomes_sha256"}
    public_manifest["predictions_sha256"] = hashlib.sha256(public_predictions.read_bytes()).hexdigest()
    public_manifest["stimuli_sha256"] = hashlib.sha256(public_stimuli.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(public_manifest, indent=2, sort_keys=True) + "\n")
    audit = {
        "checks": {
            "complete_1152_new_judgments": len(public_rows) == 1152,
            "balanced_96_positive_96_negative_stimuli": sum(row["gold"] for row in stimuli) == 96,
            "each_judge_has_384_paired_decisions": all(
                len({(row["id"], row["budget"]) for row in public_rows if row["judge"] == judge})
                == 384
                for judge in JUDGES
            ),
            "experiment_035_direct_subset_only": all(row["form"] == "direct" for row in stimuli),
            "protocol_hash_matches": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
            == manifest["protocol_sha256"],
        }
    }
    audit["checks_passed"] = all(audit["checks"].values())
    (output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["checks_passed"]:
        raise ValueError("Experiment 036 output audit failed")
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    (output / "README.md").write_text(
        "# Experiment 036 result\n\n"
        "This paired abstention-wrapper test reuses the direct, balanced subset of "
        "experiment 035. See [`analysis.json`](analysis.json), synthetic "
        "[`stimuli.json`](stimuli.json), label-only [`predictions.json`](predictions.json), "
        "and provenance [`manifest.json`](manifest.json) / [`audit.json`](audit.json). "
        "Protocol: [`036-evidence-aware-abstention.md`](../../docs/experiments/036-evidence-aware-abstention.md).\n"
    )
    print(json.dumps(analysis, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", type=Path, default=PRIVATE)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.private, args.output)


if __name__ == "__main__":
    main()
