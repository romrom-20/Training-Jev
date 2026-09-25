"""Analyze the preregistered controlled cue-position polarity experiment."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PRIVATE = Path(".context/cue-position-polarity-035")
OUT = Path("results/cue-position-polarity-v1")
PROTOCOL = Path("docs/experiments/035-cue-position-polarity.md")
JUDGES = ("laya", "qwen2.5-3b", "phi3-mini")
BINARIES = ("qwen2.5-3b", "phi3-mini")
BUDGETS = (8, 12)
REPS = 10_000
SEED = 20260935


def is_correct(label, gold):
    return label == ("positive" if gold else "negative") if isinstance(label, str) else label == gold


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


def mean_accuracy(rows):
    return float(np.mean([is_correct(row["label"], row["gold"]) for row in rows]))


def analyze(stimuli, outcomes):
    indexed = {}
    for row in outcomes:
        key = (row["judge"], row["id"], row["budget"])
        if key in indexed:
            raise ValueError(f"Duplicate judgment row {key}")
        indexed[key] = row
    expected = len(stimuli) * len(BUDGETS) * len(JUDGES)
    if len(indexed) != expected:
        raise ValueError(f"Expected {expected} outcomes, got {len(indexed)}")

    long_rows = []
    for judge in JUDGES:
        for stimulus in stimuli:
            for budget in BUDGETS:
                row = indexed[(judge, stimulus["id"], budget)]
                if row["cluster_id"] != stimulus["cluster_id"] or row["gold"] != stimulus["gold"]:
                    raise ValueError("Judgment metadata does not match frozen stimulus")
                long_rows.append(row)

    primary_rows = []
    judge_primary = {judge: [] for judge in BINARIES}
    for judge in BINARIES:
        for stimulus in stimuli:
            short = indexed[(judge, stimulus["id"], 8)]
            long = indexed[(judge, stimulus["id"], 12)]
            primary_rows.append(
                {
                    "cluster_id": stimulus["cluster_id"],
                    "judge": judge,
                    "position": stimulus["position"],
                    "delta": int(is_correct(long["label"], long["gold"]))
                    - int(is_correct(short["label"], short["gold"])),
                }
            )
            judge_primary[judge].append(primary_rows[-1])

    def timing_contrast(rows):
        late = [row["delta"] for row in rows if row["position"] == "late"]
        early = [row["delta"] for row in rows if row["position"] == "early"]
        return float(np.mean(late) - np.mean(early))

    primary = cluster_bootstrap(primary_rows, timing_contrast, seed=SEED)
    per_judge_primary = {
        judge: cluster_bootstrap(
            rows, timing_contrast, seed=SEED + 1 + index
        )
        for index, (judge, rows) in enumerate(judge_primary.items())
    }

    cells = {}
    for judge in JUDGES:
        cells[judge] = {}
        for polarity in ("negative", "positive"):
            cells[judge][polarity] = {}
            for form in ("direct", "negated"):
                cells[judge][polarity][form] = {}
                for position in ("early", "late"):
                    cells[judge][polarity][form][position] = {}
                    for budget in BUDGETS:
                        subset = [
                            row
                            for row in long_rows
                            if row["judge"] == judge
                            and row["polarity"] == polarity
                            and row["form"] == form
                            and row["position"] == position
                            and row["budget"] == budget
                        ]
                        cells[judge][polarity][form][position][str(budget)] = (
                            cluster_bootstrap(
                                subset,
                                mean_accuracy,
                                seed=SEED
                                + 10
                                + JUDGES.index(judge) * 100
                                + (0 if polarity == "negative" else 20)
                                + (0 if form == "direct" else 4)
                                + (0 if position == "early" else 2)
                                + budget,
                            )
                        )

    laya_ambiguity = {}
    for position in ("early", "late"):
        laya_ambiguity[position] = {}
        for budget in BUDGETS:
            subset = [
                row
                for row in long_rows
                if row["judge"] == "laya" and row["position"] == position and row["budget"] == budget
            ]
            laya_ambiguity[position][str(budget)] = cluster_bootstrap(
                subset,
                lambda sample: float(
                    np.mean([row["label"] in {"mixed", "unclear"} for row in sample])
                ),
                seed=SEED + 500 + (0 if position == "early" else 10) + budget,
            )

    by_budget = {}
    for budget in BUDGETS:
        binary_rows = [row for row in long_rows if row["judge"] in BINARIES and row["budget"] == budget]
        by_budget[str(budget)] = {
            "binary_accuracy": cluster_bootstrap(binary_rows, mean_accuracy, seed=SEED + 600 + budget),
            "binary_parse_coverage": {
                judge: float(
                    np.mean(
                        [
                            isinstance(row["label"], int)
                            for row in binary_rows
                            if row["judge"] == judge
                        ]
                    )
                )
                for judge in BINARIES
            },
            "early_accuracy": cluster_bootstrap(
                [row for row in binary_rows if row["position"] == "early"],
                mean_accuracy,
                seed=SEED + 610 + budget,
            ),
        }
    capability = {
        "twelve_word_accuracy_at_least_90_percent": by_budget["12"]["binary_accuracy"]["estimate"]
        >= 0.90,
        "early_eight_word_accuracy_at_least_85_percent": by_budget["8"]["early_accuracy"]["estimate"]
        >= 0.85,
    }
    return {
        "primary_endpoint": "binary accuracy improvement 8-to-12 late minus binary accuracy improvement 8-to-12 early, equal-weight pooled over Qwen/Phi",
        "primary_late_minus_early_difference_in_differences": primary,
        "primary_by_binary_judge": per_judge_primary,
        "accuracy_by_judge_polarity_form_position_and_prefix": cells,
        "laya_mixed_unclear_by_position_and_prefix": laya_ambiguity,
        "overall_binary_accuracy_by_prefix": by_budget,
        "capability_check": capability,
        "capability_check_passed": all(capability.values()),
    }


def run(private=PRIVATE, output=OUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest_path = private / "manifest.json"
    outcomes_path = private / "outcomes.json"
    stimuli_path = private / "stimuli.json"
    manifest = json.loads(manifest_path.read_text())
    for path, hash_field in ((outcomes_path, "outcomes_sha256"), (stimuli_path, "stimuli_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[hash_field]:
            raise ValueError(f"Private file hash mismatch: {path.name}")
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != manifest["protocol_sha256"]:
        raise ValueError("Protocol changed after collection")
    stimuli = json.loads(stimuli_path.read_text())
    outcomes = json.loads(outcomes_path.read_text())
    if len(stimuli) != 384 or len({row["cluster_id"] for row in stimuli}) != 48:
        raise ValueError("Frozen 035 stimulus factorial changed")
    analysis = {
        "experiment": "035",
        "design": "controlled cue-position × polarity × expression-form factorial",
        "stimuli": len(stimuli),
        "clusters": len({row["cluster_id"] for row in stimuli}),
        "outcomes": len(outcomes),
        "analysis": analyze(stimuli, outcomes),
        "limitations": [
            "The texts are short deterministic templates, not natural generated answers.",
            "Gold follows the intended ordinary-language meaning of good/bad and not good/not bad; pragmatics may differ from this construction.",
            "The two binary evaluators are fixed model families, not a random sample of judges.",
            "The late condition changes clause order as well as cue position; the estimand is the full cue-placement intervention.",
        ],
    }
    output.mkdir(parents=True)
    public_stimuli_path = output / "stimuli.json"
    public_stimuli_path.write_text(json.dumps(stimuli, indent=2, ensure_ascii=False) + "\n")
    public_outcomes = [
        {key: value for key, value in row.items() if key != "visible_text"}
        for row in outcomes
    ]
    public_outcomes_path = output / "predictions.json"
    public_outcomes_path.write_text(json.dumps(public_outcomes, indent=2, sort_keys=True) + "\n")
    public_manifest = {key: value for key, value in manifest.items() if key != "outcomes_sha256"}
    public_manifest["predictions_sha256"] = hashlib.sha256(public_outcomes_path.read_bytes()).hexdigest()
    public_manifest["stimuli_sha256"] = hashlib.sha256(public_stimuli_path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(public_manifest, indent=2, sort_keys=True) + "\n")
    audit = {
        "checks": {
            "complete_2304_judgment_factorial": len(public_outcomes) == 2304,
            "balanced_192_positive_192_negative_stimuli": sum(row["gold"] for row in stimuli) == 192,
            "all_judges_cover_all_stimuli_and_prefixes": all(
                len({(row["id"], row["budget"]) for row in public_outcomes if row["judge"] == judge})
                == 768
                for judge in JUDGES
            ),
            "protocol_hash_matches": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
            == manifest["protocol_sha256"],
        }
    }
    audit["checks_passed"] = all(audit["checks"].values())
    (output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["checks_passed"]:
        raise ValueError("Experiment 035 output audit failed")
    print(json.dumps(analysis, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", type=Path, default=PRIVATE)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.private, args.output)


if __name__ == "__main__":
    main()
