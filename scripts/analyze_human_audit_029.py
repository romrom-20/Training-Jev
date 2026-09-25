"""Analyze experiment 029 labels without exporting generated answer text."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PRIVATE = Path(".context/human-coding-029")
JUDGES = Path("results/independent-judge-check-v1/analysis.json")
DEFAULT_OUT = Path("results/human-audit-029-v1")
CODE_MAP = {"positive": 1, "negative": 0}
LABEL_CODES = set(CODE_MAP) | {"mixed", "unclear"}
MODELS = ("granite-3.1-2b", "qwen-1.5b", "smollm2-1.7b")
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261029


def sentence_bootstrap(rows, prediction, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    by_sentence = defaultdict(list)
    for row in rows:
        if row["human_label"] in CODE_MAP:
            by_sentence[row["sentence_id"]].append(row)
    sentence_ids = np.asarray(sorted(by_sentence))
    if not len(sentence_ids):
        return None
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps)
    for index in range(reps):
        selected = rng.choice(sentence_ids, size=len(sentence_ids), replace=True)
        sample = [row for sentence_id in selected for row in by_sentence[sentence_id]]
        eligible = [row for row in sample if prediction(row) is not None]
        estimates[index] = (
            np.mean([prediction(row) == row["human_numeric"] for row in eligible])
            if eligible
            else np.nan
        )
    valid = estimates[np.isfinite(estimates)]
    eligible = [
        row for row in rows if row["human_label"] in CODE_MAP and prediction(row) is not None
    ]
    point = float(np.mean([prediction(row) == row["human_numeric"] for row in eligible]))
    return {
        "point_accuracy": point,
        "sentence_cluster_bootstrap_95_ci": [
            float(np.quantile(valid, 0.025)),
            float(np.quantile(valid, 0.975)),
        ],
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
    }


def load_joined(labels_path):
    mapping = json.loads((PRIVATE / "private-key.json").read_text())
    annotation_doc = json.loads(labels_path.read_text())
    if annotation_doc.get("experiment") != "029":
        raise ValueError("Annotation file must identify experiment 029")
    annotations = annotation_doc.get("annotations")
    expected = {row["blind_id"] for row in mapping}
    if not isinstance(annotations, dict) or set(annotations) != expected:
        missing = sorted(expected - set(annotations or {}))
        unknown = sorted(set(annotations or {}) - expected)
        raise ValueError(f"Need exactly one label per row; missing={missing}, unknown={unknown}")
    if any(value not in LABEL_CODES for value in annotations.values()):
        raise ValueError("Labels must be positive, negative, mixed, or unclear")
    judge_rows = json.loads(JUDGES.read_text())["outcomes"]
    judge_by_key = {(row["model"], row["id"]): row for row in judge_rows}
    joined = []
    for private in mapping:
        key = (private["model"], private["stimulus_id"])
        if key not in judge_by_key:
            raise ValueError(f"Missing frozen judge row for {key}")
        judge = judge_by_key[key]
        human = annotations[private["blind_id"]]
        joined.append(
            {
                "blind_id": private["blind_id"],
                "model": private["model"],
                "stimulus_id": private["stimulus_id"],
                "sentence_id": private["sentence_id"],
                "category": private["category"],
                "gold": private["gold"],
                "human_label": human,
                "human_numeric": CODE_MAP.get(human),
                "candidate_pair_prediction": judge["candidate_pair_prediction"],
                "qwen_label": judge["qwen_label"],
                "phi_label": judge["phi_label"],
            }
        )
    if len(joined) != 60 or len({row["stimulus_id"] for row in joined}) != 20:
        raise ValueError("Frozen paired design changed: expected 60 rows from 20 items")
    return joined, annotations


def summarize(joined, annotations):
    counts = Counter(annotations.values())
    output = {
        "experiment": "029",
        "n_answers": len(joined),
        "n_unique_source_sentences": len({row["sentence_id"] for row in joined}),
        "human_code_counts": {
            code: counts[code] for code in ("positive", "negative", "mixed", "unclear")
        },
        "per_target": {},
        "limitations": [
            "One blinded annotator; no inter-rater reliability estimate.",
            "Twenty stratified source sentences; bootstrap intervals are descriptive and not population-representative.",
            "Human/automated agreement assesses this annotation of these regenerated answers, not general model quality.",
        ],
    }
    predictors = {
        "candidate_pair": lambda row: row["candidate_pair_prediction"],
        "qwen2_5_3b": lambda row: row["qwen_label"],
        "phi3_mini": lambda row: row["phi_label"],
    }
    for model in MODELS:
        rows = [row for row in joined if row["model"] == model]
        clear = [row for row in rows if row["human_label"] in CODE_MAP]
        gold_accuracy = (
            float(np.mean([row["gold"] == row["human_numeric"] for row in clear]))
            if clear
            else None
        )
        metrics = {
            "n_answers": len(rows),
            "human_clear_coverage": len(clear) / len(rows),
            "human_agreement_with_review_gold": gold_accuracy,
            "signals": {},
            "paired_disagreements": [],
        }
        for name, predictor in predictors.items():
            eligible = [row for row in clear if predictor(row) is not None]
            correct = sum(predictor(row) == row["human_numeric"] for row in eligible)
            metrics["signals"][name] = {
                "n_scored": len(eligible),
                "coverage_of_all_answers": len(eligible) / len(rows),
                "accuracy_on_clear_human_codes": correct / len(eligible) if eligible else None,
                "sentence_cluster_bootstrap": sentence_bootstrap(rows, predictor)
                if eligible
                else None,
            }
        for row in rows:
            metrics["paired_disagreements"].append(
                {
                    key: row[key]
                    for key in (
                        "blind_id",
                        "stimulus_id",
                        "category",
                        "human_label",
                        "gold",
                        "candidate_pair_prediction",
                        "qwen_label",
                        "phi_label",
                    )
                }
            )
        metrics["paired_disagreements"] = [
            row
            for row in metrics["paired_disagreements"]
            if row["human_label"] not in CODE_MAP
            or row["gold"] != CODE_MAP[row["human_label"]]
            or row["candidate_pair_prediction"] != CODE_MAP[row["human_label"]]
            or row["qwen_label"] != CODE_MAP[row["human_label"]]
            or row["phi_label"] != CODE_MAP[row["human_label"]]
        ]
        metrics["n_paired_disagreements"] = len(metrics["paired_disagreements"])
        output["per_target"][model] = metrics
    return output


def run(labels_path, output=DEFAULT_OUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    joined, annotations = load_joined(labels_path)
    summary = summarize(joined, annotations)
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "audited-labels.json").write_text(json.dumps(joined, indent=2) + "\n")
    lines = [
        "# Blinded human audit of generated-answer polarity",
        "",
        "This report contains item IDs and labels only; generated answer text is excluded.",
        "The primary descriptive accuracy is computed only on human-coded positive/negative answers; coverage and mixed/unclear counts are reported explicitly.",
        "See `summary.json` and `audited-labels.json` for the machine-readable results.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    source_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    print(f"Wrote {output}; annotation file SHA-256: {source_hash}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=PRIVATE / "labels.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.labels, args.output)


if __name__ == "__main__":
    main()
