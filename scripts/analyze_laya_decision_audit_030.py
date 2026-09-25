"""Analyze Laya's local typed decisions without exporting text."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PRIVATE_RUN = Path(".context/laya-decision-audit-030")
OUT = Path("results/laya-decision-audit-v1")
QWEN = Path("results/local-open-judge-v1/analysis.json")
PHI = Path("results/independent-judge-check-v1/analysis.json")
MODELS = ("granite-3.1-2b", "qwen-1.5b", "smollm2-1.7b")
LABELS = {"positive": 1, "negative": 0}
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261030


def bootstrap_ratio(rows, numerator, denominator, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sentence_id"]].append(row)
    sentence_ids = np.asarray(sorted(grouped))
    if not len(sentence_ids):
        return None

    def rate(sample):
        den = sum(denominator(row) for row in sample)
        if den == 0:
            return np.nan
        return sum(numerator(row) for row in sample) / den

    point = rate(rows)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for index in range(reps):
        selected = rng.choice(sentence_ids, size=len(sentence_ids), replace=True)
        sample = [row for sid in selected for row in grouped[sid]]
        draws[index] = rate(sample)
    valid = draws[np.isfinite(draws)]
    return {
        "estimate": float(point),
        "sentence_cluster_bootstrap_95_ci": [
            float(np.quantile(valid, 0.025)),
            float(np.quantile(valid, 0.975)),
        ],
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
    }


def join_outcomes(raw_rows):
    qwen_rows = json.loads(QWEN.read_text())["outcomes"]
    phi_rows = json.loads(PHI.read_text())["outcomes"]
    qwen_by_key = {(row["model"], row["id"]): row for row in qwen_rows}
    phi_by_key = {(row["model"], row["id"]): row for row in phi_rows}
    joined = []
    for raw in raw_rows:
        if raw["phase"] != "generated":
            continue
        key = (raw["model"], raw["id"])
        if key not in qwen_by_key or key not in phi_by_key:
            raise ValueError(f"Missing prior judge output for {key}")
        joined.append(
            raw
            | {
                "qwen_label": qwen_by_key[key]["judge_label"],
                "phi_label": phi_by_key[key]["phi_label"],
                "candidate_pair_prediction": qwen_by_key[key]["candidate_pair_prediction"],
            }
        )
    if len(joined) != 60 or len({row["id"] for row in joined}) != 20:
        raise ValueError("Expected 60 generated decisions on 20 paired items")
    return joined


def signal_metric(rows, predictor, conditional_clear=False):
    scored = [row for row in rows if predictor(row) in (0, 1)]
    metric = {
        "n_scored": len(scored),
        "coverage_of_20": len(scored) / len(rows),
        "accuracy_against_review_gold": (
            sum(predictor(row) == row["gold"] for row in scored) / len(scored) if scored else None
        ),
    }
    if scored:
        metric["accuracy_sentence_cluster_bootstrap"] = bootstrap_ratio(
            rows,
            lambda row: int(predictor(row) == row["gold"]),
            lambda row: int(predictor(row) in (0, 1)),
        )
    return metric


def summarize(raw_rows):
    source = [row for row in raw_rows if row["phase"] == "source"]
    generated = join_outcomes(raw_rows)
    if len(source) != 233:
        raise ValueError("Source screen must include all 233 frozen items")
    source_clear = [row for row in source if row["laya_label"] in LABELS]
    source_negative = [row for row in source if row["gold"] == 0]
    report = {
        "experiment": "030",
        "source_screen": {
            "n": len(source),
            "four_way_counts": dict(Counter(row["laya_label"] for row in source)),
            "clear_polarity_coverage": len(source_clear) / len(source),
            "clear_polarity_accuracy": (
                sum(LABELS[row["laya_label"]] == row["gold"] for row in source_clear)
                / len(source_clear)
                if source_clear
                else None
            ),
            "strict_binary_accuracy_all_items": sum(
                row["laya_label"] in LABELS and LABELS[row["laya_label"]] == row["gold"]
                for row in source
            )
            / len(source),
            "negative_recall_all_negative_items": sum(
                row["laya_label"] == "negative" for row in source_negative
            )
            / len(source_negative),
            "always_positive_baseline": sum(row["gold"] == 1 for row in source) / len(source),
        },
        "per_target_generated_answers": {},
        "limitations": [
            "Laya is an automated decision engine, not a human reference annotator.",
            "Review-level gold is only a proxy for meaning of generated answers.",
            "The 20-item generated-answer sample is stratified; intervals are descriptive.",
            "Model-card calibration results do not validate confidence on aspect sentiment.",
        ],
    }
    predictors = {
        "laya": lambda row: LABELS.get(row["laya_label"]),
        "qwen2_5_3b": lambda row: row["qwen_label"],
        "phi3_mini": lambda row: row["phi_label"],
        "candidate_pair": lambda row: row["candidate_pair_prediction"],
    }
    for model in MODELS:
        rows = [row for row in generated if row["model"] == model]
        counts = Counter(row["laya_label"] for row in rows)
        laya_clear = [row for row in rows if row["laya_label"] in LABELS]
        item_report = {
            "n": len(rows),
            "laya_four_way_counts": {
                label: counts[label] for label in ("positive", "negative", "mixed", "unclear")
            },
            "laya_four_way_proportions": {
                label: bootstrap_ratio(
                    rows,
                    lambda row, label=label: int(row["laya_label"] == label),
                    lambda row: 1,
                    seed=BOOTSTRAP_SEED + 20 + MODELS.index(model),
                )
                for label in ("positive", "negative", "mixed", "unclear")
            },
            "laya_clear_polarity_coverage": len(laya_clear) / len(rows),
            "laya_clear_polarity_coverage_bootstrap": bootstrap_ratio(
                rows,
                lambda row: int(row["laya_label"] in LABELS),
                lambda row: 1,
                seed=BOOTSTRAP_SEED + MODELS.index(model),
            ),
            "laya_strict_accuracy_against_review_gold": sum(
                row["laya_label"] in LABELS and LABELS[row["laya_label"]] == row["gold"]
                for row in rows
            )
            / len(rows),
            "automated_signals_against_review_gold": {},
            "laya_agreement": {},
            "mean_laya_reported_confidence": float(
                np.mean(
                    [row["laya_confidence"] for row in rows if row["laya_confidence"] is not None]
                )
            ),
            "disagreements": [],
        }
        for signal, predict in predictors.items():
            item_report["automated_signals_against_review_gold"][signal] = signal_metric(
                rows, predict
            )
        laya_binary = predictors["laya"]
        for signal in ("qwen2_5_3b", "phi3_mini", "candidate_pair"):
            predict = predictors[signal]
            both = [row for row in rows if laya_binary(row) is not None and predict(row) in (0, 1)]
            item_report["laya_agreement"][signal] = {
                "n_both_binary": len(both),
                "coverage_of_20": len(both) / len(rows),
                "agreement": (
                    sum(laya_binary(row) == predict(row) for row in both) / len(both)
                    if both
                    else None
                ),
                "agreement_sentence_cluster_bootstrap": (
                    bootstrap_ratio(
                        rows,
                        lambda row: int(laya_binary(row) == predict(row)),
                        lambda row: int(laya_binary(row) is not None and predict(row) in (0, 1)),
                        seed=BOOTSTRAP_SEED + 10 + MODELS.index(model),
                    )
                    if both
                    else None
                ),
            }
        for row in rows:
            item_report["disagreements"].append(
                {
                    key: row[key]
                    for key in (
                        "id",
                        "category",
                        "gold",
                        "laya_label",
                        "laya_probabilities",
                        "qwen_label",
                        "phi_label",
                        "candidate_pair_prediction",
                    )
                }
            )
        item_report["disagreements"] = [
            row
            for row in item_report["disagreements"]
            if row["laya_label"] not in LABELS
            or LABELS[row["laya_label"]] != row["gold"]
            or row["qwen_label"] != LABELS.get(row["laya_label"])
            or row["phi_label"] != LABELS.get(row["laya_label"])
            or row["candidate_pair_prediction"] != LABELS.get(row["laya_label"])
        ]
        report["per_target_generated_answers"][model] = item_report
    return report, generated


def run(raw_dir=PRIVATE_RUN, output=OUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    raw_rows = json.loads((raw_dir / "predictions.json").read_text())
    manifest = json.loads((raw_dir / "manifest.json").read_text())
    report, generated = summarize(raw_rows)
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "analysis.json").write_text(json.dumps(generated, indent=2) + "\n")
    (output / "source-analysis.json").write_text(
        json.dumps([row for row in raw_rows if row["phase"] == "source"], indent=2) + "\n"
    )
    (output / "audit.json").write_text(
        json.dumps(
            {
                "experiment": "030",
                "n_source_reviews": sum(row["phase"] == "source" for row in raw_rows),
                "n_generated_answers": len(generated),
                "model_revision": manifest["model_revision"],
                "laya_package_version": manifest["package"]["version"],
                "device": manifest["device"],
                "protocol_sha256": manifest["protocol_sha256"],
                "runner_sha256": manifest["runner_sha256"],
                "private_answer_key_sha256": manifest["private_answer_key_sha256"],
                "raw_generated_text_in_bundle": False,
                "unique_sampled_source_sentences": len({row["sentence_id"] for row in generated}),
            },
            indent=2,
        )
        + "\n"
    )
    (output / "README.md").write_text(
        "# Laya decision-engine audit (experiment 030)\n\n"
        "This local Laya run classifies source reviews and paired generated answers. "
        "The report contains IDs, labels, and probabilities, not review or answer text. "
        "Laya is an additional automated judge, not human ground truth; review labels "
        "are only a proxy for generated-answer meaning. See `summary.json`, `analysis.json`, "
        "`source-analysis.json`, `audit.json`, and the [preregistered protocol]"
        "(../../docs/experiments/030-laya-decision-audit.md).\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=PRIVATE_RUN)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.raw_dir, args.output)


if __name__ == "__main__":
    main()
