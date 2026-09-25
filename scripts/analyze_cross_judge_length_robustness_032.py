"""Analyze paired Qwen/Phi judgments with sentence-cluster intervals."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PRIVATE = Path(".context/answer-length-crossjudge-032")
OUT = Path("results/cross-judge-length-robustness-v1")
QWEN_PRIOR = Path("results/local-open-judge-v1/analysis.json")
PHI_PRIOR = Path("results/independent-judge-check-v1/analysis.json")
MODELS = ("granite-3.1-2b", "qwen-1.5b", "smollm2-1.7b")
JUDGES = ("qwen2.5-3b", "phi3-mini")
REPS = 10_000
SEED = 20260932


def is_correct(label, gold):
    if label is None:
        return False
    prediction = label if label in (0, 1) else int(label == "positive")
    return prediction == gold


def paired_accuracy_bootstrap(rows, reps=REPS, seed=SEED):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["sentence_id"]].append(row)
    clusters = np.asarray(sorted(grouped))
    if not len(clusters):
        return {"difference": None, "sentence_cluster_bootstrap_95_ci": None}

    def effect(sample):
        return float(
            np.mean(
                [
                    int(is_correct(row["label_32"], row["gold"]))
                    - int(is_correct(row["label_8"], row["gold"]))
                    for row in sample
                ]
            )
        )

    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for index in range(reps):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        sample = [row for cluster in chosen for row in grouped[cluster]]
        draws[index] = effect(sample)
    return {
        "difference_strict_accuracy_32_minus_8": effect(rows),
        "sentence_cluster_bootstrap_95_ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "bootstrap_reps": reps,
        "bootstrap_seed": seed,
        "n_sentence_clusters": len(clusters),
    }


def normalized_label(label):
    if label is None:
        return "unparseable"
    if label == 1:
        return "positive"
    if label == 0:
        return "negative"
    return label


def label_for_row(row):
    return normalized_label(row["judge_label"])


def join_pairs(predictions):
    indexed = {
        (row["judge"], row["model"], row["id"], row["budget"]): row
        for row in predictions
    }
    pairs = []
    for judge in JUDGES:
        for model in MODELS:
            keys = sorted(
                key[2] for key in indexed if key[0] == judge and key[1] == model and key[3] == 8
            )
            for item_id in keys:
                short = indexed[(judge, model, item_id, 8)]
                long = indexed[(judge, model, item_id, 32)]
                pairs.append(
                    {
                        "judge": judge,
                        "model": model,
                        "id": item_id,
                        "sentence_id": short["sentence_id"],
                        "category": short["category"],
                        "gold": short["gold"],
                        "label_8": short["judge_label"],
                        "label_32": long["judge_label"],
                        "label_changed": short["judge_label"] != long["judge_label"],
                    }
                )
    if len(pairs) != len(JUDGES) * len(MODELS) * 233:
        raise ValueError(f"Expected 1,398 judged pairs, received {len(pairs)}")
    return pairs


def label_reproduction(pairs):
    prior_by_judge = {
        "qwen2.5-3b": json.loads(QWEN_PRIOR.read_text())["outcomes"],
        "phi3-mini": json.loads(PHI_PRIOR.read_text())["outcomes"],
    }
    key_fields = {"qwen2.5-3b": "judge_label", "phi3-mini": "phi_label"}
    result = {}
    for judge in JUDGES:
        prior = {
            (row["model"], row["id"]): row[key_fields[judge]]
            for row in prior_by_judge[judge]
        }
        candidates = [row for row in pairs if row["judge"] == judge]
        fresh = {(row["model"], row["id"]): row["label_8"] for row in candidates}
        common = [key for key in prior if key in fresh]
        result[judge] = {
            "n_overlap": len(common),
            "exact_8_token_label_agreement": sum(prior[key] == fresh[key] for key in common)
            / len(common),
            "n_label_disagreements": sum(prior[key] != fresh[key] for key in common),
        }
    return result


def summarize(pairs):
    reports = {}
    for judge in JUDGES:
        judge_pairs = [row for row in pairs if row["judge"] == judge]
        targets = {}
        for model in MODELS:
            rows = [row for row in judge_pairs if row["model"] == model]
            targets[model] = {
                "n_pairs": len(rows),
                "coverage_8": sum(row["label_8"] is not None for row in rows) / len(rows),
                "coverage_32": sum(row["label_32"] is not None for row in rows) / len(rows),
                "strict_accuracy_vs_review_gold": {
                    "budget_8": sum(is_correct(row["label_8"], row["gold"]) for row in rows)
                    / len(rows),
                    "budget_32": sum(is_correct(row["label_32"], row["gold"]) for row in rows)
                    / len(rows),
                    "paired_sentence_cluster_bootstrap": paired_accuracy_bootstrap(
                        rows, seed=SEED + MODELS.index(model)
                    ),
                },
                "parseable_accuracy_vs_review_gold": {
                    "budget_8": (
                        sum(is_correct(row["label_8"], row["gold"]) for row in rows)
                        / sum(row["label_8"] is not None for row in rows)
                        if any(row["label_8"] is not None for row in rows)
                        else None
                    ),
                    "budget_32": (
                        sum(is_correct(row["label_32"], row["gold"]) for row in rows)
                        / sum(row["label_32"] is not None for row in rows)
                        if any(row["label_32"] is not None for row in rows)
                        else None
                    ),
                },
                "label_flip_matrix": {
                    label: {
                        other: sum(
                            normalized_label(row["label_8"]) == label
                            and normalized_label(row["label_32"]) == other
                            for row in rows
                        )
                        for other in ("positive", "negative", "unparseable")
                    }
                    for label in ("positive", "negative", "unparseable")
                },
            }
        reports[judge] = {
            "n_pairs": len(judge_pairs),
            "coverage_8": sum(row["label_8"] is not None for row in judge_pairs)
            / len(judge_pairs),
            "coverage_32": sum(row["label_32"] is not None for row in judge_pairs)
            / len(judge_pairs),
            "strict_accuracy_vs_review_gold": {
                "budget_8": sum(is_correct(row["label_8"], row["gold"]) for row in judge_pairs)
                / len(judge_pairs),
                "budget_32": sum(is_correct(row["label_32"], row["gold"]) for row in judge_pairs)
                / len(judge_pairs),
                "paired_sentence_cluster_bootstrap": paired_accuracy_bootstrap(judge_pairs),
            },
            "label_change_rate": sum(row["label_changed"] for row in judge_pairs)
            / len(judge_pairs),
            "by_target": targets,
        }

    agreement = {}
    by_judge_key = {
        (row["judge"], row["model"], row["id"]): row
        for row in pairs
    }
    for budget, label_key in ((8, "label_8"), (32, "label_32")):
        common = [
            (model, item_id)
            for model in MODELS
            for item_id in sorted(
                row["id"] for row in pairs if row["judge"] == "qwen2.5-3b" and row["model"] == model
            )
            if by_judge_key[("qwen2.5-3b", model, item_id)][label_key] is not None
            and by_judge_key[("phi3-mini", model, item_id)][label_key] is not None
        ]
        agreement[str(budget)] = {
            "n_both_parseable": len(common),
            "agreement": (
                sum(
                    by_judge_key[("qwen2.5-3b", model, item_id)][label_key]
                    == by_judge_key[("phi3-mini", model, item_id)][label_key]
                    for model, item_id in common
                )
                / len(common)
                if common
                else None
            ),
        }
    return {"by_judge": reports, "qwen_phi_agreement_by_budget": agreement}


def run(private=PRIVATE, output=OUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest = json.loads((private / "manifest.json").read_text())
    predictions = json.loads((private / "predictions.json").read_text())
    if hashlib.sha256((private / "predictions.json").read_bytes()).hexdigest() != manifest[
        "predictions_sha256"
    ]:
        raise ValueError("Judge predictions fail source manifest hash")
    pairs = join_pairs(predictions)
    analysis = {
        "experiment": "032",
        "status": "post-result exploratory robustness follow-up",
        "n_paired_judge_target_item_outcomes": len(pairs),
        "analysis": summarize(pairs),
        "exact_8_token_reproduction": label_reproduction(pairs),
        "limitations": [
            "Qwen and Phi are automated judges, not answer-level human ground truth.",
            "Review-level gold is only a proxy for generated-answer meaning.",
            "The answer texts and items are exactly those used in experiment 031, so this is judge replication rather than an independent sample replication.",
            "The length intervention adds semantic content as well as tokens; it does not isolate a generic preference for verbosity.",
        ],
    }
    output.mkdir(parents=True)
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    (output / "predictions.json").write_text(json.dumps(predictions, indent=2, sort_keys=True) + "\n")
    public_manifest = {key: value for key, value in manifest.items() if key != "source_answer_file_sha256"}
    public_manifest["predictions_sha256"] = hashlib.sha256(
        (output / "predictions.json").read_bytes()
    ).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(public_manifest, indent=2, sort_keys=True) + "\n"
    )
    audit = {
        "checks": {
            "complete_1398_paired_labels": len(pairs) == 2 * 3 * 233,
            "no_raw_text_or_token_ids_in_public_predictions": all(
                not ({"answer", "text", "review", "user", "token_ids"} & row.keys())
                for row in predictions
            ),
            "protocol_hash_matches": hashlib.sha256(
                Path("docs/experiments/032-cross-judge-length-robustness.md").read_bytes()
            ).hexdigest()
            == manifest["protocol_sha256"],
        }
    }
    audit["checks_passed"] = all(audit["checks"].values())
    (output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if not audit["checks_passed"]:
        raise ValueError("Audit failed")
    readme = """# Experiment 032 — cross-judge length robustness\n\nThis post-result follow-up applies the Qwen2.5-3B and Phi-3 Mini judges to the exact 8- and 32-token texts from experiment 031. It is a judge replication, not a new sample. Raw reviews and generated answer text remain in ignored `.context/`. Read [`analysis.json`](analysis.json), [`predictions.json`](predictions.json), and the frozen [protocol](../../docs/experiments/032-cross-judge-length-robustness.md). Review-level labels are only proxies for the meaning of generated answers.\n"""
    (output / "README.md").write_text(readme)
    print(json.dumps(analysis, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", type=Path, default=PRIVATE)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.private, args.output)


if __name__ == "__main__":
    main()
