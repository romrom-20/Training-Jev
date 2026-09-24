"""Analyze candidate-pair/generation agreement across model families."""

import argparse
import json
from pathlib import Path

import numpy as np
from cross_family_score_generation import MODELS, ROOT
from sklearn.metrics import roc_auc_score

from latent_decisions.experiment import write_json

BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261025


def summarize_model(root, name):
    rows = json.loads((root / name / "outcomes.json").read_text())
    valid = [row for row in rows if row["generated_valid_exact_one_word"]]
    agreements = [row["candidate_pair_prediction"] == row["generated_label"] for row in valid]
    by_sentence = {}
    for row in valid:
        by_sentence.setdefault(row["sentence_id"], []).append(
            float(row["candidate_pair_prediction"] == row["generated_label"])
        )
    return rows, valid, by_sentence, {
        "n_prompts": len(rows),
        "generated_exact_one_word_rate": float(
            np.mean([row["generated_valid_exact_one_word"] for row in rows])
        ),
        "candidate_pair_accuracy": float(
            np.mean([row["candidate_pair_prediction"] == row["label"] for row in rows])
        ),
        "generated_strict_accuracy": float(np.mean([row["generated_strict_correct"] for row in rows])),
        "all_vocabulary_top1_accuracy": float(
            np.mean([row["all_vocabulary_top1_correct"] for row in rows])
        ),
        "candidate_pair_generation_agreement": float(np.mean(agreements)) if agreements else None,
        "n_generation_errors": int(sum(not row["generated_strict_correct"] for row in rows)),
        "absolute_margin_error_detection_auc_secondary": float(
            roc_auc_score(
                [1 - row["generated_strict_correct"] for row in rows],
                [-abs(row["candidate_margin"]) for row in rows],
            )
        )
        if len({row["generated_strict_correct"] for row in rows}) == 2
        else None,
    }


def analyze(root=ROOT):
    details = {}
    sentence_agreements = {}
    for name in MODELS:
        rows, valid, grouped, summary = summarize_model(root, name)
        sentence_ids = sorted(grouped)
        rng = np.random.default_rng(BOOTSTRAP_SEED + MODELS.index(name))
        draws = []
        for _ in range(BOOTSTRAP_REPS):
            selected = rng.choice(sentence_ids, size=len(sentence_ids), replace=True)
            sample = [value for sid in selected for value in grouped[sid]]
            draws.append(float(np.mean(sample)))
        summary["candidate_pair_generation_agreement_sentence_bootstrap_95_ci"] = [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ]
        summary["measurement_gate_passed"] = (
            summary["generated_exact_one_word_rate"] >= 0.95
            and summary["candidate_pair_generation_agreement_sentence_bootstrap_95_ci"][0] > 0.5
        )
        details[name] = summary
        sentence_agreements[name] = grouped

    common_sentences = sorted(
        set(sentence_agreements[MODELS[0]]).intersection(sentence_agreements[MODELS[1]])
    )
    difference_rng = np.random.default_rng(BOOTSTRAP_SEED + 2)
    differences = []
    for _ in range(BOOTSTRAP_REPS):
        selected = difference_rng.choice(common_sentences, size=len(common_sentences), replace=True)
        qwen_values = [value for sid in selected for value in sentence_agreements[MODELS[0]][sid]]
        smol_values = [value for sid in selected for value in sentence_agreements[MODELS[1]][sid]]
        differences.append(float(np.mean(smol_values) - np.mean(qwen_values)))
    result = {
        "n_sentences": 112,
        "models": details,
        "smollm2_minus_qwen_agreement_difference": details[MODELS[1]][
            "candidate_pair_generation_agreement"
        ]
        - details[MODELS[0]]["candidate_pair_generation_agreement"],
        "paired_sentence_bootstrap_difference_95_ci": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "cross_family_replication_passed": all(
            details[name]["measurement_gate_passed"] for name in MODELS
        ),
        "noncausal_observational_test": True,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
    }
    write_json(root / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), indent=2), flush=True)


if __name__ == "__main__":
    main()
