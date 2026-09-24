"""Analyze candidate-score agreement and error detection for SemEval generations."""

import argparse
import json
from pathlib import Path

import numpy as np
from granite_semeval_margin_generation import ROOT
from sklearn.metrics import roc_auc_score

from latent_decisions.experiment import write_json

BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261024


def analyze(root=ROOT):
    outcomes = json.loads((root / "outcomes.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    valid = [row for row in outcomes if row["generated_valid_exact_one_word"]]
    agreement = np.asarray(
        [row["candidate_pair_prediction"] == row["generated_label"] for row in valid],
        dtype=float,
    )
    grouped = {}
    for row in valid:
        grouped.setdefault(row["sentence_id"], []).append(
            float(row["candidate_pair_prediction"] == row["generated_label"])
        )
    sentence_ids = sorted(grouped)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPS)
    for index in range(BOOTSTRAP_REPS):
        selected = rng.choice(sentence_ids, size=len(sentence_ids), replace=True)
        sample = [value for sid in selected for value in grouped[sid]]
        draws[index] = float(np.mean(sample))
    ci = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]

    labels = np.asarray([row["label"] for row in outcomes], dtype=int)
    margins = np.asarray([row["candidate_margin"] for row in outcomes], dtype=float)
    pair_predictions = np.asarray(
        [row["candidate_pair_prediction"] for row in outcomes], dtype=int
    )
    generated_correct = np.asarray(
        [row["generated_strict_correct"] for row in outcomes], dtype=int
    )
    all_top1_correct = np.asarray(
        [row["all_vocabulary_top1_correct"] for row in outcomes], dtype=int
    )
    # Absolute pairwise margin is available at prediction time. A gold-aligned
    # margin would encode correctness directly and is therefore not a valid
    # predictor of generation error.
    error_auc = None
    if len(set(1 - generated_correct)) == 2:
        error_auc = float(roc_auc_score(1 - generated_correct, -np.abs(margins)))
    validity = float(np.mean([row["generated_valid_exact_one_word"] for row in outcomes]))
    primary_pass = validity >= 0.95 and ci[0] > 0.5
    result = {
        "n_prompts": len(outcomes),
        "n_sentences": len({row["sentence_id"] for row in outcomes}),
        "generated_exact_one_word_rate": validity,
        "candidate_pair_accuracy": float(np.mean(pair_predictions == labels)),
        "generated_strict_accuracy": float(np.mean(generated_correct)),
        "all_vocabulary_top1_accuracy": float(np.mean(all_top1_correct)),
        "candidate_pair_generation_agreement": float(np.mean(agreement)) if len(agreement) else None,
        "candidate_pair_generation_agreement_sentence_bootstrap_95_ci": ci,
        "absolute_margin_error_detection_auc_secondary": error_auc,
        "n_generation_errors": int(np.sum(1 - generated_correct)),
        "n_valid_generations": len(valid),
        "primary_measurement_gate_passed": primary_pass,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "manifest_model": manifest["model"],
        "noncausal_observational_test": True,
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
