"""Analyze frozen MAMS intervention captures using experiment 014 endpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
from mams_aspect_selectivity import MODELS
from natural_aspect_selectivity import analyze_model, bootstrap_ci, sentence_contrasts

from latent_decisions.experiment import write_json


def analyze_one(root, model):
    result = analyze_model(root, model)
    baseline = json.loads((root / model / "baseline.json").read_text())
    effects = json.loads((root / model / "effects.json").read_text())
    labels = (0, 1)
    result["label_balance_and_class_accuracy"] = {
        "positive_label_fraction": float(np.mean([row["label"] for row in baseline])),
        "baseline_accuracy_by_label": {
            str(label): float(
                np.mean([row["strict_correct"] for row in baseline if row["label"] == label])
            )
            for label in labels
        },
        "steered_accuracy_by_label": {
            condition: {
                str(label): float(
                    np.mean(
                        [
                            row["steered_strict_correct"]
                            for row in effects
                            if row["condition"] == condition and row["label"] == label
                        ]
                    )
                )
                for label in labels
            }
            for condition in ("native", "shared", "random")
        },
    }
    conflict_ids = {
        row["sentence_id"]
        for row in baseline
        if sum(
            other["sentence_id"] == row["sentence_id"] and other["label"] != row["label"]
            for other in baseline
        )
    }
    conflict_effects = [row for row in effects if row["sentence_id"] in conflict_ids]
    native = sentence_contrasts(conflict_effects, "native")
    random = sentence_contrasts(conflict_effects, "random")
    common = sorted(set(native) & set(random))
    deltas = [native[key] - random[key] for key in common]
    result["polarity_conflict_primary_slice"] = {
        "n_sentences": len(common),
        "native_minus_random_specificity": float(np.mean(deltas)),
        "sentence_bootstrap_95_ci": bootstrap_ci(deltas, common),
    }
    native_means = [result["effects"][f"native_{source}"]["mean_margin_delta"] for source in range(3)]
    generic_shift = float(np.mean(native_means))
    specificity = result["primary_native_minus_random_specificity"]
    lower = specificity["sentence_bootstrap_95_ci"][0]
    fraction_of_shift = specificity["mean_native_minus_random_specificity"] / generic_shift
    result["generic_shift_mean_logits"] = generic_shift
    result["specificity_fraction_of_generic_shift"] = float(fraction_of_shift)
    result["preregistered_practical_selectivity_gate"] = bool(
        lower > 0 and fraction_of_shift >= 0.05
    )
    return result


def analyze(root, models=MODELS):
    results = {model: analyze_one(root, model) for model in models}
    results["cross_model_practical_selectivity_gate_pass"] = all(
        results[model]["preregistered_practical_selectivity_gate"] for model in models
    )
    write_json(root / "analysis.json", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/mams-aspect-selectivity-v1"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.models), indent=2), flush=True)


if __name__ == "__main__":
    main()
