"""Deterministic analysis for experiment 014 captures."""

import argparse
import json
from pathlib import Path

import numpy as np
from natural_aspect_selectivity import MODELS, analyze_model

from latent_decisions.experiment import write_json


def augment_per_category(root, model, result):
    effects = json.loads((root / model / "effects.json").read_text())
    baseline = json.loads((root / model / "baseline.json").read_text())
    labels = (0, 1)
    result["label_balance_and_class_accuracy"] = {
        "positive_label_fraction": float(np.mean([row["label"] for row in baseline])),
        "baseline_accuracy_by_label": {
            str(label): float(np.mean([row["strict_correct"] for row in baseline if row["label"] == label]))
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
    categories = {}
    for category in ("food", "service", "price"):
        categories[category] = {}
        for condition in ("native", "shared", "random"):
            categories[category][condition] = {}
            for source in range(3):
                rows = [
                    row
                    for row in effects
                    if row["category"] == category
                    and row["condition"] == condition
                    and row["source"] == source
                ]
                categories[category][condition][str(source)] = {
                    "mean_margin_delta": float(np.mean([row["margin_delta"] for row in rows])),
                    "n": len(rows),
                }
    result["per_category_direction_effects"] = categories
    return result


def analyze(root, models=MODELS):
    results = {
        model: augment_per_category(root, model, analyze_model(root, model))
        for model in models
    }
    write_json(root / "analysis.json", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/natural-aspect-selectivity-v1"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.models), indent=2), flush=True)


if __name__ == "__main__":
    main()
