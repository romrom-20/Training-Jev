"""Analyze component-specific steering on TripR-2020Large."""

import argparse
import json
from pathlib import Path

import numpy as np
from natural_aspect_selectivity import sentence_contrasts
from tripr_component_decomposition import CONDITIONS, MODELS

from latent_decisions.experiment import write_json

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261002


def bootstrap_difference(left, right, ids, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPS)
    for index in range(BOOTSTRAP_REPS):
        selected = rng.integers(0, len(ids), size=len(ids))
        draws[index] = np.mean([left[ids[i]] - right[ids[i]] for i in selected])
    return {
        "mean_difference": float(np.mean([left[sid] - right[sid] for sid in ids])),
        "paired_sentence_bootstrap_95_ci": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "n_sentences": len(ids),
    }


def analyze_model(root, model):
    folder = root / model
    baseline = json.loads((folder / "baseline.json").read_text())
    effects = json.loads((folder / "effects.json").read_text())
    contrasts = {
        condition: sentence_contrasts(effects, condition) for condition in CONDITIONS
    }
    ids = sorted(set.intersection(*(set(value) for value in contrasts.values())))
    conflicts = {
        sid
        for sid in ids
        if len({row["label"] for row in baseline if row["sentence_id"] == sid}) > 1
    }
    comparisons = {
        "residual_minus_shared": bootstrap_difference(
            contrasts["residual"], contrasts["shared"], ids, BOOTSTRAP_SEED
        ),
        "residual_minus_random_residual": bootstrap_difference(
            contrasts["residual"],
            contrasts["random_residual"],
            ids,
            BOOTSTRAP_SEED + 1,
        ),
        "native_minus_shared": bootstrap_difference(
            contrasts["native"], contrasts["shared"], ids, BOOTSTRAP_SEED + 2
        ),
        "native_minus_residual": bootstrap_difference(
            contrasts["native"], contrasts["residual"], ids, BOOTSTRAP_SEED + 3
        ),
    }
    residual_gate = all(
        comparisons[key]["paired_sentence_bootstrap_95_ci"][0] > 0
        for key in ("residual_minus_shared", "residual_minus_random_residual")
    )
    summary = {}
    for condition in CONDITIONS:
        rows = [row for row in effects if row["condition"] == condition]
        summary[condition] = {
            "specificity_mean_logits": float(np.mean(list(contrasts[condition].values()))),
            "specificity_95_ci": bootstrap_difference(
                contrasts[condition],
                {sid: 0.0 for sid in ids},
                ids,
                BOOTSTRAP_SEED + CONDITIONS.index(condition) + 10,
            )["paired_sentence_bootstrap_95_ci"],
            "generic_margin_shift_mean_logits": float(
                np.mean([row["margin_delta"] for row in rows])
            ),
            "steered_strict_accuracy": float(
                np.mean([row["steered_strict_correct"] for row in rows])
            ),
        }
    conflict_summary = {
        condition: float(np.mean([contrasts[condition][sid] for sid in sorted(conflicts)]))
        for condition in CONDITIONS
    }
    baseline_accuracy = float(np.mean([row["strict_correct"] for row in baseline]))
    return {
        "n_sentences": len(ids),
        "n_conflict_sentences": len(conflicts),
        "baseline_strict_accuracy": baseline_accuracy,
        "component_specificity": summary,
        "paired_component_comparisons": comparisons,
        "conflict_only_specificity_mean_logits": conflict_summary,
        "residual_mechanism_gate": residual_gate and baseline_accuracy > 0.5,
    }


def analyze(root, models=MODELS):
    results = {model: analyze_model(root, model) for model in models}
    results["cross_model_residual_mechanism_gate"] = all(
        results[model]["residual_mechanism_gate"] for model in models
    )
    write_json(root / "analysis.json", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/tripr-component-decomposition-v1"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.models), indent=2), flush=True)


if __name__ == "__main__":
    main()
