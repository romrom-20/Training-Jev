"""Analyze preregistered MAMS layer-specificity outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np
from mams_aspect_selectivity import MODELS
from mams_layer_selectivity import LAYERS
from natural_aspect_selectivity import bootstrap_ci, sentence_contrasts

from latent_decisions.experiment import write_json

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260924
CONDITIONS = ("native", "shared", "random")


def analyze_cell(root, model, layer):
    folder = root / f"layer-{layer:02d}" / model
    baseline = json.loads((folder / "baseline.json").read_text())
    effects = json.loads((folder / "effects.json").read_text())
    native = sentence_contrasts(effects, "native")
    random = sentence_contrasts(effects, "random")
    shared = sentence_contrasts(effects, "shared")
    sentence_ids = sorted(set(native) & set(random))
    specificity = {sid: native[sid] - random[sid] for sid in sentence_ids}
    generic = {
        sid: float(
            np.mean(
                [
                    row["margin_delta"]
                    for row in effects
                    if row["sentence_id"] == sid and row["condition"] == "native"
                ]
            )
        )
        for sid in sentence_ids
    }
    point_specificity = float(np.mean(list(specificity.values())))
    point_generic = float(np.mean(list(generic.values())))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    spec_draws, ratio_draws = [], []
    for _ in range(BOOTSTRAP_REPS):
        chosen = rng.choice(sentence_ids, size=len(sentence_ids), replace=True)
        spec = float(np.mean([specificity[sid] for sid in chosen]))
        gen = float(np.mean([generic[sid] for sid in chosen]))
        spec_draws.append(spec)
        ratio_draws.append(spec / gen if gen else float("nan"))
    ratio = point_specificity / point_generic
    conflict_ids = {
        sid
        for sid in sentence_ids
        if len({row["label"] for row in baseline if row["sentence_id"] == sid}) > 1
    }
    conflict_values = [specificity[sid] for sid in sorted(conflict_ids)]
    accuracy_by_condition = {
        condition: float(
            np.mean(
                [
                    row["steered_strict_correct"]
                    for row in effects
                    if row["condition"] == condition
                ]
            )
        )
        for condition in CONDITIONS
    }
    return {
        "model": model,
        "layer": layer,
        "n_sentences": len(sentence_ids),
        "n_polarity_conflict_sentences": len(conflict_ids),
        "baseline_accuracy": float(np.mean([row["strict_correct"] for row in baseline])),
        "baseline_valid_label_rate": float(
            np.mean([row["valid_polarity_token"] for row in baseline])
        ),
        "steered_accuracy_by_condition": accuracy_by_condition,
        "generic_native_shift_mean_logits": point_generic,
        "native_minus_random_specificity_mean_logits": point_specificity,
        "specificity_95_ci": [
            float(np.quantile(spec_draws, 0.025)),
            float(np.quantile(spec_draws, 0.975)),
        ],
        "specificity_fraction_of_generic_shift": ratio,
        "ratio_95_ci": [
            float(np.quantile(ratio_draws, 0.025)),
            float(np.quantile(ratio_draws, 0.975)),
        ],
        "conflict_only_specificity_mean_logits": float(np.mean(conflict_values)),
        "conflict_only_95_ci": bootstrap_ci(
            conflict_values, sorted(conflict_ids), reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED
        ),
        "shared_specificity_mean_logits": float(np.mean(list(shared.values()))),
        "practical_layer_gate": bool(
            ratio >= 0.05
            and np.quantile(spec_draws, 0.025) > 0
            and np.mean([row["strict_correct"] for row in baseline]) > 0.5
        ),
    }


def analyze(root, models=MODELS):
    results = {
        model: {str(layer): analyze_cell(root, model, layer) for layer in LAYERS}
        for model in models
    }
    results["any_practical_layer_gate_by_model"] = {
        model: any(results[model][str(layer)]["practical_layer_gate"] for layer in LAYERS)
        for model in models
    }
    results["cross_model_any_practical_layer_gate"] = all(
        results["any_practical_layer_gate_by_model"].values()
    )
    write_json(root / "analysis.json", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/mams-layer-selectivity-v1"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.models), indent=2), flush=True)


if __name__ == "__main__":
    main()
