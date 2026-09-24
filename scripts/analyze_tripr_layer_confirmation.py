"""Analyze the preregistered paired layer comparison on TripR-2020Large."""

import argparse
import json
from pathlib import Path

import numpy as np
from natural_aspect_selectivity import sentence_contrasts
from tripr_layer_confirmation import LAYERS, MODELS

from latent_decisions.experiment import write_json

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260924
CONDITIONS = ("native", "shared", "random")


def load_cell(root, model, layer):
    folder = root / f"layer-{layer:02d}" / model
    baseline = json.loads((folder / "baseline.json").read_text())
    effects = json.loads((folder / "effects.json").read_text())
    native = sentence_contrasts(effects, "native")
    random = sentence_contrasts(effects, "random")
    common = sorted(set(native) & set(random))
    specificity = {sid: native[sid] - random[sid] for sid in common}
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
        for sid in common
    }
    shared = sentence_contrasts(effects, "shared")
    conflicts = {
        sid
        for sid in common
        if len(
            {
                row["label"]
                for row in baseline
                if row["sentence_id"] == sid
            }
        )
        > 1
    }
    baseline_accuracy = float(np.mean([row["strict_correct"] for row in baseline]))
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
    ids = sorted(set(specificity) & set(generic))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    specificity_draws = np.empty(BOOTSTRAP_REPS)
    generic_draws = np.empty(BOOTSTRAP_REPS)
    ratio_draws = np.full(BOOTSTRAP_REPS, np.nan)
    for index in range(BOOTSTRAP_REPS):
        chosen = rng.integers(0, len(ids), size=len(ids))
        sample = [ids[item] for item in chosen]
        spec = float(np.mean([specificity[sid] for sid in sample]))
        shift = float(np.mean([generic[sid] for sid in sample]))
        specificity_draws[index] = spec
        generic_draws[index] = shift
        if shift > 0:
            ratio_draws[index] = spec / shift
    generic_ci = [
        float(np.quantile(generic_draws, 0.025)),
        float(np.quantile(generic_draws, 0.975)),
    ]
    ratio_defined = float(np.mean(list(generic.values()))) > 0 and generic_ci[0] > 0
    ratio = (
        float(np.mean(list(specificity.values())))
        / float(np.mean(list(generic.values())))
        if ratio_defined
        else None
    )
    conflict_values = [specificity[sid] for sid in sorted(conflicts)]
    conflict_rng = np.random.default_rng(BOOTSTRAP_SEED + 81)
    conflict_draws = np.empty(BOOTSTRAP_REPS)
    for index in range(BOOTSTRAP_REPS):
        chosen = conflict_rng.integers(
            0, len(conflict_values), size=len(conflict_values)
        )
        conflict_draws[index] = np.mean(np.asarray(conflict_values)[chosen])
    return {
        "model": model,
        "layer": layer,
        "n_sentences": len(ids),
        "n_conflict_sentences": len(conflicts),
        "baseline_accuracy": baseline_accuracy,
        "baseline_valid_label_rate": float(
            np.mean([row["valid_polarity_token"] for row in baseline])
        ),
        "steered_accuracy_by_condition": accuracy_by_condition,
        "generic_native_shift_mean_logits": float(np.mean(list(generic.values()))),
        "generic_native_shift_95_ci": generic_ci,
        "native_minus_random_specificity_mean_logits": float(
            np.mean(list(specificity.values()))
        ),
        "specificity_95_ci": [
            float(np.quantile(specificity_draws, 0.025)),
            float(np.quantile(specificity_draws, 0.975)),
        ],
        "specificity_fraction_of_generic_shift": ratio,
        "ratio_95_ci": (
            [
                float(np.quantile(ratio_draws, 0.025)),
                float(np.quantile(ratio_draws, 0.975)),
            ]
            if ratio_defined
            and np.isfinite(ratio_draws).sum() >= int(0.975 * BOOTSTRAP_REPS)
            else None
        ),
        "conflict_only_specificity_mean_logits": float(np.mean(conflict_values)),
        "conflict_only_95_ci": [
            float(np.quantile(conflict_draws, 0.025)),
            float(np.quantile(conflict_draws, 0.975)),
        ],
        "shared_specificity_mean_logits": float(np.mean(list(shared.values()))),
        "_specificity_by_sentence": specificity,
        "_generic_by_sentence": generic,
    }


def paired_ratio_difference(left, right):
    ids = sorted(
        set(left["_specificity_by_sentence"])
        & set(right["_specificity_by_sentence"])
        & set(left["_generic_by_sentence"])
        & set(right["_generic_by_sentence"])
    )
    point_left = left["specificity_fraction_of_generic_shift"]
    point_right = right["specificity_fraction_of_generic_shift"]
    if point_left is None or point_right is None:
        return {"defined": False, "n_sentences": len(ids)}
    rng = np.random.default_rng(BOOTSTRAP_SEED + 200)
    draws = np.full(BOOTSTRAP_REPS, np.nan)
    for index in range(BOOTSTRAP_REPS):
        chosen = rng.integers(0, len(ids), size=len(ids))
        sample = [ids[item] for item in chosen]
        left_shift = float(
            np.mean([left["_generic_by_sentence"][sid] for sid in sample])
        )
        right_shift = float(
            np.mean([right["_generic_by_sentence"][sid] for sid in sample])
        )
        if left_shift > 0 and right_shift > 0:
            left_spec = float(
                np.mean([left["_specificity_by_sentence"][sid] for sid in sample])
            )
            right_spec = float(
                np.mean([right["_specificity_by_sentence"][sid] for sid in sample])
            )
            draws[index] = left_spec / left_shift - right_spec / right_shift
    valid = draws[np.isfinite(draws)]
    defined = len(valid) >= int(0.975 * BOOTSTRAP_REPS)
    return {
        "defined": defined,
        "n_sentences": len(ids),
        "layer16_minus_layer24_specificity_fraction": point_left - point_right,
        "sentence_bootstrap_95_ci": (
            [
                float(np.quantile(valid, 0.025)),
                float(np.quantile(valid, 0.975)),
            ]
            if defined
            else None
        ),
    }


def analyze(root, models=MODELS):
    results = {}
    for model in models:
        cells = {
            str(layer): load_cell(root, model, layer)
            for layer in LAYERS
        }
        paired = paired_ratio_difference(cells["16"], cells["24"])
        layer16 = cells["16"]
        support = bool(
            paired["defined"]
            and paired["sentence_bootstrap_95_ci"][0] > 0
            and layer16["specificity_95_ci"][0] > 0
            and layer16["baseline_accuracy"] > 0.5
        )
        practical = bool(
            layer16["specificity_fraction_of_generic_shift"] is not None
            and layer16["specificity_fraction_of_generic_shift"] >= 0.05
            and layer16["specificity_95_ci"][0] > 0
            and layer16["baseline_accuracy"] > 0.5
        )
        for cell in cells.values():
            cell.pop("_specificity_by_sentence")
            cell.pop("_generic_by_sentence")
        results[model] = {
            **cells,
            "primary_layer16_minus_layer24_ratio_difference": paired,
            "localization_replication_gate": support,
            "layer16_practical_specificity_gate": practical,
        }
    results["cross_model_localization_replication_gate"] = all(
        results[model]["localization_replication_gate"] for model in models
    )
    results["cross_model_layer16_practical_gate"] = all(
        results[model]["layer16_practical_specificity_gate"] for model in models
    )
    write_json(root / "analysis.json", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/tripr-layer-confirmation-v1"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.models), indent=2), flush=True)


if __name__ == "__main__":
    main()
