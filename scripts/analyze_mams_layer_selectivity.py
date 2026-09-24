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
    spec_draws, generic_draws, ratio_draws = [], [], []
    for _ in range(BOOTSTRAP_REPS):
        chosen = rng.choice(sentence_ids, size=len(sentence_ids), replace=True)
        spec = float(np.mean([specificity[sid] for sid in chosen]))
        gen = float(np.mean([generic[sid] for sid in chosen]))
        spec_draws.append(spec)
        generic_draws.append(gen)
        if gen > 0:
            ratio_draws.append(spec / gen)
    generic_ci = [
        float(np.quantile(generic_draws, 0.025)),
        float(np.quantile(generic_draws, 0.975)),
    ]
    ratio_defined = point_generic > 0 and generic_ci[0] > 0
    ratio = point_specificity / point_generic if ratio_defined else None
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
        "generic_native_shift_95_ci": generic_ci,
        "native_minus_random_specificity_mean_logits": point_specificity,
        "specificity_95_ci": [
            float(np.quantile(spec_draws, 0.025)),
            float(np.quantile(spec_draws, 0.975)),
        ],
        "specificity_fraction_of_generic_shift": ratio,
        "ratio_95_ci": (
            [
                float(np.quantile(ratio_draws, 0.025)),
                float(np.quantile(ratio_draws, 0.975)),
            ]
            if ratio_defined
            else None
        ),
        "conflict_only_specificity_mean_logits": float(np.mean(conflict_values)),
        "conflict_only_95_ci": bootstrap_ci(
            conflict_values, sorted(conflict_ids), reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED
        ),
        "shared_specificity_mean_logits": float(np.mean(list(shared.values()))),
        "practical_layer_gate": bool(
            ratio is not None
            and ratio >= 0.05
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
    # Post-hoc paired comparisons are reported for every layer pair. They were not
    # included in protocol 017 and are descriptive, with no multiplicity correction.
    pairwise = {}
    for model in models:
        pairwise[model] = {}
        contrasts, generic_effects, conflict_ids = {}, {}, {}
        for layer in LAYERS:
            folder = root / f"layer-{layer:02d}" / model
            effects = json.loads((folder / "effects.json").read_text())
            baseline = json.loads((folder / "baseline.json").read_text())
            native = sentence_contrasts(effects, "native")
            random = sentence_contrasts(effects, "random")
            contrasts[layer] = {
                sid: native[sid] - random[sid]
                for sid in set(native) & set(random)
            }
            generic_effects[layer] = {
                sid: float(
                    np.mean(
                        [
                            row["margin_delta"]
                            for row in effects
                            if row["sentence_id"] == sid
                            and row["condition"] == "native"
                        ]
                    )
                )
                for sid in contrasts[layer]
            }
            conflict_ids[layer] = {
                sid
                for sid in contrasts[layer]
                if len(
                    {
                        row["label"]
                        for row in baseline
                        if row["sentence_id"] == sid
                    }
                )
                > 1
            }
        for left_ix, left in enumerate(LAYERS):
            for right in LAYERS[left_ix + 1 :]:
                ids = sorted(set(contrasts[left]) & set(contrasts[right]))
                values = np.asarray(
                    [contrasts[left][sid] - contrasts[right][sid] for sid in ids]
                )
                rng = np.random.default_rng(BOOTSTRAP_SEED + left * 100 + right)
                draws = np.empty(BOOTSTRAP_REPS)
                for index in range(BOOTSTRAP_REPS):
                    draws[index] = np.mean(
                        values[rng.integers(0, len(values), size=len(values))]
                    )
                conflict = sorted(conflict_ids[left] & conflict_ids[right])
                conflict_values = np.asarray(
                    [contrasts[left][sid] - contrasts[right][sid] for sid in conflict]
                )
                conflict_rng = np.random.default_rng(
                    BOOTSTRAP_SEED + left * 100 + right + 1
                )
                conflict_draws = np.empty(BOOTSTRAP_REPS)
                for index in range(BOOTSTRAP_REPS):
                    conflict_draws[index] = np.mean(
                        conflict_values[
                            conflict_rng.integers(
                                0, len(conflict_values), size=len(conflict_values)
                            )
                        ]
                    )
                ratio_difference = None
                ratio_difference_ci = None
                left_generic = float(np.mean(list(generic_effects[left].values())))
                right_generic = float(np.mean(list(generic_effects[right].values())))
                if left_generic > 0 and right_generic > 0:
                    ratio_rng = np.random.default_rng(
                        BOOTSTRAP_SEED + left * 100 + right + 2
                    )
                    ratio_draws = np.empty(BOOTSTRAP_REPS)
                    for index in range(BOOTSTRAP_REPS):
                        chosen = ratio_rng.integers(0, len(ids), size=len(ids))
                        sampled_ids = [ids[item] for item in chosen]
                        left_specificity = float(
                            np.mean([contrasts[left][sid] for sid in sampled_ids])
                        )
                        right_specificity = float(
                            np.mean([contrasts[right][sid] for sid in sampled_ids])
                        )
                        left_shift = float(
                            np.mean([generic_effects[left][sid] for sid in sampled_ids])
                        )
                        right_shift = float(
                            np.mean([generic_effects[right][sid] for sid in sampled_ids])
                        )
                        if left_shift > 0 and right_shift > 0:
                            ratio_draws[index] = left_specificity / left_shift - (
                                right_specificity / right_shift
                            )
                        else:
                            ratio_draws[index] = np.nan
                    valid = ratio_draws[np.isfinite(ratio_draws)]
                    if len(valid) >= int(0.975 * BOOTSTRAP_REPS):
                        ratio_difference = (
                            results[model][str(left)][
                                "specificity_fraction_of_generic_shift"
                            ]
                            - results[model][str(right)][
                                "specificity_fraction_of_generic_shift"
                            ]
                        )
                        ratio_difference_ci = [
                            float(np.quantile(valid, 0.025)),
                            float(np.quantile(valid, 0.975)),
                        ]
                pairwise[model][f"{left}_minus_{right}"] = {
                    "post_hoc_unadjusted": True,
                    "n_sentences": len(ids),
                    "mean_specificity_difference_logits": float(np.mean(values)),
                    "sentence_bootstrap_95_ci": [
                        float(np.quantile(draws, 0.025)),
                        float(np.quantile(draws, 0.975)),
                    ],
                    "n_conflict_sentences": len(conflict),
                    "conflict_mean_difference_logits": float(np.mean(conflict_values)),
                    "conflict_sentence_bootstrap_95_ci": [
                        float(np.quantile(conflict_draws, 0.025)),
                        float(np.quantile(conflict_draws, 0.975)),
                    ],
                    "specificity_fraction_difference": ratio_difference,
                    "specificity_fraction_difference_95_ci": ratio_difference_ci,
                }
    results["post_hoc_pairwise_layer_differences"] = pairwise
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
