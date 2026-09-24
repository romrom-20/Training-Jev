"""Sentence-bootstrap dose curves for experiment 016."""

import argparse
import json
from pathlib import Path

import numpy as np
from mams_aspect_selectivity import MODELS
from mams_dose_response import DOSES, dose_name
from natural_aspect_selectivity import sentence_contrasts

from latent_decisions.experiment import write_json

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20260924


def dose_records(root, model, fraction):
    folder = root / dose_name(fraction) / model
    baseline = json.loads((folder / "baseline.json").read_text())
    effects = json.loads((folder / "effects.json").read_text())
    return baseline, effects


def analyze_model(root, model):
    per_dose = {}
    caches = {}
    for fraction in DOSES:
        baseline, effects = dose_records(root, model, fraction)
        native = sentence_contrasts(effects, "native")
        random = sentence_contrasts(effects, "random")
        common = sorted(set(native) & set(random))
        specificity = {sentence: native[sentence] - random[sentence] for sentence in common}
        generic = {}
        for sentence in common:
            rows = [
                row
                for row in effects
                if row["sentence_id"] == sentence and row["condition"] == "native"
            ]
            generic[sentence] = float(np.mean([row["margin_delta"] for row in rows]))
        caches[fraction] = {"specificity": specificity, "generic": generic, "groups": common}
        per_dose[f"{fraction:.4f}"] = {
            "n_sentences": len(common),
            "baseline_accuracy": float(np.mean([row["strict_correct"] for row in baseline])),
            "baseline_valid_label_rate": float(
                np.mean([row["valid_polarity_token"] for row in baseline])
            ),
            "steered_accuracy_by_condition": {
                condition: float(
                    np.mean(
                        [
                            row["steered_strict_correct"]
                            for row in effects
                            if row["condition"] == condition
                        ]
                    )
                )
                for condition in ("native", "shared", "random")
            },
        }

    ids = sorted(set.intersection(*(set(caches[d]["groups"]) for d in DOSES)))
    for fraction in DOSES:
        caches[fraction]["groups"] = ids
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    slope_draws, ratio_draws = {d: [] for d in DOSES}, {d: [] for d in DOSES}
    specific_draws, generic_draws = {d: [] for d in DOSES}, {d: [] for d in DOSES}
    x = np.log2(np.asarray(DOSES, dtype=float))
    for _ in range(BOOTSTRAP_REPS):
        chosen = rng.choice(ids, size=len(ids), replace=True)
        ratios = []
        for fraction in DOSES:
            s = caches[fraction]["specificity"]
            g = caches[fraction]["generic"]
            specificity = float(np.mean([s[sid] for sid in chosen]))
            generic = float(np.mean([g[sid] for sid in chosen]))
            ratio = specificity / generic if generic else float("nan")
            specific_draws[fraction].append(specificity)
            generic_draws[fraction].append(generic)
            ratio_draws[fraction].append(ratio)
            ratios.append(ratio)
        slope_draws[DOSES[0]].append(float(np.polyfit(x, ratios, 1)[0]))
    for fraction in DOSES:
        s = caches[fraction]["specificity"]
        g = caches[fraction]["generic"]
        mean_s = float(np.mean([s[sid] for sid in ids]))
        mean_g = float(np.mean([g[sid] for sid in ids]))
        ratios = np.asarray(ratio_draws[fraction])
        per_dose[f"{fraction:.4f}"].update(
            {
                "generic_native_shift_mean_logits": mean_g,
                "native_minus_random_specificity_mean_logits": mean_s,
                "specificity_95_ci": [
                    float(np.quantile(specific_draws[fraction], 0.025)),
                    float(np.quantile(specific_draws[fraction], 0.975)),
                ],
                "specificity_fraction_of_generic_shift": mean_s / mean_g,
                "ratio_95_ci": [
                    float(np.quantile(ratios, 0.025)),
                    float(np.quantile(ratios, 0.975)),
                ],
                "practical_specificity_gate_at_dose": bool(
                    mean_s / mean_g >= 0.05
                    and per_dose[f"{fraction:.4f}"]["baseline_accuracy"] > 0.5
                ),
            }
        )
    slope_values = slope_draws[DOSES[0]]
    slope_point = float(
        np.polyfit(
            x,
            [per_dose[f"{dose:.4f}"]["specificity_fraction_of_generic_shift"] for dose in DOSES],
            1,
        )[0]
    )
    return {
        "model": model,
        "doses": per_dose,
        "ratio_vs_log2_dose_slope": {
            "point_estimate": slope_point,
            "sentence_bootstrap_95_ci": [
                float(np.quantile(slope_values, 0.025)),
                float(np.quantile(slope_values, 0.975)),
            ],
            "trend_gate_pass": bool(np.quantile(slope_values, 0.025) > 0),
        },
        "any_practical_dose_pass": any(
            row["practical_specificity_gate_at_dose"] for row in per_dose.values()
        ),
    }


def analyze(root, models=MODELS):
    results = {model: analyze_model(root, model) for model in models}
    results["cross_model_positive_ratio_trend_pass"] = all(
        results[model]["ratio_vs_log2_dose_slope"]["trend_gate_pass"] for model in models
    )
    results["cross_model_any_practical_dose_pass"] = all(
        results[model]["any_practical_dose_pass"] for model in models
    )
    write_json(root / "analysis.json", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/mams-dose-response-v1"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.models), indent=2), flush=True)


if __name__ == "__main__":
    main()
