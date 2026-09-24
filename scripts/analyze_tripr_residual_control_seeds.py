"""Compare the learned aspect residual with a seeded random-residual ensemble."""

import argparse
import json
from pathlib import Path

import numpy as np
from tripr_residual_control_seeds import MODELS, SEEDS

from latent_decisions.experiment import write_json

BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261003


def sentence_contrast(rows):
    by_sentence = {}
    for row in rows:
        by_sentence.setdefault(row["sentence_id"], []).append(row)
    values = {}
    for sid, group in by_sentence.items():
        diagonal, off = [], []
        for source in range(3):
            source_rows = [row for row in group if row["source"] == source]
            match = [row["margin_delta"] for row in source_rows if row["target"] == source]
            other = [row["margin_delta"] for row in source_rows if row["target"] != source]
            if match and other:
                diagonal.extend(match)
                off.extend(other)
        if diagonal and off:
            values[sid] = float(np.mean(diagonal) - np.mean(off))
    return values


def analyze_model(root, model):
    folder = root / model
    baseline = json.loads((folder / "baseline.json").read_text())
    effects = json.loads((folder / "effects.json").read_text())
    trained_rows = [row for row in effects if row["seed"] is None]
    trained = sentence_contrast(trained_rows)
    random = {
        seed: sentence_contrast([row for row in effects if row["seed"] == seed])
        for seed in SEEDS
    }
    ids = sorted(set(trained).intersection(*(set(value) for value in random.values())))
    seed_means = {
        seed: float(np.mean([values[sid] for sid in ids])) for seed, values in random.items()
    }
    trained_mean = float(np.mean([trained[sid] for sid in ids]))
    control_mean = float(np.mean(list(seed_means.values())))
    empirical_p = (1 + sum(value >= trained_mean for value in seed_means.values())) / (
        len(seed_means) + 1
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPS)
    seed_list = list(SEEDS)
    for index in range(BOOTSTRAP_REPS):
        chosen_sentences = [ids[i] for i in rng.integers(0, len(ids), size=len(ids))]
        chosen_seeds = [seed_list[i] for i in rng.integers(0, len(seed_list), size=len(seed_list))]
        observed = float(np.mean([trained[sid] for sid in chosen_sentences]))
        controls = [
            random[seed][sid] for seed in chosen_seeds for sid in chosen_sentences
        ]
        draws[index] = observed - float(np.mean(controls))
    ci = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
    accuracy = float(np.mean([row["strict_correct"] for row in baseline]))
    control_values = list(seed_means.values())
    return {
        "n_conflict_sentences": len(ids),
        "n_conflict_queries": len(baseline),
        "baseline_strict_accuracy": accuracy,
        "trained_residual_specificity_mean_logits": trained_mean,
        "random_seed_specificity_mean_logits": control_mean,
        "trained_minus_random_seed_mean": trained_mean - control_mean,
        "nested_sentence_seed_bootstrap_95_ci": ci,
        "monte_carlo_one_sided_p": empirical_p,
        "random_seed_specificity_range": [min(control_values), max(control_values)],
        "random_seed_specificity_median": float(np.median(control_values)),
        "random_seed_specificity_by_seed": {
            str(seed): value for seed, value in seed_means.items()
        },
        "control_robustness_gate": ci[0] > 0 and empirical_p <= 0.05 and accuracy > 0.5,
    }


def analyze(root, models=MODELS):
    result = {model: analyze_model(root, model) for model in models}
    result["cross_model_control_robustness_gate"] = all(
        result[model]["control_robustness_gate"] for model in models
    )
    write_json(root / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/tripr-residual-control-seeds-v1"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.models), indent=2), flush=True)


if __name__ == "__main__":
    main()
