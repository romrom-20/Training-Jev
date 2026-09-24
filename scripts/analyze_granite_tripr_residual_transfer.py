"""Analyze Granite score, random-control, and greedy-generation outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np
from granite_tripr_residual_transfer import CONDITIONS, CONTROL_SEEDS, MODEL
from natural_aspect_selectivity import sentence_contrasts

from latent_decisions.experiment import write_json

BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261003


def contrast_by_sentence(rows):
    result = {}
    grouped = {}
    for row in rows:
        grouped.setdefault(row["sentence_id"], []).append(row)
    for sid, records in grouped.items():
        diagonal, off_diagonal = [], []
        for source in range(3):
            source_rows = [row for row in records if row["source"] == source]
            match = [row["margin_delta"] for row in source_rows if row["target"] == source]
            other = [row["margin_delta"] for row in source_rows if row["target"] != source]
            if match and other:
                diagonal.extend(match)
                off_diagonal.extend(other)
        if diagonal and off_diagonal:
            result[sid] = float(np.mean(diagonal) - np.mean(off_diagonal))
    return result


def analyze(root):
    folder = root / "evaluation"
    manifest = json.loads((folder / "manifest.json").read_text())
    baseline = json.loads((folder / "baseline.json").read_text())
    effects = json.loads((folder / "effects.json").read_text())
    random_effects = json.loads((folder / "random_seed_effects.json").read_text())
    generations = json.loads((folder / "generation.json").read_text())
    all_conflict_ids = {
        sid for sid in {row["sentence_id"] for row in baseline}
        if len({row["label"] for row in baseline if row["sentence_id"] == sid}) > 1
    }
    baseline_accuracy = float(np.mean([row["strict_correct"] for row in baseline]))
    conflict_baseline = [row for row in baseline if row["sentence_id"] in all_conflict_ids]
    conflict_accuracy = float(np.mean([row["strict_correct"] for row in conflict_baseline]))
    all_candidate_pair_accuracy = float(
        np.mean([(row["margin"] > 0) == bool(row["label"]) for row in baseline])
    )
    conflict_candidate_pair_accuracy = float(
        np.mean([(row["margin"] > 0) == bool(row["label"]) for row in conflict_baseline])
    )

    contrasts = {
        condition: sentence_contrasts(effects, condition) for condition in CONDITIONS
    }
    full_specificity = {}
    for condition in CONDITIONS:
        values = contrasts[condition]
        ids = sorted(values)
        rng = np.random.default_rng(BOOTSTRAP_SEED + CONDITIONS.index(condition))
        draws = np.empty(BOOTSTRAP_REPS)
        arr = np.asarray([values[sid] for sid in ids])
        for index in range(BOOTSTRAP_REPS):
            draws[index] = np.mean(arr[rng.integers(0, len(arr), size=len(arr))])
        full_specificity[condition] = {
            "mean_logits": float(np.mean(arr)),
            "sentence_bootstrap_95_ci": [
                float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
            ],
            "n_sentences": len(ids),
        }

    trained_rows = [row for row in effects if row["condition"] == "residual"]
    trained = contrast_by_sentence(
        [row for row in trained_rows if row["sentence_id"] in all_conflict_ids]
    )
    random_by_seed = {
        seed: contrast_by_sentence(
            [row for row in random_effects if row["seed"] == seed]
        )
        for seed in CONTROL_SEEDS
    }
    ids = sorted(set(trained).intersection(*(set(values) for values in random_by_seed.values())))
    seed_means = {
        seed: float(np.mean([values[sid] for sid in ids]))
        for seed, values in random_by_seed.items()
    }
    trained_mean = float(np.mean([trained[sid] for sid in ids]))
    random_mean = float(np.mean(list(seed_means.values())))
    monte_carlo_p = (1 + sum(value >= trained_mean for value in seed_means.values())) / (
        len(seed_means) + 1
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    nested = np.empty(BOOTSTRAP_REPS)
    seed_list = list(CONTROL_SEEDS)
    for index in range(BOOTSTRAP_REPS):
        sentence_sample = [ids[i] for i in rng.integers(0, len(ids), size=len(ids))]
        seed_sample = [seed_list[i] for i in rng.integers(0, len(seed_list), size=len(seed_list))]
        actual = float(np.mean([trained[sid] for sid in sentence_sample]))
        control = float(
            np.mean([random_by_seed[seed][sid] for seed in seed_sample for sid in sentence_sample])
        )
        nested[index] = actual - control
    nested_ci = [float(np.quantile(nested, 0.025)), float(np.quantile(nested, 0.975))]
    score_gate = nested_ci[0] > 0 and monte_carlo_p <= 0.05 and conflict_accuracy > 0.5

    gen_summary = {}
    for condition in ("baseline", "trained_residual", "random_residual"):
        rows = [row for row in generations if row["condition"] == condition]
        gen_summary[condition] = {
            "n": len(rows),
            "exact_one_word_rate": float(np.mean([row["valid_exact_one_word"] for row in rows])),
            "strict_gold_accuracy": float(np.mean([row["strict_correct"] for row in rows])),
        }

    generation_by_condition = {
        condition: {row["id"]: row for row in generations if row["condition"] == condition}
        for condition in ("baseline", "trained_residual", "random_residual")
    }
    labels = {row["id"]: row["label"] for row in generation_by_condition["baseline"].values()}

    def generated_label(row):
        if not row["valid_exact_one_word"]:
            return None
        return labels[row["id"]] if row["strict_correct"] else 1 - labels[row["id"]]

    generation_comparisons = {}
    for condition in ("trained_residual", "random_residual"):
        paired = [
            (generation_by_condition["baseline"][stimulus_id], row)
            for stimulus_id, row in generation_by_condition[condition].items()
        ]
        flips = [
            (baseline_row, treatment_row)
            for baseline_row, treatment_row in paired
            if generated_label(baseline_row) != generated_label(treatment_row)
        ]
        gains = [
            (baseline_row, treatment_row)
            for baseline_row, treatment_row in paired
            if not baseline_row["strict_correct"] and treatment_row["strict_correct"]
        ]
        harms = [
            (baseline_row, treatment_row)
            for baseline_row, treatment_row in paired
            if baseline_row["strict_correct"] and not treatment_row["strict_correct"]
        ]
        generation_comparisons[condition] = {
            "paired_n": len(paired),
            "label_flips": len(flips),
            "accuracy_gains": len(gains),
            "accuracy_harms": len(harms),
        }

    result = {
        "model": MODEL,
        "layer": manifest["layer"],
        "relative_depth": manifest["relative_depth"],
        "n_sentences": manifest["n_sentences"],
        "n_conflict_sentences": manifest["n_conflict_sentences"],
        "n_conflict_queries": manifest["n_conflict_queries"],
        "baseline_strict_accuracy_all": baseline_accuracy,
        "baseline_strict_accuracy_conflicts": conflict_accuracy,
        "baseline_candidate_pair_accuracy_all": all_candidate_pair_accuracy,
        "baseline_candidate_pair_accuracy_conflicts": conflict_candidate_pair_accuracy,
        "score_specificity_by_condition": full_specificity,
        "conflict_trained_residual_specificity_mean": trained_mean,
        "conflict_random_seed_specificity_mean": random_mean,
        "conflict_trained_minus_random_mean": trained_mean - random_mean,
        "conflict_nested_sentence_seed_bootstrap_95_ci": nested_ci,
        "conflict_monte_carlo_one_sided_p": monte_carlo_p,
        "random_seed_specificity_range": [min(seed_means.values()), max(seed_means.values())],
        "random_seed_specificity_by_seed": {str(seed): value for seed, value in seed_means.items()},
        "score_gate": score_gate,
        "generated_answer_behavior": gen_summary,
        "generated_answer_paired_comparisons": generation_comparisons,
        "generation_tested_target_matched_directions_only": True,
    }
    write_json(folder / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/granite-tripr-residual-transfer-v1"))
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), indent=2), flush=True)


if __name__ == "__main__":
    main()
