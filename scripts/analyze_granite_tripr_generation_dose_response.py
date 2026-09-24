"""Analyze paired answer utility under Granite residual dose escalation."""

import argparse
import json
from pathlib import Path

import numpy as np
from granite_tripr_generation_dose_response import BOOTSTRAP_SEED, DOSES
from granite_tripr_residual_transfer import CONTROL_SEEDS, MODEL

from latent_decisions.experiment import write_json

BOOTSTRAP_REPS = 10_000


def answer_label(row):
    if not row["valid_exact_one_word"]:
        return None
    return row["label"] if row["strict_correct"] else 1 - row["label"]


def summarize(root):
    folder = root / "evaluation"
    manifest = json.loads((folder / "manifest.json").read_text())
    outcomes = json.loads((folder / "outcomes.json").read_text())
    baseline_rows = json.loads((folder / "baseline.json").read_text())
    baseline = {row["id"]: row for row in baseline_rows}
    base_correct = {sid: row["strict_correct"] for sid, row in baseline.items()}
    sentences = sorted({row["sentence_id"] for row in baseline_rows})
    result_by_dose = {}
    for dose in DOSES:
        dose_rows = [row for row in outcomes if row["dose_fraction"] == dose]
        trained = {row["id"]: row for row in dose_rows if row["condition"] == "trained"}
        random_rows = [row for row in dose_rows if row["condition"] == "random"]
        if len(trained) != 63 or len(random_rows) != 63 * len(CONTROL_SEEDS):
            raise ValueError(f"Incomplete generation outcomes at dose {dose}")

        trained_utility = {
            sid: float(
                np.mean(
                    [trained[row["id"]]["strict_correct"] - base_correct[row["id"]]
                     for row in baseline_rows if row["sentence_id"] == sid]
                )
            )
            for sid in sentences
        }
        random_by_seed = {}
        for seed in CONTROL_SEEDS:
            rows_by_id = {
                row["id"]: row for row in random_rows if row["seed"] == seed
            }
            random_by_seed[seed] = {
                sid: float(
                    np.mean(
                        [rows_by_id[row["id"]]["strict_correct"] - base_correct[row["id"]]
                         for row in baseline_rows if row["sentence_id"] == sid]
                    )
                )
                for sid in sentences
            }
        trained_mean = float(np.mean(list(trained_utility.values())))
        random_means = {
            seed: float(np.mean(list(values.values())))
            for seed, values in random_by_seed.items()
        }
        random_mean = float(np.mean(list(random_means.values())))
        rank_p = (1 + sum(value >= trained_mean for value in random_means.values())) / (
            len(random_means) + 1
        )

        rng = np.random.default_rng(BOOTSTRAP_SEED + int(dose * 100))
        draws = np.empty(BOOTSTRAP_REPS)
        seed_list = list(CONTROL_SEEDS)
        for index in range(BOOTSTRAP_REPS):
            sentence_sample = [sentences[i] for i in rng.integers(0, len(sentences), len(sentences))]
            seed_sample = [seed_list[i] for i in rng.integers(0, len(seed_list), len(seed_list))]
            trained_value = float(np.mean([trained_utility[sid] for sid in sentence_sample]))
            control_value = float(
                np.mean(
                    [random_by_seed[seed][sid] for seed in seed_sample for sid in sentence_sample]
                )
            )
            draws[index] = trained_value - control_value
        ci = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]

        gains = harms = flips = 0
        for sid, row in trained.items():
            prior = baseline[sid]
            gains += int(not prior["strict_correct"] and row["strict_correct"])
            harms += int(prior["strict_correct"] and not row["strict_correct"])
            flips += int(answer_label(prior) != answer_label(row))
        validity = float(np.mean([row["valid_exact_one_word"] for row in trained.values()]))
        gate = ci[0] > 0 and rank_p <= 0.05 and validity >= 0.95
        result_by_dose[str(dose)] = {
            "trained_mean_utility_change": trained_mean,
            "random_seed_mean_utility_change": random_mean,
            "trained_minus_random_utility": trained_mean - random_mean,
            "nested_sentence_seed_bootstrap_95_ci": ci,
            "random_seed_monte_carlo_one_sided_p": rank_p,
            "random_seed_utility_by_seed": {str(k): v for k, v in random_means.items()},
            "trained_generated_strict_accuracy": float(
                np.mean([row["strict_correct"] for row in trained.values()])
            ),
            "trained_exact_one_word_rate": validity,
            "trained_label_flips_vs_baseline": flips,
            "trained_accuracy_gains": gains,
            "trained_accuracy_harms": harms,
            "behavioral_gate": gate,
        }

    result = {
        "model": MODEL,
        "layer": manifest["layer"],
        "n_sentences": len(sentences),
        "n_queries": len(baseline_rows),
        "primary_dose_fraction": "0.4",
        "dose_results": result_by_dose,
        "behavioral_gate_passed": result_by_dose["0.4"]["behavioral_gate"],
        "baseline_generated_accuracy": float(
            np.mean([row["strict_correct"] for row in baseline_rows])
        ),
    }
    write_json(folder / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/granite-tripr-generation-dose-response-v1"))
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2), flush=True)


if __name__ == "__main__":
    main()
