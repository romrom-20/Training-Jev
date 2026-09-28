"""Analyze paired 1.5B/3B decoder factorials on English laptop reviews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONDITIONS = ("aspect_only", "opinion_masked")
DECODERS = ("finite_grid", "free_greedy")
MODELS = ("qwen-1.5b", "qwen-3b")
SEED = 20260950
BOOTSTRAPS = 10_000
EXPECTED_IDS = 943


def load_rows(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_key = {
        (row["case_id"], row["condition"], row["decoder"]): row
        for row in rows
    }
    expected = EXPECTED_IDS * len(CONDITIONS) * len(DECODERS)
    if len(rows) != expected or len(by_key) != expected:
        raise ValueError(f"Expected {expected} unique model rows, found {len(rows)}")
    if {row["condition"] for row in rows} != set(CONDITIONS):
        raise ValueError("Unexpected input-condition set")
    if {row["decoder"] for row in rows} != set(DECODERS):
        raise ValueError("Unexpected decoder set")
    return by_key


def _keys(rows: dict) -> set:
    return {
        (case_id, condition, decoder)
        for case_id in {key[0] for key in rows}
        for condition in CONDITIONS
        for decoder in DECODERS
    }


def analyze(parent_15: dict, model_3b: dict, manifest: dict | None = None) -> dict:
    ids_15 = {key[0] for key in parent_15}
    ids_3b = {key[0] for key in model_3b}
    if len(ids_15) != EXPECTED_IDS or ids_3b != ids_15:
        raise ValueError("1.5B and 3B outputs must cover the same 943 source IDs")
    if set(parent_15) != _keys(parent_15) or set(model_3b) != _keys(model_3b):
        raise ValueError("Each model's factorial must have all four cells per source ID")
    for key in parent_15:
        if parent_15[key]["gold"] != model_3b[key]["gold"]:
            raise ValueError("Paired model runs disagree on a source ID's gold VA")

    invalid = {
        model: {
            decoder: {
                condition: sum(
                    rows[(case_id, condition, decoder)]["prediction"] is None
                    for case_id in ids_15
                )
                for condition in CONDITIONS
            }
            for decoder in DECODERS
        }
        for model, rows in (("qwen-1.5b", parent_15), ("qwen-3b", model_3b))
    }
    free_invalid_3b = sum(invalid["qwen-3b"]["free_greedy"].values())
    free_rate_3b = free_invalid_3b / (EXPECTED_IDS * len(CONDITIONS))
    summary = {
        "experiment": "050-laptop-model-size-factorial",
        "interpretation": "adaptive paired model-size diagnostic; exploratory",
        "n_clusters": EXPECTED_IDS,
        "n_outputs_3b": len(model_3b),
        "invalid_by_model_decoder_condition": invalid,
        "invalid_free_3b_outputs": free_invalid_3b,
        "invalid_free_3b_rate": free_rate_3b,
        "analysis_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_unit": "source sentence ID, retaining all model, decoder, condition and VA cells",
        "score_analysis_performed": False,
    }
    if manifest is not None:
        summary["run_provenance"] = {
            key: manifest[key]
            for key in (
                "protocol_sha256", "parent_049_output_sha256", "source_revision",
                "source_file", "source_sha256", "model", "model_revision", "device",
                "n_grid_candidates", "invalid_by_decoder_condition", "output_sha256",
                "generation_seconds_this_process_only",
            )
        }
    if free_rate_3b > 0.02:
        summary["status"] = "protocol_execution_failure"
        summary["reason"] = "Qwen2.5-3B free-output invalid rate exceeded 2%; 3B score contrasts withheld"
        return summary

    complete_3b = [
        case_id for case_id in sorted(ids_15)
        if all(model_3b[(case_id, condition, decoder)]["prediction"] is not None
               for condition in CONDITIONS for decoder in DECODERS)
    ]
    complete_paired = [
        case_id for case_id in complete_3b
        if all(parent_15[(case_id, condition, decoder)]["prediction"] is not None
               for condition in CONDITIONS for decoder in DECODERS)
    ]
    for rows in (parent_15, model_3b):
        for row in rows.values():
            prediction = row["prediction"]
            if prediction is not None and (
                len(prediction) != 2 or not np.isfinite(prediction).all()
            ):
                raise ValueError("Non-finite VA output reached the analyzer")

    def make_errors(rows: dict, ids: list[str]) -> dict:
        return {
            decoder: {
                condition: {
                    case_id: np.asarray(
                        [
                            (
                                float(rows[(case_id, condition, decoder)]["prediction"][dim])
                                - float(rows[(case_id, condition, decoder)]["gold"][dim])
                            ) ** 2
                            for dim in (0, 1)
                        ],
                        dtype=float,
                    )
                    for case_id in ids
                }
                for condition in CONDITIONS
            }
            for decoder in DECODERS
        }

    errors_15 = make_errors(parent_15, complete_paired)
    errors_3b_all = make_errors(model_3b, complete_3b)
    errors_3b_paired = make_errors(model_3b, complete_paired)

    def context_gain(errors: dict, decoder: str, sample: list[str]) -> float:
        aspect = np.sqrt(
            np.mean(np.concatenate([errors[decoder]["aspect_only"][case_id] for case_id in sample]))
        )
        context = np.sqrt(
            np.mean(np.concatenate([errors[decoder]["opinion_masked"][case_id] for case_id in sample]))
        )
        return float(aspect - context)

    def interaction(errors: dict, sample: list[str]) -> float:
        return context_gain(errors, "finite_grid", sample) - context_gain(errors, "free_greedy", sample)

    gains_3b = {}
    for offset, decoder in enumerate(DECODERS):
        point = context_gain(errors_3b_all, decoder, complete_3b)
        rng = np.random.default_rng(SEED + offset)
        draws = np.empty(BOOTSTRAPS)
        for draw in range(BOOTSTRAPS):
            sample = rng.choice(complete_3b, size=len(complete_3b), replace=True).tolist()
            draws[draw] = context_gain(errors_3b_all, decoder, sample)
        ci = [float(value) for value in np.quantile(draws, [0.025, 0.975])]
        gains_3b[decoder] = {
            "estimate": point,
            "ci95": ci,
            "n_clusters": len(complete_3b),
            "registered_context_gain_gate_passed": point >= 0.25 and ci[0] > 0,
        }

    primary = interaction(errors_3b_all, complete_3b)
    rng_primary = np.random.default_rng(SEED + 2)
    primary_draws = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        sample = rng_primary.choice(complete_3b, size=len(complete_3b), replace=True).tolist()
        primary_draws[draw] = interaction(errors_3b_all, sample)
    primary_ci = [float(value) for value in np.quantile(primary_draws, [0.025, 0.975])]

    model_15_point = interaction(errors_15, complete_paired)
    model_3b_point = interaction(errors_3b_paired, complete_paired)
    size_delta = model_3b_point - model_15_point
    rng_size = np.random.default_rng(SEED + 3)
    size_draws = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        sample = rng_size.choice(complete_paired, size=len(complete_paired), replace=True).tolist()
        size_draws[draw] = interaction(errors_3b_paired, sample) - interaction(errors_15, sample)
    size_ci = [float(value) for value in np.quantile(size_draws, [0.025, 0.975])]

    summary.update(
        {
            "status": "scored",
            "score_analysis_performed": True,
            "n_complete_clusters_3b": len(complete_3b),
            "n_complete_clusters_paired_size": len(complete_paired),
            "context_gain_3b": gains_3b,
            "primary_3b_interaction": {
                "contrast": "context_gain(finite_grid) - context_gain(free_greedy)",
                "estimate": primary,
                "ci95": primary_ci,
                "n_clusters": len(complete_3b),
                "registered_interaction_gate_passed": primary >= 0.25 and primary_ci[0] > 0,
            },
            "paired_size_moderation_secondary": {
                "contrast": "3B decoder interaction - 1.5B decoder interaction",
                "interaction_1_5b": model_15_point,
                "interaction_3b_same_complete_ids": model_3b_point,
                "estimate": size_delta,
                "ci95": size_ci,
                "n_clusters": len(complete_paired),
                "confirmatory": False,
            },
        }
    )
    return summary


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] == "protocol_execution_failure":
        result = (
            f"The 3B free-output invalid gate failed ({summary['invalid_free_3b_outputs']}/"
            f"1,886 invalid); all 3B score contrasts were withheld."
        )
    else:
        primary = summary["primary_3b_interaction"]
        delta = summary["paired_size_moderation_secondary"]
        result = (
            f"The Qwen2.5-3B laptop decoder interaction was {primary['estimate']:.3f} "
            f"(95% source-ID interval [{primary['ci95'][0]:.3f}, {primary['ci95'][1]:.3f}]); "
            f"its registered practical gate {'passed' if primary['registered_interaction_gate_passed'] else 'did not pass'}. "
            f"The descriptive paired difference from 1.5B was {delta['estimate']:.3f} "
            f"(95% interval [{delta['ci95'][0]:.3f}, {delta['ci95'][1]:.3f}])."
        )
    readme = f"""# Experiment 050: Qwen2.5 model-size check on laptop reviews

## Result

{result}

This repeats Experiment 049's exact English laptop IDs, target aspects, gold labels, and prompt bytes at Qwen2.5-3B. It is a paired model-size diagnostic within one public benchmark split, not an independent sample or corpus replication.

No review text, target aspects, IDs, raw generations, or item predictions are published; private outputs remain in ignored `.context/`.

- Protocol: `docs/experiments/050-laptop-model-size-factorial.md`
- Runner/analyzer: `scripts/run_laptop_model_size_050.py`, `scripts/analyze_laptop_model_size_050.py`
- Aggregate result and provenance: `summary.json`
"""
    (output / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-1-5b", type=Path, default=Path(".context/exp049-private-predictions.jsonl"))
    parser.add_argument("--predictions-3b", type=Path, default=Path(".context/exp050-private-predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(".context/exp050-run-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/laptop-model-size-v1"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else None
    summary = analyze(load_rows(args.predictions_1_5b), load_rows(args.predictions_3b), manifest)
    write_report(summary, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
