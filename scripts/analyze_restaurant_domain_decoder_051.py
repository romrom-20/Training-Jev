"""Analyze the 3B decoder factorial and laptop-domain moderation contrast."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_laptop_model_size_050 import load_rows as load_laptop_rows

CONDITIONS = ("aspect_only", "opinion_masked")
DECODERS = ("finite_grid", "free_greedy")
SEED = 20260951
BOOTSTRAPS = 10_000
EXPECTED_RESTAURANT_IDS = 963
EXPECTED_LAPTOP_IDS = 943


def load_restaurant_rows(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_key = {
        (row["case_id"], row["condition"], row["decoder"]): row
        for row in rows
    }
    expected = EXPECTED_RESTAURANT_IDS * len(CONDITIONS) * len(DECODERS)
    if len(rows) != expected or len(by_key) != expected:
        raise ValueError(f"Expected {expected} unique restaurant outputs, found {len(rows)}")
    if {row["condition"] for row in rows} != set(CONDITIONS):
        raise ValueError("Unexpected restaurant condition set")
    if {row["decoder"] for row in rows} != set(DECODERS):
        raise ValueError("Unexpected restaurant decoder set")
    return by_key


def _expected_keys(rows: dict) -> set:
    return {
        (case_id, condition, decoder)
        for case_id in {key[0] for key in rows}
        for condition in CONDITIONS
        for decoder in DECODERS
    }


def _invalid_counts(rows: dict) -> dict:
    return {
        decoder: {
            condition: sum(
                row["prediction"] is None
                for row in rows.values()
                if row["decoder"] == decoder and row["condition"] == condition
            )
            for condition in CONDITIONS
        }
        for decoder in DECODERS
    }


def _errors(rows: dict, ids: list[str]) -> dict:
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


def _context_gain(errors: dict, decoder: str, ids: list[str]) -> float:
    aspect = np.sqrt(
        np.mean(np.concatenate([errors[decoder]["aspect_only"][case_id] for case_id in ids]))
    )
    masked = np.sqrt(
        np.mean(np.concatenate([errors[decoder]["opinion_masked"][case_id] for case_id in ids]))
    )
    return float(aspect - masked)


def _interaction(errors: dict, ids: list[str]) -> float:
    return _context_gain(errors, "finite_grid", ids) - _context_gain(errors, "free_greedy", ids)


def analyze(restaurant: dict, laptop: dict, manifest: dict | None = None) -> dict:
    restaurant_ids = sorted({key[0] for key in restaurant})
    laptop_ids = sorted({key[0] for key in laptop})
    if len(restaurant_ids) != EXPECTED_RESTAURANT_IDS or set(restaurant) != _expected_keys(restaurant):
        raise ValueError("Restaurant run must contain all four cells for 963 source IDs")
    if len(laptop_ids) != EXPECTED_LAPTOP_IDS or set(laptop) != _expected_keys(laptop):
        raise ValueError("Laptop parent run must contain all four cells for 943 source IDs")

    for rows, domain in ((restaurant, "restaurant"), (laptop, "laptop")):
        for row in rows.values():
            prediction = row["prediction"]
            if prediction is not None and (
                len(prediction) != 2 or not np.isfinite(prediction).all()
            ):
                raise ValueError(f"Non-finite VA prediction in {domain} run")

    invalid_restaurant = _invalid_counts(restaurant)
    invalid_laptop = _invalid_counts(laptop)
    invalid_free = sum(invalid_restaurant["free_greedy"].values())
    invalid_rate = invalid_free / (EXPECTED_RESTAURANT_IDS * len(CONDITIONS))
    summary = {
        "experiment": "051-restaurant-domain-decoder-transfer-3b",
        "interpretation": "adaptive same-language domain transfer; exploratory",
        "n_restaurant_clusters": len(restaurant_ids),
        "n_laptop_clusters": len(laptop_ids),
        "n_restaurant_outputs": len(restaurant),
        "invalid_by_decoder_condition": invalid_restaurant,
        "invalid_laptop_by_decoder_condition": invalid_laptop,
        "invalid_free_outputs": invalid_free,
        "invalid_free_rate": invalid_rate,
        "analysis_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_unit": "source sentence ID, retaining evidence, decoders and both VA dimensions",
        "score_analysis_performed": False,
    }
    if manifest is not None:
        summary["run_provenance"] = {
            key: manifest[key]
            for key in (
                "protocol_sha256", "source_revision", "source_file", "source_sha256",
                "selected_valence_buckets", "model", "model_revision", "device",
                "n_grid_candidates", "invalid_by_decoder_condition", "output_sha256",
                "generation_seconds_this_process_only",
            )
        }
    if invalid_rate > 0.02:
        summary["status"] = "protocol_execution_failure"
        summary["reason"] = "free-output invalid rate exceeded 2%; all restaurant score contrasts withheld"
        return summary

    restaurant_complete = [
        case_id for case_id in restaurant_ids
        if all(restaurant[(case_id, condition, decoder)]["prediction"] is not None
               for condition in CONDITIONS for decoder in DECODERS)
    ]
    laptop_complete = [
        case_id for case_id in laptop_ids
        if all(laptop[(case_id, condition, decoder)]["prediction"] is not None
               for condition in CONDITIONS for decoder in DECODERS)
    ]
    if not restaurant_complete or not laptop_complete:
        raise ValueError("No complete source clusters remain in a domain")

    err_restaurant = _errors(restaurant, restaurant_complete)
    err_laptop = _errors(laptop, laptop_complete)
    gains = {}
    for offset, decoder in enumerate(DECODERS):
        estimate = _context_gain(err_restaurant, decoder, restaurant_complete)
        rng = np.random.default_rng(SEED + offset)
        draws = np.empty(BOOTSTRAPS)
        for draw in range(BOOTSTRAPS):
            sample = rng.choice(restaurant_complete, size=len(restaurant_complete), replace=True).tolist()
            draws[draw] = _context_gain(err_restaurant, decoder, sample)
        ci = [float(value) for value in np.quantile(draws, [0.025, 0.975])]
        gains[decoder] = {
            "estimate": estimate,
            "ci95": ci,
            "n_clusters": len(restaurant_complete),
            "registered_context_gain_gate_passed": estimate >= 0.25 and ci[0] > 0,
        }

    primary = _interaction(err_restaurant, restaurant_complete)
    rng_primary = np.random.default_rng(SEED + 2)
    primary_draws = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        sample = rng_primary.choice(
            restaurant_complete, size=len(restaurant_complete), replace=True
        ).tolist()
        primary_draws[draw] = _interaction(err_restaurant, sample)
    primary_ci = [float(value) for value in np.quantile(primary_draws, [0.025, 0.975])]

    restaurant_interaction = primary
    laptop_interaction = _interaction(err_laptop, laptop_complete)
    domain_delta = laptop_interaction - restaurant_interaction
    rng_laptop = np.random.default_rng(SEED + 3)
    rng_restaurant = np.random.default_rng(SEED + 4)
    delta_draws = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        laptop_sample = rng_laptop.choice(
            laptop_complete, size=len(laptop_complete), replace=True
        ).tolist()
        restaurant_sample = rng_restaurant.choice(
            restaurant_complete, size=len(restaurant_complete), replace=True
        ).tolist()
        delta_draws[draw] = _interaction(err_laptop, laptop_sample) - _interaction(
            err_restaurant, restaurant_sample
        )
    delta_ci = [float(value) for value in np.quantile(delta_draws, [0.025, 0.975])]

    summary.update(
        {
            "status": "scored",
            "score_analysis_performed": True,
            "n_complete_restaurant_clusters": len(restaurant_complete),
            "n_complete_laptop_clusters": len(laptop_complete),
            "context_gain_by_decoder": gains,
            "primary_decoder_interaction": {
                "contrast": "context_gain(finite_grid) - context_gain(free_greedy)",
                "estimate": primary,
                "ci95": primary_ci,
                "n_clusters": len(restaurant_complete),
                "registered_interaction_gate_passed": primary >= 0.25 and primary_ci[0] > 0,
            },
            "domain_moderation_secondary": {
                "contrast": "laptop interaction - restaurant interaction",
                "laptop_interaction": laptop_interaction,
                "restaurant_interaction": restaurant_interaction,
                "estimate": domain_delta,
                "ci95": delta_ci,
                "n_laptop_clusters": len(laptop_complete),
                "n_restaurant_clusters": len(restaurant_complete),
                "confirmatory": False,
            },
        }
    )
    return summary


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] == "protocol_execution_failure":
        sentence = (
            f"The free-output invalid gate failed ({summary['invalid_free_outputs']}/"
            f"1,926 invalid); all restaurant score contrasts were withheld."
        )
    else:
        finite = summary["context_gain_by_decoder"]["finite_grid"]
        free = summary["context_gain_by_decoder"]["free_greedy"]
        primary = summary["primary_decoder_interaction"]
        secondary = summary["domain_moderation_secondary"]
        sentence = (
            f"On English restaurant reviews, the finite-grid context gain was {finite['estimate']:.3f} "
            f"(95% source-ID interval [{finite['ci95'][0]:.3f}, {finite['ci95'][1]:.3f}]) and the "
            f"free-greedy gain was {free['estimate']:.3f} (95% interval "
            f"[{free['ci95'][0]:.3f}, {free['ci95'][1]:.3f}]). The registered interaction was "
            f"{primary['estimate']:.3f} (95% interval [{primary['ci95'][0]:.3f}, "
            f"{primary['ci95'][1]:.3f}]); its practical gate "
            f"{'passed' if primary['registered_interaction_gate_passed'] else 'did not pass'}. "
            f"The descriptive laptop-minus-restaurant interaction difference was "
            f"{secondary['estimate']:.3f} (95% independent-domain bootstrap interval "
            f"[{secondary['ci95'][0]:.3f}, {secondary['ci95'][1]:.3f}])."
        )
    readme = f"""# Experiment 051: Qwen2.5-3B restaurant-domain transfer

## Result

{sentence}

This uses the full eligible English restaurant test split with the same 3B model, prompts and decoders as the English laptop run. It is a same-language product-domain transfer within one public DimABSA release, not an independent corpus replication.

No review text, target aspects, IDs, raw generations, or item predictions are published; private outputs remain in ignored `.context/`.

- Protocol: `docs/experiments/051-restaurant-domain-decoder-transfer-3b.md`
- Runner/analyzer: `scripts/run_restaurant_domain_decoder_051.py`, `scripts/analyze_restaurant_domain_decoder_051.py`
- Aggregate result and provenance: `summary.json`
"""
    (output / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restaurant-predictions", type=Path, default=Path(".context/exp051-private-predictions.jsonl"))
    parser.add_argument("--laptop-predictions", type=Path, default=Path(".context/exp050-private-predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(".context/exp051-run-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/restaurant-domain-decoder-3b-v1"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else None
    summary = analyze(load_restaurant_rows(args.restaurant_predictions), load_laptop_rows(args.laptop_predictions), manifest)
    write_report(summary, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
