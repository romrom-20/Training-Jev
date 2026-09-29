"""Analyze whether numeric-anchor shifts depend on anchor location."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import run_anchor_location_control_075 as runner
import run_prefix_score_distribution_audit_069 as exp069
from analyze_prefix_score_distribution_audit_069 import bootstrap_mean
from run_small_model_decoder_factorial_048 import sha256

OUT = Path("results/anchor-location-control-v1")
N_BOOTSTRAPS = 10_000


def _distribution(row: dict) -> tuple[np.ndarray, float]:
    logprobs = np.asarray(row["canonical_logprobs"], dtype=np.float64)
    if logprobs.shape != (81,) or not np.isfinite(logprobs).all():
        raise ValueError("Expected 81 finite complete-score log probabilities")
    mass = float(np.exp(logprobs).sum())
    if mass <= 0 or mass > 1.0001:
        raise ValueError("Invalid probability mass on complete canonical scores")
    probabilities = np.exp(logprobs - np.max(logprobs))
    probabilities /= probabilities.sum()
    return probabilities, mass


def summarize(predictions: Path = runner.OUT, manifest_path: Path = runner.MANIFEST,
              bootstrap_replicates: int = N_BOOTSTRAPS) -> dict:
    manifest = json.loads(manifest_path.read_text())
    output_hash = sha256(predictions.read_bytes())
    if (manifest.get("experiment") != "075-anchor-location-control"
            or manifest.get("protocol_sha256") != runner.PROTOCOL_SHA256
            or manifest.get("output_sha256") != output_hash
            or manifest.get("n_total_model_contexts") != 1024):
        raise ValueError("Experiment 075 output is incomplete or hash-mismatched")
    cases, metadata = runner.select_cases(Path(".context/dimabsa"))
    jobs = runner.build_jobs(cases)
    expected = {(model, job["case_id"], job["anchor_location"], job["order"],
                 job["forced_value"]) for model in runner.MODEL_CONFIGS for job in jobs}
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    index = {}
    for row in rows:
        key = (row["model_key"], row["case_id"], row["anchor_location"], row["order"],
               float(row["forced_value"]))
        if key in index or key not in expected:
            raise ValueError(f"Unexpected or duplicate Experiment 075 row: {key}")
        config = runner.MODEL_CONFIGS[row["model_key"]]
        if row["model"] != config["model"] or row["model_revision"] != config["revision"]:
            raise ValueError(f"Unexpected model revision for cell: {key}")
        if row["target_axis"] != ("arousal" if row["order"] == "valence_first" else "valence"):
            raise ValueError(f"Unexpected target axis for cell: {key}")
        _distribution(row)
        index[key] = row
    if set(index) != expected:
        raise ValueError("Experiment 075 is missing model/recipient/condition cells")

    invalid = manifest["invalid_by_model_location_order_value"]
    invalid_rates = {
        model: {location: {order: {value: count / 64 for value, count in arms.items()}
                           for order, arms in orders.items()}
                for location, orders in locations.items()}
        for model, locations in invalid.items()
    }
    gate_pass = all(rate <= 0.02 for locations in invalid_rates.values()
                    for orders in locations.values() for arms in orders.values()
                    for rate in arms.values())
    values = exp069.VALUES
    per_location, location_contrasts, masses = {}, {}, {}
    greedy = {}
    ids = [case["case_id"] for case in cases]
    for model_index, model in enumerate(runner.MODEL_CONFIGS):
        per_location[model], location_contrasts[model], masses[model], greedy[model] = {}, {}, {}, {}
        for order_index, order in enumerate(runner.ORDERS):
            order_data = {}
            arrays = {}
            greedy_arrays = {}
            for location in ("assistant_prefix", "user_message"):
                shifts, greedy_shifts, low_masses, high_masses = [], [], [], []
                for case_id in ids:
                    low = index[(model, case_id, location, order, 2.0)]
                    high = index[(model, case_id, location, order, 8.0)]
                    p_low, mass_low = _distribution(low)
                    p_high, mass_high = _distribution(high)
                    shifts.append(float((p_high - p_low) @ values))
                    low_masses.append(mass_low)
                    high_masses.append(mass_high)
                    if low["greedy_target_score"] is None or high["greedy_target_score"] is None:
                        greedy_shifts.append(np.nan)
                    else:
                        greedy_shifts.append(float(
                            high["greedy_target_score"] - low["greedy_target_score"]))
                arrays[location] = np.asarray(shifts)
                greedy_arrays[location] = np.asarray(greedy_shifts)
                seed = 20260975 + model_index * 20 + order_index
                valid_greedy = np.isfinite(greedy_arrays[location])
                order_data[location] = {
                    "n_recipients": len(ids),
                    "expected_score_shift_8_minus_2": bootstrap_mean(
                        arrays[location], seed, bootstrap_replicates),
                    "greedy_score_shift_8_minus_2": bootstrap_mean(
                        greedy_arrays[location][valid_greedy], seed, bootstrap_replicates),
                    "n_greedy_complete": int(valid_greedy.sum()),
                    "mean_valid_canonical_mass": {
                        "if_anchor_2": float(np.mean(low_masses)),
                        "if_anchor_8": float(np.mean(high_masses)),
                    },
                }
            per_location[model][order] = order_data
            complete_greedy = (np.isfinite(greedy_arrays["user_message"])
                               & np.isfinite(greedy_arrays["assistant_prefix"]))
            location_contrasts[model][order] = {
                "user_message_minus_assistant_prefix_expected_shift": bootstrap_mean(
                    arrays["user_message"] - arrays["assistant_prefix"],
                    20260975 + model_index * 20 + order_index,
                    bootstrap_replicates),
                "user_message_minus_assistant_prefix_greedy_shift": bootstrap_mean(
                    (greedy_arrays["user_message"] - greedy_arrays["assistant_prefix"])[complete_greedy],
                    20260975 + model_index * 20 + order_index,
                    bootstrap_replicates),
                "n_paired_greedy_recipients": int(complete_greedy.sum()),
            }
    return {
        "experiment": "075-anchor-location-control",
        "protocol_sha256": runner.PROTOCOL_SHA256,
        "output_sha256": output_hash,
        "recipient_ids_sha256": metadata["recipient_ids_sha256"],
        "n_recipients": len(ids), "n_models": len(runner.MODEL_CONFIGS),
        "n_total_model_contexts": len(rows),
        "invalid_by_model_location_order_value": invalid,
        "invalid_rates_by_model_location_order_value": invalid_rates,
        "invalid_gate_passed": gate_pass,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": 20260975,
        "expected_and_greedy_shifts_by_location": per_location if gate_pass else None,
        "paired_location_contrasts": location_contrasts if gate_pass else None,
        "interpretation_limit": "The partial-assistant-prefix and user-anchor prompts use different answer schemas and text; this compares two registered formats and does not isolate a schema-free causal effect.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=runner.OUT)
    parser.add_argument("--manifest", dest="manifest_path", type=Path, default=runner.MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=N_BOOTSTRAPS)
    args = parser.parse_args()
    summary = summarize(args.predictions, args.manifest_path, args.bootstrap_replicates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
