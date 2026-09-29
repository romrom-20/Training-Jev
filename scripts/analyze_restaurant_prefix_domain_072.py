"""Analyze restaurant prefix coupling and describe domain contrasts with Experiment 072."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import run_cross_family_prefix_coupling_071 as laptop_runner
import run_prefix_score_distribution_audit_069 as exp069
import run_restaurant_prefix_domain_072 as runner
from analyze_prefix_score_distribution_audit_069 import bootstrap_mean, validate_distribution
from run_small_model_decoder_factorial_048 import sha256

OUT = Path("results/restaurant-prefix-domain-v1")
N_BOOTSTRAPS = 10_000


def independent_difference(restaurants: np.ndarray, laptops: np.ndarray,
                           seed: int, replicates: int) -> dict:
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        draws[i] = (restaurants[rng.integers(0, len(restaurants), len(restaurants))].mean()
                    - laptops[rng.integers(0, len(laptops), len(laptops))].mean())
    estimate = float(restaurants.mean() - laptops.mean())
    return {
        "restaurant_minus_laptop": estimate,
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "n_restaurant": int(len(restaurants)), "n_laptop": int(len(laptops)),
    }


def paired_effect(low: dict, high: dict, values: np.ndarray) -> dict:
    p_low = validate_distribution(low)
    p_high = validate_distribution(high)
    return {
        "expected_shift": float((p_high - p_low) @ values),
        "greedy_shift": None if low["greedy_second_score"] is None
        or high["greedy_second_score"] is None
        else float(high["greedy_second_score"] - low["greedy_second_score"]),
        "total_variation": float(0.5 * np.abs(p_high - p_low).sum()),
        "wasserstein1": float(np.abs(np.cumsum(p_high - p_low)).sum() * 0.1),
        "grid_mass_low": float(np.exp(np.asarray(low["score_logprobs"])).sum()),
        "grid_mass_high": float(np.exp(np.asarray(high["score_logprobs"])).sum()),
    }


def summarize(predictions: Path = runner.OUT, manifest_path: Path = runner.MANIFEST,
              bootstrap_replicates: int = N_BOOTSTRAPS) -> dict:
    manifest = json.loads(manifest_path.read_text())
    output_hash = sha256(predictions.read_bytes())
    if manifest.get("experiment") != "072-restaurant-prefix-domain-replication":
        raise ValueError("Unexpected Experiment 072 manifest")
    if manifest.get("protocol_sha256") != runner.PROTOCOL_SHA256:
        raise ValueError("Run manifest does not match the frozen protocol")
    if manifest.get("output_sha256") != output_hash or manifest.get("n_total_model_contexts") != 768:
        raise ValueError("Experiment 072 raw outputs are incomplete or hash-mismatched")
    cases, sample_stats = runner.select_cases(Path(".context/dimabsa"))
    jobs = runner.build_jobs(cases)
    case_ids = [case["case_id"] for case in cases]
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    keys = [(row["model_key"], row["case_id"], row["order"],
             float(row["forced_first_value"])) for row in rows]
    expected = {(model_key, job["case_id"], job["order"], job["forced_first_value"])
                for model_key in runner.MODEL_CONFIGS for job in jobs}
    if len(rows) != 768 or len(set(keys)) != 768 or set(keys) != expected:
        raise ValueError("Experiment 072 must have every model × recipient × prefix cell once")
    index = {}
    prompt_hashes = {}
    for row in rows:
        key = (row["model_key"], row["case_id"], row["order"],
               float(row["forced_first_value"]))
        expected_model = runner.MODEL_CONFIGS[row["model_key"]]
        if row["model"] != expected_model["model"] or row["model_revision"] != expected_model["revision"]:
            raise ValueError(f"Unexpected model revision in row: {key}")
        validate_distribution(row)
        if row["user_prompt_sha256"] != next(
            job["user_prompt_sha256"] for job in jobs if job["case_id"] == row["case_id"]
        ):
            raise ValueError(f"User prompt body mismatch: {key}")
        pkey = (row["case_id"], row["order"], float(row["forced_first_value"]))
        prompt_hashes.setdefault(pkey, set()).add(row["user_prompt_sha256"])
        axis = 0 if row["order"] == "valence_first" else 1
        if row["greedy_prediction"] is not None and float(row["greedy_prediction"][axis]) != float(row["forced_first_value"]):
            raise ValueError(f"Forced first coordinate mismatch: {key}")
        index[key] = row
    if any(len(value) != 1 for value in prompt_hashes.values()):
        raise ValueError("Models were not given matching system/user message bodies")

    invalid = manifest["invalid_by_model_order_forced_value"]
    invalid_rates = {
        model: {order: {value: count / 64 for value, count in arms.items()}
                for order, arms in orders.items()}
        for model, orders in invalid.items()
    }
    gate_pass = all(rate <= 0.02 for orders in invalid_rates.values()
                    for arms in orders.values() for rate in arms.values())
    values = exp069.VALUES
    model_metrics = {}
    effect_arrays = {}
    seed = 20260972
    if gate_pass:
        for model_key in runner.MODEL_CONFIGS:
            model_metrics[model_key] = {}
            for order in runner.ORDERS:
                effects = [paired_effect(
                    index[(model_key, case_id, order, 2.0)],
                    index[(model_key, case_id, order, 8.0)], values,
                ) for case_id in case_ids]
                exp_shift = np.asarray([e["expected_shift"] for e in effects])
                greedy_shift = np.asarray([np.nan if e["greedy_shift"] is None else e["greedy_shift"]
                                           for e in effects])
                effect_arrays[(model_key, order)] = {
                    "expected": exp_shift, "greedy": greedy_shift,
                }
                valid_greedy = np.isfinite(greedy_shift)
                greedy_report = bootstrap_mean(greedy_shift[valid_greedy], seed,
                                               bootstrap_replicates)
                greedy_report["n_complete"] = int(valid_greedy.sum())
                model_metrics[model_key][order] = {
                    "n_recipients": len(case_ids),
                    "expected_score_shift_8_minus_2": bootstrap_mean(
                        exp_shift, seed, bootstrap_replicates),
                    "greedy_score_shift_8_minus_2": greedy_report,
                    "expected_minus_greedy_shift": bootstrap_mean(
                        exp_shift[valid_greedy] - greedy_shift[valid_greedy],
                        seed, bootstrap_replicates),
                    "mean_total_variation_distance": float(np.mean([
                        e["total_variation"] for e in effects])),
                    "mean_wasserstein1_score_points": float(np.mean([
                        e["wasserstein1"] for e in effects])),
                    "valid_score_string_mass": {
                        "mean_if_first_2": float(np.mean([e["grid_mass_low"] for e in effects])),
                        "mean_if_first_8": float(np.mean([e["grid_mass_high"] for e in effects])),
                    },
                }
    size_contrasts = {}
    family_contrasts = {}
    domain_contrasts = {}
    if gate_pass:
        for order in runner.ORDERS:
            qwen_difference = (effect_arrays[("qwen-1.5b", order)]["expected"]
                               - effect_arrays[("qwen-3b", order)]["expected"])
            size_contrasts[order] = bootstrap_mean(qwen_difference, seed, bootstrap_replicates)
            family_contrasts[order] = {
                "smollm2_minus_qwen_1_5b": bootstrap_mean(
                    effect_arrays[("smollm2-1.7b", order)]["expected"]
                    - effect_arrays[("qwen-1.5b", order)]["expected"], seed,
                    bootstrap_replicates),
                "smollm2_minus_qwen_3b": bootstrap_mean(
                    effect_arrays[("smollm2-1.7b", order)]["expected"]
                    - effect_arrays[("qwen-3b", order)]["expected"], seed,
                    bootstrap_replicates),
            }
        laptop_manifest = json.loads(laptop_runner.MANIFEST.read_text())
        laptop_hash = sha256(laptop_runner.OUT.read_bytes())
        if (laptop_manifest.get("output_sha256") != laptop_hash
                or laptop_manifest.get("n_total_model_contexts") != 768):
            raise ValueError("Experiment 071 laptop artifact is incomplete or hash-mismatched")
        laptop_cases, _ = laptop_runner.select_cases(Path(".context/dimabsa"))
        laptop_jobs = laptop_runner.build_jobs(laptop_cases)
        laptop_expected = {
            (key, job["case_id"], job["order"], job["forced_first_value"])
            for key in laptop_runner.MODEL_CONFIGS for job in laptop_jobs
        }
        laptop_rows = [json.loads(line) for line in laptop_runner.OUT.read_text().splitlines()
                       if line.strip()]
        laptop_index = {
            (row["model_key"], row["case_id"], row["order"],
             float(row["forced_first_value"])): row for row in laptop_rows
        }
        if len(laptop_rows) != 768 or set(laptop_index) != laptop_expected:
            raise ValueError("Experiment 071 laptop cells are incomplete")
        laptop_case_ids = [case["case_id"] for case in laptop_cases]
        for model_key in runner.MODEL_CONFIGS:
            domain_contrasts[model_key] = {}
            for order_index, order in enumerate(runner.ORDERS):
                laptop_effects = np.asarray([
                    paired_effect(
                        laptop_index[(model_key, case_id, order, 2.0)],
                        laptop_index[(model_key, case_id, order, 8.0)], values,
                    )["expected_shift"] for case_id in laptop_case_ids
                ])
                restaurant_effects = effect_arrays[(model_key, order)]["expected"]
                domain_contrasts[model_key][order] = independent_difference(
                    restaurant_effects, laptop_effects,
                    seed + 100 + order_index + list(runner.MODEL_CONFIGS).index(model_key) * 2,
                    bootstrap_replicates,
                )
    return {
        "experiment": "072-restaurant-prefix-domain-replication",
        "protocol_sha256": runner.PROTOCOL_SHA256,
        "output_sha256": output_hash,
        "source_revision": sample_stats["source_revision"],
        "source_sha256": sample_stats["source_sha256"],
        "recipient_ids_sha256": sample_stats["recipient_ids_sha256"],
        "n_recipients": len(case_ids), "n_models": len(runner.MODEL_CONFIGS),
        "n_total_model_contexts": len(rows),
        "prior_exp051_full_split_exposure": True,
        "invalid_by_model_order_forced_value": invalid,
        "invalid_rates_by_model_order_forced_value": invalid_rates,
        "invalid_gate_passed": gate_pass,
        "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": seed,
        "by_model_and_order": model_metrics if gate_pass else None,
        "paired_qwen_size_contrast": size_contrasts if gate_pass else None,
        "smollm2_family_contrasts": family_contrasts if gate_pass else None,
        "restaurant_minus_laptop_expected_shift": domain_contrasts if gate_pass else None,
        "interpretation_limit": "Exploratory prefix intervention on 64 restaurant IDs previously scored in Experiment 051; not a fresh-sample replication. Artificial prefixes; model family, tokenizer, tuning, and chat template differ.",
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
