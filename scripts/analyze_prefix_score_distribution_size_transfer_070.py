"""Analyze Qwen2.5-1.5B score distributions and compare with Exp069."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import run_prefix_score_distribution_audit_069 as exp069
import run_prefix_score_distribution_size_transfer_070 as runner
from analyze_prefix_score_distribution_audit_069 import bootstrap_mean, validate_distribution
from run_small_model_decoder_factorial_048 import sha256

OUT = Path("results/prefix-score-distribution-size-transfer-v1")
N_BOOTSTRAPS = 10_000


def paired_order_metrics(low: dict, high: dict, values: np.ndarray) -> dict:
    p_low = validate_distribution(low)
    p_high = validate_distribution(high)
    return {
        "low_expectation": float(p_low @ values),
        "high_expectation": float(p_high @ values),
        "greedy_low": low["greedy_second_score"],
        "greedy_high": high["greedy_second_score"],
        "total_variation": float(0.5 * np.abs(p_high - p_low).sum()),
        "wasserstein1": float(np.abs(np.cumsum(p_high - p_low)).sum() * 0.1),
        "low_grid_mass": float(np.exp(np.asarray(low["score_logprobs"])).sum()),
        "high_grid_mass": float(np.exp(np.asarray(high["score_logprobs"])).sum()),
        "low_map": float(values[int(np.argmax(p_low))]),
        "high_map": float(values[int(np.argmax(p_high))]),
    }


def summarize(predictions: Path = runner.OUT, manifest_path: Path = runner.MANIFEST,
              parent_predictions: Path = exp069.OUT,
              bootstrap_replicates: int = N_BOOTSTRAPS) -> dict:
    manifest = json.loads(manifest_path.read_text())
    output_hash = sha256(predictions.read_bytes())
    if manifest.get("experiment") != "070-prefix-score-distribution-size-transfer":
        raise ValueError("Unexpected Experiment 070 manifest")
    if manifest.get("protocol_sha256") != runner.PROTOCOL_SHA256:
        raise ValueError("Manifest does not match frozen Experiment 070 protocol")
    if manifest.get("output_sha256") != output_hash or manifest.get("n_contexts") != 512:
        raise ValueError("Experiment 070 output is incomplete or hash-mismatched")
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    keys = [(r["cohort"], r["case_id"], r["order"], float(r["forced_first_value"])) for r in rows]
    if len(rows) != 512 or len(set(keys)) != 512:
        raise ValueError("Expected 512 unique Experiment 070 context cells")
    index = {}
    for row in rows:
        probs = validate_distribution(row)
        row["_probs"] = probs
        index[(row["cohort"], row["case_id"], row["order"],
               float(row["forced_first_value"]))] = row
    parent_manifest = json.loads(Path(".context/exp069-run-manifest.json").read_text())
    if sha256(parent_predictions.read_bytes()) != exp069.EXPECTED_OUTPUT_HASHES["067"]:
        # The combined Experiment 069 artifact has its own hash, checked via manifest.
        if sha256(parent_predictions.read_bytes()) != parent_manifest.get("output_sha256"):
            raise ValueError("Experiment 069 comparison artifact hash mismatch")
    parent_rows = [json.loads(line) for line in parent_predictions.read_text().splitlines()
                   if line.strip()]
    parent_index = {(r["cohort"], r["case_id"], r["order"], float(r["forced_first_value"])): r
                    for r in parent_rows}
    if len(parent_index) != 512:
        raise ValueError("Expected complete Experiment 069 comparison grid")
    for row in rows:
        key = (row["cohort"], row["case_id"], row["order"], float(row["forced_first_value"]))
        if key not in parent_index or row["prompt_sha256"] != parent_index[key]["prompt_sha256"]:
            raise ValueError(f"Experiment 070 prompt/cell differs from Exp069: {key}")

    invalid = manifest["invalid_by_cohort_order_forced_value"]
    invalid_rates = {
        cohort: {order: {forced: count / 64 for forced, count in arms.items()}
                 for order, arms in orders.items()}
        for cohort, orders in invalid.items()
    }
    gate_pass = all(rate <= 0.02 for orders in invalid_rates.values()
                    for arms in orders.values() for rate in arms.values())
    values = exp069.VALUES
    by_cohort = {}
    bootstrap_seed = 20260970
    for cohort in ("067", "068"):
        case_ids = sorted({r["case_id"] for r in rows if r["cohort"] == cohort})
        if len(case_ids) != 64:
            raise ValueError(f"Expected 64 recipients in cohort {cohort}")
        by_cohort[cohort] = {}
        for order_index, order in enumerate(exp069.ORDERS):
            case_metrics = []
            size_diffs = []
            for case_id in case_ids:
                low = index[(cohort, case_id, order, 2.0)]
                high = index[(cohort, case_id, order, 8.0)]
                case_metrics.append(paired_order_metrics(low, high, values))
                low3 = parent_index[(cohort, case_id, order, 2.0)]
                high3 = parent_index[(cohort, case_id, order, 8.0)]
                low3p = validate_distribution(low3)
                high3p = validate_distribution(high3)
                size_diffs.append(float((high["_probs"] @ values - low["_probs"] @ values)
                                        - (high3p @ values - low3p @ values)))
            expected_shifts = np.asarray([
                m["high_expectation"] - m["low_expectation"] for m in case_metrics
            ])
            greedy = np.asarray([
                np.nan if m["greedy_low"] is None or m["greedy_high"] is None
                else m["greedy_high"] - m["greedy_low"] for m in case_metrics
            ])
            if not gate_pass:
                by_cohort[cohort][order] = {"score_contrasts_withheld": True}
                continue
            valid_greedy = np.isfinite(greedy)
            if valid_greedy.sum() != len(greedy):
                greedy_report = {"n_complete": int(valid_greedy.sum()), "estimate": None, "ci95": None}
                gap_report = {"n_complete": int(valid_greedy.sum()), "estimate": None, "ci95": None}
            else:
                greedy_report = bootstrap_mean(greedy, bootstrap_seed, bootstrap_replicates)
                gap_report = bootstrap_mean(expected_shifts - greedy, bootstrap_seed,
                                            bootstrap_replicates)
                greedy_report["n_complete"] = len(greedy)
                gap_report["n_complete"] = len(greedy)
            mass_low = np.asarray([m["low_grid_mass"] for m in case_metrics])
            mass_high = np.asarray([m["high_grid_mass"] for m in case_metrics])
            by_cohort[cohort][order] = {
                "n_recipients": 64,
                "expected_score_shift_8_minus_2": bootstrap_mean(
                    expected_shifts, bootstrap_seed, bootstrap_replicates),
                "greedy_score_shift_8_minus_2": greedy_report,
                "expected_minus_greedy_shift": gap_report,
                "paired_1_5b_minus_3b_expected_shift": bootstrap_mean(
                    np.asarray(size_diffs), bootstrap_seed, bootstrap_replicates),
                "mean_total_variation_distance": float(np.mean([
                    m["total_variation"] for m in case_metrics])),
                "mean_wasserstein1_score_points": float(np.mean([
                    m["wasserstein1"] for m in case_metrics])),
                "valid_score_string_mass": {
                    "mean_if_first_2": float(mass_low.mean()),
                    "mean_if_first_8": float(mass_high.mean()),
                    "high_minus_low": bootstrap_mean(
                        mass_high - mass_low, bootstrap_seed, bootstrap_replicates),
                },
                "restricted_map_shift": bootstrap_mean(
                    np.asarray([m["high_map"] - m["low_map"] for m in case_metrics]),
                    bootstrap_seed, bootstrap_replicates),
            }
    return {
        "experiment": "070-prefix-score-distribution-size-transfer",
        "protocol_sha256": runner.PROTOCOL_SHA256,
        "output_sha256": output_hash,
        "parent_069_output_sha256": manifest.get("parent_069_output_sha256"),
        "n_contexts": len(rows), "score_grid_size": 81,
        "invalid_by_cohort_order_forced_value": invalid,
        "invalid_rates_by_cohort_order_forced_value": invalid_rates,
        "invalid_gate_passed": gate_pass,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "by_cohort_and_order": by_cohort,
        "interpretation_limit": "Same-Qwen-family size transfer on reused recipients after observing 3B outcomes; this is neither independent replication nor evidence about ordinary ratings.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=runner.OUT)
    parser.add_argument("--manifest", dest="manifest_path", type=Path, default=runner.MANIFEST)
    parser.add_argument("--parent-predictions", type=Path, default=exp069.OUT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=N_BOOTSTRAPS)
    args = parser.parse_args()
    result = summarize(args.predictions, args.manifest_path, args.parent_predictions,
                       args.bootstrap_replicates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
