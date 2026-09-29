"""Analyze the additional Qwen2.5-0.5B size point against 071."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import run_cross_family_prefix_coupling_071 as exp071
import run_prefix_score_distribution_audit_069 as exp069
import run_qwen05_prefix_size_073 as runner
from analyze_cross_family_prefix_coupling_071 import paired_effect
from analyze_prefix_score_distribution_audit_069 import bootstrap_mean, validate_distribution
from run_small_model_decoder_factorial_048 import sha256

OUT = Path("results/qwen05-prefix-size-continuation-v1")
N_BOOTSTRAPS = 10_000


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(predictions: Path = runner.OUT, manifest_path: Path = runner.MANIFEST,
              bootstrap_replicates: int = N_BOOTSTRAPS) -> dict:
    manifest = json.loads(manifest_path.read_text())
    output_hash = sha256(predictions.read_bytes())
    if (manifest.get("experiment") != "073-qwen05-prefix-size-continuation"
            or manifest.get("protocol_sha256") != runner.PROTOCOL_SHA256
            or manifest.get("output_sha256") != output_hash
            or manifest.get("n_total_model_contexts") != 256):
        raise ValueError("Experiment 073 raw outputs are incomplete or hash-mismatched")

    cases, metadata = runner.select_cases(Path(".context/dimabsa"))
    jobs = runner.build_jobs(cases)
    case_ids = [case["case_id"] for case in cases]
    rows = load_rows(predictions)
    index = {}
    expected = {(job["case_id"], job["order"], job["forced_first_value"]) for job in jobs}
    for row in rows:
        key = (row["case_id"], row["order"], float(row["forced_first_value"]))
        if key in index or key not in expected:
            raise ValueError(f"Unexpected/duplicate 073 row: {key}")
        if row["model"] != runner.QWEN05_MODEL or row["model_revision"] != runner.QWEN05_REVISION:
            raise ValueError(f"Unexpected Qwen0.5B checkpoint: {key}")
        validate_distribution(row)
        index[key] = row
    if set(index) != expected:
        raise ValueError("Experiment 073 is missing recipient/prefix cells")

    reference_manifest = json.loads(exp071.MANIFEST.read_text())
    reference_hash = sha256(exp071.OUT.read_bytes())
    if (reference_manifest.get("output_sha256") != reference_hash
            or reference_manifest.get("n_total_model_contexts") != 768):
        raise ValueError("Experiment 071 reference artifact is incomplete or mismatched")
    reference_rows = load_rows(exp071.OUT)
    reference_index = {
        (row["model_key"], row["case_id"], row["order"],
         float(row["forced_first_value"])): row for row in reference_rows
    }
    if len(reference_rows) != 768 or len(reference_index) != 768:
        raise ValueError("Experiment 071 must contain 768 unique reference cells")

    invalid = manifest["invalid_by_model_order_forced_value"]["qwen-0.5b"]
    invalid_rates = {order: {value: count / 64 for value, count in arms.items()}
                     for order, arms in invalid.items()}
    gate_pass = all(rate <= 0.02 for arms in invalid_rates.values() for rate in arms.values())
    values = exp069.VALUES
    metrics = {}
    arrays = {"qwen-0.5b": {}}
    for order in runner.ORDERS:
        metrics[order] = {}
        effects = []
        for case_id in case_ids:
            lo, hi = index[(case_id, order, 2.0)], index[(case_id, order, 8.0)]
            effects.append(paired_effect(lo, hi, values))
        expected_shifts = np.asarray([effect["expected_shift"] for effect in effects])
        greedy_shifts = np.asarray([np.nan if effect["greedy_shift"] is None
                                    else effect["greedy_shift"] for effect in effects])
        arrays["qwen-0.5b"][order] = expected_shifts
        if gate_pass:
            metrics[order] = {
                "n_recipients": len(case_ids),
                "expected_score_shift_8_minus_2": bootstrap_mean(
                    expected_shifts, 20260973, bootstrap_replicates),
                "greedy_score_shift_8_minus_2": bootstrap_mean(
                    greedy_shifts, 20260973, bootstrap_replicates),
                "mean_total_variation_distance": float(np.mean(
                    [effect["total_variation"] for effect in effects])),
                "mean_wasserstein1_score_points": float(np.mean(
                    [effect["wasserstein1"] for effect in effects])),
                "valid_score_string_mass": {
                    "mean_if_first_2": float(np.mean([effect["grid_mass_low"] for effect in effects])),
                    "mean_if_first_8": float(np.mean([effect["grid_mass_high"] for effect in effects])),
                },
            }

    contrasts = {}
    if gate_pass:
        for order in runner.ORDERS:
            for comparator in ("qwen-1.5b", "qwen-3b"):
                qwen_effects = []
                for case_id in case_ids:
                    lo = reference_index[(comparator, case_id, order, 2.0)]
                    hi = reference_index[(comparator, case_id, order, 8.0)]
                    qwen_effects.append(paired_effect(lo, hi, values)["expected_shift"])
                arrays[comparator] = arrays.get(comparator, {})
                arrays[comparator][order] = np.asarray(qwen_effects)
                contrasts.setdefault(order, {})[f"qwen-0.5b_minus_{comparator}"] = bootstrap_mean(
                    arrays["qwen-0.5b"][order] - arrays[comparator][order],
                    20260973, bootstrap_replicates,
                )
    return {
        "experiment": "073-qwen05-prefix-size-continuation",
        "protocol_sha256": runner.PROTOCOL_SHA256,
        "output_sha256": output_hash,
        "reference_071_output_sha256": reference_hash,
        "recipient_ids_sha256": metadata["recipient_ids_sha256"],
        "n_recipients": len(case_ids), "n_model_contexts": len(rows),
        "invalid_by_order_forced_value": invalid,
        "invalid_rates_by_order_forced_value": invalid_rates,
        "invalid_gate_passed": gate_pass,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": 20260973,
        "qwen_0_5b_by_order": metrics if gate_pass else None,
        "paired_expected_shift_contrasts_vs_071": contrasts if gate_pass else None,
        "interpretation_limit": "Third Qwen2.5 size point on the previously observed public laptop cohort; a three-point family-specific result is not a scaling law.",
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
