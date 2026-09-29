"""Analyze full-JSON and alternate numeric score support in Experiment 074."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import run_cross_family_prefix_coupling_071 as exp071
import run_prefix_score_distribution_audit_069 as exp069
import run_qwen05_numeric_support_074 as runner
import run_qwen05_prefix_size_073 as exp073
from analyze_prefix_score_distribution_audit_069 import bootstrap_mean
from run_small_model_decoder_factorial_048 import sha256

OUT = Path("results/qwen05-numeric-support-audit-v1")
N_BOOTSTRAPS = 10_000


def _mass_expected(row: dict, family: str) -> tuple[float, float]:
    values = exp069.VALUES
    canonical = np.exp(np.asarray(row["canonical_logprobs"], dtype=float))
    extended = np.exp(np.asarray(row["extended_logprobs"], dtype=float))
    integers = np.exp(np.asarray(row["integer_logprobs"], dtype=float))
    if canonical.shape != (81,) or extended.shape != (81,) or integers.shape != (9,):
        raise ValueError("Malformed score-family likelihood array")
    if family == "canonical":
        probabilities = canonical
        mass = float(probabilities.sum())
    elif family == "all_parser_forms":
        probabilities = canonical + extended
        probabilities[np.arange(0, 81, 10)] += integers
        mass = float(probabilities.sum())
    else:
        raise ValueError(f"Unexpected surface-form family: {family}")
    if mass <= 0 or not np.isfinite(probabilities).all():
        raise ValueError("Invalid surface-form score probability mass")
    expected = float((probabilities / mass) @ values)
    return mass, expected


def _expected_shift(index: dict, model: str, case_id: str, order: str,
                    family: str) -> tuple[float, dict]:
    low = index[(model, case_id, order, 2.0)]
    high = index[(model, case_id, order, 8.0)]
    low_mass, low_expected = _mass_expected(low, family)
    high_mass, high_expected = _mass_expected(high, family)
    return high_expected - low_expected, {"mass_low": low_mass, "mass_high": high_mass}


def summarize(predictions: Path = runner.OUT, manifest_path: Path = runner.MANIFEST,
              bootstrap_replicates: int = N_BOOTSTRAPS) -> dict:
    manifest = json.loads(manifest_path.read_text())
    output_hash = sha256(predictions.read_bytes())
    if (manifest.get("experiment") != "074-qwen05-numeric-support-audit"
            or manifest.get("protocol_sha256") != runner.PROTOCOL_SHA256
            or manifest.get("output_sha256") != output_hash
            or manifest.get("n_total_model_contexts") != 192):
        raise ValueError("Experiment 074 output is incomplete or hash-mismatched")
    cases, metadata = runner.select_cases(Path(".context/dimabsa"))
    jobs = runner.build_jobs(cases)
    case_ids = [case["case_id"] for case in cases]
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    expected = {(model, job["case_id"], job["order"], job["forced_first_value"])
                for model in runner.MODEL_CONFIGS for job in jobs}
    index = {}
    for row in rows:
        key = (row["model_key"], row["case_id"], row["order"],
               float(row["forced_first_value"]))
        if key in index or key not in expected:
            raise ValueError(f"Unexpected or duplicate 074 cell: {key}")
        if row["model_revision"] != runner.MODEL_CONFIGS[row["model_key"]]["revision"]:
            raise ValueError(f"Unexpected model revision for cell {key}")
        index[key] = row
    if set(index) != expected:
        raise ValueError("Experiment 074 is missing expected cells")

    prior_manifest = json.loads(exp073.MANIFEST.read_text())
    prior_hash = sha256(exp073.OUT.read_bytes())
    if (prior_manifest.get("output_sha256") != prior_hash
            or prior_manifest.get("n_total_model_contexts") != 256):
        raise ValueError("Experiment 073 comparison artifact is incomplete")
    prior_rows = [json.loads(line) for line in exp073.OUT.read_text().splitlines() if line.strip()]
    prior_index = {(row["case_id"], row["order"], float(row["forced_first_value"])): row
                   for row in prior_rows}
    size_manifest = json.loads(exp071.MANIFEST.read_text())
    size_hash = sha256(exp071.OUT.read_bytes())
    if size_manifest.get("output_sha256") != size_hash or size_manifest.get("n_total_model_contexts") != 768:
        raise ValueError("Experiment 071 comparison artifact is incomplete")
    size_rows = [json.loads(line) for line in exp071.OUT.read_text().splitlines() if line.strip()]
    size_index = {
        (row["model_key"], row["case_id"], row["order"],
         float(row["forced_first_value"])): row for row in size_rows
    }
    if len(size_index) != 768:
        raise ValueError("Experiment 071 comparison cells are not unique")

    invalid = manifest["invalid_by_model_order_forced_value"]
    metrics, method_contrasts = {}, {}
    greedy_metrics, validation_summary = {}, {}
    values = exp069.VALUES
    for model in runner.MODEL_CONFIGS:
        metrics[model], greedy_metrics[model], method_contrasts[model] = {}, {}, {}
        model_validation = manifest["sequence_likelihood_validation"][model]
        if model_validation["n_checks"] != 32 or model_validation["max_abs_logprob_error"] > 0.001:
            raise ValueError(f"Independent sequence scoring check failed for {model}")
        validation_summary[model] = model_validation
        for order in runner.ORDERS:
            per_family = {"canonical": [], "all_parser_forms": []}
            mass_rows = {"canonical": [], "extended": [], "integer": [], "combined": []}
            greedy = []
            original = []
            for case_id in case_ids:
                low = index[(model, case_id, order, 2.0)]
                high = index[(model, case_id, order, 8.0)]
                for family in per_family:
                    shift, _ = _expected_shift(index, model, case_id, order, family)
                    per_family[family].append(shift)
                for row in (low, high):
                    canonical_mass = float(np.exp(row["canonical_logprobs"]).sum())
                    extended_mass = float(np.exp(row["extended_logprobs"]).sum())
                    integer_mass = float(np.exp(row["integer_logprobs"]).sum())
                    if canonical_mass + extended_mass + integer_mass > 1.0001:
                        raise ValueError("Disjoint full continuations exceed unit probability mass")
                    mass_rows["canonical"].append(canonical_mass)
                    mass_rows["extended"].append(extended_mass)
                    mass_rows["integer"].append(integer_mass)
                    mass_rows["combined"].append(canonical_mass + extended_mass + integer_mass)
                greedy.append(
                    None if low["greedy_second_score"] is None or high["greedy_second_score"] is None
                    else float(high["greedy_second_score"] - low["greedy_second_score"])
                )
                if model == "qwen-0.5b":
                    p_low, p_high = prior_index[(case_id, order, 2.0)], prior_index[(case_id, order, 8.0)]
                else:
                    p_low = size_index[(model, case_id, order, 2.0)]
                    p_high = size_index[(model, case_id, order, 8.0)]
                original.append(float((np.exp(p_high["score_logprobs"]) /
                                       np.exp(p_high["score_logprobs"]).sum()) @ values
                                      - (np.exp(p_low["score_logprobs"]) /
                                         np.exp(p_low["score_logprobs"]).sum()) @ values))
            seed = 20260974 + list(runner.MODEL_CONFIGS).index(model) * 10 + runner.ORDERS.index(order)
            arrays = {family: np.asarray(shifts) for family, shifts in per_family.items()}
            metrics[model][order] = {
                family: bootstrap_mean(shifts, seed, bootstrap_replicates)
                for family, shifts in arrays.items()
            }
            metrics[model][order]["original_number_only_from_073"] = bootstrap_mean(
                np.asarray(original), seed, bootstrap_replicates)
            method_contrasts[model][order] = {
                "all_parser_forms_minus_canonical_and_close": bootstrap_mean(
                    arrays["all_parser_forms"] - arrays["canonical"],
                    seed, bootstrap_replicates),
                "canonical_and_close_minus_original_number_only": bootstrap_mean(
                    arrays["canonical"] - np.asarray(original), seed, bootstrap_replicates),
            }
            complete_greedy = np.asarray([np.nan if value is None else value for value in greedy])
            if not np.isfinite(complete_greedy).all():
                raise ValueError("Experiment 074 contains invalid greedy outputs")
            greedy_metrics[model][order] = bootstrap_mean(
                complete_greedy, seed, bootstrap_replicates)
            metrics[model][order]["mean_unnormalized_mass_by_form"] = {
                form: float(np.mean(mass)) for form, mass in mass_rows.items()
            }
    invalid_rates = {model: {order: {value: count / 24 for value, count in arms.items()}
                             for order, arms in orders.items()}
                     for model, orders in invalid.items()}
    return {
        "experiment": "074-qwen05-numeric-support-audit",
        "protocol_sha256": runner.PROTOCOL_SHA256,
        "output_sha256": output_hash,
        "reference_073_output_sha256": prior_hash,
        "reference_071_output_sha256": size_hash,
        "recipient_ids_sha256": metadata["recipient_ids_sha256"],
        "n_recipients": len(case_ids), "n_model_contexts": len(rows),
        "candidate_surfaces_per_context": {"canonical": 81, "extended": 81, "integer": 9},
        "invalid_by_model_order_forced_value": invalid,
        "invalid_rates_by_model_order_forced_value": invalid_rates,
        "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": 20260974,
        "expected_shift_by_model_and_order": metrics,
        "greedy_shift_by_model_and_order": greedy_metrics,
        "paired_method_contrasts": method_contrasts,
        "full_sequence_likelihood_validation": validation_summary,
        "interpretation_limit": "Support audit on 24 previously observed laptop reviews; alternate forms are accepted by the parser but violate one-decimal instructions, and the set still does not enumerate all valid parser outputs.",
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
