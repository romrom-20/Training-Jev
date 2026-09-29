"""Analyze frozen score distributions from Experiment 069."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import run_prefix_score_distribution_audit_069 as runner
from run_small_model_decoder_factorial_048 import sha256

OUT = Path("results/prefix-score-distribution-audit-v1")
N_BOOTSTRAPS = 10_000


def bootstrap_mean(values: np.ndarray, seed: int, replicates: int = N_BOOTSTRAPS) -> dict:
    if values.ndim != 1 or not len(values):
        raise ValueError("Bootstrap requires a nonempty vector of recipient-level values")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for i in range(replicates):
        draws[i] = np.mean(values[rng.integers(0, len(values), size=len(values))])
    return {"estimate": float(values.mean()),
            "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])]}


def validate_distribution(row: dict) -> np.ndarray:
    logp = np.asarray(row["score_logprobs"], dtype=np.float64)
    probs = np.asarray(row["restricted_probabilities"], dtype=np.float64)
    if logp.shape != (81,) or probs.shape != (81,):
        raise ValueError("Every context must contain 81 candidate scores and probabilities")
    if not np.isfinite(logp).all() or not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("Non-finite or negative candidate probability")
    expected = np.exp(logp - np.max(logp))
    expected /= expected.sum()
    if not np.allclose(probs.sum(), 1.0, atol=1e-10):
        raise ValueError("Restricted candidate probabilities do not sum to one")
    if not np.allclose(probs, expected, rtol=1e-7, atol=1e-10):
        raise ValueError("Stored probabilities do not match the token log-probabilities")
    return probs


def summarize(predictions: Path = runner.OUT, manifest_path: Path = runner.MANIFEST,
              bootstrap_replicates: int = N_BOOTSTRAPS) -> dict:
    manifest = json.loads(manifest_path.read_text())
    output_hash = sha256(predictions.read_bytes())
    if manifest.get("experiment") != "069-prefix-score-distribution-audit":
        raise ValueError("Unexpected Experiment 069 manifest")
    if manifest.get("protocol_sha256") != runner.PROTOCOL_SHA256:
        raise ValueError("Manifest does not match frozen Experiment 069 protocol")
    if manifest.get("output_sha256") != output_hash or manifest.get("n_contexts") != 512:
        raise ValueError("Score-distribution output is incomplete or hash-mismatched")
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    keys = [(r["cohort"], r["case_id"], r["order"], r["forced_first_value"]) for r in rows]
    expected = {(cohort, case, order, forced)
                for cohort in ("067", "068") for case in {
                    r["case_id"] for r in rows if r["cohort"] == cohort
                } for order in runner.ORDERS for forced in runner.FORCED_VALUES}
    if len(rows) != 512 or len(set(keys)) != 512 or set(keys) != expected:
        raise ValueError("Experiment 069 output must have the complete 2×64×2×2 grid")
    index = {}
    for row in rows:
        probs = validate_distribution(row)
        if len(row.get("prompt_sha256", "")) != 64:
            raise ValueError("Missing reconstructed-prompt hash")
        row["_probs"] = probs
        index[(row["cohort"], row["case_id"], row["order"],
               float(row["forced_first_value"]))] = row

    values = runner.VALUES
    metrics = {}
    paired_arrays = {}
    for cohort in ("067", "068"):
        metrics[cohort] = {}
        case_ids = sorted({r["case_id"] for r in rows if r["cohort"] == cohort})
        if len(case_ids) != 64:
            raise ValueError(f"Experiment {cohort} must contain 64 recipients")
        for order_index, order in enumerate(runner.ORDERS):
            low_expect, high_expect, greedy_delta, map_delta, tv, w1 = [], [], [], [], [], []
            low_mass, high_mass = [], []
            for case_id in case_ids:
                low = index[(cohort, case_id, order, 2.0)]
                high = index[(cohort, case_id, order, 8.0)]
                p_low, p_high = low["_probs"], high["_probs"]
                low_expect.append(float(p_low @ values))
                high_expect.append(float(p_high @ values))
                low_mass.append(float(np.exp(np.asarray(low["score_logprobs"])).sum()))
                high_mass.append(float(np.exp(np.asarray(high["score_logprobs"])).sum()))
                greedy_delta.append(float(high["observed_second"] - low["observed_second"]))
                map_delta.append(float(values[int(np.argmax(p_high))] - values[int(np.argmax(p_low))]))
                tv.append(float(0.5 * np.abs(p_high - p_low).sum()))
                cdf_delta = np.cumsum(p_high) - np.cumsum(p_low)
                w1.append(float(np.abs(cdf_delta).sum() * 0.1))
            shift = np.asarray(high_expect) - np.asarray(low_expect)
            greedy_delta_array = np.asarray(greedy_delta)
            seed = runner.BOOTSTRAP_SEED
            paired_arrays[(cohort, order)] = shift
            metrics[cohort][order] = {
                "n_recipients": len(case_ids),
                "mean_expected_score_if_first_2": float(np.mean(low_expect)),
                "mean_expected_score_if_first_8": float(np.mean(high_expect)),
                "expected_score_shift_8_minus_2": bootstrap_mean(
                    shift, seed, bootstrap_replicates),
                "observed_greedy_score_shift_8_minus_2": bootstrap_mean(
                    greedy_delta_array, seed, bootstrap_replicates),
                "expected_minus_greedy_shift_exploratory": bootstrap_mean(
                    shift - greedy_delta_array, seed, bootstrap_replicates),
                "restricted_distribution_map_shift_8_minus_2": bootstrap_mean(
                    np.asarray(map_delta), seed, bootstrap_replicates),
                "mean_total_variation_distance": float(np.mean(tv)),
                "mean_wasserstein1_score_points": float(np.mean(w1)),
                "posthoc_valid_score_string_mass": {
                    "mean_if_first_2": float(np.mean(low_mass)),
                    "mean_if_first_8": float(np.mean(high_mass)),
                    "high_minus_low": bootstrap_mean(
                        np.asarray(high_mass) - np.asarray(low_mass), seed, bootstrap_replicates),
                    "interpretation": "Unnormalized model probability mass on the 81 canonical numeric prefixes; post-hoc support check.",
                },
                "mean_restricted_entropy_bits_low_prefix": float(np.mean([
                    -(p * np.log2(np.maximum(p, 1e-300))).sum()
                    for p in (index[(cohort, case_id, order, 2.0)]["_probs"] for case_id in case_ids)
                ])),
                "mean_restricted_entropy_bits_high_prefix": float(np.mean([
                    -(p * np.log2(np.maximum(p, 1e-300))).sum()
                    for p in (index[(cohort, case_id, order, 8.0)]["_probs"] for case_id in case_ids)
                ])),
            }
    primary_resolved = all(
        metrics[cohort]["arousal_first"]["expected_score_shift_8_minus_2"]["ci95"][1] < 0
        for cohort in ("067", "068")
    )
    return {
        "experiment": "069-prefix-score-distribution-audit",
        "protocol_sha256": runner.PROTOCOL_SHA256,
        "output_sha256": output_hash,
        "n_contexts": len(rows), "n_recipients_per_cohort": 64,
        "score_support": {"n_values": len(values), "min": float(values[0]),
                           "max": float(values[-1]), "step": 0.1,
                           "normalization": "within 81 canonical one-decimal number strings only"},
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": runner.BOOTSTRAP_SEED,
        "primary_diagnostic": "paired conditional expected-score shift after arousal-first high vs low forced prefix, evaluated separately in both cohorts",
        "primary_distribution_shift_resolved_in_both_cohorts": primary_resolved,
        "posthoc_addendum": "The valid-number support mass and paired expected-minus-greedy contrasts are exploratory additions prompted by the frozen-support interpretation; they do not replace the registered primary endpoint.",
        "by_cohort_and_order": metrics,
        "interpretation_limit": "This is a post-hoc restricted-support continuation audit, not a new independent confirmation or evidence about ordinary ratings.",
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
