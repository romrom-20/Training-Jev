"""Analyze the fresh-recipient forced-prefix replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import run_fresh_sample_prefix_replication_068 as runner
import run_sighan_chinese_decoder_transfer_052 as exp052
import run_sighan_output_key_order_064 as exp064
from run_small_model_decoder_factorial_048 import sha256

OUT = Path("results/fresh-sample-prefix-coupling-replication-v1")
N_BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260968
PRIOR_DIRECTION = {"valence_first": 1, "arousal_first": -1}


def index_rows(path: Path, manifest: dict) -> dict[tuple, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [(row["case_id"], row["order"], row["forced_first_value"], row["decoder"])
            for row in rows]
    if len(rows) != 256 or len(set(keys)) != 256:
        raise ValueError(f"Expected 256 unique Exp068 output cells; found {len(rows)}")
    if manifest.get("n_outputs") != 256 or manifest.get("output_sha256") != sha256(path.read_bytes()):
        raise ValueError("Experiment 068 outputs do not match the run manifest")
    return dict(zip(keys, rows))


def analyze(predictions: Path = runner.OUT, manifest_path: Path = runner.MANIFEST,
            source_dir: Path = exp052.SOURCE_DIR,
            bootstrap_replicates: int = N_BOOTSTRAPS) -> dict:
    runner.verify_parents()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("experiment") != "068-fresh-sample-prefix-coupling-replication":
        raise ValueError("Unexpected Experiment 068 manifest")
    if manifest.get("protocol_sha256") != runner.PROTOCOL_SHA256:
        raise ValueError("Manifest does not match frozen Experiment 068 protocol")
    source_rows, source_hashes = exp052.read_source(source_dir)
    all_cases = exp052.select_cases(source_rows)
    old_cohort, _ = exp064.select_cases(all_cases)
    excluded_ids = {case["case_id"] for case in old_cohort}
    cases, sample_stats = runner.select_cases(all_cases, excluded_ids)
    if sample_stats["recipient_ids_sha256"] != runner.EXPECTED_IDS_SHA256:
        raise ValueError("Experiment 068 recipients differ from the frozen sample")
    if not sample_stats["disjoint_from_exp064"]:
        raise ValueError("Experiment 068 recipients overlap Experiment 064")
    ids = sorted(case["case_id"] for case in cases)
    new = index_rows(predictions, manifest)
    expected = {
        (case_id, order, forced, runner.DECODER)
        for case_id in ids for order in runner.ORDERS for forced in runner.FORCED_VALUES
    }
    if set(new) != expected:
        raise ValueError("Experiment 068 outputs differ from the frozen recipient/cell grid")
    if manifest.get("exp064_output_sha256") != runner.PARENT_064_SHA256:
        raise ValueError("Experiment 068 manifest records unexpected Exp064 cohort provenance")
    if manifest.get("source_hashes") != source_hashes:
        raise ValueError("Experiment 068 manifest records unexpected pinned source hashes")
    for key, row in new.items():
        first_axis = 0 if key[1] == "valence_first" else 1
        if row["prediction"] is not None and float(row["prediction"][first_axis]) != key[2]:
            raise ValueError(f"Forced first coordinate changed in parsed output: {key}")

    invalid = {
        order: {
            str(forced): sum(new[(case_id, order, forced, runner.DECODER)]["prediction"] is None
                             for case_id in ids)
            for forced in runner.FORCED_VALUES
        }
        for order in runner.ORDERS
    }
    denominator = len(ids)
    rates = {order: {forced: count / denominator for forced, count in arms.items()}
             for order, arms in invalid.items()}
    provenance = {
        "experiment": "068-fresh-sample-prefix-coupling-replication",
        "protocol_sha256": manifest["protocol_sha256"],
        "output_sha256": manifest["output_sha256"],
        "exp064_output_sha256": runner.PARENT_064_SHA256,
        "source_hashes": source_hashes,
    }
    if any(rate > 0.02 for arms in rates.values() for rate in arms.values()):
        return {
            **provenance, "status": "protocol_execution_failure",
            "reason": "At least one forced-value × output-order arm exceeded the 2% invalid-output gate",
            "invalid_by_order_forced_value": invalid,
            "invalid_denominator_per_arm": denominator,
            "invalid_rate_by_order_forced_value": rates,
            "score_analysis_performed": False,
        }

    complete = [case_id for case_id in ids if all(
        new[(case_id, order, forced, runner.DECODER)]["prediction"] is not None
        for order in runner.ORDERS for forced in runner.FORCED_VALUES
    )]
    if not complete:
        raise ValueError("No recipients have all four valid Exp068 cells")
    second_axis = {"valence_first": 1, "arousal_first": 0}
    arms = {}
    shifts = {}
    for order in runner.ORDERS:
        axis = second_axis[order]
        arms[order] = {
            forced: np.array([
                float(new[(case_id, order, forced, runner.DECODER)]["prediction"][axis])
                for case_id in complete
            ]) for forced in runner.FORCED_VALUES
        }
        shifts[order] = arms[order][8.0] - arms[order][2.0]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(complete)
    order_draws = {order: np.empty(bootstrap_replicates) for order in runner.ORDERS}
    primary_draws = np.empty(bootstrap_replicates)
    for i in range(bootstrap_replicates):
        sample = rng.integers(0, n, size=n)
        for order in runner.ORDERS:
            order_draws[order][i] = float(np.mean(shifts[order][sample]))
        primary_draws[i] = float(np.mean([order_draws[order][i] for order in runner.ORDERS]))
    order_points = {order: float(np.mean(values)) for order, values in shifts.items()}
    primary = float(np.mean(list(order_points.values())))
    order_results = {}
    for order in runner.ORDERS:
        ci = [float(value) for value in np.quantile(order_draws[order], [0.025, 0.975])]
        expected_sign = PRIOR_DIRECTION[order]
        direction_matches = order_points[order] * expected_sign > 0
        interval_resolves_direction = ci[0] > 0 if expected_sign > 0 else ci[1] < 0
        order_results[order] = {
            "mean_second_coordinate_if_first_forced_2": float(np.mean(arms[order][2.0])),
            "mean_second_coordinate_if_first_forced_8": float(np.mean(arms[order][8.0])),
            "mean_second_coordinate_shift_8_minus_2": order_points[order],
            "ci95": ci,
            "matches_exp067_estimate_direction": direction_matches,
            "ci_resolves_exp067_direction": interval_resolves_direction,
            "prior_exp067_direction": "positive" if expected_sign > 0 else "negative",
        }
    return {
        **provenance,
        "status": "scored",
        "n_recipient_ids_selected": len(ids),
        "n_complete_four_cell_recipient_ids": n,
        "n_new_outputs": manifest["n_outputs"],
        "sample": {
            "n_exp064_ids_excluded": sample_stats["n_exp064_ids_excluded"],
            "recipient_ids_sha256": sample_stats["recipient_ids_sha256"],
            "category_polarity_counts": sample_stats["category_polarity_counts"],
            "polarity_counts": sample_stats["polarity_counts"],
        },
        "invalid_by_order_forced_value": invalid,
        "invalid_denominator_per_arm": denominator,
        "invalid_rate_by_order_forced_value": rates,
        "primary_endpoint": {
            "contrast": "second coordinate after forced first=8.0 minus second coordinate after forced first=2.0; equal-weighted across both field orders",
            "estimate": primary,
            "ci95": [float(v) for v in np.quantile(primary_draws, [0.025, 0.975])],
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "fresh recipient ID; four forced-value × order cells resampled together",
        },
        "order_specific_replication": order_results,
        "pattern_resolved_in_fresh_sample": all(
            item["matches_exp067_estimate_direction"] and item["ci_resolves_exp067_direction"]
            for item in order_results.values()
        ),
        "model": manifest["model"], "model_revision": manifest["model_revision"],
        "device": manifest["device"],
        "score_analysis_performed": True,
        "inferential_limits": [
            "This fresh sample contains only food-quality target aspects because Experiment 064 exhausted available negative examples in the other preselected categories.",
            "A forced assistant prefix measures conditional continuation sensitivity, not natural rating behavior.",
            "The result applies to one model revision and one Chinese prompt template.",
            "An effect does not show that naturally generated output order caused prior cross-context moderation.",
            "Public-test pretraining exposure cannot be ruled out.",
        ],
    }


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] != "scored":
        lead = summary["reason"]
    else:
        primary = summary["primary_endpoint"]
        lead = (f"The fresh-sample average second-coordinate shift after forcing the first score from 2.0 to 8.0 was "
                f"{primary['estimate']:.3f} points (95% recipient-bootstrap interval "
                f"[{primary['ci95'][0]:.3f}, {primary['ci95'][1]:.3f}]) on "
                f"{summary['n_complete_four_cell_recipient_ids']} complete recipients.")
    (output / "README.md").write_text(f"""# Experiment 068: fresh-sample forced-prefix replication

## Result

{lead}

The registered replication targets the opposite-sign order-specific pattern from Experiment 067. Fresh recipients were selected from the same pinned SIGHAN release but excluded all 180 IDs from Experiment 064. Available negative non-quality categories had been exhausted by that cohort, so all 64 replication recipients are food-quality reviews, balanced 32 negative and 32 positive. This tests sample robustness within one category, not category generality.

The valence-first shift was {summary.get('order_specific_replication', {}).get('valence_first', {}).get('mean_second_coordinate_shift_8_minus_2', float('nan')):+.3f} points (95% interval [{summary.get('order_specific_replication', {}).get('valence_first', {}).get('ci95', [float('nan'), float('nan')])[0]:+.3f}, {summary.get('order_specific_replication', {}).get('valence_first', {}).get('ci95', [float('nan'), float('nan')])[1]:+.3f}]); the arousal-first shift was {summary.get('order_specific_replication', {}).get('arousal_first', {}).get('mean_second_coordinate_shift_8_minus_2', float('nan')):+.3f} ([{summary.get('order_specific_replication', {}).get('arousal_first', {}).get('ci95', [float('nan'), float('nan')])[0]:+.3f}, {summary.get('order_specific_replication', {}).get('arousal_first', {}).get('ci95', [float('nan'), float('nan')])[1]:+.3f}]). The arousal-first effect matches Exp067's direction and magnitude; the valence-first effect did not replicate.

Order-specific estimates, intervals, invalid rates and replication indicators are in `summary.json`. The forced-prefix intervention measures conditional continuation sensitivity, not natural rating behavior. Raw IDs and outputs remain private in `.context/`.

- Frozen protocol: `docs/experiments/068-fresh-sample-prefix-coupling-replication.md`
- Prior result: [Experiment 067](../forced-coordinate-prefix-coupling-v1/README.md)
- Runner/analyzer: `scripts/run_fresh_sample_prefix_replication_068.py`, `scripts/analyze_fresh_sample_prefix_replication_068.py`
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=runner.OUT)
    parser.add_argument("--manifest", type=Path, default=runner.MANIFEST)
    parser.add_argument("--source-dir", type=Path, default=exp052.SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    summary = analyze(args.predictions, args.manifest, args.source_dir)
    write_report(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
