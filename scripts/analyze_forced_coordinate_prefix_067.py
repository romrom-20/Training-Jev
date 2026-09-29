"""Analyze forced-prefix effects on the second generated VA coordinate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import run_forced_coordinate_prefix_067 as runner
import run_sighan_chinese_decoder_transfer_052 as exp052
import run_sighan_instruction_language_066 as exp066
import run_sighan_output_key_order_064 as exp064
from run_small_model_decoder_factorial_048 import sha256

OUT = Path("results/forced-coordinate-prefix-coupling-v1")
PARENT = runner.PARENT
N_BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260967


def index_rows(path: Path, manifest: dict) -> dict[tuple, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [(row["case_id"], row["order"], row["forced_first_value"], row["decoder"])
            for row in rows]
    if len(rows) != 256 or len(set(keys)) != 256:
        raise ValueError(f"Expected 256 unique Exp067 output cells; found {len(rows)}")
    if manifest.get("n_outputs") != 256 or manifest.get("output_sha256") != sha256(path.read_bytes()):
        raise ValueError("Experiment 067 outputs do not match the run manifest")
    return dict(zip(keys, rows))


def analyze(predictions: Path = runner.OUT, manifest_path: Path = runner.MANIFEST,
            source_dir: Path = exp052.SOURCE_DIR,
            bootstrap_replicates: int = N_BOOTSTRAPS) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("experiment") != "067-forced-coordinate-prefix-coupling":
        raise ValueError("Unexpected Experiment 067 manifest")
    if manifest.get("protocol_sha256") != runner.PROTOCOL_SHA256:
        raise ValueError("Manifest does not match frozen Experiment 067 protocol")
    parent = runner.validate_parent(PARENT)
    source_rows, source_hashes = exp052.read_source(source_dir)
    cohort_180, _ = exp064.select_cases(exp052.select_cases(source_rows))
    cohort_88, _ = exp066.select_cases(cohort_180)
    cases, sample_stats = runner.select_cases(cohort_88)
    if sample_stats["recipient_ids_sha256"] != runner.EXPECTED_IDS_SHA256:
        raise ValueError("Experiment 067 IDs differ from the frozen protocol")
    ids = sorted(case["case_id"] for case in cases)
    new = index_rows(predictions, manifest)
    expected = {
        (case_id, order, forced, runner.DECODER)
        for case_id in ids for order in runner.ORDERS for forced in runner.FORCED_VALUES
    }
    if set(new) != expected:
        raise ValueError("Experiment 067 outputs differ from the frozen recipient/cell grid")
    if manifest.get("parent_output_sha256") != runner.PARENT_SHA256:
        raise ValueError("Run manifest records an unexpected Exp066 parent hash")
    if manifest.get("source_hashes") != source_hashes:
        raise ValueError("Run manifest records unexpected pinned SIGHAN source hashes")
    for key, row in new.items():
        order, forced = key[1], key[2]
        first_axis = 0 if order == "valence_first" else 1
        if row["prediction"] is not None and float(row["prediction"][first_axis]) != forced:
            raise ValueError(f"Forced first coordinate changed in parsed output: {key}")

    invalid_by_arm = {
        order: {
            str(forced): sum(new[(case_id, order, forced, runner.DECODER)]["prediction"] is None
                             for case_id in ids)
            for forced in runner.FORCED_VALUES
        }
        for order in runner.ORDERS
    }
    denominator = len(ids)
    rate_by_arm = {
        order: {forced: count / denominator for forced, count in arms.items()}
        for order, arms in invalid_by_arm.items()
    }
    provenance = {
        "experiment": "067-forced-coordinate-prefix-coupling",
        "protocol_sha256": manifest["protocol_sha256"],
        "output_sha256": manifest["output_sha256"],
        "parent_output_sha256": runner.PARENT_SHA256,
        "source_hashes": source_hashes,
    }
    if any(rate > 0.02 for arms in rate_by_arm.values() for rate in arms.values()):
        return {
            **provenance, "status": "protocol_execution_failure",
            "reason": "At least one forced-value × output-order arm exceeded the 2% invalid-output gate",
            "invalid_by_order_forced_value": invalid_by_arm,
            "invalid_denominator_per_arm": denominator,
            "invalid_rate_by_order_forced_value": rate_by_arm,
            "score_analysis_performed": False,
        }

    complete = [case_id for case_id in ids if all(
        new[(case_id, order, forced, runner.DECODER)]["prediction"] is not None
        for order in runner.ORDERS for forced in runner.FORCED_VALUES
    )]
    if not complete:
        raise ValueError("No recipient has all four valid forced-prefix cells")
    second_axis = {"valence_first": 1, "arousal_first": 0}
    shifts = {}
    for order in runner.ORDERS:
        axis = second_axis[order]
        shifts[order] = np.array([
            float(new[(case_id, order, 8.0, runner.DECODER)]["prediction"][axis])
            - float(new[(case_id, order, 2.0, runner.DECODER)]["prediction"][axis])
            for case_id in complete
        ])
    order_draws = {order: np.empty(bootstrap_replicates) for order in runner.ORDERS}
    primary_draws = np.empty(bootstrap_replicates)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(complete)
    for i in range(bootstrap_replicates):
        sample = rng.integers(0, n, size=n)
        for order in runner.ORDERS:
            order_draws[order][i] = float(np.mean(shifts[order][sample]))
        primary_draws[i] = float(np.mean([
            order_draws[order][i] for order in runner.ORDERS
        ]))
    point_by_order = {order: float(np.mean(values)) for order, values in shifts.items()}
    primary = float(np.mean(list(point_by_order.values())))

    unforced = {}
    for order in runner.ORDERS:
        values = []
        for case_id in ids:
            row = parent.get((case_id, order, 0, exp066.OWN, runner.DECODER))
            if row is not None and row["prediction"] is not None:
                values.append(float(row["prediction"][second_axis[order]]))
        unforced[order] = {
            "n_valid_unforced_ids": len(values),
            "mean_unforced_second_coordinate": float(np.mean(values)) if values else None,
        }

    return {
        **provenance,
        "status": "scored",
        "n_recipient_ids_selected": len(ids),
        "n_complete_four_cell_recipient_ids": n,
        "n_new_outputs": manifest["n_outputs"],
        "invalid_by_order_forced_value": invalid_by_arm,
        "invalid_denominator_per_arm": denominator,
        "invalid_rate_by_order_forced_value": rate_by_arm,
        "primary_endpoint": {
            "contrast": "second coordinate after forced first=8.0 minus second coordinate after forced first=2.0; equal-weighted across both field orders",
            "estimate": primary,
            "ci95": [float(v) for v in np.quantile(primary_draws, [0.025, 0.975])],
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "recipient ID; all four forced-value × order cells resampled together",
        },
        "order_specific_secondary": {
            order: {
                "mean_second_coordinate_shift_8_minus_2": point_by_order[order],
                "ci95": [float(v) for v in np.quantile(order_draws[order], [0.025, 0.975])],
            } for order in runner.ORDERS
        },
        "unforced_exp066_context": unforced,
        "model": manifest["model"], "model_revision": manifest["model_revision"],
        "device": manifest["device"],
        "score_analysis_performed": True,
        "inferential_limits": [
            "A forced assistant prefix measures conditional continuation sensitivity, not natural rating behavior.",
            "The result applies to one model revision, one Chinese prompt template and this selected SIGHAN cohort.",
            "An effect does not show that naturally generated output order caused Experiment 064's cross-context moderation.",
            "The Experiment 066 unforced comparison is secondary and may have fewer valid IDs.",
        ],
    }


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] != "scored":
        lead = summary["reason"]
    else:
        primary = summary["primary_endpoint"]
        lead = (f"The average second-coordinate shift after forcing the first value from 2.0 to 8.0 was "
                f"{primary['estimate']:.3f} VA points (95% recipient-bootstrap interval "
                f"[{primary['ci95'][0]:.3f}, {primary['ci95'][1]:.3f}]) on "
                f"{summary['n_complete_four_cell_recipient_ids']} complete recipients.")
    (output / "README.md").write_text(f"""# Experiment 067: forced-coordinate prefix coupling

## Result

{lead}

This is a counterfactual continuation test: the review and user prompt stay fixed, while an assistant output prefix supplies either a low or high first VA score and the model generates the other score. It measures how the continuation responds to a forced prefix, not how accurate a normal score is. The test uses 64 hash-selected SIGHAN reviews and 256 new greedy continuations on Qwen2.5-3B.

The primary contrast and order-specific results are in `summary.json`. Raw review text, IDs, prefixes and model outputs remain private in `.context/`.

- Frozen protocol: `docs/experiments/067-forced-coordinate-prefix-coupling.md`
- Literature boundary: `docs/experiments/064-literature-boundary-audit.md`
- Runner/analyzer: `scripts/run_forced_coordinate_prefix_067.py`, `scripts/analyze_forced_coordinate_prefix_067.py`
- Related prompt-language experiment: [066 aggregate bundle](../sighan-instruction-language-control-v1/README.md)
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
