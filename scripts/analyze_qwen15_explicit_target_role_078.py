"""Analyze Experiment 078's named-target role control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

INPUT = Path(".context/exp078-private-qwen15-role-control.jsonl")
OUT_DIR = Path("results/qwen15-explicit-target-role-v1")
PROTOCOL = Path("docs/experiments/078-qwen15-explicit-target-role-replication.md")
VALUES = np.arange(1.0, 9.0 + .05, .1, dtype=np.float64)
BOOTSTRAPS = 10_000
SEED = 20260978
ORDERS = ("valence_first", "arousal_first")
LOCATIONS = ("prior_assistant", "prior_user")


def _score(row: dict) -> tuple[float, float]:
    logp = np.asarray(row["canonical_logprobs"], dtype=np.float64)
    if logp.shape != (81,) or not np.isfinite(logp).all():
        raise ValueError("Expected 81 finite score likelihoods")
    mass = float(np.exp(np.clip(logp, -745, 50)).sum())
    probabilities = np.exp(logp - logp.max())
    probabilities /= probabilities.sum()
    return float(probabilities @ VALUES), mass


def _interval(values: np.ndarray, rng: np.random.Generator) -> dict:
    values = np.asarray(values, dtype=np.float64)
    indexes = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[indexes].mean(axis=1)
    return {"estimate": float(values.mean()), "ci95": [
        float(np.quantile(means, .025)), float(np.quantile(means, .975))
    ], "n_recipients": int(len(values))}


def analyze(input_path: Path = INPUT, output_dir: Path = OUT_DIR) -> dict:
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    index = {}
    for row in rows:
        key = (row["case_id"], row["order"], row["anchor_location"], float(row["forced_value"]))
        if key in index:
            raise ValueError(f"Duplicate cell {key}")
        index[key] = row
    if len(index) != 192:
        raise ValueError(f"Expected 192 rows, found {len(index)}")

    invalid_rates = {}
    gates = {}
    for order in ORDERS:
        invalid_rates[order] = {}
        bad_arm = False
        for location in LOCATIONS:
            for value in (2.0, 8.0):
                arm = [row for row in rows if row["order"] == order
                       and row["anchor_location"] == location
                       and float(row["forced_value"]) == value]
                invalid = sum(row["greedy_target_score"] is None for row in arm)
                rate = invalid / len(arm)
                invalid_rates[order][f"{location}/{value:.1f}"] = {
                    "invalid": invalid, "n": len(arm), "rate": rate
                }
                bad_arm |= rate > .02
        gates[order] = not bad_arm

    rng = np.random.default_rng(SEED)
    results = {}
    for order in ORDERS:
        case_ids = sorted({row["case_id"] for row in rows if row["order"] == order})
        effects, greedy, masses, by_case = {}, {}, {}, {}
        for location in LOCATIONS:
            shifts, greedy_shifts, valid_masses = [], [], []
            for case_id in case_ids:
                low = index[(case_id, order, location, 2.0)]
                high = index[(case_id, order, location, 8.0)]
                low_mean, low_mass = _score(low)
                high_mean, high_mass = _score(high)
                shift = high_mean - low_mean
                shifts.append(shift)
                valid_masses.append((low_mass + high_mass) / 2)
                by_case.setdefault(case_id, {})[location] = shift
                if low["greedy_target_score"] is not None and high["greedy_target_score"] is not None:
                    greedy_shifts.append(float(high["greedy_target_score"])
                                          - float(low["greedy_target_score"]))
            effects[location] = _interval(np.asarray(shifts), rng) if gates[order] else None
            greedy[location] = (_interval(np.asarray(greedy_shifts), rng)
                                if gates[order] and len(greedy_shifts) == len(case_ids) else None)
            masses[location] = float(np.mean(valid_masses))
        differences = np.asarray([by_case[case]["prior_assistant"]
                                  - by_case[case]["prior_user"] for case in case_ids])
        results[order] = {
            "validity_gate_passed": gates[order],
            "expected_shift_8_minus_2": effects,
            "assistant_minus_user_shift": _interval(differences, rng) if gates[order] else None,
            "greedy_shift_8_minus_2": greedy,
            "mean_valid_canonical_mass": masses,
        }
    summary = {
        "experiment": "078-qwen15-explicit-target-role-replication",
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "n_rows": len(rows), "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": SEED, "invalid_rates": invalid_rates, "results": results,
        "interpretation_boundary": (
            "One 1.5B model on 24 reused public reviews. The within-dialogue source comparison "
            "still bundles role with the acknowledgement turn."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "README.md").write_text(render_readme(summary))
    return summary


def render_readme(summary: dict) -> str:
    def display(item):
        return "withheld by >2% invalid gate" if item is None else (
            f"{item['estimate']:.3f} [{item['ci95'][0]:.3f}, {item['ci95'][1]:.3f}]")

    lines = [
        "# Experiment 078: Qwen-1.5B named-target role control",
        "",
        "## Result",
        "",
        "The final request explicitly names the fixed and target coordinates. Recipient-bootstrap intervals use 24 paired reviews.",
        "",
        "| Target order | Validity gate | Assistant-prior shift | User-prior shift | Assistant − user |",
        "|---|---|---:|---:|---:|",
    ]
    for order, stats in summary["results"].items():
        label = "Valence first" if order == "valence_first" else "Arousal first"
        assistant = stats["expected_shift_8_minus_2"]["prior_assistant"]
        user = stats["expected_shift_8_minus_2"]["prior_user"]
        diff = stats["assistant_minus_user_shift"]
        lines.append(f"| {label} | {stats['validity_gate_passed']} | {display(assistant)} | {display(user)} | {display(diff)} |")
    lines += [
        "",
        "Greedy shifts, valid canonical probability mass, and per-arm invalid rates are in `summary.json`.",
        "",
        "## Interpretation and limits",
        "",
        summary["interpretation_boundary"],
        "",
        "Compare with the [Qwen-0.5B Experiment 077 result](../explicit-target-role-control-v1/README.md) and [078 protocol](../../docs/experiments/078-qwen15-explicit-target-role-replication.md). This same-item model-size comparison is descriptive, not independent replication.",
        "",
        "Raw prompts and outputs remain in ignored `.context/`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
