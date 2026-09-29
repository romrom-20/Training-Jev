"""Analyze Experiment 077's schema-matched role comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

INPUT = Path(".context/exp077-private-explicit-target-role.jsonl")
OUT_DIR = Path("results/explicit-target-role-control-v1")
VALUES = np.arange(1.0, 9.0 + 0.05, 0.1, dtype=np.float64)
BOOTSTRAPS = 10_000
SEED = 20260977
LOCATIONS = ("prior_assistant", "prior_user")
ORDERS = ("valence_first", "arousal_first")


def _score(row: dict) -> tuple[float, float]:
    logp = np.asarray(row["canonical_logprobs"], dtype=np.float64)
    if logp.shape != (81,) or not np.isfinite(logp).all():
        raise ValueError("Expected 81 finite score likelihoods")
    mass = float(np.exp(np.clip(logp, -745, 50)).sum())
    probs = np.exp(logp - logp.max())
    probs /= probs.sum()
    return float(probs @ VALUES), mass


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
    for order in ORDERS:
        invalid_rates[order] = {}
        for location in LOCATIONS:
            for value in (2.0, 8.0):
                arm = [r for r in rows if r["order"] == order
                       and r["anchor_location"] == location
                       and float(r["forced_value"]) == value]
                bad = sum(row["greedy_target_score"] is None for row in arm)
                invalid_rates[order][f"{location}/{value:.1f}"] = {
                    "invalid": bad, "n": len(arm), "rate": bad / len(arm)
                }
    gates = {order: all(invalid_rates[order][f"{location}/{value:.1f}"]["rate"] <= .02
                        for location in LOCATIONS for value in (2.0, 8.0))
             for order in ORDERS}

    rng = np.random.default_rng(SEED)
    results = {}
    for order in ORDERS:
        recipient_ids = sorted({row["case_id"] for row in rows if row["order"] == order})
        effects, greedy, mass_by_location = {}, {}, {}
        effects_by_case = {}
        for location in LOCATIONS:
            loc_effects, loc_greedy, masses = [], [], []
            for case_id in recipient_ids:
                low = index[(case_id, order, location, 2.0)]
                high = index[(case_id, order, location, 8.0)]
                low_mean, low_mass = _score(low)
                high_mean, high_mass = _score(high)
                effect = high_mean - low_mean
                loc_effects.append(effect)
                masses.append((low_mass + high_mass) / 2)
                g0, g1 = low["greedy_target_score"], high["greedy_target_score"]
                if g0 is not None and g1 is not None:
                    loc_greedy.append(float(g1) - float(g0))
                effects_by_case.setdefault(case_id, {})[location] = effect
            effects[location] = (_interval(np.asarray(loc_effects), rng)
                                 if gates[order] else None)
            greedy[location] = (_interval(np.asarray(loc_greedy), rng)
                                if gates[order] and len(loc_greedy) == len(recipient_ids) else None)
            mass_by_location[location] = float(np.mean(masses))
        difference = np.asarray([
            effects_by_case[case_id]["prior_assistant"] - effects_by_case[case_id]["prior_user"]
            for case_id in recipient_ids
        ])
        results[order] = {
            "validity_gate_passed": gates[order],
            "expected_shift_8_minus_2": effects,
            "assistant_minus_user_shift": _interval(difference, rng) if gates[order] else None,
            "greedy_shift_8_minus_2": greedy,
            "mean_valid_canonical_mass": mass_by_location,
        }
    protocol = Path("docs/experiments/077-explicit-target-role-control.md")
    summary = {
        "experiment": "077-explicit-target-role-control",
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "n_rows": len(rows), "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": SEED, "invalid_rates": invalid_rates, "results": results,
        "interpretation_boundary": (
            "24 reused public reviews, one model, and a two-location conversation manipulation. "
            "The named-target request diagnoses Experiment 076's invalid response format; "
            "it does not isolate a general role effect."
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
        "# Experiment 077: name the target axis",
        "",
        "## Result",
        "",
        "The final request names the fixed and target dimensions and requires the target dimension as the sole JSON key. Intervals resample 24 paired recipients.",
        "",
        "| Target order | Validity gate | Assistant-prior shift | User-prior shift | Assistant − user |",
        "|---|---|---:|---:|---:|",
    ]
    for order, stats in summary["results"].items():
        short_order = "Valence first" if order == "valence_first" else "Arousal first"
        a = stats["expected_shift_8_minus_2"]["prior_assistant"]
        u = stats["expected_shift_8_minus_2"]["prior_user"]
        d = stats["assistant_minus_user_shift"]
        lines.append(f"| {short_order} | {stats['validity_gate_passed']} | {display(a)} | {display(u)} | {display(d)} |")
    lines += [
        "",
        "Greedy shifts, valid canonical probability mass, and each arm's invalid rate are in `summary.json`.",
        "",
        "## Interpretation and limits",
        "",
        summary["interpretation_boundary"],
        "",
        "Compare with the [Experiment 076 result](../schema-corpus-controls-v1/README.md) and the [077 protocol](../../docs/experiments/077-explicit-target-role-control.md). A validity improvement would show the earlier generic target wording contributed to a format problem; it would not make the role contrast a pure source manipulation.",
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
