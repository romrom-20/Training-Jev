"""Exploratory paired analysis combining Experiment 076–078 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

INPUTS = {
    "generic": Path(".context/exp076-private-schema-corpus.jsonl"),
    "explicit_0.5b": Path(".context/exp077-private-explicit-target-role.jsonl"),
    "explicit_1.5b": Path(".context/exp078-private-qwen15-role-control.jsonl"),
}
OUT_DIR = Path("results/target-wording-role-factorial-v1")
PROTOCOL_NOTE = Path("docs/analysis/076-078-target-wording-factorial.md")
VALUES = np.arange(1.0, 9.0 + .05, .1, dtype=np.float64)
BOOTSTRAPS = 10_000
SEED = 20260979
MODELS = ("qwen-0.5b", "qwen-1.5b")
WORDINGS = ("generic", "explicit")
LOCATIONS = ("prior_assistant", "prior_user")
ORDERS = ("valence_first", "arousal_first")


def _expected(row: dict) -> float:
    logp = np.asarray(row["canonical_logprobs"], dtype=np.float64)
    if logp.shape != (81,) or not np.isfinite(logp).all():
        raise ValueError("Expected 81 finite canonical score log probabilities")
    probs = np.exp(logp - logp.max())
    probs /= probs.sum()
    return float(probs @ VALUES)


def _interval(values: np.ndarray, rng: np.random.Generator) -> dict:
    values = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[draws].mean(axis=1)
    return {"estimate": float(values.mean()), "ci95": [
        float(np.quantile(means, .025)), float(np.quantile(means, .975))
    ], "n_recipients": int(len(values))}


def load_rows(paths: dict[str, Path] = INPUTS) -> tuple[list[dict], dict]:
    rows = []
    input_hashes = {}
    for label, path in paths.items():
        raw_bytes = path.read_bytes()
        input_hashes[label] = hashlib.sha256(raw_bytes).hexdigest()
        source_rows = [json.loads(line) for line in raw_bytes.decode().splitlines() if line.strip()]
        for row in source_rows:
            if label == "generic":
                if row.get("study") != "role_control":
                    continue
                model = row["model_key"]
                wording = "generic"
            else:
                model = row["model_key"]
                wording = "explicit"
                expected_model = "qwen-0.5b" if label == "explicit_0.5b" else "qwen-1.5b"
                if model != expected_model:
                    raise ValueError(f"Unexpected model in {label}: {model}")
            rows.append({**row, "model": model, "wording": wording})
    return rows, input_hashes


def analyze(paths: dict[str, Path] = INPUTS, output_dir: Path = OUT_DIR) -> dict:
    rows, input_hashes = load_rows(paths)
    index = {}
    for row in rows:
        key = (row["model"], row["wording"], row["case_id"], row["order"],
               row["anchor_location"], float(row["forced_value"]))
        if key in index:
            raise ValueError(f"Duplicate factorial cell {key}")
        index[key] = row
    expected_n = 24 * 2 * 2 * 2 * 2 * 2
    if len(index) != expected_n:
        raise ValueError(f"Expected {expected_n} factorial rows, found {len(index)}")

    invalid = {}
    for model in MODELS:
        invalid[model] = {}
        for wording in WORDINGS:
            invalid[model][wording] = {}
            for order in ORDERS:
                invalid[model][wording][order] = {}
                for location in LOCATIONS:
                    invalid[model][wording][order][location] = {}
                    for value in (2.0, 8.0):
                        arm = [row for row in rows if row["model"] == model
                               and row["wording"] == wording and row["order"] == order
                               and row["anchor_location"] == location
                               and float(row["forced_value"]) == value]
                        n_bad = sum(row["greedy_target_score"] is None for row in arm)
                        invalid[model][wording][order][location][str(value)] = {
                            "invalid": n_bad, "n": len(arm),
                            "rate": n_bad / len(arm) if arm else None,
                        }

    def gate(model: str, wording: str, order: str, location: str) -> bool:
        return all(invalid[model][wording][order][location][str(value)]["rate"] <= .02
                   for value in (2.0, 8.0))

    recipient_ids = sorted({row["case_id"] for row in rows})
    rng = np.random.default_rng(SEED)
    results = {}
    for model in MODELS:
        results[model] = {}
        for order in ORDERS:
            per_case = {}
            cell_means = {}
            for wording in WORDINGS:
                cell_means[wording] = {}
                for location in LOCATIONS:
                    shifts = []
                    for case_id in recipient_ids:
                        low = index[(model, wording, case_id, order, location, 2.0)]
                        high = index[(model, wording, case_id, order, location, 8.0)]
                        effect = _expected(high) - _expected(low)
                        per_case.setdefault(case_id, {}).setdefault(wording, {})[location] = effect
                        shifts.append(effect)
                    cell_means[wording][location] = (
                        _interval(np.asarray(shifts), rng) if gate(model, wording, order, location)
                        else None
                    )

            wording_effects = {}
            for location in LOCATIONS:
                valid = all(gate(model, wording, order, location) for wording in WORDINGS)
                deltas = np.asarray([
                    per_case[case]["explicit"][location] - per_case[case]["generic"][location]
                    for case in recipient_ids
                ])
                wording_effects[location] = _interval(deltas, rng) if valid else None

            role_effects = {}
            for wording in WORDINGS:
                valid = all(gate(model, wording, order, location) for location in LOCATIONS)
                deltas = np.asarray([
                    per_case[case][wording]["prior_assistant"]
                    - per_case[case][wording]["prior_user"] for case in recipient_ids
                ])
                role_effects[wording] = _interval(deltas, rng) if valid else None

            did_valid = all(gate(model, wording, order, location)
                            for wording in WORDINGS for location in LOCATIONS)
            did = np.asarray([
                (per_case[case]["explicit"]["prior_assistant"]
                 - per_case[case]["explicit"]["prior_user"])
                - (per_case[case]["generic"]["prior_assistant"]
                   - per_case[case]["generic"]["prior_user"])
                for case in recipient_ids
            ])
            results[model][order] = {
                "expected_shift_8_minus_2": cell_means,
                "explicit_minus_generic_shift": wording_effects,
                "assistant_minus_user_shift": role_effects,
                "wording_by_role_difference_in_differences": _interval(did, rng) if did_valid else None,
            }

    summary = {
        "analysis": "post-hoc target-wording-by-anchor-source factorial, Experiments 076–078",
        "preregistered": False,
        "analysis_note_sha256": hashlib.sha256(PROTOCOL_NOTE.read_bytes()).hexdigest(),
        "input_sha256": input_hashes, "n_rows": len(rows),
        "n_recipients": len(recipient_ids), "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": SEED, "invalid_rates": invalid, "results": results,
        "interpretation_boundary": (
            "Same 24 items and two fixed models; post-hoc cell contrasts. Generic-wording "
            "Qwen-0.5B prior-user/arousal-first cell fails the preregistered validity gate, "
            "so dependent contrasts are withheld."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "README.md").write_text(render_readme(summary))
    return summary


def _format(item: dict | None) -> str:
    if item is None:
        return "withheld by validity gate"
    return f"{item['estimate']:.3f} [{item['ci95'][0]:.3f}, {item['ci95'][1]:.3f}]"


def render_readme(summary: dict) -> str:
    lines = [
        "# Exploratory target-wording × anchor-source factorial",
        "",
        "## Result",
        "",
        "This is a post-hoc analysis of all role-control cells in Experiments 076–078. Each shift is the paired 8−2 expected-score change over normalized canonical one-decimal JSON score completions. Intervals bootstrap the 24 recipients.",
        "",
        "### Role-source contrast by model and order",
        "",
        "| Model | Target order | Generic wording: assistant − user | Explicit wording: assistant − user | Explicit − generic difference-in-differences |",
        "|---|---|---:|---:|---:|",
    ]
    for model, orders in summary["results"].items():
        for order, stats in orders.items():
            label = "Valence first" if order == "valence_first" else "Arousal first"
            generic = stats["assistant_minus_user_shift"]["generic"]
            explicit = stats["assistant_minus_user_shift"]["explicit"]
            did = stats["wording_by_role_difference_in_differences"]
            lines.append(
                f"| {model} | {label} | {_format(generic)} | {_format(explicit)} | {_format(did)} |"
            )
    lines += [
        "",
        "Full cell shifts, wording contrasts, and invalid rates are in `summary.json`.",
        "",
        "## Interpretation and limits",
        "",
        summary["interpretation_boundary"],
        "",
        "The experiment components are [076](../../docs/experiments/076-schema-and-corpus-controls.md), [077](../../docs/experiments/077-explicit-target-role-control.md), and [078](../../docs/experiments/078-qwen15-explicit-target-role-replication.md). This exploratory combined analysis cannot establish that the prompt components act independently, and the user-role comparison still includes an acknowledgement turn.",
        "",
        "Literature already documents generic score anchoring, output-order effects, and sensitivity to structured formats; this is a narrow audit of how those factors interact in one forced-coordinate affect task. See [Kapetanovic et al. (2026)](https://arxiv.org/abs/2608.25869), [Chen et al. (2024)](https://arxiv.org/abs/2406.02863), and [Parikh (2026)](https://arxiv.org/abs/2607.18476).",
        "",
        "Raw model outputs remain in ignored `.context/`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    print(json.dumps(analyze(output_dir=args.output_dir), indent=2))


if __name__ == "__main__":
    main()
