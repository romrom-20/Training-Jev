"""Analyze the preregistered Experiment 076 paired score distributions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

VALUES = np.arange(1.0, 9.0 + 0.05, 0.1, dtype=np.float64)
INPUT = Path(".context/exp076-private-schema-corpus.jsonl")
OUT_DIR = Path("results/schema-corpus-controls-v1")
BOOTSTRAPS = 10_000
SEED = 20260976
ROLE_LOCATIONS = ("prior_assistant", "prior_user")


def _distribution(row: dict) -> tuple[np.ndarray, float]:
    logp = np.asarray(row["canonical_logprobs"], dtype=np.float64)
    if logp.shape != (81,) or not np.isfinite(logp).all():
        raise ValueError("Every row must have 81 finite canonical numeric-score logprobs")
    probabilities = np.exp(logp - logp.max())
    probabilities /= probabilities.sum()
    raw_mass = np.exp(np.clip(logp, -745, 50)).sum()
    return probabilities, float(raw_mass)


def _expected(row: dict) -> tuple[float, float]:
    p, mass = _distribution(row)
    return float(p @ VALUES), mass


def _interval(values: np.ndarray, rng: np.random.Generator) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap input must be a nonempty finite vector")
    indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[indices].mean(axis=1)
    return {"estimate": float(values.mean()), "ci95": [
        float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    ], "n_recipients": int(len(values))}


def _paired_shift(index: dict, model: str, study: str, case_id: str,
                  order: str, location: str) -> tuple[float, float, float | None]:
    low = index[(model, study, case_id, order, location, 2.0)]
    high = index[(model, study, case_id, order, location, 8.0)]
    low_mean, low_mass = _expected(low)
    high_mean, high_mass = _expected(high)
    low_greedy = low["greedy_target_score"]
    high_greedy = high["greedy_target_score"]
    greedy_shift = (float(high_greedy) - float(low_greedy)
                    if low_greedy is not None and high_greedy is not None else None)
    return high_mean - low_mean, (low_mass + high_mass) / 2.0, greedy_shift


def analyze(input_path: Path = INPUT, output_dir: Path = OUT_DIR) -> dict:
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    index = {}
    for row in rows:
        key = (row["model_key"], row["study"], row["case_id"], row["order"],
               row["anchor_location"], float(row["forced_value"]))
        if key in index:
            raise ValueError(f"Duplicate raw outcome {key}")
        index[key] = row
    if len(index) != 576:
        raise ValueError(f"Expected 576 model-context rows, found {len(index)}")
    models = sorted({row["model_key"] for row in rows})
    if models != ["qwen-0.5b", "qwen-1.5b"]:
        raise ValueError(f"Unexpected model set: {models}")
    if any(row["study"] == "role_control" for row in rows):
        studies = ("role_control", "fresh_aste_prefix")
    else:
        raise ValueError("Missing Experiment 076 studies")

    invalid = {}
    for model in models:
        invalid[model] = {}
        for study in studies:
            study_rows = [row for row in rows if row["model_key"] == model and row["study"] == study]
            invalid[model][study] = {}
            arms = sorted({(row["order"], row["anchor_location"], float(row["forced_value"]))
                           for row in study_rows})
            for order, location, value in arms:
                arm = [row for row in study_rows if row["order"] == order
                       and row["anchor_location"] == location
                       and float(row["forced_value"]) == value]
                n_invalid = sum(row["greedy_target_score"] is None for row in arm)
                invalid[model][study][f"{location}/{order}/{value:.1f}"] = {
                    "invalid": n_invalid, "n": len(arm),
                    "rate": n_invalid / len(arm) if arm else None,
                }

    rng = np.random.default_rng(SEED)

    def invalid_rate(model: str, study: str, order: str, location: str, value: float) -> float:
        key = f"{location}/{order}/{value:.1f}"
        return invalid[model][study][key]["rate"]

    part_a = {}
    part_b = {}
    for model in models:
        part_a[model] = {}
        for order in ("valence_first", "arousal_first"):
            location_effects = {}
            location_greedy = {}
            location_mass = {}
            for location in ("prior_assistant", "prior_user"):
                case_ids = sorted({row["case_id"] for row in rows
                                   if row["model_key"] == model and row["study"] == "role_control"
                                   and row["order"] == order and row["anchor_location"] == location})
                effects, greedy, masses = [], [], []
                for case_id in case_ids:
                    effect, mass, greedy_shift = _paired_shift(
                        index, model, "role_control", case_id, order, location
                    )
                    effects.append(effect)
                    masses.append(mass)
                    if greedy_shift is not None:
                        greedy.append(greedy_shift)
                location_pass = all(invalid_rate(model, "role_control", order, location, value)
                                    <= 0.02 for value in (2.0, 8.0))
                location_effects[location] = (_interval(np.asarray(effects), rng)
                                              if location_pass else None)
                location_mass[location] = float(np.mean(masses))
                location_greedy[location] = (_interval(np.asarray(greedy), rng)
                                             if greedy and location_pass else None)
            assistant_by_case = {}
            user_by_case = {}
            for case_id in sorted({row["case_id"] for row in rows
                                   if row["model_key"] == model and row["study"] == "role_control"
                                   and row["order"] == order}):
                assistant_by_case[case_id] = _paired_shift(
                    index, model, "role_control", case_id, order, "prior_assistant"
                )[0]
                user_by_case[case_id] = _paired_shift(
                    index, model, "role_control", case_id, order, "prior_user"
                )[0]
            paired_diffs = np.asarray([assistant_by_case[key] - user_by_case[key]
                                       for key in sorted(assistant_by_case)])
            contrast_pass = all(
                invalid_rate(model, "role_control", order, location, value) <= 0.02
                for location in ROLE_LOCATIONS for value in (2.0, 8.0)
            )
            part_a[model][order] = {
                "expected_shift_8_minus_2": location_effects,
                "assistant_minus_user_shift": (_interval(paired_diffs, rng)
                                               if contrast_pass else None),
                "mean_valid_canonical_mass": location_mass,
                "greedy_shift_8_minus_2": location_greedy,
            }

        part_b[model] = {}
        effects_by_order = {}
        for order in ("valence_first", "arousal_first"):
            case_ids = sorted({row["case_id"] for row in rows
                               if row["model_key"] == model and row["study"] == "fresh_aste_prefix"
                               and row["order"] == order})
            effects, greedy, masses = [], [], []
            for case_id in case_ids:
                effect, mass, greedy_shift = _paired_shift(
                    index, model, "fresh_aste_prefix", case_id, order, "assistant_prefix"
                )
                effects.append(effect)
                masses.append(mass)
                if greedy_shift is not None:
                    greedy.append(greedy_shift)
            effects_by_order[order] = np.asarray(effects)
            order_pass = all(invalid_rate(model, "fresh_aste_prefix", order,
                                          "assistant_prefix", value) <= 0.02
                             for value in (2.0, 8.0))
            part_b[model][order] = {
                "expected_shift_8_minus_2": (_interval(effects_by_order[order], rng)
                                             if order_pass else None),
                "mean_valid_canonical_mass": float(np.mean(masses)),
                "greedy_shift_8_minus_2": (_interval(np.asarray(greedy), rng)
                                            if greedy and order_pass else None),
            }
        order_difference = effects_by_order["valence_first"] - effects_by_order["arousal_first"]
        both_orders_pass = all(
            invalid_rate(model, "fresh_aste_prefix", order, "assistant_prefix", value) <= 0.02
            for order in ("valence_first", "arousal_first") for value in (2.0, 8.0)
        )
        part_b[model]["valence_first_minus_arousal_first_shift"] = (
            _interval(order_difference, rng) if both_orders_pass else None
        )

    protocol = Path("docs/experiments/076-schema-and-corpus-controls.md")
    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    summary = {
        "experiment": "076-schema-corpus-controls", "protocol_sha256": protocol_hash,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "n_rows": len(rows), "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": SEED, "invalid_rates": invalid,
        "part_a_role_control": part_a, "part_b_fresh_aste": part_b,
        "interpretation_boundary": (
            "Finite-set paired effects in artificial forced-coordinate prompts. "
            "Part A reuses public texts and still bundles role with an acknowledgement turn; "
            "Part B uses 24 new-to-project texts without VA gold labels."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "README.md").write_text(render_readme(summary))
    return summary


def render_readme(summary: dict) -> str:
    def display(item):
        if item is None:
            return "withheld by >2% invalid gate"
        return f"{item['estimate']:.3f} [{item['ci95'][0]:.3f}, {item['ci95'][1]:.3f}]"

    lines = [
        "# Experiment 076: role and corpus controls",
        "",
        "## Result",
        "",
        "Paired 8−2 expected-score shifts use the normalized probability distribution over the 81 registered one-decimal scores. Intervals resample recipient sentences, not output strings. The sample is small; these are finite-set estimates.",
        "",
        "### A. Same final request and one-key output schema",
        "",
        "| Model | Target order | Assistant prior shift | User prior shift | Assistant − user |",
        "|---|---|---:|---:|---:|",
    ]
    for model, by_order in summary["part_a_role_control"].items():
        for order, stats in by_order.items():
            short_order = "Valence first" if order == "valence_first" else "Arousal first"
            assistant = stats["expected_shift_8_minus_2"]["prior_assistant"]
            user = stats["expected_shift_8_minus_2"]["prior_user"]
            diff = stats["assistant_minus_user_shift"]
            lines.append(
                f"| {model} | {short_order} | {display(assistant)} | {display(user)} | {display(diff)} |"
            )
    lines += [
        "",
        "### B. Fresh SemEval-2014 laptop sentences",
        "",
        "| Model | Valence-first shift | Arousal-first shift | Difference |",
        "|---|---:|---:|---:|",
    ]
    for model, by_order in summary["part_b_fresh_aste"].items():
        vf = by_order["valence_first"]["expected_shift_8_minus_2"]
        af = by_order["arousal_first"]["expected_shift_8_minus_2"]
        diff = by_order["valence_first_minus_arousal_first_shift"]
        lines.append(
            f"| {model} | {display(vf)} | {display(af)} | {display(diff)} |"
        )
    lines += [
        "",
        "Greedy shifts, valid probability mass, and per-arm parsing rates are in `summary.json`.",
        "",
        "## Interpretation and limits",
        "",
        summary["interpretation_boundary"],
        "",
        "A shift that persists in Part A would rule out the two-key-versus-one-key schema change as the sole explanation for Experiment 075, but it would not establish a pure causal role effect. A Part B pattern would be a small transfer signal; a null or interval spanning zero would leave the interaction unresolved rather than prove equivalence.",
        "",
        "The results do not establish novelty against the broader literature on anchoring, output order, or structured-output effects. See [Experiment 076 protocol](../../docs/experiments/076-schema-and-corpus-controls.md), [Kapetanovic et al. (2026)](https://arxiv.org/abs/2608.25869), [Chen et al. (2024)](https://arxiv.org/abs/2406.02863), and [Parikh (2026)](https://arxiv.org/abs/2607.18476).",
        "",
        "## Reproduction",
        "",
        "Run the frozen protocol with `PYTHONPATH=scripts:src .venv/bin/python scripts/run_schema_corpus_controls_076.py`, then analyze with `PYTHONPATH=scripts:src .venv/bin/python scripts/analyze_schema_corpus_controls_076.py`.",
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
