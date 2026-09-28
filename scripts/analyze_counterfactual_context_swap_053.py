"""Analyze the frozen 3B counterfactual review-context swap experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DECODERS = ("finite_grid", "free_greedy")
CONDITIONS = ("opinion_masked", "swapped_context")
EXPECTED_IDS = 217
SEED = 20260953
BOOTSTRAPS = 10_000


def load_rows(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keyed = {(row["case_id"], row["condition"], row["decoder"]): row for row in rows}
    expected = EXPECTED_IDS * len(DECODERS)
    if len(rows) != expected or len(keyed) != expected:
        raise ValueError(f"Expected {expected} unique swapped outputs, got {len(rows)}")
    if {row["condition"] for row in rows} != {"swapped_context"}:
        raise ValueError("Unexpected counterfactual condition")
    if {row["decoder"] for row in rows} != set(DECODERS):
        raise ValueError("Unexpected decoder set")
    return keyed


def _score(rows: dict, ids: list[str], decoder: str, condition: str) -> float:
    squared = [
        (float(rows[(case_id, condition, decoder)]["prediction"][dim])
         - float(rows[(case_id, condition, decoder)]["gold"][dim])) ** 2
        for case_id in ids
        for dim in (0, 1)
    ]
    return float(np.sqrt(np.mean(squared)))


def analyze(swapped: dict, matched: dict, manifest: dict | None = None) -> dict:
    ids = sorted({key[0] for key in swapped})
    if len(ids) != EXPECTED_IDS:
        raise ValueError(f"Expected {EXPECTED_IDS} recipients, got {len(ids)}")
    expected_swapped = {
        (case_id, "swapped_context", decoder)
        for case_id in ids for decoder in DECODERS
    }
    expected_matched = {
        (case_id, "opinion_masked", decoder)
        for case_id in ids for decoder in DECODERS
    }
    if set(swapped) != expected_swapped or not expected_matched.issubset(matched):
        raise ValueError("Every recipient must have both decoder arms in matched and swapped conditions")
    for case_id in ids:
        for decoder in DECODERS:
            own = matched[(case_id, "opinion_masked", decoder)]
            wrong = swapped[(case_id, "swapped_context", decoder)]
            if own["gold"] != wrong["gold"]:
                raise ValueError(f"Recipient gold mismatch for {case_id}/{decoder}")
            for row in (own, wrong):
                prediction = row["prediction"]
                if prediction is not None and (
                    len(prediction) != 2 or not np.isfinite(prediction).all()
                ):
                    raise ValueError("Non-finite VA prediction")

    invalid = {
        decoder: sum(swapped[(case_id, "swapped_context", decoder)]["prediction"] is None
                     for case_id in ids)
        for decoder in DECODERS
    }
    invalid_free = invalid["free_greedy"]
    invalid_free_rate = invalid_free / (EXPECTED_IDS * 2)
    summary = {
        "experiment": "053-counterfactual-review-context-swap",
        "interpretation": "adaptive fixed-donor counterfactual diagnostic; exploratory",
        "n_recipient_ids": len(ids),
        "n_new_outputs": len(swapped),
        "invalid_swapped_by_decoder": invalid,
        "invalid_free_outputs": invalid_free,
        "invalid_free_rate": invalid_free_rate,
        "bootstrap_unit": "recipient sentence ID, conditional on the frozen donor permutation",
        "analysis_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "score_analysis_performed": False,
    }
    if manifest is not None:
        summary["run_provenance"] = {
            key: manifest[key]
            for key in (
                "protocol_sha256", "source_revision", "source_sha256", "parent_output_sha256",
                "model", "model_revision", "device", "donor_assignment_sha256", "output_sha256",
                "generation_seconds_this_process_only",
            )
        }
    if invalid_free_rate > 0.02:
        summary.update({
            "status": "protocol_execution_failure",
            "reason": "Swapped-context free-output invalid rate exceeded 2%; score contrasts withheld",
        })
        return summary

    complete = [
        case_id for case_id in ids
        if all(
            swapped[(case_id, "swapped_context", decoder)]["prediction"] is not None
            and matched[(case_id, "opinion_masked", decoder)]["prediction"] is not None
            for decoder in DECODERS
        )
    ]
    if not complete:
        raise ValueError("No complete recipient IDs available for paired scoring")

    def advantage(decoder: str, sample: list[str]) -> float:
        return _score(swapped, sample, decoder, "swapped_context") - _score(
            matched, sample, decoder, "opinion_masked"
        )

    estimates = {}
    for index, decoder in enumerate(DECODERS):
        point = advantage(decoder, complete)
        rng = np.random.default_rng(SEED + index)
        draws = np.empty(BOOTSTRAPS)
        for draw in range(BOOTSTRAPS):
            sample = rng.choice(complete, size=len(complete), replace=True).tolist()
            draws[draw] = advantage(decoder, sample)
        estimates[decoder] = {
            "matched_review_advantage_rmse": point,
            "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
            "n_recipient_ids": len(complete),
        }

    rng = np.random.default_rng(SEED + 2)
    interaction_draws = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        sample = rng.choice(complete, size=len(complete), replace=True).tolist()
        interaction_draws[draw] = advantage("finite_grid", sample) - advantage("free_greedy", sample)
    interaction = estimates["finite_grid"]["matched_review_advantage_rmse"] - estimates[
        "free_greedy"
    ]["matched_review_advantage_rmse"]
    summary.update({
        "status": "scored",
        "score_analysis_performed": True,
        "n_complete_recipient_ids": len(complete),
        "matched_review_advantage": estimates,
        "primary_decoder_interaction": {
            "contrast": "matched-review advantage(finite_grid) - matched-review advantage(free_greedy)",
            "estimate": interaction,
            "ci95": [float(value) for value in np.quantile(interaction_draws, [0.025, 0.975])],
            "n_recipient_ids": len(complete),
            "confirmatory": False,
        },
        "limitations": [
            "One deterministic donor permutation; intervals condition on that assignment.",
            "Polarity-balanced 217-case subset is not prevalence-representative.",
            "Public benchmark labels and test text may have appeared in model training data.",
        ],
    })
    return summary


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] == "protocol_execution_failure":
        sentence = summary["reason"] + "."
    else:
        primary = summary["primary_decoder_interaction"]
        sentence = (
            f"On {summary['n_complete_recipient_ids']} complete recipient IDs, the finite-minus-free "
            f"matched-review interaction was {primary['estimate']:.3f} VA RMSE points "
            f"(95% recipient-bootstrap interval [{primary['ci95'][0]:.3f}, "
            f"{primary['ci95'][1]:.3f}]). This exploratory interval conditions on one frozen donor assignment."
        )
    readme = f"""# Experiment 053: counterfactual review-context swap

## Result

{sentence}

The test keeps each target aspect and gold fixed, replacing its opinion-masked review with a length-matched masked review from another selected laptop case. A positive decoder-specific matched-review advantage means the model did better with the recipient's own review. The experiment asks whether that advantage differs between finite-grid and free-greedy decoding; it does not establish general context understanding.

The 217-case sample was balanced across negative, neutral, and positive valence buckets and is not prevalence-representative. Intervals condition on a single deterministic donor permutation. The public test may have been seen during pretraining. No raw review text, IDs, donor assignments, or item-level outputs are published.

- Protocol: `docs/experiments/053-counterfactual-review-context-swap.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_counterfactual_context_swap_053.py`, `scripts/analyze_counterfactual_context_swap_053.py`
"""
    (output / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swapped", type=Path, default=Path(".context/exp053-private-predictions.jsonl"))
    parser.add_argument("--matched", type=Path, default=Path(".context/exp050-private-predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(".context/exp053-run-manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/counterfactual-context-swap-v1"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else None
    summary = analyze(load_rows(args.swapped), {
        (row["case_id"], row["condition"], row["decoder"]): row
        for row in (json.loads(line) for line in args.matched.read_text().splitlines() if line.strip())
    }, manifest)
    write_report(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
