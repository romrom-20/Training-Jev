"""Analyze three fixed donor permutations for the 054 robustness test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DECODERS = ("finite_grid", "free_greedy")
PERMUTATIONS = (1, 2, 3)
EXPECTED_IDS = 217
SEED = 20260954
BOOTSTRAPS = 10_000


def load_rows(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keyed = {
        (row["case_id"], row["permutation"], row["condition"], row["decoder"]): row
        for row in rows
    }
    expected = EXPECTED_IDS * len(PERMUTATIONS) * len(DECODERS)
    if len(rows) != expected or len(keyed) != expected:
        raise ValueError(f"Expected {expected} unique outputs, found {len(rows)}")
    if {row["condition"] for row in rows} != {"swapped_context"}:
        raise ValueError("Unexpected condition in 054 output")
    if {row["decoder"] for row in rows} != set(DECODERS):
        raise ValueError("Unexpected decoder set")
    if {row["permutation"] for row in rows} != set(PERMUTATIONS):
        raise ValueError("Unexpected permutation set")
    return keyed


def _rmse(rows: dict, ids: list[str], decoder: str, condition: str, permutation: int | None = None) -> float:
    squared = []
    for case_id in ids:
        key = ((case_id, permutation, condition, decoder) if permutation is not None
               else (case_id, condition, decoder))
        row = rows[key]
        squared.extend(
            (float(row["prediction"][dim]) - float(row["gold"][dim])) ** 2
            for dim in (0, 1)
        )
    return float(np.sqrt(np.mean(squared)))


def analyze(swapped: dict, matched: dict, manifest: dict | None = None) -> dict:
    ids = sorted({key[0] for key in swapped})
    if len(ids) != EXPECTED_IDS:
        raise ValueError(f"Expected {EXPECTED_IDS} recipient IDs, found {len(ids)}")
    expected_swapped = {
        (case_id, permutation, "swapped_context", decoder)
        for case_id in ids for permutation in PERMUTATIONS for decoder in DECODERS
    }
    expected_matched = {
        (case_id, "opinion_masked", decoder)
        for case_id in ids for decoder in DECODERS
    }
    if set(swapped) != expected_swapped or not expected_matched.issubset(matched):
        raise ValueError("Incomplete paired donor-permutation design")
    for case_id in ids:
        for permutation in PERMUTATIONS:
            for decoder in DECODERS:
                own = matched[(case_id, "opinion_masked", decoder)]
                wrong = swapped[(case_id, permutation, "swapped_context", decoder)]
                if own["gold"] != wrong["gold"]:
                    raise ValueError(f"Recipient gold mismatch: {case_id}/{permutation}/{decoder}")
                for row in (own, wrong):
                    prediction = row["prediction"]
                    if prediction is not None and (
                        len(prediction) != 2 or not np.isfinite(prediction).all()
                    ):
                        raise ValueError("Non-finite prediction")

    invalid = {
        decoder: sum(
            swapped[(case_id, permutation, "swapped_context", decoder)]["prediction"] is None
            for case_id in ids for permutation in PERMUTATIONS
        )
        for decoder in DECODERS
    }
    free_denominator = EXPECTED_IDS * len(PERMUTATIONS)
    invalid_rate = invalid["free_greedy"] / free_denominator
    summary = {
        "experiment": "054-context-swap-donor-robustness",
        "interpretation": "adaptive three-assignment robustness diagnostic; exploratory",
        "n_recipient_ids": len(ids),
        "n_permutations": len(PERMUTATIONS),
        "n_new_outputs": len(swapped),
        "invalid_swapped_by_decoder": invalid,
        "invalid_free_outputs": invalid["free_greedy"],
        "invalid_free_denominator": free_denominator,
        "invalid_free_rate": invalid_rate,
        "analysis_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_unit": "recipient sentence ID, retaining all three fixed donor assignments",
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
    if invalid_rate > 0.02:
        summary.update({
            "status": "protocol_execution_failure",
            "reason": "Free-greedy invalid rate exceeded 2%; all score contrasts withheld",
        })
        return summary

    complete = [
        case_id for case_id in ids
        if all(
            swapped[(case_id, permutation, "swapped_context", decoder)]["prediction"] is not None
            and matched[(case_id, "opinion_masked", decoder)]["prediction"] is not None
            for permutation in PERMUTATIONS for decoder in DECODERS
        )
    ]
    if not complete:
        raise ValueError("No complete recipient IDs")

    def advantage(decoder: str, permutation: int, sample: list[str]) -> float:
        return _rmse(swapped, sample, decoder, "swapped_context", permutation) - _rmse(
            matched, sample, decoder, "opinion_masked"
        )

    per_permutation = {}
    for permutation in PERMUTATIONS:
        decoder_advantages = {
            decoder: advantage(decoder, permutation, complete)
            for decoder in DECODERS
        }
        per_permutation[permutation] = {
            "matched_review_advantage_by_decoder": decoder_advantages,
            "finite_minus_free_interaction": (
                decoder_advantages["finite_grid"] - decoder_advantages["free_greedy"]
            ),
        }
    interactions = np.array([
        per_permutation[permutation]["finite_minus_free_interaction"]
        for permutation in PERMUTATIONS
    ])
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        sample = rng.choice(complete, size=len(complete), replace=True).tolist()
        draw_interactions = [
            advantage("finite_grid", permutation, sample)
            - advantage("free_greedy", permutation, sample)
            for permutation in PERMUTATIONS
        ]
        draws[draw] = float(np.mean(draw_interactions))
    avg_by_decoder = {
        decoder: float(np.mean([
            per_permutation[permutation]["matched_review_advantage_by_decoder"][decoder]
            for permutation in PERMUTATIONS
        ]))
        for decoder in DECODERS
    }
    summary.update({
        "status": "scored",
        "score_analysis_performed": True,
        "n_complete_recipient_ids": len(complete),
        "per_permutation": per_permutation,
        "mean_matched_review_advantage_by_decoder": avg_by_decoder,
        "mean_primary_interaction": {
            "contrast": "mean over three permutations of [matched-review advantage(finite) - matched-review advantage(free)]",
            "estimate": float(np.mean(interactions)),
            "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
            "n_recipient_ids": len(complete),
            "confirmatory": False,
        },
        "permutation_sensitivity": {
            "range": [float(np.min(interactions)), float(np.max(interactions))],
            "sample_standard_deviation": float(np.std(interactions, ddof=1)),
            "n_permutations": len(interactions),
            "inferential_status": "descriptive only; three fixed mappings",
        },
        "limitations": [
            "All donor permutations may change coarse polarity; this design does not control it.",
            "Intervals condition on the three fixed donor mappings.",
            "The polarity-balanced subset is not prevalence-representative.",
            "Public benchmark text may have appeared in model training data.",
        ],
    })
    return summary


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] == "protocol_execution_failure":
        sentence = summary["reason"] + "."
    else:
        primary = summary["mean_primary_interaction"]
        sensitivity = summary["permutation_sensitivity"]
        sentence = (
            f"Across three fixed donor permutations and {summary['n_complete_recipient_ids']} complete IDs, "
            f"the mean finite-minus-free matched-review interaction was {primary['estimate']:.3f} VA RMSE "
            f"points (95% recipient-bootstrap interval [{primary['ci95'][0]:.3f}, {primary['ci95'][1]:.3f}]). "
            f"The three permutation estimates ranged from {sensitivity['range'][0]:.3f} to "
            f"{sensitivity['range'][1]:.3f}; this range is descriptive."
        )
    readme = f"""# Experiment 054: review-donor assignment robustness

## Result

{sentence}

Each target aspect and gold stayed fixed while its opinion-masked review was replaced by a review from another case. Three independent deterministic donor derangements test whether Experiment 053's finite-decoder sensitivity depends on one arbitrary assignment. Positive per-decoder matched-review advantage means the recipient's own review predicts better than donor reviews.

This follow-up does not control coarse gold-valence polarity, and its intervals condition on the three mappings. The 217-case subset is polarity-balanced rather than prevalence-representative. Public test examples may have appeared in model training data. No review text, IDs, donor mappings, or item-level predictions are published.

- Protocol: `docs/experiments/054-context-swap-donor-robustness.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_context_swap_donor_robustness_054.py`, `scripts/analyze_context_swap_donor_robustness_054.py`
"""
    (output / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swapped", type=Path, default=Path(".context/exp054-private-predictions.jsonl"))
    parser.add_argument("--matched", type=Path, default=Path(".context/exp050-private-predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(".context/exp054-run-manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/context-swap-donor-robustness-v1"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else None
    matched = {
        (row["case_id"], row["condition"], row["decoder"]): row
        for row in (json.loads(line) for line in args.matched.read_text().splitlines() if line.strip())
    }
    summary = analyze(load_rows(args.swapped), matched, manifest)
    write_report(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
