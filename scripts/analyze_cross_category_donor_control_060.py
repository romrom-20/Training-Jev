"""Analyze the paired same-topic versus cross-topic donor contrast."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_category_matched_context_swap_057 as category_analyzer
import analyze_context_swap_donor_robustness_054 as shared
import numpy as np

shared.EXPECTED_IDS = 184
shared.SEED = 20260961
OUT = Path("results/cross-category-donor-control-v1")


def load_cross_rows(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keyed = {}
    for row in rows:
        key = (row["case_id"], row["permutation"], "swapped_context", row["decoder"])
        if key in keyed:
            raise ValueError(f"Duplicate cross-category output: {key}")
        keyed[key] = row
    expected = shared.EXPECTED_IDS * len(shared.PERMUTATIONS) * len(shared.DECODERS)
    if len(rows) != expected or len(keyed) != expected:
        raise ValueError(f"Expected {expected} unique cross-category outputs")
    if {row["condition"] for row in rows} != {"cross_category_polarity_matched_context"}:
        raise ValueError("Unexpected 060 output condition")
    if {row["decoder"] for row in rows} != set(shared.DECODERS):
        raise ValueError("Unexpected decoder set")
    return keyed


def interaction_by_map(donors: dict, own: dict, ids: list[str]) -> dict[int, float]:
    estimates = {}
    for permutation in shared.PERMUTATIONS:
        advantages = {}
        for decoder in shared.DECODERS:
            donor_rows = {
                (case_id, "swapped_context", decoder): donors[
                    (case_id, permutation, "swapped_context", decoder)
                ] for case_id in ids
            }
            own_rows = {
                (case_id, "opinion_masked", decoder): own[
                    (case_id, "opinion_masked", decoder)
                ] for case_id in ids
            }
            advantages[decoder] = shared._rmse(
                donor_rows, ids, decoder, "swapped_context"
            ) - shared._rmse(own_rows, ids, decoder, "opinion_masked")
        estimates[permutation] = advantages["finite_grid"] - advantages["free_greedy"]
    return estimates


def analyze(cross: dict, same: dict, own: dict, manifest: dict) -> dict:
    summary = shared.analyze(cross, own, manifest)
    summary["experiment"] = "060-cross-category-donor-control"
    summary["interpretation"] = "same-recipient category-match versus category-mismatch diagnostic"
    summary["control"] = {
        "donors_match_recipient_gold_valence_polarity": True,
        "donors_differ_from_recipient_official_category": True,
        "same_category_comparator": "Experiment 057, same recipients and own-review predictions",
        "n_unique_cross_category_assignment_maps": (
            manifest.get("n_unique_assignment_maps") if manifest is not None else None
        ),
    }
    summary["limitations"] = [
        "The 184-case sample excludes category-polarity singleton cells and contains no neutral targets.",
        "The same-category comparator was collected earlier, so this is a registered follow-up to an observed result, not a blind replication.",
        "Cross-category matching changes lexical and semantic content in addition to broad topic.",
        "One public benchmark split and one model family; pretraining exposure is possible.",
        "Bootstrap intervals condition on the three fixed assignment sets.",
    ]
    if summary.get("status") != "scored":
        return summary
    ids = sorted({key[0] for key in cross})
    cross_by_map = interaction_by_map(cross, own, ids)
    same_by_map = interaction_by_map(same, own, ids)
    if set(cross_by_map) != set(same_by_map):
        raise ValueError("The paired conditions do not share assignment indices")
    point_cross = float(np.mean(list(cross_by_map.values())))
    point_same = float(np.mean(list(same_by_map.values())))
    rng = np.random.default_rng(20260961)
    draws = np.empty(10_000)
    for index in range(len(draws)):
        sample = rng.choice(ids, size=len(ids), replace=True).tolist()
        draws[index] = float(np.mean(list(interaction_by_map(cross, own, sample).values()))) - float(
            np.mean(list(interaction_by_map(same, own, sample).values()))
        )
    summary["per_assignment"] = {
        str(permutation): {
            "cross_category_interaction": cross_by_map[permutation],
            "same_category_interaction": same_by_map[permutation],
            "cross_minus_same": cross_by_map[permutation] - same_by_map[permutation],
        } for permutation in shared.PERMUTATIONS
    }
    summary["category_mismatched_interaction"] = summary["mean_primary_interaction"]
    summary["category_mismatched_interaction"]["estimate"] = point_cross
    summary["category_matched_interaction"] = {
        "estimate": point_same,
        "n_recipient_ids": len(ids),
        "source_experiment": "057-category-matched-context-swap",
    }
    summary["mean_primary_interaction"] = {
        "contrast": "cross-category interaction minus same-category interaction, paired on recipient ID",
        "estimate": point_cross - point_same,
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "n_recipient_ids": len(ids),
        "bootstrap_replicates": len(draws),
        "bootstrap_seed": 20260961,
        "confirmatory": False,
    }
    return summary


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] == "protocol_execution_failure":
        result = summary["reason"] + "."
    else:
        direct = summary["mean_primary_interaction"]
        same = summary["category_matched_interaction"]["estimate"]
        cross = summary["category_mismatched_interaction"]["estimate"]
        result = (
            f"The polarity-matched cross-category interaction was {cross:.3f}; the same-category "
            f"interaction was {same:.3f}. Their paired cross-minus-same difference was "
            f"{direct['estimate']:.3f} (95% recipient-bootstrap interval "
            f"[{direct['ci95'][0]:.3f}, {direct['ci95'][1]:.3f}])."
        )
    (output / "README.md").write_text(f"""# Experiment 060: cross-category donor control

## Result

{result}

The two conditions use the same 184 recipients, recipient gold values, own-review baselines, model and polarity constraints. Same-category results are from Experiment 057. Three cross-category one-to-one donor matchings keep polarity fixed while requiring a different official category. The bootstrap resamples recipient IDs and retains all mappings; it does not generalize over possible donor assignments.

The contrast tests broad official-category match on this sample. Category mismatches also change lexical and semantic content; this design cannot identify a specific word-level mechanism. There is one public benchmark split and one model family, with possible pretraining exposure. No text, IDs, donor mappings, or item-level outputs are published.

- Protocol: `docs/experiments/060-cross-category-donor-control.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_cross_category_donor_control_060.py`, `scripts/analyze_cross_category_donor_control_060.py`
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross", type=Path, default=Path(".context/exp060-private-predictions.jsonl"))
    parser.add_argument("--same", type=Path, default=Path(".context/exp057-private-predictions.jsonl"))
    parser.add_argument("--own", type=Path, default=Path(".context/exp050-private-predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(".context/exp060-run-manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    same = category_analyzer.load_rows(args.same)
    own_rows = [json.loads(line) for line in args.own.read_text().splitlines() if line.strip()]
    own = {
        (row["case_id"], row["condition"], row["decoder"]): row
        for row in own_rows if row["condition"] == "opinion_masked"
    }
    cross = load_cross_rows(args.cross)
    manifest = json.loads(args.manifest.read_text())
    summary = analyze(cross, same, own, manifest)
    write_report(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
