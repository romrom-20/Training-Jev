"""Analyze the 1.5B transfer and the paired cross-size diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_category_matched_context_swap_057 as category_analyzer
import analyze_context_swap_donor_robustness_054 as shared
import numpy as np

OUT = Path("results/category-match-size-transfer-v1")
SEED_SIZE_CONTRAST = 20260960


def split_1p5b(path: Path) -> tuple[dict, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    donors, own = {}, {}
    for row in rows:
        if row["condition"] == "category_polarity_matched_context":
            key = (row["case_id"], row["permutation"], "swapped_context", row["decoder"])
            if key in donors:
                raise ValueError(f"Duplicate 1.5B donor output: {key}")
            donors[key] = row
        elif row["condition"] == "opinion_masked":
            key = (row["case_id"], "opinion_masked", row["decoder"])
            if key in own:
                raise ValueError(f"Duplicate 1.5B own-review output: {key}")
            own[key] = row
        else:
            raise ValueError(f"Unknown 1.5B condition: {row['condition']}")
    expected_donors = 184 * 3 * 2
    expected_own = 184 * 2
    if len(rows) != expected_donors + expected_own or len(donors) != expected_donors or len(own) != expected_own:
        raise ValueError("Incomplete 1.5B paired design")
    return donors, own


def load_3b_donors(path: Path) -> dict:
    return category_analyzer.load_rows(path)


def load_3b_own(path: Path, ids: set[str]) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keyed = {
        (row["case_id"], row["condition"], row["decoder"]): row
        for row in rows if row["condition"] == "opinion_masked" and row["case_id"] in ids
    }
    if {key[0] for key in keyed} != ids or len(keyed) != len(ids) * 2:
        raise ValueError("Expected 184 paired 3B own-review baselines")
    return keyed


def interaction_by_size(donors: dict, own: dict, ids: list[str]) -> float:
    per_map = []
    for permutation in shared.PERMUTATIONS:
        effect = {}
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
            effect[decoder] = shared._rmse(
                donor_rows, ids, decoder, "swapped_context"
            ) - shared._rmse(own_rows, ids, decoder, "opinion_masked")
        per_map.append(effect["finite_grid"] - effect["free_greedy"])
    return float(np.mean(per_map))


def paired_size_contrast(donors_15: dict, own_15: dict,
                         donors_3: dict, own_3: dict,
                         bootstrap_replicates: int = 10_000) -> dict:
    ids = sorted({key[0] for key in donors_15})
    if ids != sorted({key[0] for key in donors_3}):
        raise ValueError("Cross-size contrast requires identical recipient IDs")
    point_15 = interaction_by_size(donors_15, own_15, ids)
    point_3 = interaction_by_size(donors_3, own_3, ids)
    rng = np.random.default_rng(SEED_SIZE_CONTRAST)
    draws = np.empty(bootstrap_replicates)
    for index in range(len(draws)):
        sample = rng.choice(ids, size=len(ids), replace=True).tolist()
        draws[index] = interaction_by_size(donors_15, own_15, sample) - interaction_by_size(
            donors_3, own_3, sample
        )
    return {
        "contrast": "1.5B interaction minus 3B interaction; same recipients and donor maps",
        "estimate_1_5b": point_15,
        "estimate_3b": point_3,
        "difference": point_15 - point_3,
        "recipient_bootstrap_ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "n_recipient_ids": len(ids),
        "bootstrap_replicates": len(draws),
        "bootstrap_seed": SEED_SIZE_CONTRAST,
        "inferential_status": "cross-size diagnostic; same model family and same public sample",
    }


def analyze(path_15: Path, manifest_path: Path, path_3donor: Path,
            path_3own: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    donor_15, own_15 = split_1p5b(path_15)
    donor_3 = load_3b_donors(path_3donor)
    own_3 = load_3b_own(path_3own, {key[0] for key in donor_3})
    three_b_manifest = json.loads(Path(".context/exp057-run-manifest.json").read_text())
    if manifest["donor_assignment_sha256"] != three_b_manifest["donor_assignment_sha256"]:
        raise ValueError("Cross-size models do not share the frozen donor maps")
    manifest_for_analysis = dict(manifest)
    manifest_for_analysis["n_unique_assignment_maps"] = len(
        set(manifest["donor_assignment_sha256"].values())
    )
    summary = category_analyzer.analyze(donor_15, own_15, manifest_for_analysis)
    summary["experiment"] = "059-category-match-size-transfer"
    summary["interpretation"] = "same-sample, same-map category-match size transfer; exploratory"
    summary["model_size"] = "Qwen2.5-1.5B"
    summary["cross_size_diagnostic"] = paired_size_contrast(donor_15, own_15, donor_3, own_3)
    summary["limitations"].append(
        "The 1.5B-vs-3B contrast reuses the same public recipients and fixed donor maps; it is not an independent replication."
    )
    return summary


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    result = summary["mean_primary_interaction"]
    size = summary["cross_size_diagnostic"]
    report = f"""# Experiment 059: category-matched context control at 1.5B

## Result

At 1.5B, the mean finite-minus-free matched-review interaction was {result['estimate']:.3f} VA RMSE points (95% recipient-bootstrap interval [{result['ci95'][0]:.3f}, {result['ci95'][1]:.3f}]). All {summary['n_new_outputs']} donor outputs were paired with own-review baselines; invalid free-greedy donor outputs: {summary['invalid_free_outputs']}/{summary['invalid_free_denominator']}.

On the same 184 recipients and the same three donor maps, the 1.5B-minus-3B interaction difference was {size['difference']:.3f} (95% paired recipient-bootstrap interval [{size['recipient_bootstrap_ci95'][0]:.3f}, {size['recipient_bootstrap_ci95'][1]:.3f}]). This is a cross-size diagnostic, not an independent replication.

The study changes model size while holding recipients and donor assignments fixed. Both sizes belong to the Qwen2.5 family and use one public test release; category matching does not control exact aspect identity or specific lexical overlap. No text, case IDs, donor maps, or item predictions are published.

- Protocol: `docs/experiments/059-category-match-size-transfer.md`
- Aggregate estimates and provenance: `summary.json`
- Runner/analyzer: `scripts/run_category_match_size_transfer_059.py`, `scripts/analyze_category_match_size_transfer_059.py`
"""
    (output / "README.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(".context/exp059-private-predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(".context/exp059-run-manifest.json"))
    parser.add_argument("--three-b-donors", type=Path, default=Path(".context/exp057-private-predictions.jsonl"))
    parser.add_argument("--three-b-own", type=Path, default=Path(".context/exp050-private-predictions.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    summary = analyze(args.predictions, args.manifest, args.three_b_donors, args.three_b_own)
    write_report(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
