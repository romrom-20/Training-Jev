"""Analyze exact-category/polarity matched swaps using the shared paired estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_context_swap_donor_robustness_054 as shared
import numpy as np

shared.EXPECTED_IDS = 184
shared.SEED = 20260957


def load_rows(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keyed = {}
    for row in rows:
        key = (row["case_id"], row["permutation"], "swapped_context", row["decoder"])
        if key in keyed:
            raise ValueError(f"Duplicate category-matched output: {key}")
        keyed[key] = row
    expected = shared.EXPECTED_IDS * len(shared.PERMUTATIONS) * len(shared.DECODERS)
    if len(keyed) != expected or {row["condition"] for row in rows} != {"category_polarity_matched_context"}:
        raise ValueError(f"Expected {expected} unique category/polarity-matched outputs")
    return keyed


def analyze(swapped: dict, matched: dict, manifest: dict | None = None) -> dict:
    summary = shared.analyze(swapped, matched, manifest)
    summary["experiment"] = "057-category-matched-context-swap"
    summary["interpretation"] = "adaptive official-category/polarity matched context diagnostic; exploratory"
    summary["analysis_seed"] = 20260957
    summary["control"] = {
        "donors_match_recipient_gold_valence_polarity": True,
        "donors_match_official_dimabsa_aspect_category": True,
        "n_category_polarity_groups": manifest.get("n_category_polarity_groups") if manifest else 45,
        "n_unique_assignment_maps": manifest.get("n_unique_assignment_maps") if manifest else None,
    }
    if manifest is not None and "run_provenance" in summary:
        summary["run_provenance"]["n_unique_assignment_maps"] = manifest.get(
            "n_unique_assignment_maps"
        )
        assignment_hashes = manifest.get("donor_assignment_sha256", {})
        unique_permutations = []
        seen_hashes = set()
        for permutation in shared.PERMUTATIONS:
            assignment_hash = assignment_hashes.get(str(permutation), assignment_hashes.get(permutation))
            if assignment_hash not in seen_hashes:
                seen_hashes.add(assignment_hash)
                unique_permutations.append(permutation)
        if summary.get("status") == "scored" and unique_permutations:
            estimates = [
                summary["per_permutation"][permutation]["finite_minus_free_interaction"]
                for permutation in unique_permutations
            ]
            summary["permutation_sensitivity"] = {
                "range": [min(estimates), max(estimates)],
                "sample_standard_deviation": float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0,
                "n_permutations": len(estimates),
                "inferential_status": "descriptive range across unique fixed donor mappings",
            }
    summary["limitations"] = [
        "The 184 eligible cases exclude category-polarity singleton cells.",
        "The fresh sample has no neutral-valence cases.",
        "Official category and polarity matching do not control exact aspect identity or relevant words.",
        "Same public test release and model family; pretraining exposure is possible.",
        "Bootstrap intervals condition on the fixed cyclic donor mappings.",
    ]
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
            f"With both donor polarity and official aspect category matched, the mean finite-minus-free "
            f"matched-review interaction was {primary['estimate']:.3f} VA RMSE points "
            f"(95% recipient-bootstrap interval [{primary['ci95'][0]:.3f}, {primary['ci95'][1]:.3f}]). "
            f"The fixed assignment estimates ranged from {sensitivity['range'][0]:.3f} "
            f"to {sensitivity['range'][1]:.3f}; {summary['run_provenance']['n_unique_assignment_maps']} "
            "unique mappings were represented."
        )
    (output / "README.md").write_text(f"""# Experiment 057: category-matched context swap

## Result

{sentence}

Donor reviews share the recipient's gold-valence polarity and official DimABSA aspect category, while the recipient's own aspect and gold remain fixed. The control tests whether the decoder-specific own-review advantage persists when donor and recipient are about the same broad topic and sentiment.

Only 184 of the 217 fresh cases could be swapped without self-donation inside their category/polarity cell; 33 singleton cases were excluded by rule. The test contains negative and positive cases only, uses one public split and one model family, and may overlap model pretraining. Category matching does not identify which exact contextual words matter. No review text, IDs, categories by case, donor mappings, or item-level predictions are published.

- Protocol: `docs/experiments/057-category-matched-context-swap.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_category_matched_context_swap_057.py`, `scripts/analyze_category_matched_context_swap_057.py`
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swapped", type=Path, default=Path(".context/exp057-private-predictions.jsonl"))
    parser.add_argument("--matched", type=Path, default=Path(".context/exp050-private-predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(".context/exp057-run-manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/category-matched-context-swap-v1"))
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
