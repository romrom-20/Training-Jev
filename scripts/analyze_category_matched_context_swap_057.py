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


def load_polarity_rows(path: Path, ids: set[str]) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keyed = {}
    for row in rows:
        if row["case_id"] not in ids:
            continue
        key = (row["case_id"], row["permutation"], "swapped_context", row["decoder"])
        if key in keyed:
            raise ValueError(f"Duplicate polarity-matched output: {key}")
        keyed[key] = row
    expected = len(ids) * len(shared.PERMUTATIONS) * len(shared.DECODERS)
    if len(keyed) != expected:
        raise ValueError(f"Expected {expected} polarity-matched rows on the shared IDs")
    return keyed


def exploratory_same_recipient_contrast(
    category: dict, polarity: dict, matched: dict, bootstrap_replicates: int = 10_000,
    seed: int = 20260958,
) -> dict:
    """Post-hoc paired contrast of category/polarity versus polarity-only donors."""
    ids = sorted({key[0] for key in category})
    if ids != sorted({key[0] for key in polarity}):
        raise ValueError("Category and polarity controls must use the same recipient IDs")

    def interaction(swapped: dict, sample: list[str]) -> float:
        assignment_effects = []
        for permutation in shared.PERMUTATIONS:
            decoder_effects = {}
            for decoder in shared.DECODERS:
                donor_rows = {
                    (case_id, "swapped_context", decoder): swapped[
                        (case_id, permutation, "swapped_context", decoder)
                    ]
                    for case_id in sample
                }
                own_rows = {
                    (case_id, "opinion_masked", decoder): matched[
                        (case_id, "opinion_masked", decoder)
                    ]
                    for case_id in sample
                }
                decoder_effects[decoder] = shared._rmse(
                    donor_rows, sample, decoder, "swapped_context"
                ) - shared._rmse(own_rows, sample, decoder, "opinion_masked")
            assignment_effects.append(
                decoder_effects["finite_grid"] - decoder_effects["free_greedy"]
            )
        return float(np.mean(assignment_effects))

    point_category = interaction(category, ids)
    point_polarity = interaction(polarity, ids)
    rng = np.random.default_rng(seed)
    differences = np.empty(bootstrap_replicates)
    for index in range(len(differences)):
        sample = rng.choice(ids, size=len(ids), replace=True).tolist()
        differences[index] = interaction(category, sample) - interaction(polarity, sample)
    return {
        "status": "post_hoc_exploratory",
        "contrast": "category-and-polarity-matched interaction minus polarity-only-matched interaction",
        "category_matched_estimate": point_category,
        "polarity_only_estimate_same_recipients": point_polarity,
        "difference": point_category - point_polarity,
        "recipient_bootstrap_ci95": [float(value) for value in np.quantile(differences, [0.025, 0.975])],
        "n_recipient_ids": len(ids),
        "bootstrap_replicates": len(differences),
        "bootstrap_seed": seed,
        "warning": "This same-recipient contrast was not prespecified; do not treat it as confirmatory.",
    }


def analyze(swapped: dict, matched: dict, manifest: dict | None = None,
            polarity: dict | None = None) -> dict:
    shared_manifest = dict(manifest) if manifest is not None else None
    if shared_manifest is not None:
        shared_manifest.setdefault("source_sha256", shared_manifest.get("task2_sha256"))
    summary = shared.analyze(swapped, matched, shared_manifest)
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
        assignment_hashes = shared_manifest.get("donor_assignment_sha256", {})
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
    if polarity is not None and summary.get("status") == "scored":
        summary["exploratory_same_recipient_contrast"] = exploratory_same_recipient_contrast(
            swapped, polarity, matched
        )
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
    comparison = summary.get("exploratory_same_recipient_contrast")
    comparison_text = ""
    if comparison:
        comparison_text = (
            "\n\nA **post-hoc, same-recipient comparison** against Experiment 056's polarity-only "
            f"donors estimated a category-control minus polarity-only interaction of "
            f"{comparison['difference']:.3f} (95% recipient-bootstrap interval "
            f"[{comparison['recipient_bootstrap_ci95'][0]:.3f}, "
            f"{comparison['recipient_bootstrap_ci95'][1]:.3f}]). This contrast was not preregistered "
            "and is exploratory."
        )
    (output / "README.md").write_text(f"""# Experiment 057: category-matched context swap

## Result

{sentence}
{comparison_text}

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
    parser.add_argument("--polarity-matched", type=Path,
                        default=Path(".context/exp056-private-predictions.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/category-matched-context-swap-v1"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else None
    matched = {
        (row["case_id"], row["condition"], row["decoder"]): row
        for row in (json.loads(line) for line in args.matched.read_text().splitlines() if line.strip())
    }
    swapped = load_rows(args.swapped)
    polarity = load_polarity_rows(args.polarity_matched, {key[0] for key in swapped})
    summary = analyze(swapped, matched, manifest, polarity)
    write_report(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
