"""Analyze the polarity-matched donor control with Experiment 054's estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_context_swap_donor_robustness_054 as shared

shared.SEED = 20260956


def load_rows(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keyed = {}
    for row in rows:
        key = (row["case_id"], row["permutation"], "swapped_context", row["decoder"])
        if key in keyed:
            raise ValueError(f"Duplicate polarity-matched output: {key}")
        keyed[key] = row
    expected = shared.EXPECTED_IDS * len(shared.PERMUTATIONS) * len(shared.DECODERS)
    if len(keyed) != expected or {row["condition"] for row in rows} != {"polarity_matched_context"}:
        raise ValueError(f"Expected {expected} unique polarity-matched outputs")
    return keyed


def analyze(swapped: dict, matched: dict, manifest: dict | None = None) -> dict:
    summary = shared.analyze(swapped, matched, manifest)
    summary["experiment"] = "056-polarity-matched-context-swap"
    summary["interpretation"] = "adaptive coarse-polarity-matched context diagnostic; exploratory"
    summary["analysis_seed"] = 20260956
    summary["control"] = {
        "donors_match_recipient_gold_valence_polarity": True,
        "donors_differ_in_exact_review_and_aspect_linkage": True,
        "polarity_buckets": "negative and positive; no neutral cases",
    }
    summary["limitations"] = [
        "Same public dataset and model family; test text may have appeared in pretraining.",
        "The fresh sample contains negative and positive cases only.",
        "Polarity matching rules out only this coarse gold-valence cue; other donor differences remain.",
        "Intervals condition on the three fixed mappings.",
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
            f"With donors matched to each recipient's gold-valence polarity, the mean finite-minus-free "
            f"matched-review interaction was {primary['estimate']:.3f} VA RMSE points "
            f"(95% recipient-bootstrap interval [{primary['ci95'][0]:.3f}, {primary['ci95'][1]:.3f}]). "
            f"The three fixed permutation estimates ranged from {sensitivity['range'][0]:.3f} "
            f"to {sensitivity['range'][1]:.3f}; that range is descriptive."
        )
    (output / "README.md").write_text(f"""# Experiment 056: polarity-matched context swap

## Result

{sentence}

Each donor review shares its recipient's negative/positive gold-valence bucket, but comes from another case. The recipient's aspect and gold stay fixed. This tests whether the decoder-specific exact-review advantage survives controlling that coarse polarity cue.

The test uses the fresh 217-case subset from Experiment 055 (108 negative and 109 positive), one public benchmark, and one model family. Donors still differ in many lexical and semantic properties besides aspect linkage; pretraining exposure is possible. Intervals condition on three fixed mappings. No text, item IDs, donor maps, or item-level outputs are published.

- Protocol: `docs/experiments/056-polarity-matched-context-swap.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_polarity_matched_context_swap_056.py`, `scripts/analyze_polarity_matched_context_swap_056.py`
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swapped", type=Path, default=Path(".context/exp056-private-predictions.jsonl"))
    parser.add_argument("--matched", type=Path, default=Path(".context/exp050-private-predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(".context/exp056-run-manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/polarity-matched-context-swap-v1"))
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
