"""Analyze the fresh disjoint sample with Experiment 054's frozen estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_context_swap_donor_robustness_054 import analyze as _analyze
from analyze_context_swap_donor_robustness_054 import load_rows


def analyze(swapped: dict, matched: dict, manifest: dict | None = None) -> dict:
    summary = _analyze(swapped, matched, manifest)
    summary["experiment"] = "055-fresh-context-swap-replication"
    summary["interpretation"] = "adaptive disjoint-sample replication; exploratory"
    summary["sample_design"] = {
        "n": summary["n_recipient_ids"],
        "selection": "fresh, SHA-256-ranked and disjoint from Exp053/054",
        "valence": "108 negative, 109 positive; no neutral cases",
        "confirmatory": False,
    }
    summary["limitations"] = [
        "Same public dataset and model family; test data may have appeared in pretraining.",
        "No neutral-valence cases in the fresh disjoint sample.",
        "Donor mappings are not polarity-matched; coarse-polarity effects remain possible.",
        "Intervals condition on three fixed donor mappings.",
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
            f"On {summary['n_complete_recipient_ids']} new IDs disjoint from Experiments 053–054, "
            f"the mean finite-minus-free matched-review interaction was {primary['estimate']:.3f} "
            f"VA RMSE points (95% recipient-bootstrap interval [{primary['ci95'][0]:.3f}, "
            f"{primary['ci95'][1]:.3f}]). The three permutation estimates ranged from "
            f"{sensitivity['range'][0]:.3f} to {sensitivity['range'][1]:.3f}; that range is descriptive."
        )
    (output / "README.md").write_text(f"""# Experiment 055: fresh-sample context-swap replication

## Result

{sentence}

The 217 recipient cases are disjoint from the first 217-case sample, with 108 negative- and 109 positive-valence cases. All neutral cases were used in the earlier sample, so this replication does not test neutral cases. Each aspect/gold pair stayed fixed while three length-binned donor reviews were assigned in deterministic derangements. The result uses the same public dataset and model family as the earlier tests.

Donor reviews were not matched by polarity, so coarse-polarity effects remain possible. The intervals condition on the three mappings, and the public test may have appeared in pretraining. No text, item IDs, donor mappings, or individual outputs are published.

- Protocol: `docs/experiments/055-fresh-context-swap-replication.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_fresh_context_swap_replication_055.py`, `scripts/analyze_fresh_context_swap_replication_055.py`
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swapped", type=Path, default=Path(".context/exp055-private-predictions.jsonl"))
    parser.add_argument("--matched", type=Path, default=Path(".context/exp050-private-predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(".context/exp055-run-manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/fresh-context-swap-replication-v1"))
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
