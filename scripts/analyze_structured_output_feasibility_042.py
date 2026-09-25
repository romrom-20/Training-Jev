"""Aggregate-only analysis for the adaptive Experiment 042 format audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DECODERS = ("free", "constrained")
GROUPS = ("prior_invalid", "prior_valid_control")


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    keys = [
        (r["case_id"], r["lang"], r["source_condition"], r["decoder"])
        for r in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate per-item decoder output")
    if len(rows) != 192:
        raise ValueError(f"Expected 192 decoder outputs; found {len(rows)}")
    return rows


def _status_counts(rows: list[dict]) -> dict:
    result = {}
    for group in GROUPS:
        for decoder in DECODERS:
            selected = [r for r in rows if r["group"] == group and r["decoder"] == decoder]
            counts = Counter(
                r["response"]["status"] if r["response"] is not None else "invalid"
                for r in selected
            )
            result[f"{group}:{decoder}"] = {
                key: counts.get(key, 0)
                for key in ("estimate", "insufficient", "invalid")
            }
    return result


def _control_agreement(rows: list[dict]) -> dict:
    by_key = defaultdict(dict)
    for row in rows:
        if row["group"] == "prior_valid_control":
            key = (row["case_id"], row["lang"], row["source_condition"])
            by_key[key][row["decoder"]] = row
    pairs = []
    for pair in by_key.values():
        free, constrained = pair["free"], pair["constrained"]
        if free["response"] is None or constrained["response"] is None:
            continue
        if free["response"]["status"] != "estimate" or constrained["response"]["status"] != "estimate":
            continue
        free_va = np.asarray(
            [free["response"]["valence"], free["response"]["arousal"]], dtype=float
        )
        constrained_va = np.asarray(
            [constrained["response"]["valence"], constrained["response"]["arousal"]],
            dtype=float,
        )
        pairs.append((free_va, constrained_va))
    if not pairs:
        return {"n_numeric_pairs": 0, "exact_pair_matches": 0, "exact_match_rate": None}
    deltas = np.stack([np.abs(left - right) for left, right in pairs])
    exact = sum(bool(np.array_equal(left, right)) for left, right in pairs)
    return {
        "n_numeric_pairs": len(pairs),
        "exact_pair_matches": exact,
        "exact_match_rate": exact / len(pairs),
        "mean_absolute_change": {
            "valence": float(deltas[:, 0].mean()),
            "arousal": float(deltas[:, 1].mean()),
        },
    }


def _cell_counts(rows: list[dict], fields: tuple[str, ...]) -> dict:
    cells = defaultdict(Counter)
    for row in rows:
        key = tuple(row[field] for field in fields) + (row["decoder"],)
        cells[key][row["response"]["status"] if row["response"] is not None else "invalid"] += 1
    return {
        "dimensions": [*fields, "decoder"],
        "cells": [
            {
                **dict(zip((*fields, "decoder"), key)),
                "n": sum(counts.values()),
                "estimate": counts.get("estimate", 0),
                "insufficient": counts.get("insufficient", 0),
                "invalid": counts.get("invalid", 0),
            }
            for key, counts in sorted(cells.items())
        ],
    }


def analyze(rows: list[dict]) -> dict:
    if len(rows) != 192:
        raise ValueError("Analysis requires the complete fixed audit set")
    counts = Counter(row["decoder"] for row in rows)
    if counts != Counter({"free": 96, "constrained": 96}):
        raise ValueError(f"Unexpected decoder counts: {dict(counts)}")
    prompt_rows = [row for row in rows if row["decoder"] == "free"]
    group_counts = Counter(row["group"] for row in prompt_rows)
    if group_counts != Counter({"prior_invalid": 48, "prior_valid_control": 48}):
        raise ValueError(f"Unexpected prompt-group counts: {dict(group_counts)}")
    control_language_counts = Counter(
        row["lang"] for row in prompt_rows if row["group"] == "prior_valid_control"
    )
    if control_language_counts != Counter({"rus": 16, "ukr": 16, "tat": 16}):
        raise ValueError(f"Unexpected valid-control language counts: {dict(control_language_counts)}")
    invalid_by_decoder = {
        decoder: sum(
            row["decoder"] == decoder and row["response"] is None for row in rows
        )
        for decoder in DECODERS
    }
    return {
        "experiment": "042-structured-output-feasibility",
        "interpretation": "adaptive technical audit; no gold-score analysis or confirmatory inference",
        "n_prompts": 96,
        "n_generations": len(rows),
        "prompt_group_counts": dict(group_counts),
        "valid_control_language_counts": dict(control_language_counts),
        "invalid_by_decoder": invalid_by_decoder,
        "valid_rate_by_decoder": {
            decoder: 1 - invalid_by_decoder[decoder] / 96 for decoder in DECODERS
        },
        "status_counts_by_prior_validity_and_decoder": _status_counts(rows),
        "status_counts_by_language_condition_and_decoder": _cell_counts(
            rows, ("lang", "source_condition")
        ),
        "status_counts_by_group_language_and_decoder": _cell_counts(
            rows, ("group", "lang")
        ),
        "free_vs_constrained_numeric_agreement_on_valid_controls": _control_agreement(rows),
    }


def write_report(summary: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    counts = summary["status_counts_by_prior_validity_and_decoder"]
    free_invalid = counts["prior_invalid:free"]
    constrained_invalid = counts["prior_invalid:constrained"]
    controls = summary["free_vs_constrained_numeric_agreement_on_valid_controls"]
    if controls["exact_match_rate"] is None:
        agreement = "No valid numeric control pairs were available."
    else:
        agreement = (
            f"On {controls['n_numeric_pairs']} paired numeric controls, exact VA-pair agreement was "
            f"{controls['exact_pair_matches']}/{controls['n_numeric_pairs']} "
            f"({controls['exact_match_rate']:.1%}); mean absolute change was "
            f"{controls['mean_absolute_change']['valence']:.3f} valence points and "
            f"{controls['mean_absolute_change']['arousal']:.3f} arousal points."
        )
    text = f"""# Experiment 042: structured-output feasibility

## Diagnostic result

The audit covers 48 prompts that failed the Experiment 041 response parser and 48 parseable aspect-plus-opinion controls, each generated once with free greedy decoding and once with a finite 82-output constraint. Free decoding returned invalid objects on {summary['invalid_by_decoder']['free']}/96 prompts; constrained decoding returned invalid objects on {summary['invalid_by_decoder']['constrained']}/96.

For the prior-invalid prompts, free generation yielded {free_invalid['estimate']} estimate objects, {free_invalid['insufficient']} explicit insufficient objects, and {free_invalid['invalid']} invalid responses. Constrained generation yielded {constrained_invalid['estimate']} estimates, {constrained_invalid['insufficient']} insufficient objects, and {constrained_invalid['invalid']} invalid responses. {agreement}

The JSON summary also reports response counts by source condition, language, and prior-validity group. Those are descriptive cells from this selected audit, not inferential comparisons.

## Interpretation

This is an adaptive technical audit selected on Experiment 041 parser outcomes. It does not measure VA accuracy and does not establish that an `insufficient` response is correct. The constrained decoder can alter numeric content; the comparison is descriptive and no score-versus-gold analysis was performed. The constrained choices use integer VA candidates and are not a general structured-generation benchmark.

No review text, aspect strings, IDs, or item-level responses are redistributed. Raw material remains in ignored `.context/`.

- Aggregate counts: `summary.json`
- Protocol: `docs/experiments/042-structured-output-feasibility.md`
- Runner/analyzer: `scripts/run_structured_output_feasibility_042.py`, `scripts/analyze_structured_output_feasibility_042.py`
"""
    (out / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(".context/exp042-private-predictions.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/structured-output-feasibility-v1"))
    args = parser.parse_args()
    rows = load_rows(args.predictions)
    summary = analyze(rows)
    summary["run_provenance"] = {
        "dataset_revision": "bdc93be1224106ae7d3eb95739c02a76ed4ae8a1",
        "source_hashes": {
            "rus": "912013b49db2bd387076f63ee9df016350180fbac621449121a27dc262d5459b",
            "ukr": "05fdf7e2dca235261060b785f969315385f962021f171abb85e45638bbadb036",
            "tat": "c4f1fb5c21f8f06f598c87e489e7adce1a953d4446ad14a60432ae2a13b85ce6",
        },
        "source_prediction_sha256": "8056a1fd91f56d0700e10968ec8c55cc0fc3ffb1de73a98375df5d21f428a9c8",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "model_revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "device": "mps",
        "protocol_sha256": hashlib.sha256(
            Path("docs/experiments/042-structured-output-feasibility.md").read_bytes()
        ).hexdigest(),
        "private_output_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "runtime_note": "Outputs span resumed generation segments; cumulative wall time was not retained.",
    }
    write_report(summary, args.out)


if __name__ == "__main__":
    main()
