"""Aggregate-only analysis for the adaptive Experiment 044 order control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LANGS = ("rus", "ukr", "tat")
SEED = 20260944
BOOTSTRAPS = 10_000


def load_parent(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by_key = {(row["case_id"], row["lang"], row["condition"]): row for row in rows}
    if len(rows) != 1302 or len(by_key) != 1302:
        raise ValueError("Expected 1,302 unique frozen Experiment 043 rows")
    if any(row["prediction"] is None for row in rows):
        raise ValueError("Experiment 043 parent contains invalid scores")
    return by_key


def load_new(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by_key = {(row["case_id"], row["lang"], row["condition"]): row for row in rows}
    if len(rows) != 651 or len(by_key) != 651:
        raise ValueError("Expected 651 unique Experiment 044 outputs")
    if {row["condition"] for row in rows} != {"opinion_shuffled"}:
        raise ValueError("Unexpected Experiment 044 condition")
    return by_key


def cluster_errors(parent: dict, shuffled: dict, case_id: str, condition: str) -> np.ndarray:
    values = []
    for lang in LANGS:
        if condition == "opinion_shuffled":
            row = shuffled[(case_id, lang, condition)]
        else:
            row = parent[(case_id, lang, condition)]
        if row["prediction"] is None:
            raise ValueError("Cannot score invalid responses")
        baseline = parent[(case_id, lang, "aspect_only")]
        gold = baseline["gold"]
        values.extend(
            (float(row["prediction"][dim]) - float(gold[dim])) ** 2
            for dim in (0, 1)
        )
    return np.asarray(values, dtype=float)


def paired_contrast(parent: dict, shuffled: dict, case_ids: list[str], left: str, right: str) -> dict:
    errors_left = {
        case_id: cluster_errors(parent, shuffled, case_id, left) for case_id in case_ids
    }
    errors_right = {
        case_id: cluster_errors(parent, shuffled, case_id, right) for case_id in case_ids
    }

    def estimate(ids: list[str]) -> float:
        rmse_left = np.sqrt(np.mean(np.concatenate([errors_left[item] for item in ids])))
        rmse_right = np.sqrt(np.mean(np.concatenate([errors_right[item] for item in ids])))
        return float(rmse_left - rmse_right)

    point = estimate(case_ids)
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        sampled = rng.choice(case_ids, size=len(case_ids), replace=True).tolist()
        draws[index] = estimate(sampled)
    return {
        "contrast": f"{left}_rmse_minus_{right}_rmse",
        "estimate": point,
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "n_clusters": len(case_ids),
    }


def analyze(parent: dict, shuffled: dict, manifest: dict | None = None) -> dict:
    case_ids = sorted({key[0] for key in parent})
    if len(case_ids) != 217:
        raise ValueError(f"Expected 217 reused clusters, found {len(case_ids)}")
    expected_parent = {
        (case_id, lang, condition)
        for case_id in case_ids
        for lang in LANGS
        for condition in ("aspect_only", "opinion_masked")
    }
    expected_new = {
        (case_id, lang, "opinion_shuffled")
        for case_id in case_ids
        for lang in LANGS
    }
    if set(parent) != expected_parent or set(shuffled) != expected_new:
        raise ValueError("Parent and shuffled outputs do not form the complete paired design")
    invalid = sum(row["prediction"] is None for row in shuffled.values())
    invalid_rate = invalid / len(shuffled)
    summary = {
        "experiment": "044-opinion-mask-word-order-control",
        "interpretation": "adaptive post-result, within-sample mechanism diagnostic; exploratory only",
        "n_reused_clusters": len(case_ids),
        "n_new_outputs": len(shuffled),
        "n_invalid_outputs": invalid,
        "invalid_rate": invalid_rate,
        "grid_resolution": 0.1,
        "bootstrap_replicates": BOOTSTRAPS,
        "analysis_seed": SEED,
        "bootstrap_unit": "shared DimABSA ID, retaining all three languages and both VA dimensions",
        "score_analysis_performed": False,
    }
    if manifest is not None:
        summary["run_provenance"] = {
            key: manifest[key]
            for key in (
                "parent_043_output_sha256", "source_revision", "source_hashes", "model",
                "model_revision", "device", "n_grid_candidates", "output_sha256",
                "generation_seconds_this_process_only",
            )
        }
    if invalid_rate > 0.02:
        summary["status"] = "protocol_execution_failure"
        summary["reason"] = "invalid output rate exceeded the registered 2% threshold"
        return summary
    complete_ids = [
        case_id
        for case_id in case_ids
        if all(
            shuffled[(case_id, lang, "opinion_shuffled")]["prediction"] is not None
            for lang in LANGS
        )
    ]
    if len(complete_ids) < 100:
        raise ValueError("Fewer than 100 complete aligned clusters remain")
    natural_order = paired_contrast(
        parent, shuffled, complete_ids, "opinion_shuffled", "opinion_masked"
    )
    natural_order["interpretation"] = "positive means natural word order has lower RMSE than shuffled"
    natural_order["practical_order_gate_passed"] = (
        natural_order["estimate"] >= 0.25 and natural_order["ci95"][0] > 0
    )
    shuffled_vs_prior = paired_contrast(
        parent, shuffled, complete_ids, "aspect_only", "opinion_shuffled"
    )
    language_rmse = {}
    for lang in LANGS:
        language_rmse[lang] = {}
        for condition in ("aspect_only", "opinion_masked", "opinion_shuffled"):
            values = []
            for case_id in complete_ids:
                row = (
                    shuffled[(case_id, lang, condition)]
                    if condition == "opinion_shuffled"
                    else parent[(case_id, lang, condition)]
                )
                gold = parent[(case_id, lang, "aspect_only")]["gold"]
                values.extend(
                    (float(row["prediction"][dim]) - float(gold[dim])) ** 2
                    for dim in (0, 1)
                )
            language_rmse[lang][f"{condition}_rmse"] = float(np.sqrt(np.mean(values)))
    summary.update(
        {
            "status": "scored",
            "score_analysis_performed": True,
            "n_complete_clusters": len(complete_ids),
            "primary_exploratory": natural_order,
            "descriptive_shuffled_vs_aspect_prior": shuffled_vs_prior,
            "by_language_rmse_descriptive": language_rmse,
        }
    )
    return summary


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] == "protocol_execution_failure":
        result = (
            "The invalid-output gate failed; score comparisons were not run "
            f"({summary['n_invalid_outputs']}/{summary['n_new_outputs']} invalid outputs)."
        )
    else:
        primary = summary["primary_exploratory"]
        result = (
            f"Shuffled-minus-natural masked RMSE was {primary['estimate']:.3f} "
            f"(95% cluster interval [{primary['ci95'][0]:.3f}, {primary['ci95'][1]:.3f}]); "
            f"the exploratory order-contribution rule {'passed' if primary['practical_order_gate_passed'] else 'did not pass'}."
        )
    (output / "README.md").write_text(
        f"""# Experiment 044: opinion-mask word-order control

## Result

{result}

This is an adaptive within-sample follow-up to the positive Experiment 043 result, not independent confirmation. The word-order shuffle keeps each sentence's whitespace-token multiset, including `[MASKED]` placeholders, but disrupts syntax and discourse. The same Qwen checkpoint and 0.1-grid constrained decoder are used. A null order contrast does not prove bag-of-words sufficiency.

No review text, aspect strings, IDs or item-level outputs are included; private material remains in ignored `.context/`.

- Protocol: `docs/experiments/044-opinion-mask-word-order-control.md`
- Runner/analyzer: `scripts/run_opinion_mask_word_order_control_044.py`, `scripts/analyze_opinion_mask_word_order_control_044.py`
- Aggregate result and provenance: `summary.json`
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=Path(".context/exp043-private-predictions.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path(".context/exp044-private-predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(".context/exp044-run-manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("results/opinion-mask-word-order-v1"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else None
    write_report(analyze(load_parent(args.parent), load_new(args.predictions), manifest), args.out)


if __name__ == "__main__":
    main()
