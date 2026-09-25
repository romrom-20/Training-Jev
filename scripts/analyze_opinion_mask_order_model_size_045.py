"""Aggregate-only paired order analysis for the adaptive Experiment 045 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LANGS = ("rus", "ukr", "tat")
CONDITIONS = ("opinion_masked", "opinion_shuffled")
SEED = 20260945
BOOTSTRAPS = 10_000


def load_rows(path: Path, expected_count: int, expected_conditions: set[str]) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    by_key = {(row["case_id"], row["lang"], row["condition"]): row for row in rows}
    if len(rows) != expected_count or len(by_key) != expected_count:
        raise ValueError(f"Expected {expected_count} unique outputs; found {len(rows)}")
    if {row["condition"] for row in rows} != expected_conditions:
        raise ValueError("Unexpected condition set")
    return by_key


def _cluster_errors(by_key: dict, gold_by_id: dict, case_id: str, condition: str) -> np.ndarray:
    values = []
    for lang in LANGS:
        row = by_key[(case_id, lang, condition)]
        if row["prediction"] is None:
            raise ValueError("Cannot score invalid model output")
        gold = gold_by_id[(case_id, lang)]
        values.extend(
            (float(row["prediction"][dim]) - float(gold[dim])) ** 2
            for dim in (0, 1)
        )
    return np.asarray(values, dtype=float)


def paired_contrast(by_key: dict, gold_by_id: dict, ids: list[str], left: str, right: str) -> dict:
    left_errors = {
        case_id: _cluster_errors(by_key, gold_by_id, case_id, left) for case_id in ids
    }
    right_errors = {
        case_id: _cluster_errors(by_key, gold_by_id, case_id, right) for case_id in ids
    }

    def estimate(sample_ids: list[str]) -> float:
        left_rmse = np.sqrt(np.mean(np.concatenate([left_errors[x] for x in sample_ids])))
        right_rmse = np.sqrt(np.mean(np.concatenate([right_errors[x] for x in sample_ids])))
        return float(left_rmse - right_rmse)

    point = estimate(ids)
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        sample = rng.choice(ids, size=len(ids), replace=True).tolist()
        draws[index] = estimate(sample)
    return {
        "contrast": f"{left}_rmse_minus_{right}_rmse",
        "estimate": point,
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "n_clusters": len(ids),
    }


def analyze(parent_043: dict, current: dict, manifest: dict | None = None) -> dict:
    ids = sorted({key[0] for key in parent_043})
    if len(ids) != 217:
        raise ValueError(f"Expected 217 frozen IDs; found {len(ids)}")
    expected_parent = {
        (case_id, lang, condition)
        for case_id in ids
        for lang in LANGS
        for condition in ("aspect_only", "opinion_masked")
    }
    expected_current = {
        (case_id, lang, condition)
        for case_id in ids
        for lang in LANGS
        for condition in CONDITIONS
    }
    if set(parent_043) != expected_parent or set(current) != expected_current:
        raise ValueError("Model-size output files do not form the frozen paired grid")
    gold_by_id = {
        (case_id, lang): parent_043[(case_id, lang, "aspect_only")]["gold"]
        for case_id in ids for lang in LANGS
    }
    invalid = sum(row["prediction"] is None for row in current.values())
    invalid_rate = invalid / len(current)
    summary = {
        "experiment": "045-order-control-model-size",
        "interpretation": "adaptive within-sample Qwen-family model-size follow-up; exploratory only",
        "n_clusters": len(ids),
        "n_outputs": len(current),
        "n_invalid_outputs": invalid,
        "invalid_rate": invalid_rate,
        "grid_resolution": 0.1,
        "analysis_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_unit": "shared DimABSA ID; keep all three languages and both VA dimensions together",
        "score_analysis_performed": False,
    }
    if manifest is not None:
        summary["run_provenance"] = {
            key: manifest[key]
            for key in (
                "parent_043_output_sha256", "parent_044_output_sha256", "model",
                "model_revision", "device", "n_grid_candidates", "output_sha256",
                "generation_seconds_this_process_only",
            )
        }
    if invalid_rate > 0.02:
        summary["status"] = "protocol_execution_failure"
        summary["reason"] = "invalid output rate exceeded the preregistered 2% threshold"
        return summary
    complete_ids = [
        case_id
        for case_id in ids
        if all(
            current[(case_id, lang, condition)]["prediction"] is not None
            for lang in LANGS for condition in CONDITIONS
        )
    ]
    if len(complete_ids) < 100:
        raise ValueError("Fewer than 100 complete clusters remain")
    primary = paired_contrast(current, gold_by_id, complete_ids, "opinion_shuffled", "opinion_masked")
    primary["interpretation"] = "positive means natural-order context has lower RMSE than shuffled context"
    primary["registered_order_gate_passed"] = (
        primary["estimate"] >= 0.25 and primary["ci95"][0] > 0
    )
    by_language = {}
    for lang in LANGS:
        by_language[lang] = {}
        for condition in CONDITIONS:
            errors = []
            for case_id in complete_ids:
                row = current[(case_id, lang, condition)]
                gold = gold_by_id[(case_id, lang)]
                errors.extend(
                    (float(row["prediction"][dim]) - float(gold[dim])) ** 2
                    for dim in (0, 1)
                )
            by_language[lang][f"{condition}_rmse"] = float(np.sqrt(np.mean(errors)))
    summary.update(
        {
            "status": "scored",
            "score_analysis_performed": True,
            "n_complete_clusters": len(complete_ids),
            "primary_exploratory": primary,
            "by_language_rmse_descriptive": by_language,
        }
    )
    return summary


def write_report(summary: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] == "protocol_execution_failure":
        text = (
            "The invalid-output gate failed, so score contrasts were withheld "
            f"({summary['n_invalid_outputs']}/{summary['n_outputs']} invalid outputs)."
        )
    else:
        primary = summary["primary_exploratory"]
        text = (
            f"For Qwen2.5-1.5B, shuffled-minus-natural masked RMSE was "
            f"{primary['estimate']:.3f} (95% cluster interval "
            f"[{primary['ci95'][0]:.3f}, {primary['ci95'][1]:.3f}]); "
            f"the exploratory natural-order rule {'passed' if primary['registered_order_gate_passed'] else 'did not pass'}."
        )
    runtime = summary.get("run_provenance", {}).get("generation_seconds_this_process_only")
    runtime_text = f"Generation took {runtime / 60:.1f} minutes after model load." if runtime else ""
    (out / "README.md").write_text(
        f"""# Experiment 045: order control by model size

## Result

{text}

{runtime_text}

This adaptive test compares both conditions under Qwen2.5-1.5B and the same finite one-decimal VA grammar. It reuses the same 217 public DimABSA IDs from Experiments 043/044, so it is a model-size diagnostic rather than an independent data replication. The grammar may change answer content, and shuffled whitespace tokens disrupt syntax and discourse together.

No review text, aspects, IDs, or per-item outputs are published; private data remain in ignored `.context/`.

- Protocol: `docs/experiments/045-order-control-model-size.md`
- Runner/analyzer: `scripts/run_opinion_mask_order_model_size_045.py`, `scripts/analyze_opinion_mask_order_model_size_045.py`
- Aggregate result and provenance: `summary.json`
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=Path(".context/exp043-private-predictions.jsonl"))
    parser.add_argument("--predictions", type=Path, default=Path(".context/exp045-private-predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(".context/exp045-run-manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("results/opinion-mask-order-model-size-v1"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else None
    write_report(analyze(load_rows(args.parent, 1302, {"aspect_only", "opinion_masked"}),
                         load_rows(args.predictions, 1302, set(CONDITIONS)), manifest), args.out)


if __name__ == "__main__":
    main()
