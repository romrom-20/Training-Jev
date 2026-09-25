"""Aggregate-only analysis for Experiment 043; do not expose private item rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LANGS = ("rus", "ukr", "tat")
CONDITIONS = ("aspect_only", "opinion_masked")
SEED = 20260943
BOOTSTRAPS = 10_000


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    keys = [(row["case_id"], row["lang"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate per-item result")
    return rows


def _cluster_errors(by_key: dict, case_id: str, condition: str, dim: int | None = None) -> np.ndarray:
    values = []
    for lang in LANGS:
        row = by_key[(case_id, lang, condition)]
        if row["prediction"] is None:
            raise ValueError("Cannot score invalid predictions")
        dims = (dim,) if dim is not None else (0, 1)
        values.extend((float(row["prediction"][i]) - float(row["gold"][i])) ** 2 for i in dims)
    return np.asarray(values, dtype=float)


def paired_contrast(
    by_key: dict,
    case_ids: list[str],
    condition_a: str,
    condition_b: str,
    dim: int | None = None,
) -> dict:
    error_a = {
        case_id: _cluster_errors(by_key, case_id, condition_a, dim) for case_id in case_ids
    }
    error_b = {
        case_id: _cluster_errors(by_key, case_id, condition_b, dim) for case_id in case_ids
    }

    def estimate(ids: list[str]) -> float:
        mse_a = np.mean(np.concatenate([error_a[x] for x in ids]))
        mse_b = np.mean(np.concatenate([error_b[x] for x in ids]))
        return float(np.sqrt(mse_a) - np.sqrt(mse_b))

    point = estimate(case_ids)
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        sample = rng.choice(case_ids, size=len(case_ids), replace=True).tolist()
        draws[index] = estimate(sample)
    return {
        "contrast": f"{condition_a}_rmse_minus_{condition_b}_rmse",
        "estimate": point,
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "n_clusters": len(case_ids),
    }


def analyze(rows: list[dict], manifest: dict | None = None) -> dict:
    if len(rows) != 1302:
        raise ValueError(f"Expected 1,302 outputs; found {len(rows)}")
    by_key = {(row["case_id"], row["lang"], row["condition"]): row for row in rows}
    case_ids = sorted({row["case_id"] for row in rows})
    if len(case_ids) != 217:
        raise ValueError(f"Expected 217 unique IDs; found {len(case_ids)}")
    expected = {
        (case_id, lang, condition)
        for case_id in case_ids for lang in LANGS for condition in CONDITIONS
    }
    if set(by_key) != expected:
        raise ValueError("Incomplete or unexpected case/language/condition grid")
    invalid = sum(row["prediction"] is None for row in rows)
    invalid_rate = invalid / len(rows)
    result = {
        "experiment": "043-expanded-opinion-mask-repair",
        "interpretation": "held-out-item execution-corrected test on one public benchmark; not an independent corpus replication",
        "n_selected_clusters": len(case_ids),
        "n_language_target_instances": len(case_ids) * len(LANGS),
        "n_expected_outputs": len(rows),
        "n_invalid_outputs": invalid,
        "invalid_rate": invalid_rate,
        "grid_resolution": 0.1,
        "analysis_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_unit": "shared DimABSA ID; resample all languages, conditions and VA dimensions together",
        "score_analysis_performed": False,
    }
    complete_ids = [
        case_id
        for case_id in case_ids
        if all(
            by_key[(case_id, lang, condition)]["prediction"] is not None
            for lang in LANGS for condition in CONDITIONS
        )
    ]
    result["n_complete_clusters"] = len(complete_ids)
    if manifest is not None:
        result["run_provenance"] = {
            key: manifest[key]
            for key in (
                "source_revision", "source_hashes", "model", "model_revision", "device",
                "n_grid_candidates", "generation_seconds_this_process_only",
                "run_code_commit", "output_sha256",
            )
        }
    if invalid_rate > 0.02:
        result["status"] = "protocol_execution_failure"
        result["reason"] = "invalid output rate exceeded the preregistered 2% threshold"
        result["invalid_by_condition"] = {
            condition: sum(
                row["condition"] == condition and row["prediction"] is None for row in rows
            )
            for condition in CONDITIONS
        }
        return result

    if len(complete_ids) < 100:
        raise ValueError("Fewer than 100 complete aligned clusters remain")
    primary = paired_contrast(by_key, complete_ids, "aspect_only", "opinion_masked")
    primary["interpretation"] = "positive favors non-opinion sentence context over aspect-only input"
    primary["registered_support_gate_passed"] = (
        primary["estimate"] >= 0.25 and primary["ci95"][0] > 0
    )
    dimension_contrasts = {
        name: paired_contrast(by_key, complete_ids, "aspect_only", "opinion_masked", dim)
        for name, dim in (("valence", 0), ("arousal", 1))
    }
    by_language = {}
    for lang in LANGS:
        errors = {condition: [] for condition in CONDITIONS}
        for case_id in complete_ids:
            for condition in CONDITIONS:
                row = by_key[(case_id, lang, condition)]
                errors[condition].extend(
                    (float(row["prediction"][dim]) - float(row["gold"][dim])) ** 2
                    for dim in (0, 1)
                )
        by_language[lang] = {
            "aspect_only_rmse": float(np.sqrt(np.mean(errors["aspect_only"]))),
            "opinion_masked_rmse": float(np.sqrt(np.mean(errors["opinion_masked"]))),
        }
    result.update(
        {
            "status": "scored",
            "score_analysis_performed": True,
            "primary": primary,
            "dimension_contrasts_descriptive": dimension_contrasts,
            "by_language_descriptive": by_language,
        }
    )
    return result


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    runtime = summary.get("run_provenance", {}).get("generation_seconds_this_process_only")
    runtime_line = (
        f"Generation took {runtime / 60:.1f} minutes after model load on "
        f"{summary['run_provenance']['device']}."
        if runtime is not None else "Generation runtime was not recorded."
    )
    if summary["status"] == "protocol_execution_failure":
        body = (
            "The preregistered invalid-output gate failed, so VA score comparisons were not run. "
            f"There were {summary['n_invalid_outputs']}/{summary['n_expected_outputs']} invalid outputs."
        )
    else:
        primary = summary["primary"]
        body = (
            f"The primary RMSE contrast (aspect-only minus opinion-masked) was "
            f"{primary['estimate']:.3f} (cluster-bootstrap 95% interval "
            f"[{primary['ci95'][0]:.3f}, {primary['ci95'][1]:.3f}]); valence and arousal "
            f"contrasts were {summary['dimension_contrasts_descriptive']['valence']['estimate']:.3f} "
            f"and {summary['dimension_contrasts_descriptive']['arousal']['estimate']:.3f}; the registered "
            f"support gate {'passed' if primary['registered_support_gate_passed'] else 'did not pass'}."
        )
    text = f"""# Experiment 043: expanded opinion-mask repair

## Result

{body}

{runtime_line}

This is a disjoint-item rerun on the same DimABSA release, language files and Qwen model. The 0.1-step constrained output grid ensures values stay in range and parse, but can alter model answers. The bootstrap resamples source IDs with all three aligned language rows kept together. This does not establish independent-corpus or human generalization.

No review text, aspects, item IDs or item-level predictions are redistributed. They remain local in ignored `.context/`.

- Protocol: `docs/experiments/043-expanded-opinion-mask-repair.md`
- Runner/analyzer: `scripts/run_expanded_opinion_mask_repair_043.py`, `scripts/analyze_expanded_opinion_mask_repair_043.py`
- Aggregate result and provenance: `summary.json`
"""
    (output / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(".context/exp043-private-predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(".context/exp043-run-manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("results/expanded-opinion-mask-repair-v1"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else None
    write_report(analyze(load_rows(args.predictions), manifest), args.out)


if __name__ == "__main__":
    main()
