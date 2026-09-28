"""Aggregate-only paired evaluation for Experiment 046."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

LANGS = ("rus", "ukr", "tat")
CONDITIONS = ("aspect_only", "masked_context")
SEED = 20260946
BOOTSTRAPS = 10_000
PARENT_043_GAIN = 1.8448904220357432


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_rows(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_key = {(row["case_id"], row["lang"], row["condition"]): row for row in rows}
    expected = 217 * len(LANGS) * len(CONDITIONS)
    if len(rows) != expected or len(by_key) != expected:
        raise ValueError(f"Expected {expected} unique paired predictions; found {len(rows)}")
    return by_key


def paired_gain(by_key: dict, ids: list[str]) -> dict:
    errors = {
        condition: {
            case_id: np.asarray(
                [
                    (
                        float(by_key[(case_id, lang, condition)]["prediction"][dim])
                        - float(by_key[(case_id, lang, condition)]["gold"][dim])
                    ) ** 2
                    for lang in LANGS
                    for dim in (0, 1)
                ],
                dtype=float,
            )
            for case_id in ids
        }
        for condition in CONDITIONS
    }

    def estimate(sampled: list[str]) -> float:
        aspect_rmse = np.sqrt(np.mean(np.concatenate([errors["aspect_only"][x] for x in sampled])))
        context_rmse = np.sqrt(np.mean(np.concatenate([errors["masked_context"][x] for x in sampled])))
        return float(aspect_rmse - context_rmse)

    point = estimate(ids)
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        draws[index] = estimate(rng.choice(ids, size=len(ids), replace=True).tolist())
    return {
        "contrast": "rmse(aspect_only) - rmse(masked_context)",
        "estimate": point,
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "n_clusters": len(ids),
        "registered_lexical_gain_gate_passed": point >= 0.25 and float(np.quantile(draws, 0.025)) > 0,
    }


def analyze(by_key: dict, manifest: dict | None = None) -> dict:
    ids = sorted({key[0] for key in by_key})
    if len(ids) != 217:
        raise ValueError(f"Expected 217 frozen source IDs; found {len(ids)}")
    expected = {
        (case_id, lang, condition)
        for case_id in ids
        for lang in LANGS
        for condition in CONDITIONS
    }
    if set(by_key) != expected:
        raise ValueError("Predictions do not form the complete frozen paired grid")
    for row in by_key.values():
        pred = row["prediction"]
        gold = row["gold"]
        if len(pred) != 2 or len(gold) != 2 or not np.isfinite(pred).all():
            raise ValueError("Invalid VA prediction or gold pair")
    summary = {
        "experiment": "046-masked-context-lexical-baseline",
        "interpretation": "adaptive sparse lexical diagnostic; exploratory only",
        "n_clusters": len(ids),
        "n_predictions": len(by_key),
        "analysis_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_unit": "source review ID, retaining all languages and both VA dimensions",
        "qwen_043_context_gain_descriptive_anchor": PARENT_043_GAIN,
        "score_analysis_performed": True,
    }
    if manifest is not None:
        summary["run_provenance"] = {
            key: manifest[key]
            for key in (
                "source_revision", "source_hashes", "training_counts", "alpha_grid",
                "cv_folds_max", "feature_map", "n_predictions", "runtime_seconds",
                "predictions_sha256",
            )
        }
    summary["primary_diagnostic"] = paired_gain(by_key, ids)
    by_language = {}
    by_dimension = {}
    for lang in LANGS:
        by_language[lang] = {}
        for condition in CONDITIONS:
            squared = [
                (
                    float(by_key[(case_id, lang, condition)]["prediction"][dim])
                    - float(by_key[(case_id, lang, condition)]["gold"][dim])
                ) ** 2
                for case_id in ids
                for dim in (0, 1)
            ]
            by_language[lang][f"{condition}_rmse"] = float(np.sqrt(np.mean(squared)))
    for dim, name in enumerate(("valence", "arousal")):
        by_dimension[name] = {}
        for condition in CONDITIONS:
            squared = [
                (
                    float(by_key[(case_id, lang, condition)]["prediction"][dim])
                    - float(by_key[(case_id, lang, condition)]["gold"][dim])
                ) ** 2
                for case_id in ids
                for lang in LANGS
            ]
            by_dimension[name][f"{condition}_rmse"] = float(np.sqrt(np.mean(squared)))
    summary["by_language_rmse_descriptive"] = by_language
    summary["by_dimension_rmse_descriptive"] = by_dimension
    summary["fraction_of_qwen_043_gain_descriptive"] = (
        summary["primary_diagnostic"]["estimate"] / PARENT_043_GAIN
    )
    return summary


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    primary = summary["primary_diagnostic"]
    result = (
        f"The sparse model's aspect-only minus opinion-masked RMSE gain was "
        f"{primary['estimate']:.3f} (95% source-ID cluster interval "
        f"[{primary['ci95'][0]:.3f}, {primary['ci95'][1]:.3f}]); "
        f"the preregistered lexical-gain rule {'passed' if primary['registered_lexical_gain_gate_passed'] else 'did not pass'}."
    )
    runtime = summary.get("run_provenance", {}).get("runtime_seconds")
    runtime_text = f"\nThe CPU run took {runtime / 60:.1f} minutes." if runtime else ""
    (output / "README.md").write_text(
        f"""# Experiment 046: sparse lexical baseline for masked VA context

## Result

{result}{runtime_text}

This adaptive diagnostic uses per-language word/character TF-IDF and ridge models trained on the official DimABSA training split. Ridge regularization was selected by grouped cross-validation over training review IDs only. Evaluation reuses the 217 held-out source IDs from Experiments 043–045 and is not an independent corpus replication. The sparse model is not a pretrained LLM; a positive gain would show predictive residual lexical information, not causal sufficiency.

The comparison with Qwen2.5-3B's Experiment 043 gain is descriptive. No review text, aspects, IDs, or per-item predictions are published; private predictions remain in ignored `.context/`.

- Protocol: `docs/experiments/046-masked-context-lexical-baseline.md`
- Runner/analyzer: `scripts/run_masked_context_lexical_baseline_046.py`, `scripts/analyze_masked_context_lexical_baseline_046.py`
- Aggregate result and training/provenance record: `summary.json`
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(".context/exp046-private/predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(".context/exp046-private/manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("results/masked-context-lexical-baseline-v1"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else None
    write_report(analyze(load_rows(args.predictions), manifest), args.out)


if __name__ == "__main__":
    main()
