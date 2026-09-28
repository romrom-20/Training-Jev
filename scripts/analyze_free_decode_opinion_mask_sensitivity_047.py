"""Aggregate-only analysis of free-greedy opinion-mask sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LANGS = ("rus", "ukr", "tat")
CONDITIONS = ("aspect_only", "opinion_masked")
SEED = 20260947
BOOTSTRAPS = 10_000


def load_rows(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_key = {(row["case_id"], row["lang"], row["condition"]): row for row in rows}
    if len(rows) != 1302 or len(by_key) != 1302:
        raise ValueError(f"Expected 1,302 unique outputs; found {len(rows)}")
    if {row["condition"] for row in rows} != set(CONDITIONS):
        raise ValueError("Unexpected output condition set")
    return by_key


def analyze(by_key: dict, manifest: dict | None = None) -> dict:
    ids = sorted({key[0] for key in by_key})
    if len(ids) != 217:
        raise ValueError(f"Expected 217 frozen IDs; found {len(ids)}")
    expected = {
        (case_id, lang, condition)
        for case_id in ids
        for lang in LANGS
        for condition in CONDITIONS
    }
    if set(by_key) != expected:
        raise ValueError("Free-decode outputs do not form the complete paired grid")
    invalid = sum(row["prediction"] is None for row in by_key.values())
    rate = invalid / len(by_key)
    summary = {
        "experiment": "047-free-decode-opinion-mask-sensitivity",
        "interpretation": "adaptive decoder-sensitivity diagnostic; exploratory only",
        "n_clusters": len(ids),
        "n_outputs": len(by_key),
        "n_invalid_outputs": invalid,
        "invalid_rate": rate,
        "invalid_by_condition": {
            condition: sum(
                row["prediction"] is None for row in by_key.values()
                if row["condition"] == condition
            )
            for condition in CONDITIONS
        },
        "analysis_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_unit": "source review ID, retaining all three languages and both VA dimensions",
        "score_analysis_performed": False,
    }
    if manifest is not None:
        summary["run_provenance"] = {
            key: manifest[key]
            for key in (
                "parent_043_output_sha256", "source_revision", "source_hashes", "model",
                "model_revision", "device", "max_new_tokens", "output_sha256",
                "generation_seconds_this_process_only", "invalid_by_condition",
            )
        }
    if rate > 0.02:
        summary["status"] = "protocol_execution_failure"
        summary["reason"] = "invalid-output rate exceeded the preregistered 2% threshold; scores withheld"
        return summary
    complete_ids = [
        case_id for case_id in ids
        if all(
            by_key[(case_id, lang, condition)]["prediction"] is not None
            for lang in LANGS for condition in CONDITIONS
        )
    ]
    if len(complete_ids) < 100:
        raise ValueError("Fewer than 100 complete source-ID clusters remain")

    def cluster_errors(case_id: str, condition: str) -> np.ndarray:
        values = []
        for lang in LANGS:
            row = by_key[(case_id, lang, condition)]
            if row["prediction"] is None:
                raise ValueError("Cannot score invalid response")
            gold = by_key[(case_id, lang, "aspect_only")]["gold"]
            values.extend(
                (float(row["prediction"][dim]) - float(gold[dim])) ** 2
                for dim in (0, 1)
            )
        return np.asarray(values, dtype=float)

    errors = {
        condition: {case_id: cluster_errors(case_id, condition) for case_id in complete_ids}
        for condition in CONDITIONS
    }

    def gain(sampled: list[str]) -> float:
        aspect = np.sqrt(np.mean(np.concatenate([errors["aspect_only"][x] for x in sampled])))
        context = np.sqrt(np.mean(np.concatenate([errors["opinion_masked"][x] for x in sampled])))
        return float(aspect - context)

    point = gain(complete_ids)
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        sample = rng.choice(complete_ids, size=len(complete_ids), replace=True).tolist()
        draws[index] = gain(sample)
    ci = [float(x) for x in np.quantile(draws, [0.025, 0.975])]
    language_rmse = {}
    for lang in LANGS:
        language_rmse[lang] = {}
        for condition in CONDITIONS:
            squared = []
            for case_id in complete_ids:
                row = by_key[(case_id, lang, condition)]
                gold = by_key[(case_id, lang, "aspect_only")]["gold"]
                squared.extend(
                    (float(row["prediction"][dim]) - float(gold[dim])) ** 2
                    for dim in (0, 1)
                )
            language_rmse[lang][f"{condition}_rmse"] = float(np.sqrt(np.mean(squared)))
    summary.update(
        {
            "status": "scored",
            "score_analysis_performed": True,
            "n_complete_clusters": len(complete_ids),
            "primary": {
                "contrast": "RMSE(aspect_only) - RMSE(opinion_masked)",
                "estimate": point,
                "ci95": ci,
                "n_clusters": len(complete_ids),
                "interpretation": "positive means free-decoded masked context improves VA RMSE",
                "registered_free_decode_gate_passed": point >= 0.25 and ci[0] > 0,
            },
            "by_language_rmse_descriptive": language_rmse,
        }
    )
    return summary


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] == "protocol_execution_failure":
        result = (
            f"The invalid-output gate failed ({summary['n_invalid_outputs']}/"
            f"{summary['n_outputs']} invalid); all score contrasts were withheld."
        )
    else:
        primary = summary["primary"]
        result = (
            f"Free-decoded aspect-only minus opinion-masked RMSE was "
            f"{primary['estimate']:.3f} (95% source-ID interval "
            f"[{primary['ci95'][0]:.3f}, {primary['ci95'][1]:.3f}]); "
            f"the preregistered free-decoding rule {'passed' if primary['registered_free_decode_gate_passed'] else 'did not pass'}."
        )
    runtime = summary.get("run_provenance", {}).get("generation_seconds_this_process_only")
    runtime_text = f"\nGeneration took {runtime / 60:.1f} minutes after model load." if runtime else ""
    (output / "README.md").write_text(
        f"""# Experiment 047: free-decoding sensitivity

## Result

{result}{runtime_text}

This adaptive test reuses the 217 source IDs from Experiment 043 and changes the output contract from finite-choice one-decimal VA to ordinary greedy JSON generation. It is a decoder-sensitivity check, not an independent replication. The free prompt/parser can still shape which generations count as valid.

No review text, aspects, IDs, raw generations, or per-item predictions are published; private outputs remain in ignored `.context/`.

- Protocol: `docs/experiments/047-free-decode-opinion-mask-sensitivity.md`
- Runner/analyzer: `scripts/run_free_decode_opinion_mask_sensitivity_047.py`, `scripts/analyze_free_decode_opinion_mask_sensitivity_047.py`
- Aggregate result and provenance: `summary.json`
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(".context/exp047-private-predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(".context/exp047-run-manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("results/free-decode-opinion-mask-sensitivity-v1"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else None
    write_report(analyze(load_rows(args.predictions), manifest), args.out)


if __name__ == "__main__":
    main()
