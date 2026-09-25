"""Aggregate-only analysis for the preregistered Experiment 041 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LANGS = ("rus", "ukr", "tat")
CONDITIONS = ("aspect_only", "aspect_opinion", "full_text", "opinion_masked")
SEED = 20260941
BOOTSTRAPS = 10_000


def load_predictions(path: Path) -> dict[tuple[str, str, str], dict]:
    predictions = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (row["case_id"], row["lang"], row["condition"])
        if key in predictions:
            raise ValueError(f"Duplicate prediction row: {key}")
        predictions[key] = row
    return predictions


def _case_ids(predictions: dict) -> list[str]:
    return sorted({key[0] for key in predictions})


def _condition_error(predictions: dict, case_ids: list[str], condition: str, lang=None, dim=None):
    errors = []
    for case_id in case_ids:
        for current_lang in ((lang,) if lang else LANGS):
            row = predictions[(case_id, current_lang, condition)]
            prediction = row["prediction"]
            if prediction is None:
                raise ValueError("Metrics require complete, parseable condition rows")
            gold = row["gold"]
            errors.extend(
                (float(prediction[i]) - float(gold[i])) ** 2
                for i in ((dim,) if dim is not None else (0, 1))
            )
    return np.asarray(errors, dtype=float)


def _rmse(squared_errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(squared_errors)))


def paired_bootstrap(
    predictions: dict,
    case_ids: list[str],
    condition_a: str,
    condition_b: str,
    *,
    lang=None,
    dim=None,
    seed=SEED,
    n_boot=BOOTSTRAPS,
) -> dict:
    rng = np.random.default_rng(seed)
    errors_a = {
        case_id: _condition_error(predictions, [case_id], condition_a, lang, dim)
        for case_id in case_ids
    }
    errors_b = {
        case_id: _condition_error(predictions, [case_id], condition_b, lang, dim)
        for case_id in case_ids
    }
    estimate = _rmse(np.concatenate([errors_a[x] for x in case_ids])) - _rmse(
        np.concatenate([errors_b[x] for x in case_ids])
    )
    draws = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(case_ids, size=len(case_ids), replace=True)
        draws[i] = _rmse(np.concatenate([errors_a[x] for x in sample])) - _rmse(
            np.concatenate([errors_b[x] for x in sample])
        )
    return {
        "contrast": f"{condition_a}_rmse_minus_{condition_b}_rmse",
        "estimate": estimate,
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "n_clusters": len(case_ids),
    }


def _complete_cases(predictions: dict) -> tuple[list[str], int]:
    invalid = sum(row["prediction"] is None for row in predictions.values())
    all_ids = _case_ids(predictions)
    complete = [
        case_id
        for case_id in all_ids
        if all(
            (case_id, lang, condition) in predictions
            and predictions[(case_id, lang, condition)]["prediction"] is not None
            for lang in LANGS
            for condition in CONDITIONS
        )
    ]
    return complete, invalid


def analyze(predictions: dict) -> dict:
    expected_count = 120 * len(LANGS) * len(CONDITIONS)
    if len(predictions) != expected_count:
        raise ValueError(f"Expected {expected_count} unique rows, found {len(predictions)}")
    if len(_case_ids(predictions)) != 120:
        raise ValueError(f"Expected 120 sampled IDs, found {len(_case_ids(predictions))}")
    complete, invalid = _complete_cases(predictions)
    invalid_rate = invalid / expected_count
    if invalid_rate > 0.02:
        invalid_by_condition = {
            condition: sum(
                row["condition"] == condition and row["prediction"] is None
                for row in predictions.values()
            )
            for condition in CONDITIONS
        }
        return {
            "experiment": "041-opinion-mask-crosslingual-dimabsa",
            "status": "protocol_execution_failure",
            "reason": "invalid output rate exceeded the preregistered 2% threshold",
            "n_selected_clusters": len(_case_ids(predictions)),
            "n_expected_outputs": expected_count,
            "n_invalid_outputs": invalid,
            "invalid_rate": invalid_rate,
            "invalid_by_condition": invalid_by_condition,
            "n_complete_clusters": len(complete),
            "score_analysis_performed": False,
            "analysis_seed": SEED,
        }
    if len(complete) < 100:
        raise ValueError(f"Only {len(complete)} complete aligned clusters")

    primary = paired_bootstrap(predictions, complete, "aspect_only", "opinion_masked")
    primary["interpretation"] = "positive means opinion-masked sentence context beats aspect-only"
    primary["registered_support_gate_passed"] = (
        primary["estimate"] >= 0.25 and primary["ci95"][0] > 0
    )
    secondary = {
        "full_vs_masked": paired_bootstrap(predictions, complete, "opinion_masked", "full_text"),
        "aspect_opinion_vs_aspect": paired_bootstrap(
            predictions, complete, "aspect_only", "aspect_opinion"
        ),
        "valence_aspect_vs_masked": paired_bootstrap(
            predictions, complete, "aspect_only", "opinion_masked", dim=0
        ),
        "arousal_aspect_vs_masked": paired_bootstrap(
            predictions, complete, "aspect_only", "opinion_masked", dim=1
        ),
        "by_language": {
            lang: {
                "aspect_vs_masked": paired_bootstrap(
                    predictions, complete, "aspect_only", "opinion_masked", lang=lang
                ),
                "full_vs_masked": paired_bootstrap(
                    predictions, complete, "opinion_masked", "full_text", lang=lang
                ),
            }
            for lang in LANGS
        },
    }
    return {
        "experiment": "041-opinion-mask-crosslingual-dimabsa",
        "n_selected_clusters": len(_case_ids(predictions)),
        "n_complete_clusters": len(complete),
        "n_language_target_instances_complete": len(complete) * len(LANGS),
        "n_expected_outputs": expected_count,
        "n_invalid_outputs": invalid,
        "invalid_rate": invalid_rate,
        "primary": primary,
        "secondary_descriptive": secondary,
        "analysis_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_unit": "shared source ID; resample all three aligned language versions together",
    }


def write_report(summary: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary.get("status") == "protocol_execution_failure":
        by_condition = ", ".join(
            f"{condition}: {count}"
            for condition, count in summary["invalid_by_condition"].items()
        )
        text = f"""# Experiment 041: opinion-mask cross-lingual DimABSA

## Execution outcome

The preregistered comparison was **not analyzed** because {summary['n_invalid_outputs']}/{summary['n_expected_outputs']} outputs were invalid ({summary['invalid_rate']:.2%}), exceeding the 2% stop threshold. Invalid counts by condition: {by_condition}. There were {summary['n_complete_clusters']} fully parseable aligned sentence clusters, but the protocol's coverage gate failed, so no RMSE, confidence interval, language contrast, or score-based claim is reported.

Some responses did not provide parseable numeric JSON, concentrated in the aspect-plus-opinion-only condition. Those responses remain invalid; they were not converted to scores or retried. This is a prompt/output feasibility failure for the registered design, not evidence for or against residual context effects.

## Limits and files

The public test split and labels are not blind. The three versions share IDs and gold labels and are treated as aligned clusters, not independent replications. Annotated-opinion masking leaves implicit and unannotated evaluative cues. This is one Qwen2.5-3B checkpoint and does not establish a property of LLMs generally. No source sentences or item-level derivatives are redistributed here.

- Aggregate execution record: `summary.json`
- Aggregate hardware/model/runtime provenance appears in `summary.json` when a private run-metadata file is available.
- Protocol: `docs/experiments/041-opinion-mask-crosslingual-dimabsa.md`
- Parser amendment: `docs/experiments/041-analysis-amendment.md`
- Per-item prompts and model outputs remain local in ignored `.context/`.
"""
        (out / "README.md").write_text(text)
        return

    p = summary["primary"]
    s = summary["secondary_descriptive"]
    text = f"""# Experiment 041: opinion-mask cross-lingual DimABSA

## Result

This run includes {summary['n_complete_clusters']} complete aligned sentence clusters across Russian, Ukrainian, and Tatar ({summary['n_language_target_instances_complete']} language-specific target instances). It produced {summary['n_expected_outputs']} outputs; {summary['n_invalid_outputs']} were invalid ({summary['invalid_rate']:.2%}).

The registered primary contrast, aspect-only RMSE minus opinion-masked-context RMSE, was **{p['estimate']:+.3f}** (95% sentence-cluster bootstrap interval {p['ci95'][0]:+.3f} to {p['ci95'][1]:+.3f}). Positive means the sentence residue helped after all annotated opinion phrases were masked. The preregistered support gate **{'passed' if p['registered_support_gate_passed'] else 'did not pass'}**.

Descriptive contrasts: opinion-masked minus full-text RMSE was {s['full_vs_masked']['estimate']:+.3f} (95% interval {s['full_vs_masked']['ci95'][0]:+.3f} to {s['full_vs_masked']['ci95'][1]:+.3f}); aspect-only minus aspect-plus-opinion RMSE was {s['aspect_opinion_vs_aspect']['estimate']:+.3f} (95% interval {s['aspect_opinion_vs_aspect']['ci95'][0]:+.3f} to {s['aspect_opinion_vs_aspect']['ci95'][1]:+.3f}).

## Interpretation limits

The public test split and its labels are not blind. The three versions share IDs and gold labels and are treated as aligned clusters, not independent replications. Annotated-opinion masking leaves implicit and unannotated evaluative cues. This is one Qwen2.5-3B checkpoint and does not establish a property of LLMs generally. Consult the preregistered protocol for the full design and limitations. No source sentences or item-level derivatives are redistributed here.

## Files

- Full aggregate metrics: `summary.json`
- Protocol: `docs/experiments/041-opinion-mask-crosslingual-dimabsa.md`
- Runner and analyzer: `scripts/run_opinion_mask_crosslingual_dimabsa_041.py`, `scripts/analyze_opinion_mask_crosslingual_dimabsa_041.py`
- Per-item prompts and model outputs remain local in ignored `.context/`.
"""
    (out / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(".context/exp041-private-predictions.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/opinion-mask-crosslingual-dimabsa-v1"))
    parser.add_argument("--run-metadata", type=Path, default=Path(".context/exp041-run-metadata.json"))
    args = parser.parse_args()
    summary = analyze(load_predictions(args.predictions))
    if args.run_metadata.exists():
        summary["execution_metadata"] = json.loads(args.run_metadata.read_text())
    write_report(summary, args.out)


if __name__ == "__main__":
    main()
