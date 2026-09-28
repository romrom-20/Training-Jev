"""Analyze the SIGHAN 2024 Chinese 3B decoder-by-context factorial."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from analyze_restaurant_domain_decoder_051 import (
    _context_gain,
    _errors,
    _expected_keys,
    _interaction,
    _invalid_counts,
    load_restaurant_rows,
)

CONDITIONS = ("aspect_only", "opinion_masked")
DECODERS = ("finite_grid", "free_greedy")
SEED = 20260952
BOOTSTRAPS = 10_000
EXPECTED_CHINESE_IDS = 1916
EXPECTED_ENGLISH_IDS = 963
PRIVATE_OUTPUT = Path(".context/exp052-private-predictions.jsonl")
PRIVATE_MANIFEST = Path(".context/exp052-run-manifest.json")
PARENT_051_OUTPUT = Path(".context/exp051-private-predictions.jsonl")
PARENT_051_MANIFEST = Path(".context/exp051-run-manifest.json")
OUTPUT_DIR = Path("results/sighan-chinese-decoder-transfer-v1")


def load_rows(path: Path) -> dict:
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_key = {
        (row["case_id"], row["condition"], row["decoder"]): row
        for row in lines
    }
    expected = EXPECTED_CHINESE_IDS * len(CONDITIONS) * len(DECODERS)
    if len(lines) != expected or len(by_key) != expected:
        raise ValueError(f"Expected {expected} unique outputs; found {len(lines)} rows")
    if {row["condition"] for row in lines} != set(CONDITIONS):
        raise ValueError("Unexpected evidence condition set")
    if {row["decoder"] for row in lines} != set(DECODERS):
        raise ValueError("Unexpected decoder set")
    return by_key


def _mode_counts(rows: dict) -> dict:
    result = {}
    for decoder in DECODERS:
        result[decoder] = {}
        for condition in CONDITIONS:
            scores = [
                tuple(row["prediction"])
                for row in rows.values()
                if row["decoder"] == decoder
                and row["condition"] == condition
                and row["prediction"] is not None
            ]
            result[decoder][condition] = [
                {"va": list(value), "count": count}
                for value, count in Counter(scores).most_common(8)
            ]
    return result


def analyze(chinese: dict, english: dict, manifest: dict | None = None) -> dict:
    chinese_ids = sorted({key[0] for key in chinese})
    english_ids = sorted({key[0] for key in english})
    expected = {
        (case_id, condition, decoder)
        for case_id in chinese_ids
        for condition in CONDITIONS
        for decoder in DECODERS
    }
    if len(chinese_ids) != EXPECTED_CHINESE_IDS or set(chinese) != expected:
        raise ValueError("Chinese run must contain all four cells for 1,925 source IDs")
    if len(english_ids) != EXPECTED_ENGLISH_IDS or set(english) != _expected_keys(english):
        raise ValueError("Experiment 051 parent must contain all four cells for 963 source IDs")

    for rows, label in ((chinese, "Chinese"), (english, "English")):
        for row in rows.values():
            prediction = row["prediction"]
            if prediction is not None and (
                len(prediction) != 2 or not np.isfinite(prediction).all()
            ):
                raise ValueError(f"Non-finite VA prediction in {label} run")

    invalid_chinese = _invalid_counts(chinese)
    invalid_english = _invalid_counts(english)
    invalid_free = sum(invalid_chinese["free_greedy"].values())
    invalid_rate = invalid_free / (EXPECTED_CHINESE_IDS * len(CONDITIONS))
    summary = {
        "experiment": "052-sighan-chinese-decoder-transfer",
        "interpretation": "adaptive independent-release cross-language transfer; exploratory",
        "n_chinese_clusters": len(chinese_ids),
        "n_english_restaurant_clusters": len(english_ids),
        "n_chinese_outputs": len(chinese),
        "invalid_by_decoder_condition": invalid_chinese,
        "invalid_english_parent_by_decoder_condition": invalid_english,
        "invalid_free_outputs": invalid_free,
        "invalid_free_rate": invalid_rate,
        "analysis_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_unit": "source sentence ID, retaining evidence, decoders and both VA dimensions",
        "condition_semantics_correction": {
            "aspect_only": "Review-text field is [NOT PROVIDED]; target aspect is supplied.",
            "opinion_masked": "Full review is supplied with all annotated non-null opinion spans replaced by [MASKED]; target aspect is supplied.",
            "context_gain_estimand": "RMSE(aspect-only prompt) - RMSE(opinion-masked full-review prompt): utility or harm of residual non-opinion review context.",
            "does_not_measure": "The effect of visible/unmasked opinion words.",
            "note": "The frozen protocol's prose incorrectly said aspect_only showed the full review. The executed runner's inherited prompt builder and analyzed rows implement the semantics above; the estimates are unchanged.",
        },
        "output_mode_counts_posthoc": _mode_counts(chinese),
        "score_analysis_performed": False,
    }
    if manifest is not None:
        summary["run_provenance"] = {
            key: manifest[key]
            for key in (
                "protocol_sha256", "source_revision", "source_files", "source_hashes",
                "selected_valence_buckets", "model", "model_revision", "device",
                "n_grid_candidates", "invalid_by_decoder_condition", "output_sha256",
                "generation_seconds_this_process_only",
            )
        }
    if invalid_rate > 0.02:
        summary["status"] = "protocol_execution_failure"
        summary["reason"] = "free-output invalid rate exceeded 2%; all Chinese score contrasts withheld"
        return summary

    complete_chinese = [
        case_id for case_id in chinese_ids
        if all(
            chinese[(case_id, condition, decoder)]["prediction"] is not None
            for condition in CONDITIONS for decoder in DECODERS
        )
    ]
    complete_english = [
        case_id for case_id in english_ids
        if all(
            english[(case_id, condition, decoder)]["prediction"] is not None
            for condition in CONDITIONS for decoder in DECODERS
        )
    ]
    if not complete_chinese or not complete_english:
        raise ValueError("No complete source clusters remain")

    chinese_errors = _errors(chinese, complete_chinese)
    english_errors = _errors(english, complete_english)
    gains = {}
    for offset, decoder in enumerate(DECODERS):
        estimate = _context_gain(chinese_errors, decoder, complete_chinese)
        rng = np.random.default_rng(SEED + offset)
        draws = np.empty(BOOTSTRAPS)
        for draw in range(BOOTSTRAPS):
            sample = rng.choice(complete_chinese, size=len(complete_chinese), replace=True).tolist()
            draws[draw] = _context_gain(chinese_errors, decoder, sample)
        ci = [float(value) for value in np.quantile(draws, [0.025, 0.975])]
        gains[decoder] = {
            "estimate": estimate,
            "ci95": ci,
            "n_clusters": len(complete_chinese),
            "registered_context_gain_gate_passed": estimate >= 0.25 and ci[0] > 0,
        }

    primary = _interaction(chinese_errors, complete_chinese)
    rng_primary = np.random.default_rng(SEED + 2)
    primary_draws = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        sample = rng_primary.choice(complete_chinese, size=len(complete_chinese), replace=True).tolist()
        primary_draws[draw] = _interaction(chinese_errors, sample)
    primary_ci = [float(value) for value in np.quantile(primary_draws, [0.025, 0.975])]

    chinese_interaction = primary
    english_interaction = _interaction(english_errors, complete_english)
    transfer_delta = english_interaction - chinese_interaction
    rng_english = np.random.default_rng(SEED + 3)
    rng_chinese = np.random.default_rng(SEED + 4)
    delta_draws = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        english_sample = rng_english.choice(
            complete_english, size=len(complete_english), replace=True
        ).tolist()
        chinese_sample = rng_chinese.choice(
            complete_chinese, size=len(complete_chinese), replace=True
        ).tolist()
        delta_draws[draw] = _interaction(english_errors, english_sample) - _interaction(
            chinese_errors, chinese_sample
        )
    delta_ci = [float(value) for value in np.quantile(delta_draws, [0.025, 0.975])]
    summary.update(
        {
            "status": "scored",
            "score_analysis_performed": True,
            "n_complete_chinese_clusters": len(complete_chinese),
            "n_complete_english_clusters": len(complete_english),
            "context_gain_by_decoder": gains,
            "primary_decoder_interaction": {
                "contrast": "context_gain(finite_grid) - context_gain(free_greedy)",
                "estimate": primary,
                "ci95": primary_ci,
                "n_clusters": len(complete_chinese),
                "registered_interaction_gate_passed": primary >= 0.25 and primary_ci[0] > 0,
            },
            "cross_release_language_transfer_secondary": {
                "contrast": "English restaurant interaction - Chinese restaurant interaction",
                "english_interaction": english_interaction,
                "chinese_interaction": chinese_interaction,
                "estimate": transfer_delta,
                "ci95": delta_ci,
                "n_english_clusters": len(complete_english),
                "n_chinese_clusters": len(complete_chinese),
                "confirmatory": False,
                "interpretation_limit": "Language and source release differ together; this descriptive contrast cannot identify which explains a difference.",
            },
        }
    )
    return summary


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] == "protocol_execution_failure":
        result = (
            f"The free-output invalid gate failed ({summary['invalid_free_outputs']}/"
            f"{EXPECTED_CHINESE_IDS * len(CONDITIONS)} invalid); all Chinese score contrasts were withheld."
        )
    else:
        finite = summary["context_gain_by_decoder"]["finite_grid"]
        free = summary["context_gain_by_decoder"]["free_greedy"]
        primary = summary["primary_decoder_interaction"]
        secondary = summary["cross_release_language_transfer_secondary"]
        result = (
            f"On the SIGHAN Chinese restaurant test split, finite-grid context gain was "
            f"{finite['estimate']:.3f} (95% source-ID interval [{finite['ci95'][0]:.3f}, "
            f"{finite['ci95'][1]:.3f}]) and free-greedy gain was {free['estimate']:.3f} "
            f"(95% interval [{free['ci95'][0]:.3f}, {free['ci95'][1]:.3f}]). The registered "
            f"interaction was {primary['estimate']:.3f} (95% interval "
            f"[{primary['ci95'][0]:.3f}, {primary['ci95'][1]:.3f}]); its practical gate "
            f"{'passed' if primary['registered_interaction_gate_passed'] else 'did not pass'}. "
            f"The descriptive English-minus-Chinese interaction difference was "
            f"{secondary['estimate']:.3f} (95% independent-release interval "
            f"[{secondary['ci95'][0]:.3f}, {secondary['ci95'][1]:.3f}])."
        )
    readme = f"""# Experiment 052: Qwen2.5-3B SIGHAN Chinese transfer

## Result

{result}

The test is a public, separately curated Chinese restaurant-review release with human continuous aspect-linked VA labels. It is an external release and cross-language transfer; it is not pretraining-blind, changes language and corpus at once, and remains in the restaurant domain. The English-minus-Chinese secondary interval is descriptive and cannot identify language effects separately from release effects.

**Interpretation correction:** the executed `aspect_only` prompt contains the target aspect and `[NOT PROVIDED]` for review text. The `opinion_masked` prompt contains the full review with all annotated opinion spans replaced by `[MASKED]`. Therefore the registered contrast measures utility or harm from the remaining non-opinion review context; it does not measure the value of visible opinion words. The frozen protocol's prose describing `aspect_only` as showing the complete review was incorrect; see `docs/experiments/052-interpretation-correction.md`. The run and estimates are unchanged.

Post-hoc most common numeric outputs by cell are recorded in `summary.json`. Treat them as exploratory mechanism clues.

No review text, target aspects, opinions, IDs, raw generations, or item predictions are published; private outputs remain in ignored `.context/`.

- Protocol: `docs/experiments/052-sighan-chinese-decoder-transfer.md`
- Runner/analyzer: `scripts/run_sighan_chinese_decoder_transfer_052.py`, `scripts/analyze_sighan_chinese_decoder_transfer_052.py`
- Aggregate result and provenance: `summary.json`
"""
    (output / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PRIVATE_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=PRIVATE_MANIFEST)
    parser.add_argument("--english-input", type=Path, default=PARENT_051_OUTPUT)
    parser.add_argument("--english-manifest", type=Path, default=PARENT_051_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    summary = analyze(
        load_rows(args.input),
        load_restaurant_rows(args.english_input),
        json.loads(args.manifest.read_text()) if args.manifest.exists() else None,
    )
    write_report(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
