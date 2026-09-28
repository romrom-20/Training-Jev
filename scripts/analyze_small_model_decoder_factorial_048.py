"""Aggregate analysis for the matched-prompt decoder factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LANGS = ("rus", "ukr", "tat")
CONDITIONS = ("aspect_only", "opinion_masked")
DECODERS = ("finite_grid", "free_greedy")
SEED = 20260948
BOOTSTRAPS = 10_000


def load_rows(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_key = {
        (row["case_id"], row["lang"], row["condition"], row["decoder"]): row
        for row in rows
    }
    expected = 217 * len(LANGS) * len(CONDITIONS) * len(DECODERS)
    if len(rows) != expected or len(by_key) != expected:
        raise ValueError(f"Expected {expected} unique rows, found {len(rows)}")
    if {row["condition"] for row in rows} != set(CONDITIONS):
        raise ValueError("Unexpected input-condition set")
    if {row["decoder"] for row in rows} != set(DECODERS):
        raise ValueError("Unexpected decoder set")
    return by_key


def analyze(by_key: dict, manifest: dict | None = None) -> dict:
    ids = sorted({key[0] for key in by_key})
    if len(ids) != 217:
        raise ValueError(f"Expected 217 frozen source IDs; found {len(ids)}")
    expected = {
        (case_id, lang, condition, decoder)
        for case_id in ids
        for lang in LANGS
        for condition in CONDITIONS
        for decoder in DECODERS
    }
    if set(by_key) != expected:
        raise ValueError("Decoder factorial does not form the complete paired grid")
    invalid_by_arm = {
        decoder: {
            condition: sum(
                row["prediction"] is None
                for row in by_key.values()
                if row["decoder"] == decoder and row["condition"] == condition
            )
            for condition in CONDITIONS
        }
        for decoder in DECODERS
    }
    invalid_free = sum(invalid_by_arm["free_greedy"].values())
    invalid_rate_free = invalid_free / (217 * len(LANGS) * len(CONDITIONS))
    summary = {
        "experiment": "048-small-model-decoder-factorial",
        "interpretation": "adaptive matched-prompt decoder comparison; exploratory only",
        "n_clusters": len(ids),
        "n_outputs": len(by_key),
        "invalid_by_decoder_condition": invalid_by_arm,
        "invalid_free_outputs": invalid_free,
        "invalid_free_rate": invalid_rate_free,
        "analysis_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_unit": "source review ID, retaining all languages, input conditions, and VA dimensions",
        "score_analysis_performed": False,
    }
    if manifest is not None:
        summary["run_provenance"] = {
            key: manifest[key]
            for key in (
                "parent_043_output_sha256", "source_revision", "source_hashes", "model",
                "model_revision", "device", "invalid_by_decoder_condition", "output_sha256",
                "generation_seconds_this_process_only",
            )
        }
    if invalid_rate_free > 0.02:
        summary["status"] = "protocol_execution_failure"
        summary["reason"] = "free-output invalid rate exceeded the preregistered 2% gate; all score contrasts withheld"
        return summary
    complete_ids = [
        case_id for case_id in ids
        if all(
            by_key[(case_id, lang, condition, decoder)]["prediction"] is not None
            for lang in LANGS for condition in CONDITIONS for decoder in DECODERS
        )
    ]
    if len(complete_ids) < 100:
        raise ValueError("Fewer than 100 complete source-ID clusters remain")
    for row in by_key.values():
        if row["prediction"] is not None:
            if len(row["prediction"]) != 2 or not np.isfinite(row["prediction"]).all():
                raise ValueError("Non-finite VA output reached the analyzer")

    errors = {
        decoder: {
            condition: {
                case_id: np.asarray(
                    [
                        (
                            float(by_key[(case_id, lang, condition, decoder)]["prediction"][dim])
                            - float(by_key[(case_id, lang, condition, decoder)]["gold"][dim])
                        ) ** 2
                        for lang in LANGS for dim in (0, 1)
                    ],
                    dtype=float,
                )
                for case_id in complete_ids
            }
            for condition in CONDITIONS
        }
        for decoder in DECODERS
    }

    def gain(decoder: str, sampled: list[str]) -> float:
        aspect = np.sqrt(np.mean(np.concatenate([errors[decoder]["aspect_only"][x] for x in sampled])))
        context = np.sqrt(np.mean(np.concatenate([errors[decoder]["opinion_masked"][x] for x in sampled])))
        return float(aspect - context)

    gain_reports = {}
    for offset, decoder in enumerate(DECODERS):
        point = gain(decoder, complete_ids)
        draws = np.empty(BOOTSTRAPS)
        rng_arm = np.random.default_rng(SEED + offset)
        for index in range(BOOTSTRAPS):
            sample = rng_arm.choice(complete_ids, size=len(complete_ids), replace=True).tolist()
            draws[index] = gain(decoder, sample)
        ci = [float(x) for x in np.quantile(draws, [0.025, 0.975])]
        gain_reports[decoder] = {
            "estimate": point,
            "ci95": ci,
            "n_clusters": len(complete_ids),
            "registered_context_gain_gate_passed": point >= 0.25 and ci[0] > 0,
        }
    interaction = gain("finite_grid", complete_ids) - gain("free_greedy", complete_ids)
    rng_interaction = np.random.default_rng(SEED + 2)
    interaction_draws = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        sample = rng_interaction.choice(complete_ids, size=len(complete_ids), replace=True).tolist()
        interaction_draws[index] = gain("finite_grid", sample) - gain("free_greedy", sample)
    interaction_ci = [float(x) for x in np.quantile(interaction_draws, [0.025, 0.975])]

    by_language = {}
    for lang in LANGS:
        by_language[lang] = {}
        for decoder in DECODERS:
            reports = {}
            for condition in CONDITIONS:
                squared = [
                    (
                        float(by_key[(case_id, lang, condition, decoder)]["prediction"][dim])
                        - float(by_key[(case_id, lang, condition, decoder)]["gold"][dim])
                    ) ** 2
                    for case_id in complete_ids for dim in (0, 1)
                ]
                reports[f"{condition}_rmse"] = float(np.sqrt(np.mean(squared)))
            reports["context_gain"] = reports["aspect_only_rmse"] - reports["opinion_masked_rmse"]
            by_language[lang][decoder] = reports
    summary.update(
        {
            "status": "scored",
            "score_analysis_performed": True,
            "n_complete_clusters": len(complete_ids),
            "context_gain_by_decoder": gain_reports,
            "primary_decoder_interaction": {
                "contrast": "context_gain(finite_grid) - context_gain(free_greedy)",
                "estimate": interaction,
                "ci95": interaction_ci,
                "n_clusters": len(complete_ids),
                "interpretation": "positive means the finite grid amplifies the context gain",
                "registered_interaction_gate_passed": interaction >= 0.25 and interaction_ci[0] > 0,
            },
            "by_language_descriptive": by_language,
        }
    )
    return summary


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] == "protocol_execution_failure":
        result = (
            f"The free-output invalid gate failed ({summary['invalid_free_outputs']}/"
            f"1,302 invalid); all score comparisons were withheld."
        )
    else:
        finite = summary["context_gain_by_decoder"]["finite_grid"]
        free = summary["context_gain_by_decoder"]["free_greedy"]
        primary = summary["primary_decoder_interaction"]
        result = (
            f"With identical prompt wording, the finite-grid context gain was "
            f"{finite['estimate']:.3f} (95% source-ID interval "
            f"[{finite['ci95'][0]:.3f}, {finite['ci95'][1]:.3f}]) and the free-greedy gain was "
            f"{free['estimate']:.3f} (95% interval [{free['ci95'][0]:.3f}, {free['ci95'][1]:.3f}]). "
            f"Their preregistered interaction was {primary['estimate']:.3f} "
            f"(95% interval [{primary['ci95'][0]:.3f}, {primary['ci95'][1]:.3f}]); "
            f"the interaction rule {'passed' if primary['registered_interaction_gate_passed'] else 'did not pass'}."
        )
    runtime = summary.get("run_provenance", {}).get("generation_seconds_this_process_only")
    runtime_text = f"\nGeneration took {runtime / 60:.1f} minutes after model load." if runtime else ""
    (output / "README.md").write_text(
        f"""# Experiment 048: matched-prompt decoder comparison at 1.5B

## Result

{result}{runtime_text}

Both decoder arms used the same Qwen2.5-1.5B checkpoint, sample, aspect-only/opinion-masked inputs, and one-decimal prompt wording. Only the finite 6,561-value grammar differs from free greedy generation. This adaptive test reuses the same DimABSA release and held-out IDs; it is a within-sample decoder diagnostic, not an independent replication.

No review text, aspects, IDs, raw generations, or item predictions are published; private outputs remain in ignored `.context/`.

- Protocol: `docs/experiments/048-small-model-decoder-factorial.md`
- Runner/analyzer: `scripts/run_small_model_decoder_factorial_048.py`, `scripts/analyze_small_model_decoder_factorial_048.py`
- Aggregate result and provenance: `summary.json`
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(".context/exp048-private-predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(".context/exp048-run-manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("results/small-model-decoder-factorial-v1"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else None
    write_report(analyze(load_rows(args.predictions), manifest), args.out)


if __name__ == "__main__":
    main()
