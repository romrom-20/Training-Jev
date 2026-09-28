"""Analyze Experiment 049 with paired source-ID cluster bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONDITIONS = ("aspect_only", "opinion_masked")
DECODERS = ("finite_grid", "free_greedy")
SEED = 20260949
BOOTSTRAPS = 10_000
EXPECTED_IDS = 943


def load_rows(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_key = {
        (row["case_id"], row["condition"], row["decoder"]): row
        for row in rows
    }
    expected = EXPECTED_IDS * len(CONDITIONS) * len(DECODERS)
    if len(rows) != expected or len(by_key) != expected:
        raise ValueError(f"Expected {expected} unique rows, found {len(rows)}")
    if {row["condition"] for row in rows} != set(CONDITIONS):
        raise ValueError("Unexpected input-condition set")
    if {row["decoder"] for row in rows} != set(DECODERS):
        raise ValueError("Unexpected decoder set")
    return by_key


def analyze(by_key: dict, manifest: dict | None = None) -> dict:
    ids = sorted({key[0] for key in by_key})
    if len(ids) != EXPECTED_IDS:
        raise ValueError(f"Expected {EXPECTED_IDS} source IDs; found {len(ids)}")
    expected = {
        (case_id, condition, decoder)
        for case_id in ids
        for condition in CONDITIONS
        for decoder in DECODERS
    }
    if set(by_key) != expected:
        raise ValueError("Decoder factorial does not form the complete paired grid")

    invalid = {
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
    free_invalid = sum(invalid["free_greedy"].values())
    free_invalid_rate = free_invalid / (EXPECTED_IDS * len(CONDITIONS))
    summary = {
        "experiment": "049-laptop-domain-decoder-transfer",
        "interpretation": "adaptive independent-domain transfer test; exploratory",
        "n_clusters": EXPECTED_IDS,
        "n_outputs": len(by_key),
        "invalid_by_decoder_condition": invalid,
        "invalid_free_outputs": free_invalid,
        "invalid_free_rate": free_invalid_rate,
        "analysis_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_unit": "source sentence ID, retaining both input conditions and VA dimensions",
        "score_analysis_performed": False,
    }
    if manifest is not None:
        summary["run_provenance"] = {
            key: manifest[key]
            for key in (
                "protocol_sha256", "source_revision", "source_file", "source_sha256",
                "selected_valence_buckets", "model", "model_revision", "device",
                "n_grid_candidates", "invalid_by_decoder_condition", "output_sha256",
                "generation_seconds_this_process_only",
            )
        }
    if free_invalid_rate > 0.02:
        summary["status"] = "protocol_execution_failure"
        summary["reason"] = "free-output invalid rate exceeded 2%; all score contrasts withheld"
        return summary

    complete_ids = [
        case_id for case_id in ids
        if all(
            by_key[(case_id, condition, decoder)]["prediction"] is not None
            for condition in CONDITIONS for decoder in DECODERS
        )
    ]
    for row in by_key.values():
        prediction = row["prediction"]
        if prediction is not None and (
            len(prediction) != 2 or not np.isfinite(prediction).all()
        ):
            raise ValueError("Non-finite VA output reached the analyzer")

    squared_errors = {
        decoder: {
            condition: {
                case_id: np.asarray(
                    [
                        (
                            float(by_key[(case_id, condition, decoder)]["prediction"][dim])
                            - float(by_key[(case_id, condition, decoder)]["gold"][dim])
                        ) ** 2
                        for dim in (0, 1)
                    ],
                    dtype=float,
                )
                for case_id in complete_ids
            }
            for condition in CONDITIONS
        }
        for decoder in DECODERS
    }

    def context_gain(decoder: str, sample: list[str]) -> float:
        aspect = np.sqrt(
            np.mean(np.concatenate([squared_errors[decoder]["aspect_only"][case_id] for case_id in sample]))
        )
        context = np.sqrt(
            np.mean(np.concatenate([squared_errors[decoder]["opinion_masked"][case_id] for case_id in sample]))
        )
        return float(aspect - context)

    gains = {}
    for offset, decoder in enumerate(DECODERS):
        estimate = context_gain(decoder, complete_ids)
        rng = np.random.default_rng(SEED + offset)
        draws = np.empty(BOOTSTRAPS)
        for draw in range(BOOTSTRAPS):
            sample = rng.choice(complete_ids, size=len(complete_ids), replace=True).tolist()
            draws[draw] = context_gain(decoder, sample)
        ci = [float(value) for value in np.quantile(draws, [0.025, 0.975])]
        gains[decoder] = {
            "estimate": estimate,
            "ci95": ci,
            "n_clusters": len(complete_ids),
            "registered_context_gain_gate_passed": estimate >= 0.25 and ci[0] > 0,
        }

    interaction = context_gain("finite_grid", complete_ids) - context_gain("free_greedy", complete_ids)
    rng = np.random.default_rng(SEED + 2)
    interaction_draws = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        sample = rng.choice(complete_ids, size=len(complete_ids), replace=True).tolist()
        interaction_draws[draw] = context_gain("finite_grid", sample) - context_gain("free_greedy", sample)
    interaction_ci = [float(value) for value in np.quantile(interaction_draws, [0.025, 0.975])]

    summary.update(
        {
            "status": "scored",
            "score_analysis_performed": True,
            "n_complete_clusters": len(complete_ids),
            "context_gain_by_decoder": gains,
            "primary_decoder_interaction": {
                "contrast": "context_gain(finite_grid) - context_gain(free_greedy)",
                "estimate": interaction,
                "ci95": interaction_ci,
                "n_clusters": len(complete_ids),
                "interpretation": "positive means the finite grid amplifies the opinion-masked context gain",
                "registered_interaction_gate_passed": interaction >= 0.25 and interaction_ci[0] > 0,
            },
        }
    )
    return summary


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] == "protocol_execution_failure":
        sentence = (
            f"The free-output invalid gate failed ({summary['invalid_free_outputs']}/"
            f"1,886 invalid); all score contrasts were withheld."
        )
    else:
        finite = summary["context_gain_by_decoder"]["finite_grid"]
        free = summary["context_gain_by_decoder"]["free_greedy"]
        interaction = summary["primary_decoder_interaction"]
        sentence = (
            f"On the English laptop domain, the finite-grid context gain was {finite['estimate']:.3f} "
            f"(95% source-ID interval [{finite['ci95'][0]:.3f}, {finite['ci95'][1]:.3f}]) and the "
            f"free-greedy gain was {free['estimate']:.3f} (95% interval "
            f"[{free['ci95'][0]:.3f}, {free['ci95'][1]:.3f}]). The primary decoder interaction was "
            f"{interaction['estimate']:.3f} (95% interval "
            f"[{interaction['ci95'][0]:.3f}, {interaction['ci95'][1]:.3f}]); its practical gate "
            f"{'passed' if interaction['registered_interaction_gate_passed'] else 'did not pass'}."
        )
    readme = f"""# Experiment 049: English laptop-domain decoder transfer

## Result

{sentence}

The design is an adaptive transfer test on the full eligible English laptop test split in the pinned DimABSA release. It uses the same Qwen2.5-1.5B model, one-decimal prompt, 6,561-value finite grid, and free-greedy parser as Experiment 048. This is a separate source-domain sample, but it is not an independent corpus or blind evaluation.

No review text, target aspects, IDs, raw generations, or item predictions are published; private outputs remain in ignored `.context/`.

- Protocol: `docs/experiments/049-laptop-domain-decoder-transfer.md`
- Runner/analyzer: `scripts/run_laptop_decoder_transfer_049.py`, `scripts/analyze_laptop_decoder_transfer_049.py`
- Aggregate result and provenance: `summary.json`
"""
    (output / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(".context/exp049-private-predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(".context/exp049-run-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/laptop-decoder-transfer-v1"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else None
    summary = analyze(load_rows(args.predictions), manifest)
    write_report(summary, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
