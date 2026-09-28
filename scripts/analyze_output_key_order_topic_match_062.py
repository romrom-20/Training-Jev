"""Analyze the key-order robustness of the same/cross-category interaction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_context_swap_donor_robustness_054 as shared
import numpy as np

shared.EXPECTED_IDS = 184
shared.SEED = 20260962
OUT = Path("results/output-key-order-topic-match-v1")

OWN = "opinion_masked"
SAME = "category_polarity_matched_context"
CROSS = "cross_category_polarity_matched_context"


def split_arousal_first(path: Path) -> tuple[dict, dict, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    own, same, cross = {}, {}, {}
    for row in rows:
        if row["condition"] == OWN:
            key = (row["case_id"], OWN, row["decoder"])
            target = own
        elif row["condition"] == SAME:
            key = (row["case_id"], row["permutation"], "swapped_context", row["decoder"])
            target = same
        elif row["condition"] == CROSS:
            key = (row["case_id"], row["permutation"], "swapped_context", row["decoder"])
            target = cross
        else:
            raise ValueError(f"Unknown arousal-first condition: {row['condition']}")
        if key in target:
            raise ValueError(f"Duplicate arousal-first output: {key}")
        target[key] = row
    expected_own = 184 * 2
    expected_donors = 184 * 3 * 2
    if len(rows) != 2576 or len(own) != expected_own or len(same) != expected_donors or len(cross) != expected_donors:
        raise ValueError("Incomplete arousal-first 062 design")
    return own, same, cross


def interaction_by_condition(donors: dict, own: dict, ids: list[str]) -> dict[int, float]:
    estimates = {}
    for permutation in shared.PERMUTATIONS:
        advantages = {}
        for decoder in shared.DECODERS:
            donor_rows = {
                (case_id, "swapped_context", decoder): donors[
                    (case_id, permutation, "swapped_context", decoder)
                ] for case_id in ids
            }
            own_rows = {
                (case_id, OWN, decoder): own[(case_id, OWN, decoder)] for case_id in ids
            }
            advantages[decoder] = shared._rmse(
                donor_rows, ids, decoder, "swapped_context"
            ) - shared._rmse(own_rows, ids, decoder, OWN)
        estimates[permutation] = advantages["finite_grid"] - advantages["free_greedy"]
    return estimates


def load_valence_first(paths: tuple[Path, Path, Path]) -> tuple[dict, dict, dict]:
    own_rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
    own = {
        (row["case_id"], OWN, row["decoder"]): row
        for row in own_rows if row["condition"] == OWN
    }
    same_rows = [json.loads(line) for line in paths[1].read_text().splitlines() if line.strip()]
    same = {
        (row["case_id"], row["permutation"], "swapped_context", row["decoder"]): row
        for row in same_rows if row["condition"] == SAME
    }
    cross_rows = [json.loads(line) for line in paths[2].read_text().splitlines() if line.strip()]
    cross = {
        (row["case_id"], row["permutation"], "swapped_context", row["decoder"]): row
        for row in cross_rows if row["condition"] == CROSS
    }
    ids = sorted({key[0] for key in same})
    if len(ids) != 184 or len(own) < len(ids) * 2 or len(cross) != len(ids) * 6:
        raise ValueError("Incomplete frozen valence-first comparator outputs")
    return own, same, cross


def analyze(path: Path, manifest: dict, old_outputs: tuple[Path, Path, Path],
            bootstrap_replicates: int = 10_000) -> dict:
    own, same, cross = split_arousal_first(path)
    ids = sorted({key[0] for key in own})
    for donors in (same, cross):
        for (case_id, permutation, _, decoder), row in donors.items():
            if own[(case_id, OWN, decoder)]["gold"] != row["gold"]:
                raise ValueError(f"Gold mismatch for {case_id}/{permutation}/{decoder}")
            if row["prediction"] is not None and (
                len(row["prediction"]) != 2 or not np.isfinite(row["prediction"]).all()
            ):
                raise ValueError("Non-finite arousal-first prediction")

    invalid = {
        condition: sum(row["prediction"] is None for row in donor.values())
        for condition, donor in ((SAME, same), (CROSS, cross))
    }
    invalid_free = sum(
        row["decoder"] == "free_greedy" and row["prediction"] is None
        for donor in (same, cross) for row in donor.values()
    )
    denominator = 184 * 3 * 2
    summary = {
        "experiment": "062-output-key-order-topic-match",
        "interpretation": "arousal-first serialization robustness follow-up",
        "n_recipient_ids": len(ids),
        "n_new_outputs": 2576,
        "invalid_by_condition_decoder": invalid,
        "invalid_free_outputs": invalid_free,
        "invalid_free_denominator": denominator,
        "invalid_free_rate": invalid_free / denominator,
        "analysis_seed": 20260962,
        "bootstrap_seed_key_order_moderation": 20260963,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_unit": "recipient sentence ID, retaining all conditions, decoders and fixed mappings",
        "model": manifest["model"],
        "model_revision": manifest["model_revision"],
        "device": manifest["device"],
        "protocol_sha256": manifest["protocol_sha256"],
        "parent_output_sha256": manifest["parent_output_sha256"],
        "output_sha256": manifest["output_sha256"],
        "score_analysis_performed": False,
    }
    if invalid_free / denominator > 0.02:
        summary.update({
            "status": "protocol_execution_failure",
            "reason": "Free-greedy donor invalid rate exceeded 2%; all score contrasts withheld",
        })
        return summary

    complete = [case_id for case_id in ids if all(
        own[(case_id, OWN, decoder)]["prediction"] is not None
        and all(
            donors[(case_id, permutation, "swapped_context", decoder)]["prediction"] is not None
            for donors in (same, cross) for permutation in shared.PERMUTATIONS
        ) for decoder in shared.DECODERS
    )]
    if not complete:
        raise ValueError("No complete recipient IDs")
    same_by_map = interaction_by_condition(same, own, complete)
    cross_by_map = interaction_by_condition(cross, own, complete)
    point_same = float(np.mean(list(same_by_map.values())))
    point_cross = float(np.mean(list(cross_by_map.values())))
    point_difference = point_cross - point_same

    rng = np.random.default_rng(20260962)
    draws = np.empty(bootstrap_replicates)
    same_draws = np.empty(bootstrap_replicates)
    cross_draws = np.empty(bootstrap_replicates)
    for index in range(bootstrap_replicates):
        sample = rng.choice(complete, size=len(complete), replace=True).tolist()
        same_estimate = float(np.mean(list(interaction_by_condition(same, own, sample).values())))
        cross_estimate = float(np.mean(list(interaction_by_condition(cross, own, sample).values())))
        same_draws[index] = same_estimate
        cross_draws[index] = cross_estimate
        draws[index] = cross_estimate - same_estimate

    old_own, old_same, old_cross = load_valence_first(old_outputs)
    old_same_point = float(np.mean(list(interaction_by_condition(old_same, old_own, complete).values())))
    old_cross_point = float(np.mean(list(interaction_by_condition(old_cross, old_own, complete).values())))
    old_difference = old_cross_point - old_same_point
    rng_key = np.random.default_rng(20260963)
    key_draws = np.empty(bootstrap_replicates)
    for index in range(bootstrap_replicates):
        sample = rng_key.choice(complete, size=len(complete), replace=True).tolist()
        new_difference = float(np.mean(list(interaction_by_condition(cross, own, sample).values()))) - float(
            np.mean(list(interaction_by_condition(same, own, sample).values()))
        )
        old_difference_draw = float(np.mean(list(interaction_by_condition(old_cross, old_own, sample).values()))) - float(
            np.mean(list(interaction_by_condition(old_same, old_own, sample).values()))
        )
        key_draws[index] = new_difference - old_difference_draw

    summary.update({
        "status": "scored",
        "score_analysis_performed": True,
        "n_complete_recipient_ids": len(complete),
        "same_category_interaction": {
            "estimate": point_same,
            "ci95": [float(v) for v in np.quantile(same_draws, [0.025, 0.975])],
        },
        "cross_category_interaction": {
            "estimate": point_cross,
            "ci95": [float(v) for v in np.quantile(cross_draws, [0.025, 0.975])],
        },
        "per_assignment": {
            str(p): {
                "same_category_interaction": same_by_map[p],
                "cross_category_interaction": cross_by_map[p],
                "cross_minus_same": cross_by_map[p] - same_by_map[p],
            } for p in shared.PERMUTATIONS
        },
        "mean_primary_interaction": {
            "contrast": "arousal-first cross-category interaction minus same-category interaction",
            "estimate": point_difference,
            "ci95": [float(v) for v in np.quantile(draws, [0.025, 0.975])],
            "n_recipient_ids": len(complete),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": 20260962,
            "registered_followup": True,
            "independent_replication": False,
        },
        "key_order_moderation": {
            "contrast": "arousal-first cross-minus-same minus valence-first cross-minus-same",
            "estimate_arousal_first": point_difference,
            "estimate_valence_first": old_difference,
            "difference": point_difference - old_difference,
            "ci95": [float(v) for v in np.quantile(key_draws, [0.025, 0.975])],
            "bootstrap_seed": 20260963,
            "n_recipient_ids": len(complete),
            "inferential_status": "same public sample/maps; key-order robustness diagnostic",
        },
    })
    return summary


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] == "protocol_execution_failure":
        result = summary["reason"] + "."
    else:
        primary = summary["mean_primary_interaction"]
        moderation = summary["key_order_moderation"]
        result = (
            f"With arousal-first JSON, the cross-minus-same category interaction was "
            f"{primary['estimate']:.3f} (95% recipient-bootstrap interval "
            f"[{primary['ci95'][0]:.3f}, {primary['ci95'][1]:.3f}]). Its change from the "
            f"valence-first estimate was {moderation['difference']:.3f} "
            f"(95% paired interval [{moderation['ci95'][0]:.3f}, {moderation['ci95'][1]:.3f}])."
        )
    (output / "README.md").write_text(f"""# Experiment 062: output key order and topic match

## Result

{result}

The run reversed only the numeric JSON key order. Same-category and cross-category donors use the same recipients, polarities, assignments, target aspects and own-review baselines. The key-order comparison reuses the already observed valence-first outputs, so it is a registered robustness follow-up rather than independent replication.

General output-format sensitivity is already studied; this experiment tests only whether one key-order change alters the continuous-VA decoder-by-topic-match effect on this public laptop split and Qwen2.5 family. Donor categories differ in lexical and semantic content. No text, item IDs, donor maps, or individual predictions are published.

- Protocol: `docs/experiments/062-output-key-order-topic-match.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_output_key_order_topic_match_062.py`, `scripts/analyze_output_key_order_topic_match_062.py`
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(".context/exp062-private-predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(".context/exp062-run-manifest.json"))
    parser.add_argument("--exp050", type=Path, default=Path(".context/exp050-private-predictions.jsonl"))
    parser.add_argument("--exp057", type=Path, default=Path(".context/exp057-private-predictions.jsonl"))
    parser.add_argument("--exp060", type=Path, default=Path(".context/exp060-private-predictions.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    summary = analyze(args.predictions, manifest, (args.exp050, args.exp057, args.exp060))
    write_report(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
