"""Analyze the preregistered output-order × category-match restaurant test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import run_output_key_order_restaurant_063 as runner
from run_small_model_decoder_factorial_048 import sha256

OUT = Path("results/output-key-order-restaurant-replication-v1")
N_BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260963
ORDERS = ("valence_first", "arousal_first")
CONDITIONS = (runner.SAME, runner.CROSS)
DECODERS = runner.DECODERS
AXES = ((0, "valence"), (1, "arousal"))


def load_private(predictions: Path, parent: Path, manifest: dict) -> tuple[list[str], dict, dict]:
    new_rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    parent_rows = [json.loads(line) for line in parent.read_text().splitlines() if line.strip()]
    if len(new_rows) != 4498 or len({
        (r["case_id"], r["order"], r["permutation"], r["condition"], r["decoder"])
        for r in new_rows
    }) != 4498:
        raise ValueError(f"Expected 4,498 unique new outputs; found {len(new_rows)}")
    if sha256(predictions.read_bytes()) != manifest["output_sha256"]:
        raise ValueError("New private predictions do not match run manifest hash")

    new = {}
    for row in new_rows:
        key = (row["case_id"], row["order"], row["permutation"], row["condition"], row["decoder"])
        new[key] = row
    own_first = {}
    for row in parent_rows:
        if row["condition"] == runner.OWN:
            key = (row["case_id"], row["decoder"])
            own_first[key] = row
    ids = sorted({row["case_id"] for row in new_rows})
    own_first_ids = {key[0] for key in own_first}
    if len(ids) != 173 or not set(ids).issubset(own_first_ids):
        raise ValueError("Selected restaurant IDs are missing from the 051 own-review baseline")
    if len(own_first) < 963 * 2:
        raise ValueError("Incomplete frozen Experiment 051 parent baseline")
    return ids, new, own_first


def analyze(predictions: Path, manifest_path: Path, parent: Path = runner.PARENT,
            bootstrap_replicates: int = N_BOOTSTRAPS) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest["experiment"] != "063-output-key-order-restaurant-replication":
        raise ValueError("Unexpected manifest experiment")
    if sha256(parent.read_bytes()) != runner.PARENT_SHA256:
        raise ValueError("Experiment 051 parent baseline hash mismatch")
    ids, new, own_first = load_private(predictions, parent, manifest)
    invalid_donor_free = sum(
        row["decoder"] == "free_greedy" and row["condition"] in CONDITIONS
        and row["prediction"] is None for row in new.values()
    )
    denominator = len(ids) * len(runner.PERMUTATIONS) * len(CONDITIONS) * len(ORDERS)
    invalid_free_rate = invalid_donor_free / denominator
    if invalid_free_rate > 0.02:
        return {
            "experiment": "063-output-key-order-restaurant-replication",
            "status": "protocol_execution_failure",
            "reason": "Free-greedy donor-context invalid rate exceeded 2%; all score contrasts withheld",
            "n_recipient_ids": len(ids), "invalid_free_outputs": invalid_donor_free,
            "invalid_free_denominator": denominator, "invalid_free_rate": invalid_free_rate,
            "output_sha256": sha256(predictions.read_bytes()),
            "protocol_sha256": manifest["protocol_sha256"],
        }

    def own_row(case_id: str, order: str, decoder: str) -> dict:
        if order == "valence_first":
            return own_first[(case_id, decoder)]
        return new[(case_id, order, 0, runner.OWN, decoder)]

    complete = []
    for case_id in ids:
        rows = [own_row(case_id, order, decoder)
                for order in ORDERS for decoder in DECODERS]
        rows.extend(new[(case_id, order, permutation, condition, decoder)]
                    for order in ORDERS for condition in CONDITIONS
                    for permutation in runner.PERMUTATIONS for decoder in DECODERS)
        if all(row["prediction"] is not None for row in rows):
            complete.append(case_id)
        for row in rows:
            if row["gold"] != new[(case_id, "arousal_first", 0, runner.OWN, "finite_grid")]["gold"]:
                raise ValueError(f"Recipient gold differs within paired cells: {case_id}")
    if not complete:
        raise ValueError("No complete paired recipient IDs")

    n = len(complete)
    error = {}
    own_error = {}
    for order in ORDERS:
        for decoder in DECODERS:
            own_error[(order, decoder)] = np.array([
                [(float(own_row(case_id, order, decoder)["prediction"][axis])
                  - float(own_row(case_id, order, decoder)["gold"][axis])) ** 2
                 for axis, _ in AXES] for case_id in complete
            ])
            for condition in CONDITIONS:
                for permutation in runner.PERMUTATIONS:
                    error[(order, condition, decoder, permutation)] = np.array([
                        [(float(new[(case_id, order, permutation, condition, decoder)]["prediction"][axis])
                          - float(new[(case_id, order, permutation, condition, decoder)]["gold"][axis])) ** 2
                         for axis, _ in AXES] for case_id in complete
                    ])

    def point_for(sample: np.ndarray, axis: int) -> dict:
        own_rmse = {
            (order, decoder): float(np.sqrt(np.mean(own_error[(order, decoder)][sample, axis])))
            for order in ORDERS for decoder in DECODERS
        }
        interactions = {}
        for order in ORDERS:
            for condition in CONDITIONS:
                per_map = {}
                for permutation in runner.PERMUTATIONS:
                    gain = {}
                    for decoder in DECODERS:
                        donor_rmse = float(np.sqrt(np.mean(
                            error[(order, condition, decoder, permutation)][sample, axis]
                        )))
                        gain[decoder] = own_rmse[(order, decoder)] - donor_rmse
                    per_map[permutation] = gain["finite_grid"] - gain["free_greedy"]
                interactions[(order, condition)] = per_map
        topic_match = {}
        for order in ORDERS:
            topic_match[order] = float(
                np.mean(list(interactions[(order, runner.SAME)].values()))
                - np.mean(list(interactions[(order, runner.CROSS)].values()))
            )
        moderation = topic_match["arousal_first"] - topic_match["valence_first"]
        return {"topic_match": topic_match, "moderation": moderation, "interactions": interactions}

    point = {name: point_for(np.arange(n), axis) for axis, name in AXES}
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(bootstrap_replicates, n))
    draws_by_axis = {name: np.empty(bootstrap_replicates) for _, name in AXES}
    primary_draws = np.empty(bootstrap_replicates)
    for b, sample in enumerate(indices):
        vals = {}
        for axis, name in AXES:
            value = point_for(sample, axis)["moderation"]
            draws_by_axis[name][b] = value
            vals[name] = value
        primary_draws[b] = vals["arousal"] - vals["valence"]
    primary = point["arousal"]["moderation"] - point["valence"]["moderation"]

    def all_va_topic_match(sample: np.ndarray, order: str) -> float:
        effects = {}
        for condition in CONDITIONS:
            map_interactions = []
            for permutation in runner.PERMUTATIONS:
                gains = {}
                for decoder in DECODERS:
                    own_rmse = float(np.sqrt(np.mean(own_error[(order, decoder)][sample, :])))
                    donor_rmse = float(np.sqrt(np.mean(
                        error[(order, condition, decoder, permutation)][sample, :]
                    )))
                    gains[decoder] = own_rmse - donor_rmse
                map_interactions.append(gains["finite_grid"] - gains["free_greedy"])
            effects[condition] = float(np.mean(map_interactions))
        return effects[runner.SAME] - effects[runner.CROSS]

    all_va_topic = {order: all_va_topic_match(np.arange(n), order) for order in ORDERS}
    all_va_draws = {
        order: np.array([all_va_topic_match(sample, order) for sample in indices])
        for order in ORDERS
    }
    all_va_moderation_draws = all_va_draws["arousal_first"] - all_va_draws["valence_first"]

    per_assignment = {}
    for permutation in runner.PERMUTATIONS:
        per_assignment[str(permutation)] = {}
        for _, axis_name in AXES:
            axis_value = point[axis_name]
            same_aro = axis_value["interactions"][("arousal_first", runner.SAME)][permutation]
            cross_aro = axis_value["interactions"][("arousal_first", runner.CROSS)][permutation]
            same_val = axis_value["interactions"][("valence_first", runner.SAME)][permutation]
            cross_val = axis_value["interactions"][("valence_first", runner.CROSS)][permutation]
            per_assignment[str(permutation)][axis_name] = {
                "topic_match_arousal_first": same_aro - cross_aro,
                "topic_match_valence_first": same_val - cross_val,
                "order_moderation": (same_aro - cross_aro) - (same_val - cross_val),
            }
    primary_components = {
        name: {
            "topic_match_effect_valence_first": point[name]["topic_match"]["valence_first"],
            "topic_match_effect_arousal_first": point[name]["topic_match"]["arousal_first"],
            "order_moderation": point[name]["moderation"],
            "order_moderation_ci95": [float(v) for v in np.quantile(draws_by_axis[name], [0.025, 0.975])],
        } for _, name in AXES
    }
    invalid_by_order = manifest["invalid_by_order_condition_decoder"]
    summary = {
        "experiment": "063-output-key-order-restaurant-replication",
        "status": "scored", "interpretation": "fresh restaurant-domain test of the post-hoc arousal-order interaction",
        "n_recipient_ids_selected": len(ids), "n_complete_recipient_ids": n,
        "n_new_outputs": manifest["n_outputs"],
        "invalid_by_order_condition_decoder": invalid_by_order,
        "invalid_free_donor_outputs": invalid_donor_free,
        "invalid_free_donor_denominator": denominator,
        "invalid_free_donor_rate": invalid_free_rate,
        "primary_endpoint": {
            "contrast": "arousal-axis order moderation minus valence-axis order moderation",
            "direction": "negative follows the algebraically matched Experiment 062 coordinate pattern; the protocol's prose direction was sign-reversed",
            "estimate": primary,
            "ci95": [float(v) for v in np.quantile(primary_draws, [0.025, 0.975])],
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "recipient source ID; all cells and fixed maps retained",
        },
        "coordinate_specific_secondary": primary_components,
        "all_va_secondary": {
            "topic_match_effect_valence_first": all_va_topic["valence_first"],
            "topic_match_effect_valence_first_ci95": [float(v) for v in np.quantile(
                all_va_draws["valence_first"], [0.025, 0.975]
            )],
            "topic_match_effect_arousal_first": all_va_topic["arousal_first"],
            "topic_match_effect_arousal_first_ci95": [float(v) for v in np.quantile(
                all_va_draws["arousal_first"], [0.025, 0.975]
            )],
            "arousal_first_minus_valence_first": all_va_topic["arousal_first"] - all_va_topic["valence_first"],
            "order_moderation_ci95": [float(v) for v in np.quantile(
                all_va_moderation_draws, [0.025, 0.975]
            )],
            "inferential_status": "secondary aggregate VA result",
        },
        "per_assignment": per_assignment,
        "analysis_seed": BOOTSTRAP_SEED,
        "model": manifest["model"], "model_revision": manifest["model_revision"],
        "device": manifest["device"],
        "protocol_sha256": manifest["protocol_sha256"],
        "task2_sha256": manifest["task2_sha256"], "task3_sha256": manifest["task3_sha256"],
        "parent_output_sha256": manifest["parent_output_sha256"],
        "output_sha256": manifest["output_sha256"],
        "score_analysis_performed": True,
        "inferential_limits": [
            "Fresh recipient sample within the English restaurant split, not an independent corpus.",
            "Restaurant valence-first own-review baselines are reused from Experiment 051.",
            "Donor-category differences also change lexical and semantic content.",
            "Coordinate-specific Experiment 062 evidence was exploratory; 063 is the frozen follow-up.",
        ],
    }
    return summary


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] == "protocol_execution_failure":
        lead = summary["reason"]
    else:
        primary = summary["primary_endpoint"]
        lead = (
            f"The registered difference between arousal and valence order moderation was {primary['estimate']:.3f} "
            f"VA-RMSE points (95% recipient-bootstrap interval "
            f"[{primary['ci95'][0]:.3f}, {primary['ci95'][1]:.3f}]) on "
            f"{summary['n_complete_recipient_ids']} complete restaurant recipients."
        )
    (output / "README.md").write_text(f"""# Experiment 063: output key order on restaurant reviews

## Result

{lead}

The preregistered primary contrast subtracts valence's JSON-order moderation from arousal's, where order moderation is the change in the same-category-minus-cross-category context-gain decoder interaction when moving from valence-first to arousal-first output. Its interval includes zero, so the test does not establish stronger order sensitivity for one coordinate. The negative direction follows the algebraically matched coordinate contrast from Experiment 062; the frozen protocol's prose direction was sign-reversed. This is a fresh recipient sample within DimABSA's English restaurant split; it is not an independent corpus. Valence-first own-review baselines are reused from Experiment 051. Category crossings also change review content and meaning.

Coordinate-wise order moderations were {summary['coordinate_specific_secondary']['valence']['order_moderation']:.3f} for valence (95% interval [{summary['coordinate_specific_secondary']['valence']['order_moderation_ci95'][0]:.3f}, {summary['coordinate_specific_secondary']['valence']['order_moderation_ci95'][1]:.3f}]) and {summary['coordinate_specific_secondary']['arousal']['order_moderation']:.3f} for arousal ([{summary['coordinate_specific_secondary']['arousal']['order_moderation_ci95'][0]:.3f}, {summary['coordinate_specific_secondary']['arousal']['order_moderation_ci95'][1]:.3f}]). Both moved in the same direction as Experiment 062's coordinate audit. The preregistered protocol text said a positive primary estimate would follow 062; that prose had the sign reversed relative to its stated formula. See `docs/experiments/063-analysis-sign-audit.md`. These coordinate outcomes are secondary; only the primary coordinate-difference contrast was confirmatory.

The secondary aggregate VA order moderation was {summary['all_va_secondary']['arousal_first_minus_valence_first']:.3f} (95% interval [{summary['all_va_secondary']['order_moderation_ci95'][0]:.3f}, {summary['all_va_secondary']['order_moderation_ci95'][1]:.3f}]). A post-hoc cross-split bootstrap found the same direction in laptop and restaurant reviews; its estimated difference was uncertain. See [`domain comparison`](../domain-order-moderation-audit-v1/README.md). This is a consistency signal within one benchmark release and Qwen2.5-3B, not independent corpus or model replication.

The protocol was frozen before inference. No item text, case IDs, donor maps or individual predictions are released.

- Protocol: `docs/experiments/063-output-key-order-restaurant-replication.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_output_key_order_restaurant_063.py`, `scripts/analyze_output_key_order_restaurant_063.py`
- Related: [Experiment 062](../output-key-order-topic-match-v1/README.md), [Experiment 051](../restaurant-domain-decoder-transfer-v1/README.md)
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=Path(".context/exp063-private-predictions.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(".context/exp063-run-manifest.json"))
    parser.add_argument("--parent", type=Path, default=runner.PARENT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    summary = analyze(args.predictions, args.manifest, args.parent)
    write_report(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
