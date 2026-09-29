"""Post-hoc check of the output-order moderation's direction across two domains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from run_small_model_decoder_factorial_048 import sha256

ROOT = Path(".context")
OUT = Path("results/domain-order-moderation-audit-v1")
FILES = {
    "exp050": ROOT / "exp050-private-predictions.jsonl",
    "exp051": ROOT / "exp051-private-predictions.jsonl",
    "exp057": ROOT / "exp057-private-predictions.jsonl",
    "exp060": ROOT / "exp060-private-predictions.jsonl",
    "exp062": ROOT / "exp062-private-predictions.jsonl",
    "exp063": ROOT / "exp063-private-predictions.jsonl",
}
HASHES = {
    "exp050": "890a16d6363e72549c7ccad9099dece3ce021ab78ab3d8071acde18774500f5f",
    "exp051": "3f37ff8d630fde9776552389a8d41a4aab9d9b6ea127e24142337de68fd99ca7",
    "exp057": "21e7fe02b43433fa4b61b5dfe790efef38c57e8a0e240e4443ffe40ffb110e31",
    "exp060": "ab721276ec320c546da2757a16575acc30eb36ce86df3a911402b3e5a91f9d94",
    "exp062": "30cb805ffc61beb2f67b78a0017a06781117de2b3f546475eb270e7e86655c8f",
    "exp063": "0640dc9f3894cec1f890c270bea7a1680311345abfb670c97b5b80ad5229f080",
}
ORDERS = ("valence_first", "arousal_first")
CONDITIONS = ("same", "cross")
DECODERS = ("finite_grid", "free_greedy")
PERMUTATIONS = (1, 2, 3)
SEED = 20260965


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def index_rows(rows: list[dict], condition: str, order: str | None = None) -> dict:
    indexed = {}
    for row in rows:
        if row["condition"] != condition:
            continue
        if order is not None and row.get("order") != order:
            continue
        if condition == "opinion_masked":
            key = (row["case_id"], row["decoder"])
        else:
            key = (row["case_id"], row.get("permutation", 0), row["decoder"])
        indexed[key] = row
    return indexed


def build_laptop_data() -> tuple[list[str], dict, dict]:
    rows = {name: read_rows(FILES[name]) for name in ("exp050", "exp057", "exp060", "exp062")}
    own_val = index_rows(rows["exp050"], "opinion_masked")
    # Experiment 062's entire output file uses arousal-first serialization; the
    # order was fixed by its protocol and is not repeated on each row.
    own_ar = index_rows(rows["exp062"], "opinion_masked")
    same_val = index_rows(rows["exp057"], "category_polarity_matched_context")
    cross_val = index_rows(rows["exp060"], "cross_category_polarity_matched_context")
    same_ar = index_rows(rows["exp062"], "category_polarity_matched_context")
    cross_ar = index_rows(rows["exp062"], "cross_category_polarity_matched_context")
    ids = sorted({key[0] for key in same_ar})
    data = {
        ("valence_first", "own"): own_val,
        ("arousal_first", "own"): own_ar,
        ("valence_first", "same"): same_val,
        ("valence_first", "cross"): cross_val,
        ("arousal_first", "same"): same_ar,
        ("arousal_first", "cross"): cross_ar,
    }
    if len(ids) != 184:
        raise ValueError(f"Expected 184 laptop recipients, found {len(ids)}")
    return ids, data, rows


def build_restaurant_data() -> tuple[list[str], dict, dict]:
    rows = {name: read_rows(FILES[name]) for name in ("exp051", "exp063")}
    own_val = index_rows(rows["exp051"], "opinion_masked")
    own_ar = index_rows(rows["exp063"], "opinion_masked", "arousal_first")
    same_val = index_rows(rows["exp063"], "same_category_context", "valence_first")
    cross_val = index_rows(rows["exp063"], "cross_category_context", "valence_first")
    same_ar = index_rows(rows["exp063"], "same_category_context", "arousal_first")
    cross_ar = index_rows(rows["exp063"], "cross_category_context", "arousal_first")
    ids = sorted({key[0] for key in same_ar})
    data = {
        ("valence_first", "own"): own_val,
        ("arousal_first", "own"): own_ar,
        ("valence_first", "same"): same_val,
        ("valence_first", "cross"): cross_val,
        ("arousal_first", "same"): same_ar,
        ("arousal_first", "cross"): cross_ar,
    }
    complete = []
    for case_id in ids:
        okay = all(
            data[(order, "own")][(case_id, decoder)]["prediction"] is not None
            for order in ORDERS for decoder in DECODERS
        ) and all(
            data[(order, condition)].get((case_id, permutation, decoder), {}).get("prediction") is not None
            for order in ORDERS for condition in CONDITIONS
            for permutation in PERMUTATIONS for decoder in DECODERS
        )
        if okay:
            complete.append(case_id)
    if len(ids) != 173 or len(complete) != 172:
        raise ValueError(f"Unexpected restaurant completeness: {len(ids)} selected, {len(complete)} complete")
    return complete, data, rows


def errors(ids: list[str], data: dict, order: str, condition: str,
           decoder: str, permutation: int | None) -> np.ndarray:
    rows = data[(order, condition)]
    values = []
    for case_id in ids:
        row = rows[(case_id, decoder)] if permutation is None else rows[(case_id, permutation, decoder)]
        if row["prediction"] is None:
            raise ValueError(f"Invalid prediction in complete set: {case_id}/{order}/{condition}/{decoder}")
        values.append([
            (float(row["prediction"][axis]) - float(row["gold"][axis])) ** 2
            for axis in (0, 1)
        ])
    return np.asarray(values, dtype=np.float64)


def prepared_errors(ids: list[str], data: dict) -> dict:
    arrays = {}
    for order in ORDERS:
        for decoder in DECODERS:
            arrays[(order, "own", decoder, None)] = errors(
                ids, data, order, "own", decoder, None
            )
            for condition in CONDITIONS:
                for permutation in PERMUTATIONS:
                    arrays[(order, condition, decoder, permutation)] = errors(
                        ids, data, order, condition, decoder, permutation
                    )
    return arrays


def order_moderation(arrays: dict, sample: np.ndarray) -> float:
    topic = {}
    for order in ORDERS:
        own_rmse = {
            decoder: np.sqrt(np.mean(arrays[(order, "own", decoder, None)][sample]))
            for decoder in DECODERS
        }
        effects = {}
        for condition in CONDITIONS:
            per_map = []
            for permutation in PERMUTATIONS:
                gains = {}
                for decoder in DECODERS:
                    donor_rmse = np.sqrt(np.mean(
                        arrays[(order, condition, decoder, permutation)][sample]
                    ))
                    gains[decoder] = own_rmse[decoder] - donor_rmse
                per_map.append(gains["finite_grid"] - gains["free_greedy"])
            effects[condition] = float(np.mean(per_map))
        topic[order] = effects["same"] - effects["cross"]
    return topic["arousal_first"] - topic["valence_first"]


def analyze(bootstrap_replicates: int = 10_000) -> dict:
    for name, path in FILES.items():
        if sha256(path.read_bytes()) != HASHES[name]:
            raise ValueError(f"Frozen {name} output hash mismatch")
    laptop_ids, laptop, _ = build_laptop_data()
    restaurant_ids, restaurant, _ = build_restaurant_data()
    laptop_arrays = prepared_errors(laptop_ids, laptop)
    restaurant_arrays = prepared_errors(restaurant_ids, restaurant)
    laptop_point = order_moderation(laptop_arrays, np.arange(len(laptop_ids)))
    restaurant_point = order_moderation(restaurant_arrays, np.arange(len(restaurant_ids)))
    rng = np.random.default_rng(SEED)
    laptop_draws = np.empty(bootstrap_replicates)
    restaurant_draws = np.empty(bootstrap_replicates)
    domain_draws = np.empty(bootstrap_replicates)
    for index in range(bootstrap_replicates):
        laptop_sample = rng.integers(0, len(laptop_ids), size=len(laptop_ids))
        restaurant_sample = rng.integers(0, len(restaurant_ids), size=len(restaurant_ids))
        laptop_draws[index] = order_moderation(laptop_arrays, laptop_sample)
        restaurant_draws[index] = order_moderation(restaurant_arrays, restaurant_sample)
        domain_draws[index] = restaurant_draws[index] - laptop_draws[index]
    return {
        "experiment": "063-domain-order-moderation-audit",
        "status": "post-hoc exploratory domain comparison",
        "estimand": "restaurant minus laptop difference in (same-minus-cross category decoder interaction, arousal-first minus valence-first), all-VA RMSE",
        "laptop": {
            "n_recipient_ids": len(laptop_ids), "estimate": laptop_point,
            "ci95": [float(v) for v in np.quantile(laptop_draws, [0.025, 0.975])],
        },
        "restaurant": {
            "n_complete_recipient_ids": len(restaurant_ids), "estimate": restaurant_point,
            "ci95": [float(v) for v in np.quantile(restaurant_draws, [0.025, 0.975])],
        },
        "restaurant_minus_laptop": {
            "estimate": restaurant_point - laptop_point,
            "ci95": [float(v) for v in np.quantile(domain_draws, [0.025, 0.975])],
            "bootstrap_replicates": bootstrap_replicates,
            "seed": SEED,
            "resampling": "independent recipient-cluster resampling within each domain; fixed donor maps retained",
        },
        "provenance": {name: HASHES[name] for name in FILES},
        "limits": [
            "This domain comparison was not preregistered and follows the Experiment 063 outcomes.",
            "Both sources are splits within the DimABSA release; this is not independent corpus replication.",
            "Valence-first restaurant own-review baselines are reused from Experiment 051.",
            "The contrast does not isolate a pure category label or generalize beyond Qwen2.5-3B.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    summary = analyze(args.bootstrap_replicates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    laptop, restaurant, delta = summary["laptop"], summary["restaurant"], summary["restaurant_minus_laptop"]
    readme = f"""# Post-hoc audit: output-order moderation across laptop and restaurant reviews

This comparison was devised after Experiment 063. It is exploratory and does not change the preregistered Experiment 063 endpoint.

The same-minus-cross-category context-gain decoder interaction changed by **{laptop['estimate']:.3f}** points when output order changed in laptop reviews (95% recipient-bootstrap interval [{laptop['ci95'][0]:.3f}, {laptop['ci95'][1]:.3f}]) and by **{restaurant['estimate']:.3f}** in restaurant reviews ([{restaurant['ci95'][0]:.3f}, {restaurant['ci95'][1]:.3f}]). Both estimates are negative, matching direction across the two DimABSA splits. The restaurant-minus-laptop difference was **{delta['estimate']:.3f}** ([{delta['ci95'][0]:.3f}, {delta['ci95'][1]:.3f}]); this interval includes zero, so the data do not establish a different effect size across splits. Treat this as a post-hoc consistency signal, not a general result.

The 062 and 063 private outputs, own-review baselines, three fixed donor maps, decoders and two VA coordinates were kept together by recipient in an independent domain bootstrap. All six source output hashes are pinned in `summary.json`.
"""
    (args.output_dir / "README.md").write_text(readme)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
