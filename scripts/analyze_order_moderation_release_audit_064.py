"""Post-hoc release comparison of the all-VA output-order moderation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_domain_order_moderation_063 as english
import numpy as np
from run_small_model_decoder_factorial_048 import sha256

ROOT = Path(".context")
EXP052 = ROOT / "exp052-private-predictions.jsonl"
EXP064 = ROOT / "exp064-private-predictions.jsonl"
EXP052_SHA256 = "76671a3330e35fddfd915f15c86b49b90f8113af74041651162b7a81051765fe"
EXP064_SHA256 = "48f7a8e2fe9d245d7742c7955b6349fe08a3935934f7ff0cd1db01e1eec9ace7"
OUT = Path("results/order-moderation-release-audit-v1")
BOOTSTRAPS = 10_000
SEED = 20260966


def build_sighan_data() -> tuple[list[str], dict]:
    if sha256(EXP052.read_bytes()) != EXP052_SHA256:
        raise ValueError("Pinned Experiment 052 baseline hash mismatch")
    if sha256(EXP064.read_bytes()) != EXP064_SHA256:
        raise ValueError("Pinned Experiment 064 prediction hash mismatch")
    rows_052 = [json.loads(line) for line in EXP052.read_text().splitlines() if line.strip()]
    rows_064 = [json.loads(line) for line in EXP064.read_text().splitlines() if line.strip()]
    if len(rows_064) != 4680:
        raise ValueError(f"Expected 4,680 Experiment 064 outputs, found {len(rows_064)}")
    ids = sorted({row["case_id"] for row in rows_064})
    if len(ids) != 180:
        raise ValueError(f"Expected 180 frozen SIGHAN IDs, found {len(ids)}")
    parent_own = {(row["case_id"], row["decoder"]): row for row in rows_052
                  if row["condition"] == "opinion_masked"}
    own_ar = {(row["case_id"], row["decoder"]): row for row in rows_064
              if row["condition"] == "opinion_masked" and row["order"] == "arousal_first"}
    data = {
        ("valence_first", "own"): parent_own,
        ("arousal_first", "own"): own_ar,
    }
    for order in english.ORDERS:
        for condition, source in (("same", "same_category_context"),
                                  ("cross", "cross_category_context")):
            data[(order, condition)] = {
                (row["case_id"], row["permutation"], row["decoder"]): row
                for row in rows_064 if row["order"] == order and row["condition"] == source
            }
    complete = []
    for case_id in ids:
        rows = [data[(order, "own")].get((case_id, decoder))
                for order in english.ORDERS for decoder in english.DECODERS]
        rows.extend(data[(order, condition)].get((case_id, permutation, decoder))
                    for order in english.ORDERS for condition in english.CONDITIONS
                    for permutation in english.PERMUTATIONS for decoder in english.DECODERS)
        if all(row is not None and row["prediction"] is not None for row in rows):
            complete.append(case_id)
    if len(complete) != 173:
        raise ValueError(f"Expected 173 complete SIGHAN recipients, found {len(complete)}")
    return complete, data


def analyze(bootstrap_replicates: int = BOOTSTRAPS) -> dict:
    laptop_ids, laptop, _ = english.build_laptop_data()
    restaurant_ids, restaurant, _ = english.build_restaurant_data()
    sighan_ids, sighan = build_sighan_data()
    datasets = {
        "dimabsa_laptop": (laptop_ids, english.prepared_errors(laptop_ids, laptop)),
        "dimabsa_restaurant": (restaurant_ids, english.prepared_errors(restaurant_ids, restaurant)),
        "sighan_chinese": (sighan_ids, english.prepared_errors(sighan_ids, sighan)),
    }
    estimates = {
        name: english.order_moderation(arrays, np.arange(len(ids)))
        for name, (ids, arrays) in datasets.items()
    }
    rng = np.random.default_rng(SEED)
    draws = {name: np.empty(bootstrap_replicates) for name in datasets}
    delta_draws = np.empty(bootstrap_replicates)
    weight_laptop = len(laptop_ids) / (len(laptop_ids) + len(restaurant_ids))
    for i in range(bootstrap_replicates):
        sampled = {}
        for name, (ids, arrays) in datasets.items():
            sample = rng.integers(0, len(ids), size=len(ids))
            sampled[name] = english.order_moderation(arrays, sample)
            draws[name][i] = sampled[name]
        english_weighted = (
            weight_laptop * sampled["dimabsa_laptop"]
            + (1.0 - weight_laptop) * sampled["dimabsa_restaurant"]
        )
        delta_draws[i] = sampled["sighan_chinese"] - english_weighted
    return {
        "experiment": "posthoc-order-moderation-release-audit-064",
        "status": "exploratory, post-hoc",
        "estimand": "SIGHAN 3B order moderation minus recipient-count-weighted mean of two English DimABSA split estimates",
        "estimate_definition": "aggregate all-VA arousal-first minus valence-first change in same-minus-cross finite-minus-free context-gain interaction",
        "dataset_estimates": {
            name: {
                "n_complete_recipient_ids": len(datasets[name][0]),
                "estimate": estimates[name],
                "ci95": [float(v) for v in np.quantile(draws[name], [0.025, 0.975])],
            } for name in datasets
        },
        "sighan_minus_weighted_english": {
            "estimate": estimates["sighan_chinese"]
            - (weight_laptop * estimates["dimabsa_laptop"]
               + (1.0 - weight_laptop) * estimates["dimabsa_restaurant"]),
            "ci95": [float(v) for v in np.quantile(delta_draws, [0.025, 0.975])],
            "bootstrap_replicates": bootstrap_replicates,
            "seed": SEED,
            "resampling": "independent recipient-cluster resampling within each of three cohorts; fixed maps retained",
            "english_weights": {
                "dimabsa_laptop": weight_laptop,
                "dimabsa_restaurant": 1.0 - weight_laptop,
            },
        },
        "provenance": {
            "exp050": english.HASHES["exp050"], "exp051": english.HASHES["exp051"],
            "exp057": english.HASHES["exp057"], "exp060": english.HASHES["exp060"],
            "exp062": english.HASHES["exp062"], "exp063": english.HASHES["exp063"],
            "exp052": EXP052_SHA256, "exp064": EXP064_SHA256,
        },
        "limits": [
            "This comparison was not registered and was motivated by the observed 064 outcome.",
            "SIGHAN differs from DimABSA in language, domain, annotation release, and selected cohort; this does not isolate which factor differs.",
            "The English mean is a recipient-count-weighted summary of two within-release splits, not a prespecified population estimate.",
            "All data are public test releases and all estimates use one Qwen2.5-3B setup.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAPS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    summary = analyze(args.bootstrap_replicates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    en = summary["dataset_estimates"]
    diff = summary["sighan_minus_weighted_english"]
    text = f"""# Post-hoc release audit: output-order moderation across English and Chinese

This audit was added after Experiment 064 and is exploratory. It does not change the registered SIGHAN or model-size outcomes.

The all-VA order moderation estimates were {en['dimabsa_laptop']['estimate']:.3f} for English laptop reviews ([{en['dimabsa_laptop']['ci95'][0]:.3f}, {en['dimabsa_laptop']['ci95'][1]:.3f}]), {en['dimabsa_restaurant']['estimate']:.3f} for English restaurant reviews ([{en['dimabsa_restaurant']['ci95'][0]:.3f}, {en['dimabsa_restaurant']['ci95'][1]:.3f}]), and {en['sighan_chinese']['estimate']:.3f} for Chinese SIGHAN reviews ([{en['sighan_chinese']['ci95'][0]:.3f}, {en['sighan_chinese']['ci95'][1]:.3f}]). The post-hoc SIGHAN-minus-weighted-English difference was {diff['estimate']:.3f} (95% independent cohort-bootstrap interval [{diff['ci95'][0]:.3f}, {diff['ci95'][1]:.3f}]).

The nominal post-hoc interval narrowly excludes zero, but this only suggests that these selected benchmark cohorts differ under one prompt/model setup. Language, domain, dataset release, and cohort remain confounded. The two English sources share the DimABSA release; the Chinese source is a separate public release. Private rows and review text remain in ignored `.context/`; `summary.json` pins aggregate source checksums.
"""
    (args.output_dir / "README.md").write_text(text)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
