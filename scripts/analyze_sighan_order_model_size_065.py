"""Paired analysis of the SIGHAN order-by-category interaction at two sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import run_sighan_output_key_order_065 as runner
from run_small_model_decoder_factorial_048 import sha256

OUT = Path("results/sighan-order-model-size-replication-v1")
PARENT_3B = Path(".context/exp052-private-predictions.jsonl")
PARENT_3B_SHA256 = "76671a3330e35fddfd915f15c86b49b90f8113af74041651162b7a81051765fe"
RUN_3B = Path(".context/exp064-private-predictions.jsonl")
RUN_3B_SHA256 = "48f7a8e2fe9d245d7742c7955b6349fe08a3935934f7ff0cd1db01e1eec9ace7"
MANIFEST_3B = Path(".context/exp064-run-manifest.json")
N_BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260965
ORDERS = runner.ORDERS
CONDITIONS = (runner.SAME, runner.CROSS)
DECODERS = runner.DECODERS
AXES = ((0, "valence"), (1, "arousal"))
MODELS = ("qwen-1.5b", "qwen-3b")


def read_unique(path: Path, expected: int) -> dict[tuple, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [(row["case_id"], row["order"], row["permutation"],
             row["condition"], row["decoder"]) for row in rows]
    if len(rows) != expected or len(set(keys)) != expected:
        raise ValueError(f"Expected {expected} unique rows from {path}; found {len(rows)}")
    return dict(zip(keys, rows))


def load_private(predictions: Path, manifest_path: Path, exp064: Path = RUN_3B,
                 exp052: Path = PARENT_3B) -> tuple[list[str], dict, dict, dict]:
    manifest = json.loads(manifest_path.read_text())
    if manifest["experiment"] != "065-sighan-order-model-size-replication":
        raise ValueError("Unexpected Experiment 065 manifest")
    if manifest["protocol_sha256"] != runner.PROTOCOL_SHA256:
        raise ValueError("Run manifest does not match frozen Experiment 065 protocol")
    if manifest["output_sha256"] != sha256(predictions.read_bytes()):
        raise ValueError("Experiment 065 outputs do not match the run manifest")
    if sha256(exp064.read_bytes()) != RUN_3B_SHA256:
        raise ValueError("Experiment 064 output hash mismatch")
    if sha256(exp052.read_bytes()) != PARENT_3B_SHA256:
        raise ValueError("Experiment 052 baseline hash mismatch")
    old = read_unique(exp064, 4680)
    new = read_unique(predictions, 5040)
    parent_rows = [json.loads(line) for line in exp052.read_text().splitlines() if line.strip()]
    parent_own = {}
    for row in parent_rows:
        if row["condition"] == runner.OWN:
            key = (row["case_id"], row["decoder"])
            if key in parent_own:
                raise ValueError(f"Duplicate Exp052 own-review baseline: {key}")
            parent_own[key] = row
    ids = sorted({key[0] for key in new})
    if len(ids) != 180:
        raise ValueError(f"Expected the frozen 180 SIGHAN IDs, found {len(ids)}")
    if {key[0] for key in old} != set(ids):
        raise ValueError("Experiments 064 and 065 do not use identical recipient IDs")
    for key, row in new.items():
        case_id, order, permutation, condition, decoder = key
        if condition in CONDITIONS:
            old_row = old[(case_id, order, permutation, condition, decoder)]
            if row["donor_id"] != old_row["donor_id"]:
                raise ValueError(f"Donor assignment changed across models: {key}")
    if not all((case_id, decoder) in parent_own for case_id in ids for decoder in DECODERS):
        raise ValueError("Exp052 is missing a 3B valence-first own-review baseline")
    return ids, new, old, parent_own


def analyze(predictions: Path, manifest_path: Path, exp064: Path = RUN_3B,
            exp052: Path = PARENT_3B, bootstrap_replicates: int = N_BOOTSTRAPS) -> dict:
    manifest = json.loads(manifest_path.read_text())
    ids, new, old, parent_own = load_private(predictions, manifest_path, exp064, exp052)
    invalid = {
        "qwen-1.5b": sum(row["decoder"] == "free_greedy" and row["condition"] in CONDITIONS
                          and row["prediction"] is None for row in new.values()),
        "qwen-3b": sum(row["decoder"] == "free_greedy" and row["condition"] in CONDITIONS
                       and row["prediction"] is None for row in old.values()),
    }
    denominator = len(ids) * len(runner.PERMUTATIONS) * len(CONDITIONS) * len(ORDERS)
    invalid_rate = {model: invalid[model] / denominator for model in MODELS}
    provenance = {
        "experiment": "065-sighan-order-model-size-replication",
        "protocol_sha256": manifest["protocol_sha256"],
        "output_sha256": manifest["output_sha256"],
        "exp064_output_sha256": RUN_3B_SHA256,
        "exp052_baseline_sha256": PARENT_3B_SHA256,
        "source_hashes": manifest["source_hashes"],
    }
    if any(rate > 0.02 for rate in invalid_rate.values()):
        return {
            **provenance, "status": "protocol_execution_failure",
            "reason": "A model's donor-context free-greedy invalid rate exceeded 2%; score comparison withheld",
            "invalid_free_outputs": invalid, "invalid_free_denominator": denominator,
            "invalid_free_rate": invalid_rate, "score_analysis_performed": False,
        }

    def own_row(model: str, case_id: str, order: str, decoder: str) -> dict:
        if model == "qwen-1.5b":
            return new[(case_id, order, 0, runner.OWN, decoder)]
        if order == "valence_first":
            return parent_own[(case_id, decoder)]
        return old[(case_id, order, 0, runner.OWN, decoder)]

    def donor_row(model: str, case_id: str, order: str, permutation: int,
                  condition: str, decoder: str) -> dict:
        table = new if model == "qwen-1.5b" else old
        return table[(case_id, order, permutation, condition, decoder)]

    complete = []
    for case_id in ids:
        cells = [own_row(model, case_id, order, decoder)
                 for model in MODELS for order in ORDERS for decoder in DECODERS]
        cells.extend(donor_row(model, case_id, order, permutation, condition, decoder)
                     for model in MODELS for order in ORDERS for condition in CONDITIONS
                     for permutation in runner.PERMUTATIONS for decoder in DECODERS)
        gold_vectors = {tuple(row["gold"]) for row in cells}
        if len(gold_vectors) != 1:
            raise ValueError(f"Gold VA values differ across paired cells for {case_id}")
        if all(row["prediction"] is not None for row in cells):
            complete.append(case_id)
    if not complete:
        raise ValueError("No complete IDs shared across the 1.5B and 3B studies")

    n = len(complete)
    own_sqerr, donor_sqerr = {}, {}
    for model in MODELS:
        for order in ORDERS:
            for decoder in DECODERS:
                own_sqerr[(model, order, decoder)] = np.array([
                    [(float(own_row(model, case_id, order, decoder)["prediction"][axis])
                      - float(own_row(model, case_id, order, decoder)["gold"][axis])) ** 2
                     for axis, _ in AXES] for case_id in complete
                ])
                for condition in CONDITIONS:
                    for permutation in runner.PERMUTATIONS:
                        donor_sqerr[(model, order, condition, decoder, permutation)] = np.array([
                            [(float(donor_row(model, case_id, order, permutation, condition, decoder)["prediction"][axis])
                              - float(donor_row(model, case_id, order, permutation, condition, decoder)["gold"][axis])) ** 2
                             for axis, _ in AXES] for case_id in complete
                        ])

    def category_effect(model: str, sample: np.ndarray, order: str) -> float:
        effects = {}
        for condition in CONDITIONS:
            interactions = []
            for permutation in runner.PERMUTATIONS:
                gain = {}
                for decoder in DECODERS:
                    own_rmse = float(np.sqrt(np.mean(own_sqerr[(model, order, decoder)][sample, :])))
                    donor_rmse = float(np.sqrt(np.mean(
                        donor_sqerr[(model, order, condition, decoder, permutation)][sample, :]
                    )))
                    gain[decoder] = own_rmse - donor_rmse
                interactions.append(gain["finite_grid"] - gain["free_greedy"])
            effects[condition] = float(np.mean(interactions))
        return effects[runner.SAME] - effects[runner.CROSS]

    def moderation(model: str, sample: np.ndarray) -> float:
        return (category_effect(model, sample, "arousal_first")
                - category_effect(model, sample, "valence_first"))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, n, size=(bootstrap_replicates, n))
    draws = {model: np.array([moderation(model, sample) for sample in samples])
             for model in MODELS}
    difference_draws = draws["qwen-1.5b"] - draws["qwen-3b"]
    point = {model: moderation(model, np.arange(n)) for model in MODELS}
    difference = point["qwen-1.5b"] - point["qwen-3b"]
    invalid_by_model = {
        "qwen-1.5b": manifest["invalid_by_order_condition_decoder"],
        "qwen-3b": json.loads(MANIFEST_3B.read_text())["invalid_by_order_condition_decoder"],
    }
    return {
        **provenance,
        "status": "scored",
        "interpretation": "same-recipient, same-map model-size comparison on SIGHAN 2024",
        "n_recipient_ids_selected": len(ids), "n_complete_paired_recipient_ids": n,
        "n_new_outputs": manifest["n_outputs"],
        "invalid_by_order_condition_decoder": invalid_by_model,
        "invalid_free_donor_outputs": invalid,
        "invalid_free_donor_denominator": denominator,
        "invalid_free_donor_rate": invalid_rate,
        "primary_endpoint": {
            "contrast": "Qwen2.5-1.5B aggregate order moderation minus Qwen2.5-3B aggregate order moderation",
            "estimate": difference,
            "ci95": [float(x) for x in np.quantile(difference_draws, [0.025, 0.975])],
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "recipient source ID; all model/order/decoder/condition/map cells paired",
        },
        "model_specific_secondary": {
            model: {
                "aggregate_order_moderation": point[model],
                "ci95": [float(x) for x in np.quantile(draws[model], [0.025, 0.975])],
            } for model in MODELS
        },
        "model_revisions": {
            "qwen-1.5b": manifest["model_revision"],
            "qwen-3b": json.loads(MANIFEST_3B.read_text())["model_revision"],
        },
        "device": manifest["device"],
        "score_analysis_performed": True,
        "inferential_limits": [
            "A model-size comparison within one Qwen family does not establish cross-family behavior.",
            "Both models use the same SIGHAN 2024 public test release and recipient set; pretraining exposure cannot be excluded.",
            "Both models are evaluated with the same English-language instruction on Chinese input.",
            "The confidence interval tests a size difference, not whether either model has a nonzero order effect.",
        ],
    }


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] != "scored":
        lead = summary["reason"]
    else:
        primary = summary["primary_endpoint"]
        lead = (f"The 1.5B-minus-3B difference in aggregate order moderation was {primary['estimate']:.3f} "
                f"VA-RMSE points (95% paired recipient-bootstrap interval [{primary['ci95'][0]:.3f}, "
                f"{primary['ci95'][1]:.3f}]) on {summary['n_complete_paired_recipient_ids']} complete IDs.")
    effects = summary.get("model_specific_secondary", {})
    (output / "README.md").write_text(f"""# Experiment 065: output-order model-size comparison on SIGHAN

## Result

{lead}

This paired difference is the registered primary endpoint. A confidence interval containing zero means the test did not detect a model-size difference; it does not prove transfer, especially if both model-specific estimates are near zero.

The model-specific aggregate order moderations were {effects.get('qwen-1.5b', {}).get('aggregate_order_moderation', float('nan')):.3f} for Qwen2.5-1.5B and {effects.get('qwen-3b', {}).get('aggregate_order_moderation', float('nan')):.3f} for Qwen2.5-3B. The study reuses the 180 SIGHAN recipients and exact donor maps from Experiment 064, adding {summary.get('n_new_outputs', 'no')} 1.5B generations. Invalid-output and complete-case counts are in `summary.json`.

This is a same-release, same-family model-size comparison, not an independent corpus replication. Chinese text is scored with English instructions, public-test pretraining exposure cannot be excluded, and donor category changes also alter review content. The experiment cannot explain the English/Chinese difference by itself.

- Frozen protocol: `docs/experiments/065-sighan-order-model-size-replication.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_sighan_output_key_order_065.py`, `scripts/analyze_sighan_order_model_size_065.py`
- Prior Chinese release test: [Experiment 064](../sighan-output-key-order-replication-v1/README.md)
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=runner.OUT)
    parser.add_argument("--manifest", type=Path, default=runner.MANIFEST)
    parser.add_argument("--exp064", type=Path, default=RUN_3B)
    parser.add_argument("--exp052", type=Path, default=PARENT_3B)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    summary = analyze(args.predictions, args.manifest, args.exp064, args.exp052)
    write_report(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
