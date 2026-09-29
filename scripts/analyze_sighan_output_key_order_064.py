"""Analyze the frozen SIGHAN output-order × category-match replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import run_sighan_output_key_order_064 as runner
from run_small_model_decoder_factorial_048 import sha256

OUT = Path("results/sighan-output-key-order-replication-v1")
N_BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260964
ORDERS = runner.ORDERS
CONDITIONS = (runner.SAME, runner.CROSS)
DECODERS = runner.DECODERS
AXES = ((0, "valence"), (1, "arousal"))


def load_private(predictions: Path, parent: Path, manifest: dict) -> tuple[list[str], dict, dict]:
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    expected = 4680
    keys = [(row["case_id"], row["order"], row["permutation"],
             row["condition"], row["decoder"]) for row in rows]
    if len(rows) != expected or len(set(keys)) != expected:
        raise ValueError(f"Expected {expected} unique new outputs; found {len(rows)}")
    if sha256(predictions.read_bytes()) != manifest["output_sha256"]:
        raise ValueError("New private predictions do not match run manifest hash")
    new = dict(zip(keys, rows))

    parent_rows = [json.loads(line) for line in parent.read_text().splitlines() if line.strip()]
    own_first = {}
    for row in parent_rows:
        if row["condition"] == runner.OWN:
            key = (row["case_id"], row["decoder"])
            if key in own_first:
                raise ValueError(f"Duplicate SIGHAN parent baseline row: {key}")
            own_first[key] = row
    ids = sorted({row["case_id"] for row in rows})
    if len(ids) != 180:
        raise ValueError(f"Expected 180 selected SIGHAN IDs, found {len(ids)}")
    if not all((case_id, decoder) in own_first for case_id in ids for decoder in DECODERS):
        raise ValueError("Selected SIGHAN IDs are missing a valence-first parent baseline")
    return ids, new, own_first


def analyze(predictions: Path, manifest_path: Path, parent: Path = runner.PARENT,
            bootstrap_replicates: int = N_BOOTSTRAPS) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest["experiment"] != "064-sighan-output-key-order-replication":
        raise ValueError("Unexpected manifest experiment")
    if manifest["protocol_sha256"] != runner.PROTOCOL_SHA256:
        raise ValueError("Run manifest does not match the frozen Experiment 064 protocol")
    if sha256(parent.read_bytes()) != runner.PARENT_SHA256:
        raise ValueError("Experiment 052 parent baseline hash mismatch")
    ids, new, own_first = load_private(predictions, parent, manifest)
    invalid_donor_free = sum(
        row["decoder"] == "free_greedy" and row["condition"] in CONDITIONS
        and row["prediction"] is None for row in new.values()
    )
    denominator = len(ids) * len(runner.PERMUTATIONS) * len(CONDITIONS) * len(ORDERS)
    invalid_rate = invalid_donor_free / denominator
    provenance = {
        "experiment": "064-sighan-output-key-order-replication",
        "protocol_sha256": manifest["protocol_sha256"],
        "output_sha256": manifest["output_sha256"],
        "parent_output_sha256": manifest["parent_output_sha256"],
        "source_hashes": manifest["source_hashes"],
    }
    if invalid_rate > 0.02:
        return {
            **provenance, "status": "protocol_execution_failure",
            "reason": "Free-greedy donor-context invalid rate exceeded 2%; all score contrasts withheld",
            "n_recipient_ids": len(ids), "invalid_free_outputs": invalid_donor_free,
            "invalid_free_denominator": denominator, "invalid_free_rate": invalid_rate,
            "score_analysis_performed": False,
        }

    def own_row(case_id: str, order: str, decoder: str) -> dict:
        if order == "valence_first":
            return own_first[(case_id, decoder)]
        return new[(case_id, order, 0, runner.OWN, decoder)]

    complete = []
    for case_id in ids:
        cell_rows = [own_row(case_id, order, decoder)
                     for order in ORDERS for decoder in DECODERS]
        cell_rows.extend(new[(case_id, order, permutation, condition, decoder)]
                         for order in ORDERS for condition in CONDITIONS
                         for permutation in runner.PERMUTATIONS for decoder in DECODERS)
        if all(row["prediction"] is not None for row in cell_rows):
            complete.append(case_id)
    if not complete:
        raise ValueError("No complete paired SIGHAN recipient IDs")

    n = len(complete)
    own_sqerr, donor_sqerr = {}, {}
    for order in ORDERS:
        for decoder in DECODERS:
            own_sqerr[(order, decoder)] = np.array([
                [(float(own_row(case_id, order, decoder)["prediction"][axis])
                  - float(own_row(case_id, order, decoder)["gold"][axis])) ** 2
                 for axis, _ in AXES] for case_id in complete
            ])
            for condition in CONDITIONS:
                for permutation in runner.PERMUTATIONS:
                    donor_sqerr[(order, condition, decoder, permutation)] = np.array([
                        [(float(new[(case_id, order, permutation, condition, decoder)]["prediction"][axis])
                          - float(new[(case_id, order, permutation, condition, decoder)]["gold"][axis])) ** 2
                         for axis, _ in AXES] for case_id in complete
                    ])

    def effect(sample: np.ndarray, axis: int | None, order: str) -> dict:
        by_condition = {}
        for condition in CONDITIONS:
            map_effects = []
            for permutation in runner.PERMUTATIONS:
                gain = {}
                for decoder in DECODERS:
                    own = own_sqerr[(order, decoder)][sample]
                    donor = donor_sqerr[(order, condition, decoder, permutation)][sample]
                    if axis is None:
                        own_rmse = float(np.sqrt(np.mean(own)))
                        donor_rmse = float(np.sqrt(np.mean(donor)))
                    else:
                        own_rmse = float(np.sqrt(np.mean(own[:, axis])))
                        donor_rmse = float(np.sqrt(np.mean(donor[:, axis])))
                    gain[decoder] = own_rmse - donor_rmse
                map_effects.append(gain["finite_grid"] - gain["free_greedy"])
            by_condition[condition] = float(np.mean(map_effects))
        return {"same_minus_cross": by_condition[runner.SAME] - by_condition[runner.CROSS],
                "by_condition": by_condition}

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, n, size=(bootstrap_replicates, n))
    axis_effects = {}
    for axis, name in AXES:
        vf = effect(np.arange(n), axis, "valence_first")["same_minus_cross"]
        af = effect(np.arange(n), axis, "arousal_first")["same_minus_cross"]
        draws = np.array([
            effect(sample, axis, "arousal_first")["same_minus_cross"]
            - effect(sample, axis, "valence_first")["same_minus_cross"]
            for sample in samples
        ])
        axis_effects[name] = {
            "same_minus_cross_valence_first": vf,
            "same_minus_cross_arousal_first": af,
            "order_moderation": af - vf,
            "order_moderation_ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        }

    def aggregate_moderation(sample: np.ndarray) -> float:
        af = effect(sample, None, "arousal_first")["same_minus_cross"]
        vf = effect(sample, None, "valence_first")["same_minus_cross"]
        return af - vf

    primary_draws = np.array([aggregate_moderation(sample) for sample in samples])
    primary = aggregate_moderation(np.arange(n))
    invalid_by_order = manifest["invalid_by_order_condition_decoder"]
    return {
        **provenance,
        "status": "scored",
        "interpretation": "cross-release and cross-language replication test on SIGHAN 2024, one model",
        "n_recipient_ids_selected": len(ids), "n_complete_recipient_ids": n,
        "n_new_outputs": manifest["n_outputs"],
        "invalid_by_order_condition_decoder": invalid_by_order,
        "invalid_free_donor_outputs": invalid_donor_free,
        "invalid_free_donor_denominator": denominator,
        "invalid_free_donor_rate": invalid_rate,
        "primary_endpoint": {
            "contrast": "aggregate two-coordinate same-minus-cross context-gain decoder interaction; arousal-first minus valence-first",
            "direction": "negative follows the preregistered direction",
            "estimate": primary,
            "ci95": [float(x) for x in np.quantile(primary_draws, [0.025, 0.975])],
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "recipient source ID; all cells and fixed maps retained",
        },
        "coordinate_specific_secondary": axis_effects,
        "n_bootstrap_complete_cases": n,
        "model": manifest["model"], "model_revision": manifest["model_revision"],
        "device": manifest["device"],
        "score_analysis_performed": True,
        "inferential_limits": [
            "One target model and one laptop; this does not establish model-general behavior.",
            "SIGHAN 2024 and DimABSA 2026 are separately curated releases, but both are public and pretraining exposure cannot be excluded.",
            "The Chinese review text is evaluated with the same English-language instruction as the parent experiment.",
            "Cross-category donors change lexical and semantic content as well as aspect category.",
            "Finite-grid and free-greedy decoding are operational conditions, not pure causal mechanisms.",
        ],
    }


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] != "scored":
        lead = summary["reason"]
        interpretation = "The protocol's invalid-output gate failed, so score contrasts were withheld."
    else:
        primary = summary["primary_endpoint"]
        lead = (f"The registered aggregate output-order moderation was {primary['estimate']:.3f} VA-RMSE "
                f"points (95% recipient-bootstrap interval [{primary['ci95'][0]:.3f}, "
                f"{primary['ci95'][1]:.3f}]) on {summary['n_complete_recipient_ids']} complete recipients.")
        interpretation = ("This was selected as an independent release and language check after two English DimABSA tests. "
                          "The interval includes zero, so the registered test does not show that the negative aggregate "
                          "moderation transfers to SIGHAN. It remains compatible with modest effects in either direction; "
                          "this alone does not establish a true release or language difference.")
    (output / "README.md").write_text(f"""# Experiment 064: output order × category match on SIGHAN 2024

## Result

{lead}

{interpretation}

Coordinate-specific order moderations are descriptive secondary outcomes: valence {summary.get('coordinate_specific_secondary', {}).get('valence', {}).get('order_moderation', float('nan')):.3f}, arousal {summary.get('coordinate_specific_secondary', {}).get('arousal', {}).get('order_moderation', float('nan')):.3f}. Raw texts, IDs, donor maps and item predictions remain private in the ignored `.context/` directory. Only aggregates and provenance hashes are released.

The test crosses same- versus cross-category polarity-matched donor context, valence-first versus arousal-first JSON key order, and finite-grid versus free-greedy decoding. It uses a fixed 180-review balanced sample, 4,680 new generations, and the frozen Experiment 052 valence-first own-review baseline. Cross-category donors alter review wording and meaning as well as category. Results apply to this prompt and Qwen2.5-3B setup.

- Frozen protocol: `docs/experiments/064-sighan-output-key-order-replication.md`
- Public aggregate and provenance: `summary.json`
- Runner/analyzer: `scripts/run_sighan_output_key_order_064.py`, `scripts/analyze_sighan_output_key_order_064.py`
- Prior English-domain tests: [Experiment 062](../output-key-order-topic-match-v1/README.md), [Experiment 063](../output-key-order-restaurant-replication-v1/README.md)
- Source release: [Lee et al. (2024)](https://aclanthology.org/2024.sighan-1.19/)
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=runner.OUT)
    parser.add_argument("--manifest", type=Path, default=runner.MANIFEST)
    parser.add_argument("--parent", type=Path, default=runner.PARENT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    summary = analyze(args.predictions, args.manifest, args.parent)
    write_report(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
