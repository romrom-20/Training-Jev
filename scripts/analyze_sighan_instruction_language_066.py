"""Analyze the paired English-versus-Chinese instruction-language test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import run_sighan_chinese_decoder_transfer_052 as exp052
import run_sighan_instruction_language_066 as runner
import run_sighan_output_key_order_064 as exp064
from run_small_model_decoder_factorial_048 import sha256

OUT = Path("results/sighan-instruction-language-control-v1")
PARENT_052 = Path(".context/exp052-private-predictions.jsonl")
PARENT_052_SHA256 = "76671a3330e35fddfd915f15c86b49b90f8113af74041651162b7a81051765fe"
PARENT_064 = Path(".context/exp064-private-predictions.jsonl")
PARENT_064_SHA256 = "48f7a8e2fe9d245d7742c7955b6349fe08a3935934f7ff0cd1db01e1eec9ace7"
N_BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260966
ORDERS = runner.ORDERS
CONDITIONS = (runner.SAME, runner.CROSS)
DECODERS = runner.DECODERS
PERMUTATIONS = runner.PERMUTATIONS
LANGUAGES = ("english", "chinese")


def index_new(path: Path, expected: int, manifest: dict) -> dict[tuple, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [(row["case_id"], row["order"], row["permutation"],
             row["condition"], row["decoder"]) for row in rows]
    if len(rows) != expected or len(set(keys)) != expected:
        raise ValueError(f"Expected {expected} unique Experiment 066 outputs; found {len(rows)}")
    if sha256(path.read_bytes()) != manifest["output_sha256"]:
        raise ValueError("Experiment 066 output hash mismatch")
    return dict(zip(keys, rows))


def build_english(new_ids: set[str]) -> tuple[dict, dict]:
    rows_052 = [json.loads(line) for line in PARENT_052.read_text().splitlines() if line.strip()]
    rows_064 = [json.loads(line) for line in PARENT_064.read_text().splitlines() if line.strip()]
    own_valence = {}
    for row in rows_052:
        if row["condition"] == runner.OWN and row["case_id"] in new_ids:
            key = (row["case_id"], row["decoder"])
            if key in own_valence:
                raise ValueError(f"Duplicate Experiment 052 own baseline: {key}")
            own_valence[key] = row
    english = {}
    for row in rows_064:
        if row["case_id"] not in new_ids:
            continue
        if row["condition"] == runner.OWN and row["order"] == "arousal_first":
            key = (row["case_id"], row["order"], 0, runner.OWN, row["decoder"])
            if key in english:
                raise ValueError(f"Duplicate Experiment 064 own baseline: {key}")
            english[key] = row
        elif row["condition"] in CONDITIONS:
            key = (row["case_id"], row["order"], row["permutation"],
                   row["condition"], row["decoder"])
            if key in english:
                raise ValueError(f"Duplicate Experiment 064 donor cell: {key}")
            english[key] = row
    if not all((case_id, decoder) in own_valence for case_id in new_ids for decoder in DECODERS):
        raise ValueError("Experiment 052 is missing an English-instruction own baseline")
    if len(rows_064) != 4680:
        raise ValueError("Unexpected Experiment 064 output count")
    return english, own_valence


def analyze(predictions: Path = runner.OUT, manifest_path: Path = runner.MANIFEST,
            source_dir: Path = exp052.SOURCE_DIR,
            bootstrap_replicates: int = N_BOOTSTRAPS) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest["experiment"] != "066-sighan-instruction-language-control":
        raise ValueError("Unexpected Experiment 066 manifest")
    if manifest["protocol_sha256"] != runner.PROTOCOL_SHA256:
        raise ValueError("Manifest does not match frozen Experiment 066 protocol")
    if sha256(PARENT_052.read_bytes()) != PARENT_052_SHA256:
        raise ValueError("Experiment 052 baseline hash mismatch")
    if sha256(PARENT_064.read_bytes()) != PARENT_064_SHA256:
        raise ValueError("Experiment 064 output/map hash mismatch")
    source_rows, source_hashes = exp052.read_source(source_dir)
    cases_180, _ = exp064.select_cases(exp052.select_cases(source_rows))
    expected_cases, _ = runner.select_cases(cases_180)
    expected_ids = {case["case_id"] for case in expected_cases}
    new = index_new(predictions, 2464, manifest)
    ids = sorted({key[0] for key in new})
    if set(ids) != expected_ids:
        raise ValueError("Experiment 066 IDs differ from the frozen hash-ranked subset")
    expected_keys = {
        (case_id, order, permutation, condition, decoder)
        for case_id in expected_ids for order in ORDERS for decoder in DECODERS
        for condition, permutations in ((runner.OWN, (0,)),
                                        (runner.SAME, PERMUTATIONS),
                                        (runner.CROSS, PERMUTATIONS))
        for permutation in permutations
    }
    if set(new) != expected_keys:
        raise ValueError("Experiment 066 output cells do not match the complete frozen factorial")
    old, own_english_valence = build_english(expected_ids)
    for key, row in new.items():
        if row["condition"] not in CONDITIONS:
            continue
        english_row = old[(key[0], key[1], key[2], key[3], key[4])]
        if row["donor_id"] != english_row["donor_id"]:
            raise ValueError(f"Donor mapping changed across prompt languages: {key}")

    invalid = {language: 0 for language in LANGUAGES}
    for language, table in (("english", old), ("chinese", new)):
        invalid[language] = sum(
            row["decoder"] == "free_greedy" and row["condition"] in CONDITIONS
            and row["prediction"] is None for row in table.values()
        )
    denominator = len(ids) * len(PERMUTATIONS) * len(CONDITIONS) * len(ORDERS)
    invalid_rate = {language: invalid[language] / denominator for language in LANGUAGES}
    provenance = {
        "experiment": "066-sighan-instruction-language-control",
        "protocol_sha256": manifest["protocol_sha256"],
        "output_sha256": manifest["output_sha256"],
        "exp064_output_sha256": PARENT_064_SHA256,
        "exp052_baseline_sha256": PARENT_052_SHA256,
        "source_hashes": source_hashes,
    }
    if any(rate > 0.02 for rate in invalid_rate.values()):
        return {
            **provenance, "status": "protocol_execution_failure",
            "reason": "An instruction condition's free-greedy donor invalid rate exceeded 2%; score contrast withheld",
            "invalid_free_outputs": invalid,
            "invalid_free_denominator": denominator,
            "invalid_free_rate": invalid_rate,
            "score_analysis_performed": False,
        }

    def row_for(language: str, case_id: str, order: str, permutation: int,
                condition: str, decoder: str) -> dict:
        if language == "chinese":
            return new[(case_id, order, permutation, condition, decoder)]
        if condition == runner.OWN and order == "valence_first":
            return own_english_valence[(case_id, decoder)]
        return old[(case_id, order, permutation, condition, decoder)]

    complete = []
    for case_id in ids:
        rows = [row_for(language, case_id, order, 0, runner.OWN, decoder)
                for language in LANGUAGES for order in ORDERS for decoder in DECODERS]
        rows.extend(row_for(language, case_id, order, permutation, condition, decoder)
                    for language in LANGUAGES for order in ORDERS for condition in CONDITIONS
                    for permutation in PERMUTATIONS for decoder in DECODERS)
        if len({tuple(row["gold"]) for row in rows}) != 1:
            raise ValueError(f"Gold labels differ across prompt-language cells: {case_id}")
        if all(row["prediction"] is not None for row in rows):
            complete.append(case_id)
    if not complete:
        raise ValueError("No complete paired recipients across instruction languages")

    n = len(complete)
    own_sqerr, donor_sqerr = {}, {}
    for language in LANGUAGES:
        for order in ORDERS:
            for decoder in DECODERS:
                own_sqerr[(language, order, decoder)] = np.array([
                    [(float(row_for(language, case_id, order, 0, runner.OWN, decoder)["prediction"][axis])
                      - float(row_for(language, case_id, order, 0, runner.OWN, decoder)["gold"][axis])) ** 2
                     for axis in (0, 1)] for case_id in complete
                ])
                for condition in CONDITIONS:
                    for permutation in PERMUTATIONS:
                        donor_sqerr[(language, order, condition, decoder, permutation)] = np.array([
                            [(float(row_for(language, case_id, order, permutation, condition, decoder)["prediction"][axis])
                              - float(row_for(language, case_id, order, permutation, condition, decoder)["gold"][axis])) ** 2
                             for axis in (0, 1)] for case_id in complete
                        ])

    def category_effect(language: str, sample: np.ndarray, order: str) -> float:
        effects = {}
        for condition in CONDITIONS:
            interactions = []
            for permutation in PERMUTATIONS:
                gains = {}
                for decoder in DECODERS:
                    own_rmse = float(np.sqrt(np.mean(own_sqerr[(language, order, decoder)][sample, :])))
                    donor_rmse = float(np.sqrt(np.mean(
                        donor_sqerr[(language, order, condition, decoder, permutation)][sample, :]
                    )))
                    gains[decoder] = own_rmse - donor_rmse
                interactions.append(gains["finite_grid"] - gains["free_greedy"])
            effects[condition] = float(np.mean(interactions))
        return effects[runner.SAME] - effects[runner.CROSS]

    def moderation(language: str, sample: np.ndarray) -> float:
        return category_effect(language, sample, "arousal_first") - category_effect(
            language, sample, "valence_first"
        )

    point = {language: moderation(language, np.arange(n)) for language in LANGUAGES}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    language_draws = {language: np.empty(bootstrap_replicates) for language in LANGUAGES}
    difference_draws = np.empty(bootstrap_replicates)
    for i in range(bootstrap_replicates):
        sample = rng.integers(0, n, size=n)
        for language in LANGUAGES:
            language_draws[language][i] = moderation(language, sample)
        difference_draws[i] = language_draws["chinese"][i] - language_draws["english"][i]
    estimate = point["chinese"] - point["english"]
    return {
        **provenance,
        "status": "scored",
        "n_recipient_ids_selected": len(ids),
        "n_complete_paired_recipient_ids": n,
        "n_new_outputs": manifest["n_outputs"],
        "invalid_free_donor_outputs": invalid,
        "invalid_free_donor_denominator": denominator,
        "invalid_free_donor_rate": invalid_rate,
        "primary_endpoint": {
            "contrast": "Chinese-instruction order moderation minus English-instruction order moderation on identical SIGHAN IDs/maps",
            "estimate": estimate,
            "ci95": [float(v) for v in np.quantile(difference_draws, [0.025, 0.975])],
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_unit": "recipient ID; both instruction languages and all cells kept paired",
        },
        "language_specific_secondary": {
            language: {
                "aggregate_order_moderation": point[language],
                "ci95": [float(v) for v in np.quantile(language_draws[language], [0.025, 0.975])],
            } for language in LANGUAGES
        },
        "model": manifest["model"], "model_revision": manifest["model_revision"],
        "device": manifest["device"],
        "score_analysis_performed": True,
        "inferential_limits": [
            "This paired test estimates prompt-language moderation within the selected SIGHAN cohort and Qwen2.5-3B only.",
            "The translated instructions were not independently rated by a professional translator.",
            "English condition outputs are reused from prior studies; Chinese condition is newly generated.",
            "Results cannot isolate the broader English/Chinese benchmark difference, where domain and release also change.",
            "Cross-category donors change lexical and semantic review content as well as category.",
        ],
    }


def write_report(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] != "scored":
        lead = summary["reason"]
    else:
        primary = summary["primary_endpoint"]
        lead = (f"The Chinese-minus-English instruction-language difference in order moderation was {primary['estimate']:.3f} "
                f"VA-RMSE points (95% paired recipient-bootstrap interval [{primary['ci95'][0]:.3f}, "
                f"{primary['ci95'][1]:.3f}]) on {summary['n_complete_paired_recipient_ids']} complete IDs.")
    effects = summary.get("language_specific_secondary", {})
    (output / "README.md").write_text(f"""# Experiment 066: instruction-language control on SIGHAN

## Result

{lead}

The model-specific aggregate order moderations were {effects.get('english', {}).get('aggregate_order_moderation', float('nan')):.3f} under the English prompt and {effects.get('chinese', {}).get('aggregate_order_moderation', float('nan')):.3f} under the translated Chinese prompt. Their intervals and the registered difference are in `summary.json`.

This paired test holds SIGHAN reviews, recipients, donor maps, output orders and decoders fixed while changing prompt language. It uses a deterministic 88-review subset and 2,464 new Qwen2.5-3B generations. Raw text, IDs, maps and predictions remain private in `.context/`.

The translation was not independently rated by a professional translator. A prompt-language effect here would not establish that language caused the earlier English/Chinese benchmark difference, because the releases and domains also differ. Results apply to this selected cohort and model setup.

- Frozen protocol: `docs/experiments/066-sighan-instruction-language-control.md`
- Aggregate result and provenance: `summary.json`
- Runner/analyzer: `scripts/run_sighan_instruction_language_066.py`, `scripts/analyze_sighan_instruction_language_066.py`
- Prior SIGHAN tests: [Experiment 064](../sighan-output-key-order-replication-v1/README.md), [Experiment 065](../sighan-order-model-size-replication-v1/README.md)
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=runner.OUT)
    parser.add_argument("--manifest", type=Path, default=runner.MANIFEST)
    parser.add_argument("--source-dir", type=Path, default=exp052.SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    summary = analyze(args.predictions, args.manifest, args.source_dir)
    write_report(summary, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
