"""Run the frozen SIGHAN-2024 order × category-match replication."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import run_counterfactual_context_swap_053 as exp053
import run_output_key_order_topic_match_062 as exp062
import run_sighan_chinese_decoder_transfer_052 as exp052
import torch
from run_expanded_opinion_mask_repair_043 import candidates, generate_batch
from run_laptop_decoder_transfer_049 import prompt_with_registered_grid
from run_small_model_decoder_factorial_048 import (
    DEFAULT_BATCH,
    MAX_NEW_TOKENS,
    build_trie,
    candidate_sequences,
    generate_free_batch,
    parse_free_va,
    sha256,
)
from scipy.optimize import linear_sum_assignment
from task_ladder import MODEL_SPECS
from transformers import AutoTokenizer, set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/064-sighan-output-key-order-replication.md")
PROTOCOL_SHA256 = "fc37efd2bf3683d80b6ca45812f918a2661d17ff0e31ad1f1f5c616baeb61fda"
PARENT = Path(".context/exp052-private-predictions.jsonl")
PARENT_SHA256 = "76671a3330e35fddfd915f15c86b49b90f8113af74041651162b7a81051765fe"
OUT = Path(".context/exp064-private-predictions.jsonl")
MANIFEST = Path(".context/exp064-run-manifest.json")
OWN = "opinion_masked"
SAME = "same_category_context"
CROSS = "cross_category_context"
DECODERS = ("finite_grid", "free_greedy")
ORDERS = ("valence_first", "arousal_first")
PERMUTATIONS = (1, 2, 3)
SAMPLE_QUOTAS = {
    "neg": {
        "食物#品质": 45,
        "食物#份量与款式": 21,
        "食物#价格": 14,
        "饮料#品质": 7,
        "餐厅#杂项": 3,
    },
    "pos": {
        "食物#品质": 45,
        "食物#份量与款式": 45,
    },
}


def polarity(case: dict) -> str:
    value = float(case["gold"][0])
    if value < 4.5:
        return "neg"
    if value > 5.5:
        return "pos"
    return "neu"


def category(case: dict) -> str:
    return case["row"]["Triplet"][case["target_index"]]["Category"]


def select_cases(all_cases: list[dict]) -> tuple[list[dict], dict]:
    selected = []
    strata_stats = {}
    for pol, quotas in SAMPLE_QUOTAS.items():
        for category_name, quota in quotas.items():
            group = [case for case in all_cases
                     if polarity(case) == pol and category(case) == category_name]
            if len(group) < quota:
                raise ValueError(f"Insufficient {pol}/{category_name}: {len(group)} < {quota}")
            group.sort(key=lambda case: hashlib.sha256(
                f"exp064-sighan-v1|{case['case_id']}".encode("utf-8")
            ).hexdigest())
            selected.extend(group[:quota])
            strata_stats[f"{pol}|{category_name}"] = quota
    selected.sort(key=lambda case: case["case_id"])
    if len(selected) != 180 or len({case["case_id"] for case in selected}) != 180:
        raise ValueError("Frozen SIGHAN sample must contain 180 unique IDs")
    pol_counts = dict(Counter(polarity(case) for case in selected))
    if pol_counts != {"neg": 90, "pos": 90}:
        raise ValueError(f"Unexpected SIGHAN polarity counts: {pol_counts}")
    groups = Counter((polarity(case), category(case)) for case in selected)
    if any(count < 2 for count in groups.values()):
        raise ValueError("Every frozen polarity/category cell must have at least two recipients")
    if max(Counter(polarity(case) for case in selected).values()) != 90:
        raise ValueError("SIGHAN polarity assignment size changed")
    return selected, {
        "n_source_cases": len(all_cases),
        "n_recipient_ids": len(selected),
        "polarity_counts": pol_counts,
        "n_category_polarity_groups": len(groups),
        "category_polarity_counts": {f"{key[0]}|{key[1]}": value
                                      for key, value in sorted(groups.items())},
        "selection_quotas": strata_stats,
    }


def tie_cost(seed: int, recipient_id: str, donor_id: str) -> float:
    raw = hashlib.sha256(f"exp064-map-{seed}|{recipient_id}|{donor_id}".encode()).digest()
    return int.from_bytes(raw[:4], "big") / (2**32) * 1e-4


def assign(rows: list[dict], lengths: dict[str, int], seed: int,
           require_different_category: bool) -> dict[str, str]:
    recipients = sorted(rows, key=lambda row: row["case_id"])
    donors = sorted(rows, key=lambda row: row["case_id"])
    cost = np.empty((len(recipients), len(donors)), dtype=np.float64)
    for i, recipient in enumerate(recipients):
        for j, donor in enumerate(donors):
            forbidden = recipient["case_id"] == donor["case_id"]
            if require_different_category:
                forbidden = forbidden or category(recipient) == category(donor)
            cost[i, j] = (1e6 if forbidden else
                          abs(lengths[recipient["case_id"]] - lengths[donor["case_id"]])
                          + tie_cost(seed, recipient["case_id"], donor["case_id"]))
    row_indices, donor_indices = linear_sum_assignment(cost)
    if np.any(cost[row_indices, donor_indices] >= 1e6):
        raise ValueError("No complete donor assignment satisfies SIGHAN category constraints")
    mapping = {recipients[i]["case_id"]: donors[j]["case_id"]
               for i, j in zip(row_indices, donor_indices)}
    if len(set(mapping.values())) != len(mapping) or any(k == v for k, v in mapping.items()):
        raise ValueError("SIGHAN donor map must be a one-to-one derangement")
    return mapping


def donor_maps(cases: list[dict], tokenizer) -> tuple[dict, dict, dict]:
    lengths = {case["case_id"]: len(tokenizer.encode(
        exp053.mask_opinions(case["row"]), add_special_tokens=False
    )) for case in cases}
    same_maps, cross_maps = {}, {}
    for permutation in PERMUTATIONS:
        same = {}
        for key in sorted({(polarity(case), category(case)) for case in cases}):
            group = [case for case in cases if (polarity(case), category(case)) == key]
            same.update(assign(group, lengths, permutation, require_different_category=False))
        cross = {}
        for pol in ("neg", "pos"):
            group = [case for case in cases if polarity(case) == pol]
            cross.update(assign(group, lengths, permutation + 100,
                                require_different_category=True))
        by_id = {case["case_id"]: case for case in cases}
        for recipient, donor in same.items():
            if polarity(by_id[recipient]) != polarity(by_id[donor]) or category(by_id[recipient]) != category(by_id[donor]):
                raise ValueError("Same-category map violated polarity or category")
        for recipient, donor in cross.items():
            if polarity(by_id[recipient]) != polarity(by_id[donor]) or category(by_id[recipient]) == category(by_id[donor]):
                raise ValueError("Cross-category map violated polarity or category")
        same_maps[permutation] = same
        cross_maps[permutation] = cross
    return same_maps, cross_maps, lengths


def jobs_for(cases: list[dict], same_maps: dict, cross_maps: dict) -> list[dict]:
    by_id = {case["case_id"]: case for case in cases}
    jobs = []
    for case in cases:
        aspect = case["row"]["Triplet"][case["target_index"]]["Aspect"]
        masked = exp053.mask_opinions(case["row"])
        prompt = prompt_with_registered_grid(masked, aspect)
        jobs.extend({
            "case_id": case["case_id"], "order": "arousal_first", "permutation": 0,
            "condition": OWN, "decoder": decoder, "gold": case["gold"],
            "prompt": exp062.arousal_first_prompt(masked, aspect),
        } for decoder in DECODERS)
    for condition, mappings in ((SAME, same_maps), (CROSS, cross_maps)):
        for order in ORDERS:
            for permutation, mapping in mappings.items():
                for case in cases:
                    donor = by_id[mapping[case["case_id"]]]
                    aspect = case["row"]["Triplet"][case["target_index"]]["Aspect"]
                    context = exp053.mask_opinions(donor["row"])
                    prompt = prompt_with_registered_grid(context, aspect)
                    if order == "arousal_first":
                        prompt = exp062.arousal_first_prompt(context, aspect)
                    for decoder in DECODERS:
                        jobs.append({
                            "case_id": case["case_id"], "order": order,
                            "permutation": permutation, "condition": condition,
                            "decoder": decoder, "gold": case["gold"],
                            "donor_id": donor["case_id"], "donor_category": category(donor),
                            "donor_polarity": polarity(donor), "prompt": prompt,
                        })
    if len(jobs) != 4680:
        raise ValueError(f"Expected 4,680 SIGHAN jobs, found {len(jobs)}")
    return sorted(jobs, key=lambda row: (
        row["order"], row["decoder"], row["condition"], row["permutation"], row["case_id"]
    ))


def hash_maps(maps: dict) -> dict[str, str]:
    return {str(seed): sha256(json.dumps(mapping, sort_keys=True).encode())
            for seed, mapping in maps.items()}


def run(source_dir: Path = exp052.SOURCE_DIR, output: Path = OUT,
        manifest_path: Path = MANIFEST, device: str = "mps", batch_size: int = DEFAULT_BATCH,
        preflight_only: bool = False) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 064 protocol hash mismatch")
    if sha256(PARENT.read_bytes()) != PARENT_SHA256:
        raise ValueError("Experiment 052 parent output hash mismatch")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    source_rows, source_hashes = exp052.read_source(source_dir)
    cases_all = exp052.select_cases(source_rows)
    cases, sample_stats = select_cases(cases_all)
    config = dict(MODEL_SPECS["qwen-3b"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True
    )
    tokenizer.padding_side = "left"
    same_maps, cross_maps, lengths = donor_maps(cases, tokenizer)
    jobs = jobs_for(cases, same_maps, cross_maps)
    values = candidates()
    valence_sequences = candidate_sequences(tokenizer, values)
    arousal_values = exp062.arousal_first_candidates()
    arousal_sequences = candidate_sequences(tokenizer, arousal_values)
    if preflight_only:
        prompt_lengths = [len(tokenizer.apply_chat_template(
            [{"role": "user", "content": job["prompt"]}],
            tokenize=True, add_generation_prompt=True,
        )) for job in jobs]
        distances = {}
        for name, maps in (("same", same_maps), ("cross", cross_maps)):
            distances[name] = max(abs(lengths[source] - lengths[donor])
                                  for mapping in maps.values()
                                  for source, donor in mapping.items())
        return {
            "experiment": "064-sighan-output-key-order-replication",
            "protocol_sha256": sha256(PROTOCOL.read_bytes()),
            "source_hashes": source_hashes,
            "parent_output_sha256": PARENT_SHA256,
            **sample_stats,
            "n_jobs": len(jobs), "n_own_arousal_first_jobs": 360,
            "n_donor_jobs": 4320, "n_candidates_per_order": 6561,
            "n_unique_candidate_token_sequences": {
                "valence_first": len(valence_sequences), "arousal_first": len(arousal_sequences)
            },
            "prompt_token_length_range": [min(prompt_lengths), max(prompt_lengths)],
            "donor_review_token_length_max_abs_difference": distances,
            "same_category_map_sha256": hash_maps(same_maps),
            "cross_category_map_sha256": hash_maps(cross_maps),
            "model": config["model"], "model_revision": config["revision"],
        }

    previous = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            key = (row["case_id"], row["order"], row["permutation"], row["condition"], row["decoder"])
            if key in previous:
                raise ValueError(f"Duplicate resumed SIGHAN output: {key}")
            previous[key] = row
    allowed = {(j["case_id"], j["order"], j["permutation"], j["condition"], j["decoder"]) for j in jobs}
    if not set(previous).issubset(allowed):
        raise ValueError("Resume file contains rows outside frozen Experiment 064")
    remaining = [job for job in jobs if (
        job["case_id"], job["order"], job["permutation"], job["condition"], job["decoder"]
    ) not in previous]
    del tokenizer
    model, tokenizer, actual_device = load_target(config, device, offline=True)
    tokenizer.padding_side = "left"
    tries = {
        "valence_first": build_trie(candidate_sequences(tokenizer, values)),
        "arousal_first": build_trie(candidate_sequences(tokenizer, arousal_values)),
    }
    set_seed(20260964)
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        offset = 0
        processed = 0
        while offset < len(remaining):
            first = remaining[offset]
            end = offset
            while end < len(remaining) and end < offset + batch_size and (
                remaining[end]["order"], remaining[end]["decoder"]
            ) == (first["order"], first["decoder"]):
                end += 1
            batch = remaining[offset:end]
            raw_outputs = (generate_batch(model, tokenizer, batch, tries[first["order"]], actual_device)
                           if first["decoder"] == "finite_grid"
                           else generate_free_batch(model, tokenizer, batch, actual_device))
            for job, raw in zip(batch, raw_outputs):
                prediction = (
                    exp062.parse_finite_arousal_first(raw)
                    if first["decoder"] == "finite_grid" and first["order"] == "arousal_first"
                    else exp052.parse_finite_va(raw)
                    if first["decoder"] == "finite_grid"
                    else parse_free_va(raw)
                )
                row = {key: job[key] for key in (
                    "case_id", "order", "permutation", "condition", "decoder", "gold"
                )}
                if job["condition"] != OWN:
                    row.update({key: job[key] for key in (
                        "donor_id", "donor_category", "donor_polarity"
                    )})
                row.update({"raw": raw, "prediction": prediction})
                key = (row["case_id"], row["order"], row["permutation"], row["condition"], row["decoder"])
                previous[key] = row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            processed += len(batch)
            offset = end
            print(f"064 {processed}/{len(remaining)} new; total {len(previous)}/{len(jobs)}", flush=True)
    manifest = {
        "experiment": "064-sighan-output-key-order-replication",
        "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        "source_hashes": source_hashes, "parent_output_sha256": PARENT_SHA256,
        "model": config["model"], "model_revision": config["revision"],
        "device": actual_device, "batch_size": batch_size,
        "max_new_tokens_free": MAX_NEW_TOKENS, **sample_stats,
        "n_outputs": len(previous),
        "invalid_by_order_condition_decoder": {
            order: {
                condition: {
                    decoder: sum(row["order"] == order and row["condition"] == condition
                                 and row["decoder"] == decoder and row["prediction"] is None
                                 for row in previous.values())
                    for decoder in DECODERS
                } for condition in (OWN, SAME, CROSS)
            } for order in ORDERS
        },
        "generation_seconds_this_process_only": time.monotonic() - started,
        "resumed_rows": len(jobs) - len(remaining),
        "output_sha256": sha256(output.read_bytes()),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    del model, tokenizer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=exp052.SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--manifest", dest="manifest_path", type=Path, default=MANIFEST)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--preflight-only", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
