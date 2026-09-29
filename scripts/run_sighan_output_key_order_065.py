"""Run the frozen SIGHAN-2024 order × category-match replication."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

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
from task_ladder import MODEL_SPECS
from transformers import AutoTokenizer, set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/065-sighan-order-model-size-replication.md")
PROTOCOL_SHA256 = "544cc2b8ecfd81bf7f340b8eb5891c3eee5eeed1152dfe0fc7111e997e32e881"
PARENT = Path(".context/exp064-private-predictions.jsonl")
PARENT_SHA256 = "48f7a8e2fe9d245d7742c7955b6349fe08a3935934f7ff0cd1db01e1eec9ace7"
OUT = Path(".context/exp065-private-predictions.jsonl")
MANIFEST = Path(".context/exp065-run-manifest.json")
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


def jobs_for(cases: list[dict], same_maps: dict, cross_maps: dict) -> list[dict]:
    by_id = {case["case_id"]: case for case in cases}
    jobs = []
    for case in cases:
        aspect = case["row"]["Triplet"][case["target_index"]]["Aspect"]
        masked = exp053.mask_opinions(case["row"])
        for order in ORDERS:
            prompt = prompt_with_registered_grid(masked, aspect)
            if order == "arousal_first":
                prompt = exp062.arousal_first_prompt(masked, aspect)
            jobs.extend({
                "case_id": case["case_id"], "order": order, "permutation": 0,
                "condition": OWN, "decoder": decoder, "gold": case["gold"],
                "prompt": prompt,
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
    if len(jobs) != 5040:
        raise ValueError(f"Expected 5,040 SIGHAN jobs, found {len(jobs)}")
    return sorted(jobs, key=lambda row: (
        row["order"], row["decoder"], row["condition"], row["permutation"], row["case_id"]
    ))


def hash_maps(maps: dict) -> dict[str, str]:
    return {str(seed): sha256(json.dumps(mapping, sort_keys=True).encode())
            for seed, mapping in maps.items()}


def extract_maps(path: Path, cases: list[dict]) -> tuple[dict, dict]:
    ids = {case["case_id"] for case in cases}
    by_id = {case["case_id"]: case for case in cases}
    same_maps = {permutation: {} for permutation in PERMUTATIONS}
    cross_maps = {permutation: {} for permutation in PERMUTATIONS}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["condition"] not in (SAME, CROSS) or row["case_id"] not in ids:
            continue
        if row["decoder"] != "finite_grid":
            continue
        if row["permutation"] not in PERMUTATIONS or row["order"] not in ORDERS:
            raise ValueError("Exp064 donor map contains an unexpected cell")
        mapping = same_maps[row["permutation"]] if row["condition"] == SAME else cross_maps[row["permutation"]]
        previous = mapping.setdefault(row["case_id"], row["donor_id"])
        if previous != row["donor_id"]:
            raise ValueError("Exp064 donor assignment changed across output orders")
    for condition, maps in ((SAME, same_maps), (CROSS, cross_maps)):
        for permutation, mapping in maps.items():
            if set(mapping) != ids:
                raise ValueError(f"Exp064 donor map is incomplete: {condition}/{permutation}")
            if set(mapping.values()) != ids or any(source == donor for source, donor in mapping.items()):
                raise ValueError(f"Exp064 donor map is not a derangement: {condition}/{permutation}")
            for source, donor in mapping.items():
                if polarity(by_id[source]) != polarity(by_id[donor]):
                    raise ValueError("Exp064 donor map did not preserve polarity")
                category_match = category(by_id[source]) == category(by_id[donor])
                if category_match != (condition == SAME):
                    raise ValueError("Exp064 donor map did not preserve its category condition")
    return same_maps, cross_maps


def run(source_dir: Path = exp052.SOURCE_DIR, output: Path = OUT,
        manifest_path: Path = MANIFEST, device: str = "mps", batch_size: int = DEFAULT_BATCH,
        preflight_only: bool = False) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 065 protocol hash mismatch")
    if sha256(PARENT.read_bytes()) != PARENT_SHA256:
        raise ValueError("Experiment 064 map artifact hash mismatch")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    source_rows, source_hashes = exp052.read_source(source_dir)
    cases_all = exp052.select_cases(source_rows)
    cases, sample_stats = select_cases(cases_all)
    config = dict(MODEL_SPECS["qwen-1.5b"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True
    )
    tokenizer.padding_side = "left"
    same_maps, cross_maps = extract_maps(PARENT, cases)
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
        return {
            "experiment": "065-sighan-order-model-size-replication",
            "protocol_sha256": sha256(PROTOCOL.read_bytes()),
            "source_hashes": source_hashes,
            "parent_output_sha256": PARENT_SHA256,
            **sample_stats,
            "n_jobs": len(jobs), "n_own_baseline_jobs": 720,
            "n_donor_jobs": 4320, "n_candidates_per_order": 6561,
            "n_unique_candidate_token_sequences": {
                "valence_first": len(valence_sequences), "arousal_first": len(arousal_sequences)
            },
            "prompt_token_length_range": [min(prompt_lengths), max(prompt_lengths)],
            "donor_review_token_length_max_abs_difference": "maps reused unchanged from Exp064",
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
        raise ValueError("Resume file contains rows outside frozen Experiment 065")
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
    set_seed(20260965)
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
            print(f"065 {processed}/{len(remaining)} new; total {len(previous)}/{len(jobs)}", flush=True)
    manifest = {
        "experiment": "065-sighan-order-model-size-replication",
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
