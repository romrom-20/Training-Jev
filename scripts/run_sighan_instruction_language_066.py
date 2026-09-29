"""Run the frozen SIGHAN-2024 instruction-language control."""

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
import run_sighan_output_key_order_064 as exp064
import torch
from run_expanded_opinion_mask_repair_043 import candidates, generate_batch
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

PROTOCOL = Path("docs/experiments/066-sighan-instruction-language-control.md")
PROTOCOL_SHA256 = "27dbd3c6d3c234b1fbb3bab7923478b465163048a18fb3fc5d1c00089dffdda5"
PARENT = Path(".context/exp064-private-predictions.jsonl")
PARENT_SHA256 = "48f7a8e2fe9d245d7742c7955b6349fe08a3935934f7ff0cd1db01e1eec9ace7"
OUT = Path(".context/exp066-private-predictions.jsonl")
MANIFEST = Path(".context/exp066-run-manifest.json")
OWN = "opinion_masked"
SAME = "same_category_context"
CROSS = "cross_category_context"
DECODERS = ("finite_grid", "free_greedy")
ORDERS = ("valence_first", "arousal_first")
PERMUTATIONS = (1, 2, 3)
SAMPLE_QUOTAS = {
    "neg": {
        "食物#品质": 22,
        "食物#份量与款式": 11,
        "食物#价格": 7,
        "饮料#品质": 4,
    },
    "pos": {
        "食物#品质": 22,
        "食物#份量与款式": 22,
    },
}
EXPECTED_MAP_SHA256 = {
    "same_category": {
        "1": "b98728e19e3bb99f3f32f1c5a9d288b2820fb0f89202a8810a009fcf5b7a7d38",
        "2": "ec4210e0429c2a7213c953a7f534609454edb6d5ec3d7223d39c13ce11f0c401",
        "3": "80a79c567e72c307bcac221291be06f0cd3bd6a695cb2b4c3e730561b874cc75",
    },
    "cross_category": {
        "1": "7cb29764333dc08e07f29675f542d5f36fa5e20c94b4f82da78ba6fe00ca408a",
        "2": "9ce3da37ae573e3d1256bdf4882c58314d7c0dceb6a6f989b3787b600120e830",
        "3": "a69470883da6fb38a00e3e023bb2fa41f613b97607aed320cef5264c4ffc857b",
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
                f"exp066-sighan-zh-instruction-v1|{case['case_id']}".encode("utf-8")
            ).hexdigest())
            selected.extend(group[:quota])
            strata_stats[f"{pol}|{category_name}"] = quota
    selected.sort(key=lambda case: case["case_id"])
    if len(selected) != 88 or len({case["case_id"] for case in selected}) != 88:
        raise ValueError("Frozen SIGHAN language-control sample must contain 88 unique IDs")
    pol_counts = dict(Counter(polarity(case) for case in selected))
    if pol_counts != {"neg": 44, "pos": 44}:
        raise ValueError(f"Unexpected SIGHAN polarity counts: {pol_counts}")
    groups = Counter((polarity(case), category(case)) for case in selected)
    if any(count < 2 for count in groups.values()):
        raise ValueError("Every frozen polarity/category cell must have at least two recipients")
    if max(Counter(polarity(case) for case in selected).values()) != 44:
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


def chinese_prompt(text: str, aspect: str, order: str) -> str:
    if order not in ORDERS:
        raise ValueError(f"Unexpected output order: {order}")
    keys = '"valence"和"arousal"' if order == "valence_first" else '"arousal"和"valence"'
    return (
        "评估作者对指定方面表达的情感倾向。"
        "效价（valence）范围从 1（极度负面）到 9（极度正面）。"
        "唤醒度（arousal）范围从 1（非常平静）到 9（非常激动或强烈）。"
        f"仅使用所提供的证据。只返回一个 JSON 对象，其中包含数值键{keys}，"
        "两项均为 1.0 至 9.0，以 0.1 递增，且恰好保留一位小数。\n"
        f"评论文本：{text}\n目标方面：{aspect}\n与目标相关的观点短语：[NOT PROVIDED]"
    )


def jobs_for(cases: list[dict], donor_cases: list[dict], same_maps: dict,
             cross_maps: dict) -> list[dict]:
    by_id = {case["case_id"]: case for case in donor_cases}
    jobs = []
    for case in cases:
        aspect = case["row"]["Triplet"][case["target_index"]]["Aspect"]
        masked = exp053.mask_opinions(case["row"])
        for order in ORDERS:
            prompt = chinese_prompt(masked, aspect, order)
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
                    prompt = chinese_prompt(context, aspect, order)
                    for decoder in DECODERS:
                        jobs.append({
                            "case_id": case["case_id"], "order": order,
                            "permutation": permutation, "condition": condition,
                            "decoder": decoder, "gold": case["gold"],
                            "donor_id": donor["case_id"], "donor_category": category(donor),
                            "donor_polarity": polarity(donor), "prompt": prompt,
                        })
    if len(jobs) != 2464:
        raise ValueError(f"Expected 2,464 Chinese-instruction jobs, found {len(jobs)}")
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
    metadata = {}
    known_ids = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        known_ids.add(row["case_id"])
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
        meta_key = (row["condition"], row["permutation"], row["case_id"])
        current_meta = (row["donor_id"], row["donor_category"], row["donor_polarity"])
        if meta_key in metadata and metadata[meta_key] != current_meta:
            raise ValueError("Exp064 donor metadata changed across output orders")
        metadata[meta_key] = current_meta
    for condition, maps in ((SAME, same_maps), (CROSS, cross_maps)):
        for permutation, mapping in maps.items():
            if set(mapping) != ids:
                raise ValueError(f"Exp064 donor map is incomplete: {condition}/{permutation}")
            if len(set(mapping.values())) != len(mapping) or any(source == donor for source, donor in mapping.items()):
                raise ValueError(f"Exp064 donor map is not a derangement: {condition}/{permutation}")
            for source, donor in mapping.items():
                if donor not in known_ids:
                    raise ValueError(f"Exp064 donor is outside the frozen cohort: {donor}")
                donor_id, donor_category, donor_polarity = metadata[(condition, permutation, source)]
                if donor_id != donor or donor_polarity != polarity(by_id[source]):
                    raise ValueError("Exp064 donor map did not preserve polarity")
                category_match = category(by_id[source]) == donor_category
                if category_match != (condition == SAME):
                    raise ValueError("Exp064 donor map did not preserve its category condition")
    return same_maps, cross_maps


def run(source_dir: Path = exp052.SOURCE_DIR, output: Path = OUT,
        manifest_path: Path = MANIFEST, device: str = "mps", batch_size: int = DEFAULT_BATCH,
        preflight_only: bool = False) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 066 protocol hash mismatch")
    if sha256(PARENT.read_bytes()) != PARENT_SHA256:
        raise ValueError("Experiment 064 English comparator/map artifact hash mismatch")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    source_rows, source_hashes = exp052.read_source(source_dir)
    cases_all = exp052.select_cases(source_rows)
    donor_cases, _ = exp064.select_cases(cases_all)
    cases, sample_stats = select_cases(donor_cases)
    config = dict(MODEL_SPECS["qwen-3b"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True
    )
    tokenizer.padding_side = "left"
    same_maps, cross_maps = extract_maps(PARENT, cases)
    actual_map_hashes = {
        "same_category": hash_maps(same_maps),
        "cross_category": hash_maps(cross_maps),
    }
    if actual_map_hashes != EXPECTED_MAP_SHA256:
        raise ValueError("Extracted Exp064 donor-map hashes differ from the frozen protocol")
    jobs = jobs_for(cases, donor_cases, same_maps, cross_maps)
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
            "experiment": "066-sighan-instruction-language-control",
            "protocol_sha256": sha256(PROTOCOL.read_bytes()),
            "source_hashes": source_hashes,
            "parent_output_sha256": PARENT_SHA256,
            **sample_stats,
            "n_jobs": len(jobs), "n_own_baseline_jobs": 352,
            "n_donor_jobs": 2112, "n_candidates_per_order": 6561,
            "n_unique_candidate_token_sequences": {
                "valence_first": len(valence_sequences), "arousal_first": len(arousal_sequences)
            },
            "prompt_token_length_range": [min(prompt_lengths), max(prompt_lengths)],
            "donor_review_token_length_max_abs_difference": "maps reused unchanged from Exp064",
            "same_category_map_sha256": actual_map_hashes["same_category"],
            "cross_category_map_sha256": actual_map_hashes["cross_category"],
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
        raise ValueError("Resume file contains rows outside frozen Experiment 066")
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
    set_seed(20260966)
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
            print(f"066 {processed}/{len(remaining)} new; total {len(previous)}/{len(jobs)}", flush=True)
    manifest = {
        "experiment": "066-sighan-instruction-language-control",
        "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        "source_hashes": source_hashes, "parent_output_sha256": PARENT_SHA256,
        "model": config["model"], "model_revision": config["revision"],
        "device": actual_device, "batch_size": batch_size,
        "max_new_tokens_free": MAX_NEW_TOKENS, **sample_stats,
        "n_outputs": len(previous),
        "n_own_baseline_outputs": len(cases) * len(ORDERS) * len(DECODERS),
        "n_donor_context_outputs": len(cases) * len(ORDERS) * len(PERMUTATIONS)
        * 2 * len(DECODERS),
        "same_category_map_sha256": actual_map_hashes["same_category"],
        "cross_category_map_sha256": actual_map_hashes["cross_category"],
        "candidate_count_per_order": len(values) ** 2,
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
