"""Run the preregistered key-order × category-match replication on restaurants."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import run_counterfactual_context_swap_053 as exp053
import run_output_key_order_topic_match_062 as exp062
import run_restaurant_domain_decoder_051 as exp051
import torch
from run_expanded_opinion_mask_repair_043 import candidates, generate_batch
from run_laptop_decoder_transfer_049 import (
    SOURCE_REVISION,
    prompt_with_registered_grid,
)
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

PROTOCOL = Path("docs/experiments/063-output-key-order-restaurant-replication.md")
PROTOCOL_SHA256 = "950689695c6d45001888098973a62eee857e7b7a77768e57e54a6505b806c0b8"
TASK3_FILE = "eng_restaurant_test_task3.jsonl"
TASK3_SHA256 = "a24cd3d6b8b2dd3bcebf5016f2e8bb90736fe2e89671aced73229062ca516f79"
TASK3_URL = (
    "https://raw.githubusercontent.com/DimABSA/DimABSA2026/"
    f"{SOURCE_REVISION}/task-dataset/track_a/subtask_3/eng/{TASK3_FILE}"
)
PARENT = Path(".context/exp051-private-predictions.jsonl")
PARENT_SHA256 = "3f37ff8d630fde9776552389a8d41a4aab9d9b6ea127e24142337de68fd99ca7"
OUT = Path(".context/exp063-private-predictions.jsonl")
MANIFEST = Path(".context/exp063-run-manifest.json")
OWN = "opinion_masked"
SAME = "same_category_context"
CROSS = "cross_category_context"
DECODERS = ("finite_grid", "free_greedy")
ORDERS = ("valence_first", "arousal_first")
PERMUTATIONS = (1, 2, 3)


def polarity(case: dict) -> str:
    value = float(case["gold"][0])
    if value < 4.5:
        return "neg"
    if value > 5.5:
        return "pos"
    return "neu"


def read_task3(source_dir: Path) -> tuple[list[dict], str]:
    path = source_dir / TASK3_FILE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(TASK3_URL, headers={"User-Agent": "Training-Jev research runner"})
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    raw = path.read_bytes()
    digest = sha256(raw)
    if digest != TASK3_SHA256:
        raise ValueError(f"Pinned restaurant Task 3 source hash mismatch: {digest}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 1000 or len({row.get("ID") for row in rows}) != 1000:
        raise ValueError("Expected 1,000 unique restaurant Task 3 rows")
    return rows, digest


def attach_categories(cases: list[dict], task3_rows: list[dict]) -> list[dict]:
    by_text = {row["Text"]: row for row in task3_rows}
    output = []
    for case in cases:
        target = case["row"]["Triplet"][case["target_index"]]
        quadruplets = by_text.get(case["row"]["Text"], {}).get("Quadruplet", [])
        matches = [row for row in quadruplets if (
            row.get("Aspect"), row.get("Opinion"), row.get("VA")
        ) == (target.get("Aspect"), target.get("Opinion"), target.get("VA"))]
        categories = {row.get("Category") for row in matches}
        if len(categories) != 1 or None in categories:
            raise ValueError(f"Expected exactly one restaurant category for {case['case_id']}")
        output.append({**case, "category": categories.pop()})
    return output


def select_cases(task2_rows: list[dict], task3_rows: list[dict]) -> tuple[list[dict], dict]:
    categorized = attach_categories(exp051.select_cases(task2_rows), task3_rows)
    ranked = {}
    for label in ("neg", "pos"):
        members = [case for case in categorized if polarity(case) == label]
        members.sort(key=lambda case: hashlib.sha256(
            f"exp063-rest-sample-v1|{case['case_id']}".encode("utf-8")
        ).hexdigest())
        if len(members) < 90:
            raise ValueError(f"Insufficient {label} restaurant cases: {len(members)}")
        ranked[label] = members[:90]
    selected = ranked["neg"] + ranked["pos"]
    counts = Counter((polarity(case), case["category"]) for case in selected)
    cases = [case for case in selected if counts[(polarity(case), case["category"])] >= 2]
    cases.sort(key=lambda case: case["case_id"])
    stats = {
        "n_source_cases": 963,
        "sample_before_singleton_exclusion": len(selected),
        "n_recipient_ids": len(cases),
        "sample_polarity_counts_before_singleton_exclusion": {key: len(value) for key, value in ranked.items()},
        "eligible_polarity_counts": dict(Counter(polarity(case) for case in cases)),
        "n_category_polarity_groups": sum(value >= 2 for value in counts.values()),
        "n_excluded_singletons": sum(value == 1 for value in counts.values()),
        "selected_category_polarity_counts": {
            f"{key[0]}|{key[1]}": value for key, value in sorted(counts.items())
        },
    }
    if len(cases) != 173 or stats["eligible_polarity_counts"] != {"neg": 86, "pos": 87}:
        raise ValueError(f"Frozen restaurant sample audit changed: {stats}")
    return cases, stats


def tie_cost(seed: int, recipient_id: str, donor_id: str) -> float:
    digest = hashlib.sha256(f"exp063-map-{seed}|{recipient_id}|{donor_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") / (2**32) * 1e-4


def assign(rows: list[dict], lengths: dict[str, int], seed: int,
           require_different_category: bool) -> dict[str, str]:
    recipients = sorted(rows, key=lambda row: row["case_id"])
    donors = sorted(rows, key=lambda row: row["case_id"])
    cost = np.empty((len(recipients), len(donors)), dtype=np.float64)
    for i, recipient in enumerate(recipients):
        for j, donor in enumerate(donors):
            forbidden = recipient["case_id"] == donor["case_id"]
            if require_different_category:
                forbidden = forbidden or recipient["category"] == donor["category"]
            if forbidden:
                cost[i, j] = 1e6
            else:
                cost[i, j] = abs(lengths[recipient["case_id"]] - lengths[donor["case_id"]]) + tie_cost(
                    seed, recipient["case_id"], donor["case_id"]
                )
    row_indices, donor_indices = linear_sum_assignment(cost)
    if np.any(cost[row_indices, donor_indices] >= 1e6):
        raise ValueError("No complete donor assignment satisfies frozen constraints")
    mapping = {recipients[i]["case_id"]: donors[j]["case_id"]
               for i, j in zip(row_indices, donor_indices)}
    if len(set(mapping.values())) != len(mapping) or any(k == v for k, v in mapping.items()):
        raise ValueError("Donor map is not a one-to-one derangement")
    return mapping


def donor_maps(cases: list[dict], tokenizer) -> tuple[dict, dict, dict]:
    lengths = {
        case["case_id"]: len(tokenizer.encode(
            exp053.mask_opinions(case["row"]), add_special_tokens=False
        )) for case in cases
    }
    by_id = {case["case_id"]: case for case in cases}
    same_maps, cross_maps = {}, {}
    for permutation in PERMUTATIONS:
        same = {}
        for key in sorted({(polarity(case), case["category"]) for case in cases}):
            group = [case for case in cases if (polarity(case), case["category"]) == key]
            same.update(assign(group, lengths, permutation, require_different_category=False))
        cross = {}
        for label in ("neg", "pos"):
            group = [case for case in cases if polarity(case) == label]
            cross.update(assign(group, lengths, permutation + 100, require_different_category=True))
        for recipient_id, donor_id in same.items():
            recipient, donor = by_id[recipient_id], by_id[donor_id]
            if polarity(recipient) != polarity(donor) or recipient["category"] != donor["category"]:
                raise ValueError("Same-category donor assignment violated its stratum")
        for recipient_id, donor_id in cross.items():
            recipient, donor = by_id[recipient_id], by_id[donor_id]
            if polarity(recipient) != polarity(donor) or recipient["category"] == donor["category"]:
                raise ValueError("Cross-category donor assignment violated its stratum")
        if set(same) != {case["case_id"] for case in cases} or set(cross) != set(same):
            raise ValueError("Incomplete donor map")
        same_maps[permutation] = same
        cross_maps[permutation] = cross
    return same_maps, cross_maps, lengths


def jobs_for(cases: list[dict], same_maps: dict, cross_maps: dict) -> list[dict]:
    by_id = {case["case_id"]: case for case in cases}
    jobs = []
    for case in cases:
        target = case["row"]["Triplet"][case["target_index"]]
        jobs.append({
            "case_id": case["case_id"], "order": "arousal_first", "permutation": 0,
            "condition": OWN, "decoder": "finite_grid", "gold": case["gold"],
            "prompt": exp062.arousal_first_prompt(
                exp053.mask_opinions(case["row"]), target["Aspect"]
            ),
        })
        jobs.append({
            "case_id": case["case_id"], "order": "arousal_first", "permutation": 0,
            "condition": OWN, "decoder": "free_greedy", "gold": case["gold"],
            "prompt": exp062.arousal_first_prompt(
                exp053.mask_opinions(case["row"]), target["Aspect"]
            ),
        })
    for condition, mappings in ((SAME, same_maps), (CROSS, cross_maps)):
        for order in ORDERS:
            for permutation, mapping in mappings.items():
                for case in cases:
                    donor = by_id[mapping[case["case_id"]]]
                    target = case["row"]["Triplet"][case["target_index"]]
                    context = exp053.mask_opinions(donor["row"])
                    prompt = prompt_with_registered_grid(context, target["Aspect"])
                    if order == "arousal_first":
                        prompt = exp062.arousal_first_prompt(context, target["Aspect"])
                    for decoder in DECODERS:
                        jobs.append({
                            "case_id": case["case_id"], "order": order,
                            "permutation": permutation, "condition": condition,
                            "decoder": decoder, "gold": case["gold"],
                            "donor_id": donor["case_id"], "donor_category": donor["category"],
                            "donor_polarity": polarity(donor), "prompt": prompt,
                        })
    expected = 4498
    if len(jobs) != expected:
        raise ValueError(f"Expected {expected} new jobs, got {len(jobs)}")
    return sorted(jobs, key=lambda row: (
        row["order"], row["decoder"], row["condition"], row["permutation"], row["case_id"]
    ))


def hash_maps(maps: dict) -> dict[str, str]:
    return {str(seed): sha256(json.dumps(mapping, sort_keys=True).encode())
            for seed, mapping in maps.items()}


def run(source_dir: Path = Path(".context/dimabsa"), output: Path = OUT,
        manifest_path: Path = MANIFEST, device: str = "mps", batch_size: int = DEFAULT_BATCH,
        preflight_only: bool = False) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 063 protocol hash mismatch")
    if sha256(PARENT.read_bytes()) != PARENT_SHA256:
        raise ValueError("Experiment 051 parent output hash mismatch")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    task2_rows, task2_hash = exp051.read_source(source_dir)
    if task2_hash != exp051.SOURCE_SHA256:
        raise ValueError("Pinned Task 2 source hash mismatch")
    task3_rows, task3_hash = read_task3(source_dir)
    cases, sample_stats = select_cases(task2_rows, task3_rows)
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
    if len(values) != 6561 or len(arousal_values) != 6561:
        raise ValueError("Expected 6,561 candidates under both key orders")
    if preflight_only:
        prompt_lengths = [len(tokenizer.apply_chat_template(
            [{"role": "user", "content": job["prompt"]}],
            tokenize=True, add_generation_prompt=True,
        )) for job in jobs]
        max_distance = {}
        for name, maps in (("same", same_maps), ("cross", cross_maps)):
            max_distance[name] = max(
                abs(lengths[key] - lengths[value])
                for mapping in maps.values() for key, value in mapping.items()
            )
        result = {
            "experiment": "063-output-key-order-restaurant-replication",
            "protocol_sha256": sha256(PROTOCOL.read_bytes()),
            "task2_sha256": task2_hash, "task3_sha256": task3_hash,
            "parent_output_sha256": PARENT_SHA256,
            **sample_stats,
            "n_jobs": len(jobs), "n_own_arousal_first_jobs": 346,
            "n_donor_jobs": 4152,
            "n_candidates_per_order": 6561,
            "n_unique_candidate_token_sequences": {
                "valence_first": len(valence_sequences), "arousal_first": len(arousal_sequences)
            },
            "prompt_token_length_range": [min(prompt_lengths), max(prompt_lengths)],
            "donor_review_token_length_max_abs_difference": max_distance,
            "same_category_map_sha256": hash_maps(same_maps),
            "cross_category_map_sha256": hash_maps(cross_maps),
            "model": config["model"], "model_revision": config["revision"],
        }
        return result

    previous = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            key = (row["case_id"], row["order"], row["permutation"], row["condition"], row["decoder"])
            if key in previous:
                raise ValueError(f"Duplicate resumed output: {key}")
            previous[key] = row
    allowed = {
        (job["case_id"], job["order"], job["permutation"], job["condition"], job["decoder"])
        for job in jobs
    }
    if not set(previous).issubset(allowed):
        raise ValueError("Resume file contains rows outside the frozen experiment")
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
    set_seed(20260963)
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        processed = 0
        offset = 0
        while offset < len(remaining):
            first = remaining[offset]
            end = offset
            while end < len(remaining) and end < offset + batch_size and (
                remaining[end]["order"], remaining[end]["decoder"]
            ) == (first["order"], first["decoder"]):
                end += 1
            batch = remaining[offset:end]
            decoder, order = first["decoder"], first["order"]
            raw_outputs = (generate_batch(model, tokenizer, batch, tries[order], actual_device)
                           if decoder == "finite_grid"
                           else generate_free_batch(model, tokenizer, batch, actual_device))
            for job, raw in zip(batch, raw_outputs):
                if decoder == "finite_grid" and order == "arousal_first":
                    prediction = exp062.parse_finite_arousal_first(raw)
                elif decoder == "finite_grid":
                    prediction = exp051.parse_finite_va(raw)
                else:
                    prediction = parse_free_va(raw)
                row = {key: job[key] for key in (
                    "case_id", "order", "permutation", "condition", "decoder", "gold"
                )}
                if job["condition"] != OWN:
                    row.update({key: job[key] for key in (
                        "donor_id", "donor_category", "donor_polarity"
                    )})
                row.update({"raw": raw, "prediction": prediction})
                key = (row["case_id"], row["order"], row["permutation"], row["condition"], decoder)
                previous[key] = row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            processed += len(batch)
            offset = end
            print(f"063 {processed}/{len(remaining)} new; total {len(previous)}/{len(jobs)}", flush=True)
    manifest = {
        "experiment": "063-output-key-order-restaurant-replication",
        "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        "task2_sha256": task2_hash, "task3_sha256": task3_hash,
        "parent_output_sha256": PARENT_SHA256,
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
    parser.add_argument("--source-dir", type=Path, default=Path(".context/dimabsa"))
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--manifest", dest="manifest_path", type=Path, default=MANIFEST)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--preflight-only", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
