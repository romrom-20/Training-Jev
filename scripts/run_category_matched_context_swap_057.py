"""Run aspect-category and polarity matched counterfactual context swaps."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

import run_counterfactual_context_swap_053 as exp053
import run_fresh_context_swap_replication_055 as exp055
from run_expanded_opinion_mask_repair_043 import candidates, generate_batch
from run_laptop_decoder_transfer_049 import SOURCE_REVISION, prompt_with_registered_grid
from run_small_model_decoder_factorial_048 import (
    DEFAULT_BATCH,
    MAX_NEW_TOKENS,
    build_trie,
    candidate_sequences,
    generate_free_batch,
    homogeneous_batches,
    parse_finite_va,
    parse_free_va,
    sha256,
)
from task_ladder import MODEL_SPECS
from transformers import AutoTokenizer, set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/057-category-matched-context-swap.md")
PROTOCOL_SHA256 = "7e0a7863d18f8d0402775a416e6d71e80377a7383dbecccdaa57c72bb117fb25"
TASK3_FILE = "eng_laptop_test_task3.jsonl"
TASK3_SHA256 = "600aeba51fb53f6bbae7f669f03796999b4609cb5eaddfef171ce390d2ff8786"
TASK3_URL = (
    "https://raw.githubusercontent.com/DimABSA/DimABSA2026/"
    f"{SOURCE_REVISION}/task-dataset/track_a/subtask_3/eng/{TASK3_FILE}"
)
PARENT_OUTPUT_SHA256 = "890a16d6363e72549c7ccad9099dece3ce021ab78ab3d8071acde18774500f5f"
PERMUTATIONS = (1, 2, 3)
DECODERS = ("finite_grid", "free_greedy")
OUT = Path(".context/exp057-private-predictions.jsonl")
MANIFEST = Path(".context/exp057-run-manifest.json")


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
        raise ValueError(f"Pinned Task 3 source hash mismatch: {digest}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 1000 or len({row["Text"] for row in rows}) != 1000:
        raise ValueError("Expected 1,000 unique Task 3 review texts")
    return rows, digest


def attach_categories(cases: list[dict], task3_rows: list[dict]) -> list[dict]:
    task3_by_text = {row["Text"]: row for row in task3_rows}
    output = []
    for case in cases:
        row = case["row"]
        triplet = row["Triplet"][case["target_index"]]
        quadruplets = task3_by_text.get(row["Text"], {}).get("Quadruplet", [])
        matches = [
            item for item in quadruplets
            if (item.get("Aspect"), item.get("Opinion"), item.get("VA"))
            == (triplet.get("Aspect"), triplet.get("Opinion"), triplet.get("VA"))
        ]
        categories = {item.get("Category") for item in matches}
        if len(categories) != 1 or None in categories:
            raise ValueError(f"Expected exactly one Task 3 category for {case['case_id']}")
        output.append({**case, "category": categories.pop()})
    return output


def select_category_cases(cases: list[dict]) -> tuple[list[dict], dict]:
    grouped = {}
    for case in cases:
        key = (exp055.bucket(case), case["category"])
        grouped.setdefault(key, []).append(case)
    selected = [case for group in grouped.values() if len(group) >= 2 for case in group]
    selected.sort(key=lambda case: case["case_id"])
    stats = {
        "n_fresh_cases": len(cases),
        "n_eligible_cases": len(selected),
        "n_category_polarity_groups": sum(len(group) >= 2 for group in grouped.values()),
        "n_excluded_singletons": len(cases) - len(selected),
        "group_sizes": dict(Counter(len(group) for group in grouped.values() if len(group) >= 2)),
        "eligible_bucket_counts": dict(Counter(exp055.bucket(case) for case in selected)),
    }
    if len(selected) != 184 or stats["n_category_polarity_groups"] != 45:
        raise ValueError(f"Frozen category/polarity sample audit changed: {stats}")
    return selected, stats


def offsets_for_group(size: int) -> list[int]:
    offsets = []
    distance = 1
    while len(offsets) < min(size - 1, len(PERMUTATIONS)):
        for candidate in (distance, size - distance):
            if 1 <= candidate < size and candidate not in offsets:
                offsets.append(candidate)
                if len(offsets) >= min(size - 1, len(PERMUTATIONS)):
                    break
        distance += 1
    return offsets


def donor_maps(cases: list[dict], tokenizer) -> tuple[dict[int, dict[str, str]], dict]:
    lengths = {
        case["case_id"]: len(tokenizer.encode(
            exp053.mask_opinions(case["row"]), add_special_tokens=False
        ))
        for case in cases
    }
    groups = {}
    for case in cases:
        groups.setdefault((exp055.bucket(case), case["category"]), []).append(case)
    maps = {permutation: {} for permutation in PERMUTATIONS}
    unique_offsets = set()
    for key, group in groups.items():
        ordered = sorted(group, key=lambda case: (lengths[case["case_id"]], case["case_id"]))
        offsets = offsets_for_group(len(ordered))
        for permutation in PERMUTATIONS:
            offset = offsets[(permutation - 1) % len(offsets)]
            unique_offsets.add((key, offset))
            for index, case in enumerate(ordered):
                donor = ordered[(index + offset) % len(ordered)]
                maps[permutation][case["case_id"]] = donor["case_id"]
    ids = {case["case_id"] for case in cases}
    by_id = {case["case_id"]: case for case in cases}
    for permutation, mapping in maps.items():
        if set(mapping) != ids or set(mapping.values()) != ids:
            raise ValueError(f"Permutation {permutation} is not a bijection")
        if any(case_id == donor_id for case_id, donor_id in mapping.items()):
            raise ValueError(f"Permutation {permutation} contains a self-donor")
        for case_id, donor_id in mapping.items():
            recipient = by_id[case_id]
            donor = by_id[donor_id]
            if exp055.bucket(recipient) != exp055.bucket(donor):
                raise ValueError("Donor and recipient polarities differ")
            if recipient["category"] != donor["category"]:
                raise ValueError("Donor and recipient official categories differ")
    unique_maps = len({tuple(sorted(mapping.items())) for mapping in maps.values()})
    meta = {
        "token_lengths": lengths,
        "n_unique_assignment_maps": unique_maps,
        "n_category_polarity_offset_pairs": len(unique_offsets),
    }
    return maps, meta


def build_jobs(cases: list[dict], mappings: dict[int, dict[str, str]]) -> list[dict]:
    by_id = {case["case_id"]: case for case in cases}
    jobs = []
    for permutation, mapping in mappings.items():
        for case in cases:
            donor = by_id[mapping[case["case_id"]]]
            target = case["row"]["Triplet"][case["target_index"]]
            prompt = prompt_with_registered_grid(exp053.mask_opinions(donor["row"]), target["Aspect"])
            for decoder in DECODERS:
                jobs.append({
                    "case_id": case["case_id"], "permutation": permutation,
                    "condition": "category_polarity_matched_context", "decoder": decoder,
                    "gold": case["gold"], "donor_id": donor["case_id"],
                    "donor_category": donor["category"],
                    "donor_polarity": exp055.bucket(donor), "prompt": prompt,
                })
    if len(jobs) != len(cases) * len(PERMUTATIONS) * len(DECODERS):
        raise ValueError("Unexpected generation count")
    prompt_map = {}
    for job in jobs:
        key = (job["case_id"], job["permutation"])
        if key in prompt_map and prompt_map[key] != job["prompt"]:
            raise ValueError("Decoder prompts differ")
        prompt_map[key] = job["prompt"]
    return sorted(jobs, key=lambda row: (row["permutation"], row["decoder"], row["case_id"]))


def verify_parent(cases: list[dict]) -> None:
    parent_path = exp053.PARENT_OUTPUT
    if sha256(parent_path.read_bytes()) != PARENT_OUTPUT_SHA256:
        raise ValueError("Experiment 050 parent output hash mismatch")
    rows = [json.loads(line) for line in parent_path.read_text().splitlines() if line.strip()]
    by_key = {(row["case_id"], row["condition"], row["decoder"]): row for row in rows}
    for case in cases:
        for decoder in DECODERS:
            key = (case["case_id"], "opinion_masked", decoder)
            if key not in by_key or by_key[key]["gold"] != case["gold"]:
                raise ValueError(f"Missing or mismatched parent row: {key}")


def run(source_dir: Path = Path(".context/dimabsa"), output: Path = OUT,
        manifest_path: Path = MANIFEST, device: str = "mps",
        batch_size: int = DEFAULT_BATCH, preflight_only: bool = False) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 057 protocol hash mismatch")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    task2_rows, task2_hash = exp053.read_source(source_dir)
    if task2_hash != exp053.SOURCE_SHA256:
        raise ValueError("Pinned Task 2 source hash mismatch")
    task3_rows, task3_hash = read_task3(source_dir)
    fresh = exp055.fresh_sample(exp053.select_cases(task2_rows))
    cases, sample_stats = select_category_cases(attach_categories(fresh, task3_rows))
    verify_parent(cases)
    model_config = dict(MODEL_SPECS["qwen-3b"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model"], revision=model_config["revision"], local_files_only=True
    )
    tokenizer.padding_side = "left"
    mappings, donor_meta = donor_maps(cases, tokenizer)
    jobs = build_jobs(cases, mappings)
    if preflight_only:
        options = candidates()
        sequences = candidate_sequences(tokenizer, options)
        prompt_lengths = [
            len(tokenizer.apply_chat_template(
                [{"role": "user", "content": job["prompt"]}],
                tokenize=True, add_generation_prompt=True,
            )) for job in jobs
        ]
        return {
            "experiment": "057-category-matched-context-swap",
            "protocol_sha256": sha256(PROTOCOL.read_bytes()),
            "source_revision": SOURCE_REVISION,
            "task2_sha256": task2_hash,
            "task3_sha256": task3_hash,
            "parent_output_sha256": PARENT_OUTPUT_SHA256,
            **sample_stats,
            "n_permutations": len(mappings),
            "donor_assignment_sha256": {
                p: sha256(json.dumps(m, sort_keys=True).encode()) for p, m in mappings.items()
            },
            "n_unique_assignment_maps": donor_meta["n_unique_assignment_maps"],
            "n_category_polarity_offset_pairs": donor_meta["n_category_polarity_offset_pairs"],
            "all_donors_match_category_and_polarity": True,
            "n_jobs": len(jobs),
            "n_candidate_values": len(options),
            "n_unique_candidate_token_sequences": len(sequences),
            "prompt_token_length_range": [min(prompt_lengths), max(prompt_lengths)],
            "model_revision": model_config["revision"],
        }
    previous = {}
    if output.exists():
        for line in output.read_text().splitlines():
            item = json.loads(line)
            key = (item["case_id"], item["permutation"], item["condition"], item["decoder"])
            if key in previous:
                raise ValueError(f"Duplicate output: {key}")
            previous[key] = item
    allowed = {
        (job["case_id"], job["permutation"], job["condition"], job["decoder"])
        for job in jobs
    }
    if not set(previous).issubset(allowed):
        raise ValueError("Resume file contains outputs outside the frozen design")
    remaining = [
        job for job in jobs
        if (job["case_id"], job["permutation"], job["condition"], job["decoder"]) not in previous
    ]
    del tokenizer
    model, tokenizer, actual_device = load_target(model_config, device, offline=True)
    tokenizer.padding_side = "left"
    sequences = candidate_sequences(tokenizer, candidates())
    trie = build_trie(sequences)
    set_seed(20260957)
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        processed = 0
        for batch in homogeneous_batches(remaining, batch_size):
            decoder = batch[0]["decoder"]
            raw_values = (generate_batch(model, tokenizer, batch, trie, actual_device)
                          if decoder == "finite_grid"
                          else generate_free_batch(model, tokenizer, batch, actual_device))
            for job, raw in zip(batch, raw_values):
                prediction = parse_finite_va(raw) if decoder == "finite_grid" else parse_free_va(raw)
                item = {key: job[key] for key in
                        ("case_id", "permutation", "condition", "decoder", "gold", "donor_id",
                         "donor_category", "donor_polarity")}
                item.update({"raw": raw, "prediction": prediction})
                key = (item["case_id"], item["permutation"], item["condition"], decoder)
                previous[key] = item
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            processed += len(batch)
            print(f"057 {processed}/{len(remaining)} new; total {len(previous)}/{len(jobs)}", flush=True)
    manifest = {
        "experiment": "057-category-matched-context-swap",
        "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        "source_revision": SOURCE_REVISION,
        "task2_sha256": task2_hash,
        "task3_sha256": task3_hash,
        "parent_output_sha256": PARENT_OUTPUT_SHA256,
        "model": model_config["model"], "model_revision": model_config["revision"],
        "device": actual_device, "batch_size": batch_size,
        "max_new_tokens_free": MAX_NEW_TOKENS,
        **sample_stats,
        "n_permutations": len(mappings),
        "donor_assignment_sha256": {
            p: sha256(json.dumps(m, sort_keys=True).encode()) for p, m in mappings.items()
        },
        "n_unique_assignment_maps": donor_meta["n_unique_assignment_maps"],
        "n_category_polarity_offset_pairs": donor_meta["n_category_polarity_offset_pairs"],
        "n_outputs": len(previous),
        "invalid_by_decoder": {
            decoder: sum(row["prediction"] is None and row["decoder"] == decoder
                         for row in previous.values()) for decoder in DECODERS
        },
        "generation_seconds_this_process_only": time.monotonic() - started,
        "resumed_rows": len(jobs) - len(remaining),
        "output_sha256": sha256(output.read_bytes()),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
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
