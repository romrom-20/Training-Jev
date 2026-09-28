"""Run arousal-first JSON generation for same- and cross-category contexts."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import run_category_matched_context_swap_057 as exp057
from run_expanded_opinion_mask_repair_043 import candidates, generate_batch
from run_laptop_decoder_transfer_049 import SOURCE_REVISION, prompt_with_registered_grid
from run_small_model_decoder_factorial_048 import (
    DEFAULT_BATCH,
    MAX_NEW_TOKENS,
    build_trie,
    candidate_sequences,
    generate_free_batch,
    homogeneous_batches,
    parse_free_va,
    sha256,
)
from task_ladder import MODEL_SPECS
from transformers import AutoTokenizer, set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/062-output-key-order-topic-match.md")
PROTOCOL_SHA256 = "111bfcc52cfd27026f74ec83c769e07b06f5d8362dac666edff7de57e8518aac"
PARENT_PATHS = {
    "exp050": Path(".context/exp050-private-predictions.jsonl"),
    "exp057": Path(".context/exp057-private-predictions.jsonl"),
    "exp060": Path(".context/exp060-private-predictions.jsonl"),
}
PARENT_HASHES = {
    "exp050": "890a16d6363e72549c7ccad9099dece3ce021ab78ab3d8071acde18774500f5f",
    "exp057": "21e7fe02b43433fa4b61b5dfe790efef38c57e8a0e240e4443ffe40ffb110e31",
    "exp060": "ab721276ec320c546da2757a16575acc30eb36ce86df3a911402b3e5a91f9d94",
}
OUT = Path(".context/exp062-private-predictions.jsonl")
MANIFEST = Path(".context/exp062-run-manifest.json")
OWN_CONDITION = "opinion_masked"
SAME_CONDITION = "category_polarity_matched_context"
CROSS_CONDITION = "cross_category_polarity_matched_context"
DECODERS = ("finite_grid", "free_greedy")


def arousal_first_candidates() -> list[tuple[str, dict]]:
    values = []
    for text, point in candidates():
        valence, arousal = point["valence"], point["arousal"]
        values.append((
            f'{{"arousal":{arousal:.1f},"valence":{valence:.1f}}}',
            {"valence": valence, "arousal": arousal},
        ))
    if len(values) != 6561 or len({text for text, _ in values}) != 6561:
        raise ValueError("Expected 6,561 unique arousal-first VA candidates")
    return values


def parse_finite_arousal_first(raw: str) -> list[float] | None:
    match = re.fullmatch(r'\{"arousal":([1-9]\.\d),"valence":([1-9]\.\d)\}', raw.strip())
    if match is None:
        return None
    arousal, valence = map(float, match.groups())
    return [valence, arousal]


def arousal_first_prompt(text: str | None, aspect: str) -> str:
    prompt = prompt_with_registered_grid(text, aspect)
    original = (
        'Return exactly one JSON object with numeric keys "valence" and "arousal", '
        "both from 1.0 to 9.0 in increments of 0.1, with exactly one decimal place."
    )
    reversed_order = (
        'Return exactly one JSON object with numeric keys "arousal" and "valence", '
        "both from 1.0 to 9.0 in increments of 0.1, with exactly one decimal place."
    )
    if original not in prompt:
        raise ValueError("Registered numeric prompt instruction changed")
    return prompt.replace(original, reversed_order, 1)


def verify_parent_files() -> dict[str, str]:
    actual = {}
    for name, path in PARENT_PATHS.items():
        digest = sha256(path.read_bytes())
        if digest != PARENT_HASHES[name]:
            raise ValueError(f"Experiment {name[-3:]} parent output hash mismatch")
        actual[name] = digest
    return actual


def extract_maps(path: Path, condition: str, ids: set[str]) -> dict[int, dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    maps = {1: {}, 2: {}, 3: {}}
    seen = set()
    for row in rows:
        if row["case_id"] not in ids or row["condition"] != condition or row["decoder"] != "finite_grid":
            continue
        key = (row["permutation"], row["case_id"])
        if key in seen:
            raise ValueError(f"Duplicate frozen donor-map row: {key}")
        seen.add(key)
        maps[row["permutation"]][row["case_id"]] = row["donor_id"]
    for permutation, mapping in maps.items():
        if set(mapping) != ids:
            raise ValueError(f"Incomplete donor map for permutation {permutation}: {condition}")
    return maps


def build_jobs(cases: list[dict], same_maps: dict, cross_maps: dict) -> list[dict]:
    by_id = {case["case_id"]: case for case in cases}
    jobs = []
    for case in cases:
        target = case["row"]["Triplet"][case["target_index"]]
        jobs.extend({
            "case_id": case["case_id"], "permutation": 0, "condition": OWN_CONDITION,
            "decoder": decoder, "gold": case["gold"],
            "prompt": arousal_first_prompt(
                exp057.exp053.mask_opinions(case["row"]), target["Aspect"]
            ),
        } for decoder in DECODERS)
    for condition, mappings in ((SAME_CONDITION, same_maps), (CROSS_CONDITION, cross_maps)):
        for permutation, mapping in mappings.items():
            for case in cases:
                donor = by_id[mapping[case["case_id"]]]
                target = case["row"]["Triplet"][case["target_index"]]
                if condition == SAME_CONDITION and donor["category"] != case["category"]:
                    raise ValueError("Frozen same-category donor map contains a category mismatch")
                if condition == CROSS_CONDITION and donor["category"] == case["category"]:
                    raise ValueError("Frozen cross-category donor map contains a category match")
                if exp057.exp055.bucket(donor) != exp057.exp055.bucket(case):
                    raise ValueError("Frozen donor maps differ in gold-valence polarity")
                prompt = arousal_first_prompt(
                    exp057.exp053.mask_opinions(donor["row"]), target["Aspect"]
                )
                for decoder in DECODERS:
                    jobs.append({
                        "case_id": case["case_id"], "permutation": permutation,
                        "condition": condition, "decoder": decoder, "gold": case["gold"],
                        "donor_id": donor["case_id"], "donor_category": donor["category"],
                        "donor_polarity": exp057.exp055.bucket(donor), "prompt": prompt,
                    })
    if len(jobs) != 2576:
        raise ValueError(f"Expected 2,576 arousal-first jobs, got {len(jobs)}")
    return sorted(jobs, key=lambda row: (
        row["decoder"], row["condition"], row["permutation"], row["case_id"]
    ))


def run(source_dir: Path = Path(".context/dimabsa"), output: Path = OUT,
        manifest_path: Path = MANIFEST, device: str = "mps",
        batch_size: int = DEFAULT_BATCH, preflight_only: bool = False) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 062 protocol hash mismatch")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    parent_hashes = verify_parent_files()
    task2_rows, task2_hash = exp057.exp053.read_source(source_dir)
    if task2_hash != exp057.exp053.SOURCE_SHA256:
        raise ValueError("Pinned Task 2 source hash mismatch")
    task3_rows, task3_hash = exp057.read_task3(source_dir)
    fresh = exp057.exp055.fresh_sample(exp057.exp053.select_cases(task2_rows))
    cases, sample_stats = exp057.select_category_cases(exp057.attach_categories(fresh, task3_rows))
    ids = {case["case_id"] for case in cases}
    same_maps = extract_maps(PARENT_PATHS["exp057"], SAME_CONDITION, ids)
    cross_maps = extract_maps(PARENT_PATHS["exp060"], CROSS_CONDITION, ids)
    jobs = build_jobs(cases, same_maps, cross_maps)
    config = dict(MODEL_SPECS["qwen-3b"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True
    )
    tokenizer.padding_side = "left"
    if preflight_only:
        values = arousal_first_candidates()
        sequences = candidate_sequences(tokenizer, values)
        prompt_lengths = [len(tokenizer.apply_chat_template(
            [{"role": "user", "content": job["prompt"]}],
            tokenize=True, add_generation_prompt=True,
        )) for job in jobs]
        return {
            "experiment": "062-output-key-order-topic-match",
            "protocol_sha256": sha256(PROTOCOL.read_bytes()),
            "parent_output_sha256": parent_hashes,
            "task2_sha256": task2_hash, "task3_sha256": task3_hash,
            **sample_stats,
            "n_jobs": len(jobs),
            "n_own_review_jobs": sum(job["condition"] == OWN_CONDITION for job in jobs),
            "n_same_category_jobs": sum(job["condition"] == SAME_CONDITION for job in jobs),
            "n_cross_category_jobs": sum(job["condition"] == CROSS_CONDITION for job in jobs),
            "n_candidate_values": len(values), "n_unique_candidate_token_sequences": len(sequences),
            "prompt_token_length_range": [min(prompt_lengths), max(prompt_lengths)],
            "same_category_map_sha256": {
                p: sha256(json.dumps(m, sort_keys=True).encode()) for p, m in same_maps.items()
            },
            "cross_category_map_sha256": {
                p: sha256(json.dumps(m, sort_keys=True).encode()) for p, m in cross_maps.items()
            },
            "model_revision": config["revision"],
        }

    previous = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            key = (row["case_id"], row["permutation"], row["condition"], row["decoder"])
            if key in previous:
                raise ValueError(f"Duplicate output row: {key}")
            previous[key] = row
    allowed = {
        (job["case_id"], job["permutation"], job["condition"], job["decoder"])
        for job in jobs
    }
    if not set(previous).issubset(allowed):
        raise ValueError("Resume file contains outputs outside the frozen design")
    remaining = [job for job in jobs if (
        job["case_id"], job["permutation"], job["condition"], job["decoder"]
    ) not in previous]
    del tokenizer
    model, tokenizer, actual_device = load_target(config, device, offline=True)
    tokenizer.padding_side = "left"
    trie = build_trie(candidate_sequences(tokenizer, arousal_first_candidates()))
    set_seed(20260962)
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        processed = 0
        for batch in homogeneous_batches(remaining, batch_size):
            decoder = batch[0]["decoder"]
            raws = (generate_batch(model, tokenizer, batch, trie, actual_device)
                    if decoder == "finite_grid"
                    else generate_free_batch(model, tokenizer, batch, actual_device))
            for job, raw in zip(batch, raws):
                prediction = parse_finite_arousal_first(raw) if decoder == "finite_grid" else parse_free_va(raw)
                row = {key: job[key] for key in (
                    "case_id", "permutation", "condition", "decoder", "gold",
                )}
                if job["condition"] != OWN_CONDITION:
                    row.update({key: job[key] for key in (
                        "donor_id", "donor_category", "donor_polarity"
                    )})
                row.update({"raw": raw, "prediction": prediction})
                key = (row["case_id"], row["permutation"], row["condition"], decoder)
                previous[key] = row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            processed += len(batch)
            print(f"062 {processed}/{len(remaining)} new; total {len(previous)}/{len(jobs)}", flush=True)
    manifest = {
        "experiment": "062-output-key-order-topic-match",
        "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        "source_revision": SOURCE_REVISION,
        "source_sha256": task2_hash, "task2_sha256": task2_hash, "task3_sha256": task3_hash,
        "parent_output_sha256": parent_hashes,
        "model": config["model"], "model_revision": config["revision"],
        "device": actual_device, "batch_size": batch_size,
        "max_new_tokens_free": MAX_NEW_TOKENS, **sample_stats,
        "n_outputs": len(previous),
        "invalid_by_decoder_condition": {
            decoder: {
                condition: sum(row["decoder"] == decoder and row["condition"] == condition
                               and row["prediction"] is None for row in previous.values())
                for condition in (OWN_CONDITION, SAME_CONDITION, CROSS_CONDITION)
            } for decoder in DECODERS
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
