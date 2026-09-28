"""Test whether the 3B decoder interaction depends on the matching review."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from run_expanded_opinion_mask_repair_043 import candidates, generate_batch
from run_laptop_decoder_transfer_049 import (
    SOURCE_REVISION,
    prompt_with_registered_grid,
    read_source,
    select_cases,
)
from run_opinion_mask_crosslingual_dimabsa_041 import mask_opinions
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

PROTOCOL = Path("docs/experiments/053-counterfactual-review-context-swap.md")
PROTOCOL_SHA256 = "ac85f068e7e2f73b4732d5e85aab7443b6f82c459b4bbaf5c62e83cb83ab0006"
PARENT_OUTPUT_SHA256 = "890a16d6363e72549c7ccad9099dece3ce021ab78ab3d8071acde18774500f5f"
PARENT_OUTPUT = Path(".context/exp050-private-predictions.jsonl")
SOURCE_SHA256 = "08a4cc197f068f6fc4c9373f61bfb86a5cd996ef16be3cb516082b434ba92543"
SAMPLE_SEED = "exp053-sample-v1|"
DONOR_SEED = "exp053-donor-v1|"
BUCKET_COUNTS = {"neg": 94, "neu": 30, "pos": 93}
DECODERS = ("finite_grid", "free_greedy")
CONDITION = "swapped_context"
OUT = Path(".context/exp053-private-predictions.jsonl")
MANIFEST = Path(".context/exp053-run-manifest.json")


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def valence_bucket(case: dict) -> str:
    valence = case["gold"][0]
    return "neg" if valence < 4.5 else "neu" if valence <= 5.5 else "pos"


def choose_sample(cases: list[dict]) -> list[dict]:
    selected = []
    by_bucket: dict[str, list[dict]] = {key: [] for key in BUCKET_COUNTS}
    for case in cases:
        by_bucket[valence_bucket(case)].append(case)
    for bucket, count in BUCKET_COUNTS.items():
        ranked = sorted(
            by_bucket[bucket], key=lambda case: stable_hash(SAMPLE_SEED + case["case_id"])
        )
        if len(ranked) < count:
            raise ValueError(f"Not enough cases in {bucket}: expected {count}, found {len(ranked)}")
        selected.extend(ranked[:count])
    selected.sort(key=lambda case: case["case_id"])
    if len(selected) != 217 or len({case["case_id"] for case in selected}) != 217:
        raise ValueError("Frozen sample must contain 217 unique IDs")
    return selected


def build_donor_map(cases: list[dict], tokenizer) -> tuple[dict[str, str], dict[str, int]]:
    lengths = {}
    for case in cases:
        row = case["row"]
        masked_text = mask_opinions(row)
        lengths[case["case_id"]] = len(tokenizer.encode(masked_text, add_special_tokens=False))
    ranked = sorted(cases, key=lambda case: (lengths[case["case_id"]], case["case_id"]))
    # Use contiguous length deciles.
    chunks = [ranked[(len(ranked) * index) // 10 : (len(ranked) * (index + 1)) // 10] for index in range(10)]
    donor_map = {}
    for chunk in chunks:
        shuffled = sorted(chunk, key=lambda case: stable_hash(DONOR_SEED + case["case_id"]))
        if len(shuffled) < 2:
            raise ValueError("Every length decile must contain at least two cases")
        for index, case in enumerate(shuffled):
            donor_map[case["case_id"]] = shuffled[(index + 1) % len(shuffled)]["case_id"]
    ids = {case["case_id"] for case in cases}
    if set(donor_map) != ids or set(donor_map.values()) != ids:
        raise ValueError("Donor assignment must be a permutation of the sample IDs")
    if any(case_id == donor_map[case_id] for case_id in ids):
        raise ValueError("Donor assignment must be a derangement")
    return donor_map, lengths


def verify_parent(cases: list[dict]) -> dict:
    if sha256(PARENT_OUTPUT.read_bytes()) != PARENT_OUTPUT_SHA256:
        raise ValueError("Experiment 050 parent output hash mismatch")
    rows = [json.loads(line) for line in PARENT_OUTPUT.read_text().splitlines() if line.strip()]
    by_key = {(row["case_id"], row["condition"], row["decoder"]): row for row in rows}
    expected_ids = {case["case_id"] for case in cases}
    for case in cases:
        for decoder in DECODERS:
            key = (case["case_id"], "opinion_masked", decoder)
            if key not in by_key:
                raise ValueError(f"Missing matched parent prediction: {key}")
            parent = by_key[key]
            if parent["gold"] != case["gold"]:
                raise ValueError(f"Parent gold mismatch: {key}")
    return {
        "parent_output_sha256": PARENT_OUTPUT_SHA256,
        "n_parent_rows": len(rows),
        "n_selected_ids_present": len(expected_ids),
    }


def build_jobs(cases: list[dict], donor_map: dict[str, str]) -> list[dict]:
    by_id = {case["case_id"]: case for case in cases}
    jobs = []
    for case in cases:
        donor = by_id[donor_map[case["case_id"]]]
        recipient_target = case["row"]["Triplet"][case["target_index"]]
        masked_text = mask_opinions(donor["row"])
        prompt = prompt_with_registered_grid(masked_text, recipient_target["Aspect"])
        for decoder in DECODERS:
            jobs.append(
                {
                    "case_id": case["case_id"],
                    "condition": CONDITION,
                    "decoder": decoder,
                    "gold": case["gold"],
                    "donor_id": donor["case_id"],
                    "prompt": prompt,
                }
            )
    if len(jobs) != 434:
        raise ValueError("Expected exactly 434 swapped-context judgments")
    prompts = {}
    for job in jobs:
        if job["case_id"] in prompts and prompts[job["case_id"]] != job["prompt"]:
            raise ValueError("Prompt bytes must match between decoder arms")
        prompts[job["case_id"]] = job["prompt"]
    return sorted(jobs, key=lambda row: (row["decoder"], row["case_id"]))


def run(
    source_dir: Path = Path(".context/dimabsa"),
    output: Path = OUT,
    manifest_path: Path = MANIFEST,
    device: str = "mps",
    batch_size: int = DEFAULT_BATCH,
    preflight_only: bool = False,
) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 053 protocol hash mismatch")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    rows, source_hash = read_source(source_dir)
    if source_hash != SOURCE_SHA256:
        raise ValueError("Pinned source hash mismatch")
    cases = choose_sample(select_cases(rows))
    parent_info = verify_parent(cases)
    model_config = dict(MODEL_SPECS["qwen-3b"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model"], revision=model_config["revision"], local_files_only=True
    )
    tokenizer.padding_side = "left"
    donor_map, text_lengths = build_donor_map(cases, tokenizer)
    jobs = build_jobs(cases, donor_map)
    if preflight_only:
        options = candidates()
        sequences = candidate_sequences(tokenizer, options)
        prompt_lengths = [
            len(tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            ))
            for row in jobs
        ]
        return {
            "experiment": "053-counterfactual-review-context-swap",
            "protocol_sha256": sha256(PROTOCOL.read_bytes()),
            "source_sha256": source_hash,
            "source_revision": SOURCE_REVISION,
            "parent_verification": parent_info,
            "sample_n": len(cases),
            "sample_buckets": dict(Counter(valence_bucket(case) for case in cases)),
            "donor_assignment_sha256": sha256(json.dumps(donor_map, sort_keys=True).encode()),
            "donor_assignment_is_derangement": True,
            "donor_token_length_range": [min(text_lengths.values()), max(text_lengths.values())],
            "prompt_token_length_range": [min(prompt_lengths), max(prompt_lengths)],
            "n_jobs": len(jobs),
            "n_candidate_values": len(options),
            "n_unique_candidate_token_sequences": len(sequences),
            "model_revision": model_config["revision"],
        }

    previous = {}
    if output.exists():
        for line in output.read_text().splitlines():
            item = json.loads(line)
            key = (item["case_id"], item["condition"], item["decoder"])
            if key in previous:
                raise ValueError(f"Duplicate 053 output: {key}")
            previous[key] = item
    allowed = {(job["case_id"], job["condition"], job["decoder"]) for job in jobs}
    if not set(previous).issubset(allowed):
        raise ValueError("Resume file contains outputs outside the frozen sample")
    remaining = [job for job in jobs if (job["case_id"], job["condition"], job["decoder"]) not in previous]
    del tokenizer
    model, tokenizer, actual_device = load_target(model_config, device, offline=True)
    tokenizer.padding_side = "left"
    options = candidates()
    sequences = candidate_sequences(tokenizer, options)
    trie = build_trie(sequences)
    set_seed(20260953)
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        processed = 0
        for batch in homogeneous_batches(remaining, batch_size):
            decoder = batch[0]["decoder"]
            raw_values = (
                generate_batch(model, tokenizer, batch, trie, actual_device)
                if decoder == "finite_grid"
                else generate_free_batch(model, tokenizer, batch, actual_device)
            )
            for job, raw in zip(batch, raw_values):
                prediction = parse_finite_va(raw) if decoder == "finite_grid" else parse_free_va(raw)
                item = {key: job[key] for key in ("case_id", "condition", "decoder", "gold", "donor_id")}
                item.update({"raw": raw, "prediction": prediction})
                key = (item["case_id"], item["condition"], decoder)
                previous[key] = item
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            processed += len(batch)
            print(f"053 {processed}/{len(remaining)} new; total {len(previous)}/{len(jobs)}", flush=True)
    manifest = {
        "experiment": "053-counterfactual-review-context-swap",
        "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        "source_revision": SOURCE_REVISION,
        "source_sha256": source_hash,
        "parent_output_sha256": PARENT_OUTPUT_SHA256,
        "model": model_config["model"],
        "model_revision": model_config["revision"],
        "device": actual_device,
        "batch_size": batch_size,
        "max_new_tokens_free": MAX_NEW_TOKENS,
        "n_sample_ids": len(cases),
        "sample_buckets": dict(Counter(valence_bucket(case) for case in cases)),
        "donor_assignment_sha256": sha256(json.dumps(donor_map, sort_keys=True).encode()),
        "n_outputs": len(previous),
        "invalid_by_decoder": {
            decoder: sum(row["prediction"] is None and row["decoder"] == decoder for row in previous.values())
            for decoder in DECODERS
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
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
