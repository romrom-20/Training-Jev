"""Transfer the category-matched context control to Qwen2.5-1.5B."""

from __future__ import annotations

import argparse
import json
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
    parse_finite_va,
    parse_free_va,
    sha256,
)
from task_ladder import MODEL_SPECS
from transformers import AutoTokenizer, set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/059-category-match-size-transfer.md")
PROTOCOL_SHA256 = "fb206d8dadd19399c84c78b9131cf69322a05fec1d5992c36697de83b4fc396d"
EXP057_OUTPUT = Path(".context/exp057-private-predictions.jsonl")
EXP057_MANIFEST = Path(".context/exp057-run-manifest.json")
EXP057_OUTPUT_SHA256 = "21e7fe02b43433fa4b61b5dfe790efef38c57e8a0e240e4443ffe40ffb110e31"
EXP057_ASSIGNMENT_HASHES = {
    "1": "3447f446f6ccdb8edbf7d86b3d40a6d88c98bf46b213d6acd3f32590b4672949",
    "2": "edfd5c94ffdca55adba488d5b573e8bb5cc8fc33db055e9592b19f0221f0b1b8",
    "3": "9e8b819b02ef7a19033e7ade367b98d64d1e46535446d9495494b46d715faa9b",
}
OUT = Path(".context/exp059-private-predictions.jsonl")
MANIFEST = Path(".context/exp059-run-manifest.json")
DECODERS = ("finite_grid", "free_greedy")
DONOR_CONDITION = "category_polarity_matched_context"
OWN_CONDITION = "opinion_masked"


def load_frozen_maps(ids: set[str]) -> tuple[dict[int, dict[str, str]], str]:
    if sha256(EXP057_OUTPUT.read_bytes()) != EXP057_OUTPUT_SHA256:
        raise ValueError("Experiment 057 parent output hash changed")
    parent_manifest = json.loads(EXP057_MANIFEST.read_text())
    if parent_manifest.get("donor_assignment_sha256") != EXP057_ASSIGNMENT_HASHES:
        raise ValueError("Experiment 057 donor-map manifest changed")
    maps = {permutation: {} for permutation in (1, 2, 3)}
    rows = [json.loads(line) for line in EXP057_OUTPUT.read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        if row["case_id"] not in ids or row["decoder"] != "finite_grid":
            continue
        key = (row["permutation"], row["case_id"])
        if key in seen:
            raise ValueError(f"Duplicate Experiment 057 donor map entry: {key}")
        seen.add(key)
        maps[row["permutation"]][row["case_id"]] = row["donor_id"]
    for permutation, mapping in maps.items():
        if set(mapping) != ids:
            raise ValueError(f"Incomplete 057 donor map {permutation}")
        if sha256(json.dumps(mapping, sort_keys=True).encode()) != EXP057_ASSIGNMENT_HASHES[str(permutation)]:
            raise ValueError(f"Experiment 057 map hash mismatch for permutation {permutation}")
    return maps, parent_manifest["output_sha256"]


def own_review_jobs(cases: list[dict]) -> list[dict]:
    jobs = []
    for case in cases:
        target = case["row"]["Triplet"][case["target_index"]]
        jobs.extend({
            "case_id": case["case_id"], "permutation": 0,
            "condition": OWN_CONDITION, "decoder": decoder,
            "gold": case["gold"],
            "prompt": prompt_with_registered_grid(
                exp057.exp053.mask_opinions(case["row"]), target["Aspect"]
            ),
        } for decoder in DECODERS)
    return jobs


def run(source_dir: Path = Path(".context/dimabsa"), output: Path = OUT,
        manifest_path: Path = MANIFEST, device: str = "mps",
        batch_size: int = DEFAULT_BATCH, preflight_only: bool = False) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 059 protocol hash mismatch")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    task2_rows, task2_hash = exp057.exp053.read_source(source_dir)
    if task2_hash != exp057.exp053.SOURCE_SHA256:
        raise ValueError("Pinned Task 2 source hash mismatch")
    task3_rows, task3_hash = exp057.read_task3(source_dir)
    fresh = exp057.exp055.fresh_sample(exp057.exp053.select_cases(task2_rows))
    cases, sample_stats = exp057.select_category_cases(
        exp057.attach_categories(fresh, task3_rows)
    )
    ids = {case["case_id"] for case in cases}
    donor_maps, exp057_output_hash = load_frozen_maps(ids)
    donor_jobs = exp057.build_jobs(cases, donor_maps)
    own_jobs = own_review_jobs(cases)
    jobs = sorted(donor_jobs + own_jobs, key=lambda row: (
        row["decoder"], row["condition"], row["permutation"], row["case_id"]
    ))
    if len(jobs) != 1472:
        raise ValueError(f"Expected 1,472 jobs, got {len(jobs)}")

    model_config = dict(MODEL_SPECS["qwen-1.5b"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model"], revision=model_config["revision"], local_files_only=True
    )
    tokenizer.padding_side = "left"
    if preflight_only:
        sequences = candidate_sequences(tokenizer, candidates())
        prompt_lengths = [len(tokenizer.apply_chat_template(
            [{"role": "user", "content": job["prompt"]}],
            tokenize=True, add_generation_prompt=True,
        )) for job in jobs]
        return {
            "experiment": "059-category-match-size-transfer",
            "protocol_sha256": sha256(PROTOCOL.read_bytes()),
            "task2_sha256": task2_hash, "task3_sha256": task3_hash,
            "exp057_output_sha256": exp057_output_hash,
            "n_jobs": len(jobs), "n_donor_jobs": len(donor_jobs), "n_own_review_jobs": len(own_jobs),
            "n_candidate_values": len(candidates()),
            "n_unique_candidate_token_sequences": len(sequences),
            "prompt_token_length_range": [min(prompt_lengths), max(prompt_lengths)],
            "model_revision": model_config["revision"], **sample_stats,
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
        raise ValueError("Resume file contains rows outside the frozen design")
    remaining = [job for job in jobs if (
        job["case_id"], job["permutation"], job["condition"], job["decoder"]
    ) not in previous]
    del tokenizer
    model, tokenizer, actual_device = load_target(model_config, device, offline=True)
    tokenizer.padding_side = "left"
    sequences = candidate_sequences(tokenizer, candidates())
    trie = build_trie(sequences)
    set_seed(20260959)
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
                prediction = parse_finite_va(raw) if decoder == "finite_grid" else parse_free_va(raw)
                row = {key: job[key] for key in (
                    "case_id", "permutation", "condition", "decoder", "gold"
                )}
                row.update({"raw": raw, "prediction": prediction})
                key = (row["case_id"], row["permutation"], row["condition"], decoder)
                previous[key] = row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            processed += len(batch)
            print(f"059 {processed}/{len(remaining)} new; total {len(previous)}/{len(jobs)}", flush=True)
    manifest = {
        "experiment": "059-category-match-size-transfer",
        "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        "source_revision": SOURCE_REVISION,
        "task2_sha256": task2_hash, "source_sha256": task2_hash,
        "task3_sha256": task3_hash, "exp057_output_sha256": exp057_output_hash,
        "parent_output_sha256": exp057_output_hash,
        "donor_assignment_sha256": EXP057_ASSIGNMENT_HASHES,
        "model": model_config["model"], "model_revision": model_config["revision"],
        "device": actual_device, "batch_size": batch_size,
        "max_new_tokens_free": MAX_NEW_TOKENS, **sample_stats,
        "n_outputs": len(previous),
        "invalid_by_decoder_condition": {
            decoder: {
                condition: sum(row["decoder"] == decoder and row["condition"] == condition
                               and row["prediction"] is None for row in previous.values())
                for condition in (OWN_CONDITION, DONOR_CONDITION)
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
