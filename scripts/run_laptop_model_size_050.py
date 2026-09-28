"""Repeat the laptop decoder factorial with Qwen2.5-3B."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from run_expanded_opinion_mask_repair_043 import candidates
from run_expanded_opinion_mask_repair_043 import generate_batch as generate_finite_batch
from run_laptop_decoder_transfer_049 import (
    CONDITIONS,
    DECODERS,
    SOURCE_REVISION,
    SOURCE_SHA256,
    read_source,
    select_cases,
)
from run_laptop_decoder_transfer_049 import (
    PROTOCOL_SHA256 as PROTOCOL_049_SHA256,
)
from run_laptop_decoder_transfer_049 import (
    build_jobs as build_jobs_049,
)
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
    verify_boundaries,
)
from task_ladder import MODEL_SPECS
from transformers import AutoTokenizer, set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/050-laptop-model-size-factorial.md")
PROTOCOL_SHA256 = "c32e01d4126f7e3f1db134cfcc0c3bf9f4c03c57cb9c0ef2128f678ac0ef489b"
PROTOCOL_049 = Path("docs/experiments/049-laptop-domain-decoder-transfer.md")
PARENT_049_OUTPUT = Path(".context/exp049-private-predictions.jsonl")
PARENT_049_MANIFEST = Path(".context/exp049-run-manifest.json")
PARENT_049_OUTPUT_SHA256 = "62107cb0b33adaf45acc5a0316921dfdab6354beeb006361a8225b87584c050c"
EXPECTED_MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SEED = 20260950
OUT = Path(".context/exp050-private-predictions.jsonl")
MANIFEST = Path(".context/exp050-run-manifest.json")


def parent_rows(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_key = {
        (row["case_id"], row["condition"], row["decoder"]): row
        for row in rows
    }
    if len(rows) != 943 * len(CONDITIONS) * len(DECODERS) or len(by_key) != len(rows):
        raise ValueError("Experiment 049 must contain 3,772 unique outputs")
    return by_key


def build_jobs(source_dir: Path) -> tuple[list[dict], str]:
    rows, source_hash = read_source(source_dir)
    if source_hash != SOURCE_SHA256:
        raise ValueError("English laptop source differs from Experiment 049")
    cases = select_cases(rows)
    jobs = build_jobs_049(cases)
    if len(cases) != 943 or len(jobs) != 3772:
        raise ValueError("Experiment 050 requires the complete 049 laptop sample")
    parent = parent_rows(PARENT_049_OUTPUT)
    expected_keys = {
        (row["case_id"], row["condition"], row["decoder"])
        for row in jobs
    }
    parent_keys = {
        (case_id, condition, decoder)
        for case_id, condition, decoder in parent
    }
    if parent_keys != expected_keys:
        raise ValueError("Experiment 049 output does not cover the same four cells")
    for job in jobs:
        key = (job["case_id"], job["condition"], job["decoder"])
        if parent[key]["gold"] != job["gold"]:
            raise ValueError("Gold VA differs from Experiment 049")
    return jobs, source_hash


def run(
    source_dir: Path = Path(".context/dimabsa"),
    output: Path = OUT,
    manifest_path: Path = MANIFEST,
    device: str = "mps",
    batch_size: int = DEFAULT_BATCH,
    preflight_only: bool = False,
) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 050 protocol hash mismatch")
    if sha256(PROTOCOL_049.read_bytes()) != PROTOCOL_049_SHA256:
        raise ValueError("Experiment 049 protocol changed")
    if sha256(PARENT_049_OUTPUT.read_bytes()) != PARENT_049_OUTPUT_SHA256:
        raise ValueError("Experiment 049 private output changed")
    parent_manifest = json.loads(PARENT_049_MANIFEST.read_text())
    if parent_manifest.get("output_sha256") != PARENT_049_OUTPUT_SHA256:
        raise ValueError("Experiment 049 manifest/output mismatch")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    jobs, source_hash = build_jobs(source_dir)
    model_config = dict(MODEL_SPECS["qwen-3b"])
    if model_config["revision"] != EXPECTED_MODEL_REVISION:
        raise ValueError("Qwen2.5-3B checkpoint revision changed")
    options = candidates()
    if preflight_only:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["model"], revision=model_config["revision"], local_files_only=True
        )
        tokenizer.padding_side = "left"
        sequences = candidate_sequences(tokenizer, options)
        verify_boundaries(tokenizer, jobs[:2], options)
        return {
            "n_clusters": 943,
            "n_prompts": len(jobs),
            "prompts_match_049": True,
            "n_candidate_values": len(options),
            "n_unique_candidate_token_sequences": len(sequences),
            "token_boundary_check": "passed for every candidate on representative prompts",
            "source_revision": SOURCE_REVISION,
            "source_sha256": source_hash,
            "parent_049_output_sha256": PARENT_049_OUTPUT_SHA256,
            "model": model_config["model"],
            "model_revision": model_config["revision"],
            "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        }

    previous = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            key = (row["case_id"], row["condition"], row["decoder"])
            if key in previous:
                raise ValueError(f"Duplicate 050 output: {key}")
            previous[key] = row
    all_keys = {
        (row["case_id"], row["condition"], row["decoder"])
        for row in jobs
    }
    if not set(previous).issubset(all_keys):
        raise ValueError("Resume file contains outputs outside the frozen laptop sample")
    remaining = [
        row for row in jobs
        if (row["case_id"], row["condition"], row["decoder"]) not in previous
    ]

    model, tokenizer, actual_device = load_target(model_config, device, offline=True)
    tokenizer.padding_side = "left"
    sequences = candidate_sequences(tokenizer, options)
    trie = build_trie(sequences)
    verify_boundaries(tokenizer, jobs[:2], options)
    set_seed(SEED)
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        processed = 0
        for batch in homogeneous_batches(remaining, batch_size):
            decoder = batch[0]["decoder"]
            raw_outputs = (
                generate_finite_batch(model, tokenizer, batch, trie, actual_device)
                if decoder == "finite_grid"
                else generate_free_batch(model, tokenizer, batch, actual_device)
            )
            for job, raw in zip(batch, raw_outputs):
                prediction = parse_finite_va(raw) if decoder == "finite_grid" else parse_free_va(raw)
                row = {
                    "case_id": job["case_id"],
                    "condition": job["condition"],
                    "decoder": decoder,
                    "gold": job["gold"],
                    "raw": raw,
                    "prediction": prediction,
                }
                key = (row["case_id"], row["condition"], decoder)
                previous[key] = row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            processed += len(batch)
            print(f"050 {processed}/{len(remaining)} new; total {len(previous)}/{len(jobs)}", flush=True)

    invalid_counts = {
        decoder: {
            condition: sum(
                row["prediction"] is None
                for row in previous.values()
                if row["decoder"] == decoder and row["condition"] == condition
            )
            for condition in CONDITIONS
        }
        for decoder in DECODERS
    }
    free_invalid = sum(invalid_counts["free_greedy"].values())
    manifest = {
        "experiment": "050-laptop-model-size-factorial",
        "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        "parent_049_output_sha256": PARENT_049_OUTPUT_SHA256,
        "source_revision": SOURCE_REVISION,
        "source_file": "eng_laptop_test_task2.jsonl",
        "source_sha256": source_hash,
        "model": model_config["model"],
        "model_revision": model_config["revision"],
        "device": actual_device,
        "batch_size": batch_size,
        "max_new_tokens_free": MAX_NEW_TOKENS,
        "n_clusters": 943,
        "n_prompts": len(jobs),
        "n_outputs": len(previous),
        "n_grid_candidates": len(options),
        "invalid_by_decoder_condition": invalid_counts,
        "invalid_free_outputs": free_invalid,
        "invalid_free_rate": free_invalid / (943 * len(CONDITIONS)),
        "generation_seconds_this_process_only": time.monotonic() - started,
        "resumed_rows": len(jobs) - len(remaining),
        "output_sha256": sha256(output.read_bytes()),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    del model, tokenizer
    gc.collect()
    if actual_device == "mps":
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
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
