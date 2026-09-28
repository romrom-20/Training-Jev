"""Test whether the 3B decoder interaction transfers to English restaurants."""

from __future__ import annotations

import argparse
import gc
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

import torch
from run_expanded_opinion_mask_repair_043 import candidates
from run_expanded_opinion_mask_repair_043 import generate_batch as generate_finite_batch
from run_laptop_decoder_transfer_049 import (
    CONDITIONS,
    DECODERS,
)
from run_laptop_decoder_transfer_049 import (
    build_jobs as build_shared_jobs,
)
from run_opinion_mask_crosslingual_dimabsa_041 import (
    SOURCE_REVISION,
    _eligible_target,
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

PROTOCOL = Path("docs/experiments/051-restaurant-domain-decoder-transfer-3b.md")
PROTOCOL_SHA256 = "fccbee4b05f938704465f0c2b42314dc7a76fe8708976fb5550780ade322f9c0"
SOURCE_FILE = "eng_restaurant_test_task2.jsonl"
SOURCE_URL = (
    "https://raw.githubusercontent.com/DimABSA/DimABSA2026/"
    f"{SOURCE_REVISION}/task-dataset/track_a/subtask_2/eng/{SOURCE_FILE}"
)
SOURCE_SHA256 = "e90386169422c84ddd42d51029f23934b54724fafe270b65138836067492857d"
SEED = 20260951
OUT = Path(".context/exp051-private-predictions.jsonl")
MANIFEST = Path(".context/exp051-run-manifest.json")
EXPECTED_MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def read_source(source_dir: Path) -> tuple[list[dict], str]:
    path = source_dir / SOURCE_FILE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Training-Jev research runner"})
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    raw = path.read_bytes()
    actual_hash = sha256(raw)
    if actual_hash != SOURCE_SHA256:
        raise ValueError(f"Pinned English restaurant source hash mismatch: {actual_hash}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 1000 or len({row.get("ID") for row in rows}) != 1000:
        raise ValueError("Expected 1,000 unique rows in the pinned English restaurant test split")
    return rows, actual_hash


def select_cases(rows: list[dict]) -> list[dict]:
    cases = []
    for row in rows:
        for target_index, target in enumerate(row["Triplet"]):
            if not _eligible_target([row], target_index):
                continue
            scores = target.get("VA")
            if not isinstance(scores, str) or scores.count("#") != 1:
                continue
            try:
                gold = [float(value) for value in scores.split("#")]
            except ValueError:
                continue
            if len(gold) != 2 or any(not 1 <= score <= 9 for score in gold):
                continue
            cases.append(
                {
                    "case_id": str(row["ID"]),
                    "target_index": target_index,
                    "gold": gold,
                    "row": row,
                }
            )
            break
    if len(cases) != 963 or len({case["case_id"] for case in cases}) != 963:
        raise ValueError(f"Expected 963 eligible unique source IDs; found {len(cases)}")
    return cases


def build_jobs(cases: list[dict]) -> list[dict]:
    return build_shared_jobs(cases)


def run(
    source_dir: Path = Path(".context/dimabsa"),
    output: Path = OUT,
    manifest_path: Path = MANIFEST,
    device: str = "mps",
    batch_size: int = DEFAULT_BATCH,
    preflight_only: bool = False,
) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 051 protocol hash mismatch")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    rows, source_hash = read_source(source_dir)
    cases = select_cases(rows)
    jobs = build_jobs(cases)
    if len(jobs) != 3852:
        raise ValueError(f"Expected 3,852 paired jobs, found {len(jobs)}")
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
            "n_source_rows": len(rows),
            "n_eligible_ids": len(cases),
            "valence_buckets": dict(
                Counter(
                    "neg" if case["gold"][0] < 4.5 else "neu" if case["gold"][0] <= 5.5 else "pos"
                    for case in cases
                )
            ),
            "n_prompts": len(jobs),
            "condition_counts_by_decoder": {
                decoder: {
                    condition: sum(
                        row["decoder"] == decoder and row["condition"] == condition for row in jobs
                    )
                    for condition in CONDITIONS
                }
                for decoder in DECODERS
            },
            "prompts_identical_across_decoders": True,
            "n_candidate_values": len(options),
            "n_unique_candidate_token_sequences": len(sequences),
            "token_boundary_check": "passed for every candidate on representative prompts",
            "source_revision": SOURCE_REVISION,
            "source_sha256": source_hash,
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
                raise ValueError(f"Duplicate 051 output: {key}")
            previous[key] = row
    all_keys = {
        (row["case_id"], row["condition"], row["decoder"])
        for row in jobs
    }
    if not set(previous).issubset(all_keys):
        raise ValueError("Resume file contains outputs outside the frozen restaurant sample")
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
                item = {
                    "case_id": job["case_id"],
                    "condition": job["condition"],
                    "decoder": decoder,
                    "gold": job["gold"],
                    "raw": raw,
                    "prediction": prediction,
                }
                key = (item["case_id"], item["condition"], decoder)
                previous[key] = item
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            processed += len(batch)
            print(f"051 {processed}/{len(remaining)} new; total {len(previous)}/{len(jobs)}", flush=True)

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
    valence_buckets = dict(
        Counter(
            "neg" if case["gold"][0] < 4.5 else "neu" if case["gold"][0] <= 5.5 else "pos"
            for case in cases
        )
    )
    manifest = {
        "experiment": "051-restaurant-domain-decoder-transfer-3b",
        "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        "source_revision": SOURCE_REVISION,
        "source_file": SOURCE_FILE,
        "source_sha256": source_hash,
        "n_source_rows": len(rows),
        "n_eligible_ids": len(cases),
        "selected_valence_buckets": valence_buckets,
        "model": model_config["model"],
        "model_revision": model_config["revision"],
        "device": actual_device,
        "batch_size": batch_size,
        "max_new_tokens_free": MAX_NEW_TOKENS,
        "n_prompts": len(jobs),
        "n_outputs": len(previous),
        "n_grid_candidates": len(options),
        "invalid_by_decoder_condition": invalid_counts,
        "invalid_free_outputs": free_invalid,
        "invalid_free_rate": free_invalid / (len(cases) * len(CONDITIONS)),
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
