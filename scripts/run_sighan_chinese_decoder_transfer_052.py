"""Run the matched decoder-by-context factorial on SIGHAN 2024 Chinese reviews."""

from __future__ import annotations

import argparse
import gc
import json
import re
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
from run_opinion_mask_crosslingual_dimabsa_041 import _eligible_target
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

PROTOCOL = Path("docs/experiments/052-sighan-chinese-decoder-transfer.md")
PROTOCOL_SHA256 = "679bf9fd850ffc2dbdd0534c2a8f5be0a3d7edbf3f02541feee06249d01bed16"
SOURCE_REVISION = "d43a482039a41fbc26087ef5b26e94ee772d0fb6"
SOURCE_FILES = {
    "test_input": "SIGHAN2024_dimABSA_Testing_Task2+3_Simplified.txt",
    "test_gold": "SIGHAN2024_dimABSA_Testing_Task2+3_Simplified_truth.txt",
}
SOURCE_SHA256 = {
    "test_input": "5053608647d06ad88ce21a2aecbef0e29dc58a8220f921561d39d4e9cf86fe28",
    "test_gold": "6ecc6949d5f77ea34a09826115a0f91327313ae33824a5c373d7322bdbb17108",
}
RAW_BASE = (
    "https://raw.githubusercontent.com/NYCU-NLP/SIGHAN2024-dimABSA/"
    f"{SOURCE_REVISION}/DataSets/dimABSA2024/Simplified"
)
SEED = 20260952
EXPECTED_SOURCE_ROWS = 2000
EXPECTED_ELIGIBLE_IDS = 1916
OUT = Path(".context/exp052-private-predictions.jsonl")
MANIFEST = Path(".context/exp052-run-manifest.json")
SOURCE_DIR = Path(".context/sighan2024-dimabsa-audit/DataSets/dimABSA2024/Simplified")
EXPECTED_MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def _read_pinned_file(source_dir: Path, key: str) -> bytes:
    path = source_dir / SOURCE_FILES[key]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            f"{RAW_BASE}/{SOURCE_FILES[key]}",
            headers={"User-Agent": "Training-Jev research runner"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    raw = path.read_bytes()
    actual = sha256(raw)
    if actual != SOURCE_SHA256[key]:
        raise ValueError(f"Pinned SIGHAN {key} source hash mismatch: {actual}")
    return raw


def parse_source(input_raw: bytes, gold_raw: bytes) -> list[dict]:
    input_lines = input_raw.decode("utf-8").splitlines()
    gold_lines = gold_raw.decode("utf-8").splitlines()
    if not input_lines or input_lines[0].strip() != "ID, Sentence":
        raise ValueError("Unexpected SIGHAN Task 2/3 input header")
    if not gold_lines or gold_lines[0].strip() != "ID Quadruples":
        raise ValueError("Unexpected SIGHAN Task 2/3 gold header")
    input_rows = [line for line in input_lines[1:] if line.strip()]
    gold_rows = [line for line in gold_lines[1:] if line.strip()]
    if len(input_rows) != len(gold_rows):
        raise ValueError("SIGHAN input and gold row counts differ")

    rows: list[dict] = []
    seen: set[str] = set()
    for input_line, gold_line in zip(input_rows, gold_rows):
        if "," not in input_line:
            raise ValueError("Malformed SIGHAN input row")
        case_id, text = input_line.split(",", 1)
        case_id, text = case_id.strip(), text.strip()
        if not case_id or case_id in seen:
            raise ValueError(f"Duplicate or empty SIGHAN test ID: {case_id}")
        seen.add(case_id)
        gold_id, separator, payload = gold_line.partition(" ")
        if not separator or gold_id.strip() != case_id:
            raise ValueError(f"SIGHAN input/gold IDs differ at {case_id}")

        triplets = []
        for match in re.finditer(r"\(([^()]*)\)", payload):
            fields = match.group(1).rsplit(",", 3)
            if len(fields) != 4:
                raise ValueError(f"Malformed SIGHAN gold quadruple for {case_id}")
            aspect, category, opinion, scores = (field.strip() for field in fields)
            if scores.count("#") != 1:
                raise ValueError(f"Malformed SIGHAN VA score for {case_id}")
            try:
                valence, arousal = (float(value) for value in scores.split("#"))
                normalized_scores = f"{valence}#{arousal}"
            except ValueError:
                # Keep a malformed gold value visible so case selection can skip
                # this target without discarding other valid targets in the row.
                normalized_scores = scores
            triplets.append({
                "Aspect": aspect,
                "Category": category,
                "Opinion": opinion,
                "VA": normalized_scores,
            })
        if not triplets:
            raise ValueError(f"No annotated triplets found for {case_id}")
        rows.append({"ID": case_id, "Text": text, "Triplet": triplets})
    return rows


def read_source(source_dir: Path = SOURCE_DIR) -> tuple[list[dict], dict[str, str]]:
    input_raw = _read_pinned_file(source_dir, "test_input")
    gold_raw = _read_pinned_file(source_dir, "test_gold")
    rows = parse_source(input_raw, gold_raw)
    if len(rows) != EXPECTED_SOURCE_ROWS:
        raise ValueError("Expected 2,000 unique SIGHAN Task 2/3 test rows")
    return rows, SOURCE_SHA256.copy()


def select_cases(rows: list[dict]) -> list[dict]:
    cases = []
    for row in rows:
        for target_index, target in enumerate(row["Triplet"]):
            if not _eligible_target([row], target_index):
                continue
            try:
                gold = [float(value) for value in target["VA"].split("#")]
            except (TypeError, ValueError):
                continue
            if len(gold) != 2 or any(not 1 <= value <= 9 for value in gold):
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
    if len(cases) != EXPECTED_ELIGIBLE_IDS or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError(
            f"Expected {EXPECTED_ELIGIBLE_IDS} eligible unique IDs; found {len(cases)}"
        )
    return cases


def build_jobs(cases: list[dict]) -> list[dict]:
    jobs = build_shared_jobs(cases)
    expected = EXPECTED_ELIGIBLE_IDS * len(CONDITIONS) * len(DECODERS)
    if len(jobs) != expected:
        raise ValueError(f"Expected {expected} paired jobs, found {len(jobs)}")
    paired_prompts: dict[tuple[str, str], str] = {}
    seen = set()
    for job in jobs:
        key = (job["case_id"], job["condition"])
        full_key = (*key, job["decoder"])
        if full_key in seen:
            raise ValueError(f"Duplicate 052 job: {full_key}")
        seen.add(full_key)
        if key in paired_prompts and paired_prompts[key] != job["prompt"]:
            raise ValueError("Prompt bytes differ between decoder arms")
        paired_prompts[key] = job["prompt"]
    if len(seen) != expected:
        raise ValueError("The paired factorial is incomplete")
    return jobs


def run(
    source_dir: Path = SOURCE_DIR,
    output: Path = OUT,
    manifest_path: Path = MANIFEST,
    device: str = "mps",
    batch_size: int = DEFAULT_BATCH,
    preflight_only: bool = False,
) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 052 protocol hash mismatch")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    rows, source_hashes = read_source(source_dir)
    cases = select_cases(rows)
    jobs = build_jobs(cases)
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
        prompt_lengths = [
            len(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
            for row in jobs
        ]
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
            "max_prompt_tokens": max(prompt_lengths),
            "token_boundary_check": "passed for representative candidates and prompts",
            "source_hashes": source_hashes,
            "model": model_config["model"],
            "model_revision": model_config["revision"],
            "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        }

    previous = {}
    if output.exists():
        for line in output.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["case_id"], row["condition"], row["decoder"])
            if key in previous:
                raise ValueError(f"Duplicate 052 output: {key}")
            previous[key] = row
    all_keys = {
        (row["case_id"], row["condition"], row["decoder"])
        for row in jobs
    }
    if not set(previous).issubset(all_keys):
        raise ValueError("Resume file contains outputs outside the frozen SIGHAN sample")
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
            print(f"052 {processed}/{len(remaining)} new; total {len(previous)}/{len(jobs)}", flush=True)

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
    manifest = {
        "experiment": "052-sighan-chinese-decoder-transfer",
        "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        "source_revision": SOURCE_REVISION,
        "source_files": SOURCE_FILES,
        "source_hashes": source_hashes,
        "n_source_rows": len(rows),
        "n_eligible_ids": len(cases),
        "selected_valence_buckets": dict(
            Counter(
                "neg" if case["gold"][0] < 4.5 else "neu" if case["gold"][0] <= 5.5 else "pos"
                for case in cases
            )
        ),
        "model": model_config["model"],
        "model_revision": model_config["revision"],
        "device": actual_device,
        "batch_size": batch_size,
        "max_new_tokens_free": MAX_NEW_TOKENS,
        "n_prompts": len(jobs),
        "n_outputs": len(previous),
        "n_grid_candidates": len(options),
        "invalid_by_decoder_condition": invalid_counts,
        "invalid_free_outputs": sum(invalid_counts["free_greedy"].values()),
        "invalid_free_rate": sum(invalid_counts["free_greedy"].values())
        / (len(cases) * len(CONDITIONS)),
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
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(
        source_dir=args.source_dir,
        output=args.output,
        manifest_path=args.manifest,
        device=args.device,
        batch_size=args.batch_size,
        preflight_only=args.preflight_only,
    ), indent=2))


if __name__ == "__main__":
    main()
