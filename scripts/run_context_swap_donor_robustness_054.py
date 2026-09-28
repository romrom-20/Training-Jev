"""Run three frozen donor permutations for the 053 review-context diagnostic."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import run_counterfactual_context_swap_053 as parent
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

PROTOCOL = Path("docs/experiments/054-context-swap-donor-robustness.md")
PROTOCOL_SHA256 = "e095fdb8292766f865d87e56cb69c59f414e9342b37ba5c0fc62a4072e9470a0"
PARENT_OUTPUT_SHA256 = "890a16d6363e72549c7ccad9099dece3ce021ab78ab3d8071acde18774500f5f"
PERMUTATIONS = (1, 2, 3)
DECODERS = ("finite_grid", "free_greedy")
CONDITION = "swapped_context"
OUT = Path(".context/exp054-private-predictions.jsonl")
MANIFEST = Path(".context/exp054-run-manifest.json")


def make_donor_map(cases: list[dict], tokenizer, permutation: int) -> tuple[dict, dict]:
    lengths = {}
    for case in cases:
        masked = parent.mask_opinions(case["row"])
        lengths[case["case_id"]] = len(tokenizer.encode(masked, add_special_tokens=False))
    ranked = sorted(cases, key=lambda case: (lengths[case["case_id"]], case["case_id"]))
    groups = [ranked[(len(ranked) * index) // 10 : (len(ranked) * (index + 1)) // 10]
              for index in range(10)]
    donor_map = {}
    for group in groups:
        shuffled = sorted(
            group,
            key=lambda case: parent.stable_hash(
                f"exp054-p{permutation}|{case['case_id']}"
            ),
        )
        if len(shuffled) < 2:
            raise ValueError("Each length bin must contain at least two cases")
        for index, case in enumerate(shuffled):
            donor_map[case["case_id"]] = shuffled[(index + 1) % len(shuffled)]["case_id"]
    ids = {case["case_id"] for case in cases}
    if set(donor_map) != ids or set(donor_map.values()) != ids:
        raise ValueError("Each permutation must assign every donor exactly once")
    if any(case_id == donor_map[case_id] for case_id in ids):
        raise ValueError("Donor mapping must be a derangement")
    return donor_map, lengths


def build_jobs(cases: list[dict], mappings: dict[int, dict[str, str]]) -> list[dict]:
    by_id = {case["case_id"]: case for case in cases}
    jobs = []
    for permutation, mapping in mappings.items():
        for case in cases:
            donor = by_id[mapping[case["case_id"]]]
            target = case["row"]["Triplet"][case["target_index"]]
            text = parent.mask_opinions(donor["row"])
            prompt = prompt_with_registered_grid(text, target["Aspect"])
            for decoder in DECODERS:
                jobs.append({
                    "case_id": case["case_id"],
                    "condition": CONDITION,
                    "permutation": permutation,
                    "decoder": decoder,
                    "gold": case["gold"],
                    "donor_id": donor["case_id"],
                    "prompt": prompt,
                })
    if len(jobs) != 1302:
        raise ValueError("Expected 1,302 swapped-context judgments")
    prompts = {}
    for job in jobs:
        key = (job["case_id"], job["permutation"])
        if key in prompts and prompts[key] != job["prompt"]:
            raise ValueError("Prompt bytes must match across decoder arms")
        prompts[key] = job["prompt"]
    return sorted(jobs, key=lambda row: (row["permutation"], row["decoder"], row["case_id"]))


def run(
    source_dir: Path = Path(".context/dimabsa"),
    output: Path = OUT,
    manifest_path: Path = MANIFEST,
    device: str = "mps",
    batch_size: int = DEFAULT_BATCH,
    preflight_only: bool = False,
) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 054 protocol hash mismatch")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    rows, source_hash = parent.read_source(source_dir)
    cases = parent.choose_sample(parent.select_cases(rows))
    if sha256(parent.PARENT_OUTPUT.read_bytes()) != PARENT_OUTPUT_SHA256:
        raise ValueError("Experiment 050 parent output hash mismatch")
    parent.verify_parent(cases)
    model_config = dict(MODEL_SPECS["qwen-3b"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model"], revision=model_config["revision"], local_files_only=True
    )
    tokenizer.padding_side = "left"
    mappings = {}
    lengths = {}
    for permutation in PERMUTATIONS:
        mappings[permutation], lengths[permutation] = make_donor_map(cases, tokenizer, permutation)
    jobs = build_jobs(cases, mappings)
    if preflight_only:
        options = candidates()
        sequences = candidate_sequences(tokenizer, options)
        prompt_lengths = [
            len(tokenizer.apply_chat_template(
                [{"role": "user", "content": job["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            ))
            for job in jobs
        ]
        return {
            "experiment": "054-context-swap-donor-robustness",
            "protocol_sha256": sha256(PROTOCOL.read_bytes()),
            "source_revision": SOURCE_REVISION,
            "source_sha256": source_hash,
            "parent_output_sha256": PARENT_OUTPUT_SHA256,
            "sample_n": len(cases),
            "sample_buckets": dict(Counter(parent.valence_bucket(case) for case in cases)),
            "n_permutations": len(mappings),
            "donor_assignment_sha256": {
                permutation: sha256(json.dumps(mapping, sort_keys=True).encode())
                for permutation, mapping in mappings.items()
            },
            "all_mappings_are_derangements": True,
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
                raise ValueError(f"Duplicate 054 output: {key}")
            previous[key] = item
    all_keys = {
        (job["case_id"], job["permutation"], job["condition"], job["decoder"])
        for job in jobs
    }
    if not set(previous).issubset(all_keys):
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
    set_seed(20260954)
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
                item = {
                    key: job[key]
                    for key in ("case_id", "condition", "permutation", "decoder", "gold", "donor_id")
                }
                item.update({"raw": raw, "prediction": prediction})
                key = (item["case_id"], item["permutation"], item["condition"], decoder)
                previous[key] = item
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            processed += len(batch)
            print(f"054 {processed}/{len(remaining)} new; total {len(previous)}/{len(jobs)}", flush=True)
    manifest = {
        "experiment": "054-context-swap-donor-robustness",
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
        "n_permutations": len(PERMUTATIONS),
        "donor_assignment_sha256": {
            permutation: sha256(json.dumps(mapping, sort_keys=True).encode())
            for permutation, mapping in mappings.items()
        },
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
