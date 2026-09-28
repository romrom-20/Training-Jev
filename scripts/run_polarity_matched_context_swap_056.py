"""Run three polarity-matched donor permutations on the fresh laptop sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
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

PROTOCOL = Path("docs/experiments/056-polarity-matched-context-swap.md")
PROTOCOL_SHA256 = "384a958d7ca15aea8b9989cab5cc3b976628a6334630b66bf397983ffccc6272"
PARENT_OUTPUT_SHA256 = "890a16d6363e72549c7ccad9099dece3ce021ab78ab3d8071acde18774500f5f"
PERMUTATIONS = (1, 2, 3)
DECODERS = ("finite_grid", "free_greedy")
OUT = Path(".context/exp056-private-predictions.jsonl")
MANIFEST = Path(".context/exp056-run-manifest.json")


def donor_maps(cases: list[dict], tokenizer) -> tuple[dict[int, dict[str, str]], dict[str, int]]:
    lengths = {
        case["case_id"]: len(tokenizer.encode(
            exp053.mask_opinions(case["row"]), add_special_tokens=False
        ))
        for case in cases
    }
    by_polarity = {
        polarity: sorted(
            [case for case in cases if exp055.bucket(case) == polarity],
            key=lambda case: (lengths[case["case_id"]], case["case_id"]),
        )
        for polarity in ("neg", "pos")
    }
    groups = {
        polarity: [
            cases_in_bucket[(len(cases_in_bucket) * index) // 10:
                            (len(cases_in_bucket) * (index + 1)) // 10]
            for index in range(10)
        ]
        for polarity, cases_in_bucket in by_polarity.items()
    }
    mappings = {}
    by_id = {case["case_id"]: case for case in cases}
    for permutation in PERMUTATIONS:
        mapping = {}
        for polarity_groups in groups.values():
            for group in polarity_groups:
                ordered = sorted(
                    group,
                    key=lambda case: hashlib.sha256(
                        f"exp056-p{permutation}|{case['case_id']}".encode("utf-8")
                    ).hexdigest(),
                )
                if len(ordered) < 2:
                    raise ValueError("Each polarity-and-length bin must contain two or more cases")
                for index, case in enumerate(ordered):
                    mapping[case["case_id"]] = ordered[(index + 1) % len(ordered)]["case_id"]
        ids = {case["case_id"] for case in cases}
        if set(mapping) != ids or set(mapping.values()) != ids:
            raise ValueError("Each mapping must be a permutation of all fresh IDs")
        if any(case_id == mapping[case_id] for case_id in ids):
            raise ValueError("Each mapping must be a derangement")
        if any(exp055.bucket(by_id[case_id]) != exp055.bucket(by_id[donor_id])
               for case_id, donor_id in mapping.items()):
            raise ValueError("Every donor must match recipient gold-valence polarity")
        mappings[permutation] = mapping
    return mappings, lengths


def build_jobs(cases: list[dict], mappings: dict[int, dict[str, str]]) -> list[dict]:
    by_id = {case["case_id"]: case for case in cases}
    jobs = []
    for permutation, mapping in mappings.items():
        for case in cases:
            donor = by_id[mapping[case["case_id"]]]
            target = case["row"]["Triplet"][case["target_index"]]
            prompt = prompt_with_registered_grid(
                exp053.mask_opinions(donor["row"]), target["Aspect"]
            )
            for decoder in DECODERS:
                jobs.append({
                    "case_id": case["case_id"], "permutation": permutation,
                    "condition": "polarity_matched_context", "decoder": decoder,
                    "gold": case["gold"], "donor_id": donor["case_id"],
                    "donor_polarity": exp055.bucket(donor), "prompt": prompt,
                })
    if len(jobs) != 1302:
        raise ValueError("Expected 1,302 generation jobs")
    prompt_by_case_perm = {}
    for job in jobs:
        key = (job["case_id"], job["permutation"])
        if key in prompt_by_case_perm and prompt_by_case_perm[key] != job["prompt"]:
            raise ValueError("Paired decoder prompts differ")
        prompt_by_case_perm[key] = job["prompt"]
    return sorted(jobs, key=lambda row: (row["permutation"], row["decoder"], row["case_id"]))


def run(source_dir: Path = Path(".context/dimabsa"), output: Path = OUT,
        manifest_path: Path = MANIFEST, device: str = "mps",
        batch_size: int = DEFAULT_BATCH, preflight_only: bool = False) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 056 protocol hash mismatch")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    rows, source_hash = exp053.read_source(source_dir)
    if source_hash != exp053.SOURCE_SHA256:
        raise ValueError("Pinned source hash mismatch")
    cases = exp055.fresh_sample(exp053.select_cases(rows))
    if sha256(exp053.PARENT_OUTPUT.read_bytes()) != PARENT_OUTPUT_SHA256:
        raise ValueError("Experiment 050 parent output hash mismatch")
    exp053.verify_parent(cases)
    model_config = dict(MODEL_SPECS["qwen-3b"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model"], revision=model_config["revision"], local_files_only=True
    )
    tokenizer.padding_side = "left"
    mappings, lengths = donor_maps(cases, tokenizer)
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
            "experiment": "056-polarity-matched-context-swap",
            "protocol_sha256": sha256(PROTOCOL.read_bytes()),
            "source_revision": SOURCE_REVISION,
            "source_sha256": source_hash,
            "parent_output_sha256": PARENT_OUTPUT_SHA256,
            "sample_n": len(cases),
            "sample_buckets": dict(Counter(exp055.bucket(case) for case in cases)),
            "sample_disjoint_from_053_054": True,
            "n_permutations": len(mappings),
            "all_mappings_are_polarity_matched_derangements": True,
            "donor_assignment_sha256": {
                p: sha256(json.dumps(m, sort_keys=True).encode()) for p, m in mappings.items()
            },
            "donor_token_length_range": [min(lengths.values()), max(lengths.values())],
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
                raise ValueError(f"Duplicate output {key}")
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
    set_seed(20260956)
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
                        ("case_id", "permutation", "condition", "decoder", "gold", "donor_id", "donor_polarity")}
                item.update({"raw": raw, "prediction": prediction})
                key = (item["case_id"], item["permutation"], item["condition"], decoder)
                previous[key] = item
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            processed += len(batch)
            print(f"056 {processed}/{len(remaining)} new; total {len(previous)}/{len(jobs)}", flush=True)
    manifest = {
        "experiment": "056-polarity-matched-context-swap",
        "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        "source_revision": SOURCE_REVISION,
        "source_sha256": source_hash,
        "parent_output_sha256": PARENT_OUTPUT_SHA256,
        "model": model_config["model"], "model_revision": model_config["revision"],
        "device": actual_device, "batch_size": batch_size,
        "max_new_tokens_free": MAX_NEW_TOKENS,
        "n_sample_ids": len(cases), "sample_buckets": dict(Counter(exp055.bucket(case) for case in cases)),
        "n_permutations": len(mappings),
        "donor_assignment_sha256": {
            p: sha256(json.dumps(m, sort_keys=True).encode()) for p, m in mappings.items()
        },
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
