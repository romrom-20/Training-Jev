"""Add a deterministic same-token word-order control to Experiment 043."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from run_expanded_opinion_mask_repair_043 import (
    MANIFEST as MANIFEST_043,
)
from run_expanded_opinion_mask_repair_043 import (
    OUT as OUT_043,
)
from run_expanded_opinion_mask_repair_043 import (
    PROTOCOL as PROTOCOL_043,
)
from run_expanded_opinion_mask_repair_043 import (
    SOURCE_REVISION,
    build_trie,
    candidate_sequences,
    candidates,
    digest,
    generate_batch,
    parse_va,
    select_repair_cases,
    verify_boundaries,
)
from run_opinion_mask_crosslingual_dimabsa_041 import (
    FILES,
    LANGS,
    SOURCE_SHA256,
    build_prompt,
    mask_opinions,
    read_source,
)
from task_ladder import MODEL_SPECS
from transformers import AutoTokenizer, set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/044-opinion-mask-word-order-control.md")
PROTOCOL_SHA256 = "188b0832cc46303afd34d803ad6b599f1ac0c71d0e1289889014f59c944bc9ed"
PROTOCOL_043_SHA256 = "526a0de535a36deda9a2b0a4fb412967fad9720f5a39f492959ab3f71e0133a8"
PREDICTIONS_043_SHA256 = "1e93d359220c533a01f9cccca827f30611a713b70d857a898f59b6560303c7c8"
OUT = Path(".context/exp044-private-predictions.jsonl")
MANIFEST = Path(".context/exp044-run-manifest.json")
SEED = 20260944
CONDITION = "opinion_shuffled"
DEFAULT_BATCH = 4


def shuffle_tokens(text: str, case_id: str, lang: str) -> str:
    tokens = text.split()
    ranked = sorted(
        enumerate(tokens),
        key=lambda item: hashlib.sha256(
            f"exp044|{SEED}|{case_id}|{lang}|{item[0]}".encode()
        ).hexdigest(),
    )
    shuffled = [token for _, token in ranked]
    if len(shuffled) > 1 and shuffled == tokens:
        shuffled = shuffled[1:] + shuffled[:1]
    result = " ".join(shuffled)
    if sorted(result.split()) != sorted(tokens):
        raise ValueError("Word-order control changed the masked sentence token multiset")
    return result


def make_jobs(data: dict[str, list[dict]]) -> list[dict]:
    cases = select_repair_cases(data)
    jobs = []
    old_instruction = (
        'Return exactly one JSON object with numeric keys "valence" and "arousal", both from 1 to 9.'
    )
    new_instruction = (
        'Return exactly one JSON object with numeric keys "valence" and "arousal", '
        "both from 1.0 to 9.0 in increments of 0.1, with exactly one decimal place."
    )
    for case in cases:
        for lang in LANGS:
            row = case["rows"][lang]
            target = row["Triplet"][case["target_index"]]
            masked = mask_opinions(row)
            shuffled = shuffle_tokens(masked, case["case_id"], lang)
            prompt = build_prompt(shuffled, target["Aspect"], None)
            if old_instruction not in prompt:
                raise ValueError("Experiment 041 prompt output instruction changed")
            prompt = prompt.replace(old_instruction, new_instruction, 1)
            jobs.append(
                {
                    "case_id": case["case_id"],
                    "lang": lang,
                    "condition": CONDITION,
                    "prompt": prompt,
                }
            )
    if len(jobs) != 651:
        raise ValueError(f"Expected 651 shuffled-context prompts, found {len(jobs)}")
    return jobs


def verify_parent_outputs() -> dict:
    if digest(PROTOCOL_043.read_bytes()) != PROTOCOL_043_SHA256:
        raise ValueError("Experiment 043 protocol changed")
    raw = OUT_043.read_bytes()
    if digest(raw) != PREDICTIONS_043_SHA256:
        raise ValueError("Experiment 043 private outputs changed")
    parent_manifest = json.loads(MANIFEST_043.read_text())
    if parent_manifest.get("output_sha256") != PREDICTIONS_043_SHA256:
        raise ValueError("Experiment 043 run manifest does not match its private outputs")
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    if len(rows) != 1302 or sum(row["prediction"] is None for row in rows) != 0:
        raise ValueError("Experiment 043 parent must have 1,302 parseable outputs")
    if {row["condition"] for row in rows} != {"aspect_only", "opinion_masked"}:
        raise ValueError("Experiment 043 parent condition set changed")
    return parent_manifest


def run(
    source_dir: Path = Path(".context/dimabsa"),
    output: Path = OUT,
    manifest_path: Path = MANIFEST,
    device: str = "mps",
    batch_size: int = DEFAULT_BATCH,
    preflight_only: bool = False,
) -> dict:
    if digest(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 044 protocol hash mismatch")
    parent_manifest = verify_parent_outputs()
    data, source_hashes = {}, {}
    for lang in LANGS:
        data[lang], source_hashes[lang] = read_source(
            source_dir / Path(FILES[lang]).name, lang
        )
    if source_hashes != SOURCE_SHA256:
        raise ValueError("DimABSA source hashes changed")
    jobs = make_jobs(data)
    old_rows = [json.loads(line) for line in OUT_043.read_text().splitlines()]
    old_ids = {row["case_id"] for row in old_rows}
    if len(old_ids) != 217 or {job["case_id"] for job in jobs} != old_ids:
        raise ValueError("044 job IDs do not match the frozen 043 sample")
    options = candidates()
    model_config = dict(MODEL_SPECS["qwen-3b"])

    if preflight_only:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["model"], revision=model_config["revision"], local_files_only=True
        )
        tokenizer.padding_side = "left"
        sequences = candidate_sequences(tokenizer, options)
        verify_boundaries(tokenizer, jobs, options)
        return {
            "n_reused_clusters": len(old_ids),
            "n_new_prompts": len(jobs),
            "n_candidate_values": len(options),
            "n_unique_candidate_token_sequences": len(sequences),
            "token_boundary_check": "passed for all candidate outputs on two prompts",
            "source_hashes": source_hashes,
            "parent_043_output_sha256": parent_manifest["output_sha256"],
            "protocol_sha256": digest(PROTOCOL.read_bytes()),
        }

    previous = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            key = (row["case_id"], row["lang"], row["condition"])
            if key in previous:
                raise ValueError(f"Duplicate 044 output: {key}")
            previous[key] = row
    keys = {(job["case_id"], job["lang"], job["condition"]) for job in jobs}
    if not set(previous).issubset(keys):
        raise ValueError("Private resume file contains an output outside the frozen sample")
    remaining = [
        job for job in jobs
        if (job["case_id"], job["lang"], job["condition"]) not in previous
    ]
    model, tokenizer, actual_device = load_target(model_config, device, offline=True)
    tokenizer.padding_side = "left"
    sequences = candidate_sequences(tokenizer, options)
    trie = build_trie(sequences)
    verify_boundaries(tokenizer, jobs, options)
    set_seed(SEED)
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        for offset in range(0, len(remaining), batch_size):
            batch = remaining[offset : offset + batch_size]
            raw_answers = generate_batch(model, tokenizer, batch, trie, actual_device)
            for job, raw in zip(batch, raw_answers):
                row = {
                    "case_id": job["case_id"],
                    "lang": job["lang"],
                    "condition": job["condition"],
                    "raw": raw,
                    "prediction": parse_va(raw),
                }
                key = (row["case_id"], row["lang"], row["condition"])
                previous[key] = row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                f"044 shuffled {min(offset + len(batch), len(remaining))}/{len(remaining)}; "
                f"total {len(previous)}/{len(jobs)}",
                flush=True,
            )
    manifest = {
        "experiment": "044-opinion-mask-word-order-control",
        "protocol_sha256": digest(PROTOCOL.read_bytes()),
        "parent_043_protocol_sha256": PROTOCOL_043_SHA256,
        "parent_043_output_sha256": parent_manifest["output_sha256"],
        "source_revision": SOURCE_REVISION,
        "source_hashes": source_hashes,
        "model": model_config["model"],
        "model_revision": model_config["revision"],
        "device": actual_device,
        "batch_size": batch_size,
        "n_new_prompts": len(jobs),
        "n_outputs": len(previous),
        "n_grid_candidates": len(options),
        "invalid_outputs": sum(row["prediction"] is None for row in previous.values()),
        "generation_seconds_this_process_only": time.monotonic() - started,
        "resumed_rows": len(jobs) - len(remaining),
        "output_sha256": digest(output.read_bytes()),
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
