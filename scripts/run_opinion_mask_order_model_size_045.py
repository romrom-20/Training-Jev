"""Compare natural and shuffled opinion-masked VA inputs on Qwen2.5-1.5B."""

from __future__ import annotations

import argparse
import gc
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
    build_trie,
    candidate_sequences,
    candidates,
    digest,
    generate_batch,
    parse_va,
    select_repair_cases,
    verify_boundaries,
)
from run_expanded_opinion_mask_repair_043 import (
    make_jobs as make_jobs_043,
)
from run_opinion_mask_crosslingual_dimabsa_041 import FILES, LANGS, SOURCE_SHA256, read_source
from run_opinion_mask_word_order_control_044 import (
    MANIFEST as MANIFEST_044,
)
from run_opinion_mask_word_order_control_044 import (
    OUT as OUT_044,
)
from run_opinion_mask_word_order_control_044 import (
    PROTOCOL as PROTOCOL_044,
)
from run_opinion_mask_word_order_control_044 import (
    make_jobs as make_shuffled_jobs_044,
)
from run_opinion_mask_word_order_control_044 import (
    verify_parent_outputs as verify_parent_043,
)
from task_ladder import MODEL_SPECS
from transformers import AutoTokenizer, set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/045-order-control-model-size.md")
PROTOCOL_SHA256 = "c1bdb548650833b7934a58a2c9caba9f48f091ebab593ff1f4a9017fa70d32f2"
PROTOCOL_043_SHA256 = "526a0de535a36deda9a2b0a4fb412967fad9720f5a39f492959ab3f71e0133a8"
PROTOCOL_044_SHA256 = "188b0832cc46303afd34d803ad6b599f1ac0c71d0e1289889014f59c944bc9ed"
PREDICTIONS_043_SHA256 = "1e93d359220c533a01f9cccca827f30611a713b70d857a898f59b6560303c7c8"
PREDICTIONS_044_SHA256 = "2e6e59ec798b6ef87f63e6088ee5602272396a7f3dd47c8f13f31f82b0387337"
OUT = Path(".context/exp045-private-predictions.jsonl")
MANIFEST = Path(".context/exp045-run-manifest.json")
SEED = 20260945
CONDITIONS = ("opinion_masked", "opinion_shuffled")
DEFAULT_BATCH = 4


def verify_parent_outputs() -> None:
    if digest(PROTOCOL_043.read_bytes()) != PROTOCOL_043_SHA256:
        raise ValueError("Experiment 043 protocol changed")
    if digest(PROTOCOL_044.read_bytes()) != PROTOCOL_044_SHA256:
        raise ValueError("Experiment 044 protocol changed")
    verify_parent_043()
    if digest(OUT_043.read_bytes()) != PREDICTIONS_043_SHA256:
        raise ValueError("Experiment 043 outputs changed")
    if digest(OUT_044.read_bytes()) != PREDICTIONS_044_SHA256:
        raise ValueError("Experiment 044 outputs changed")
    manifest_043 = json.loads(MANIFEST_043.read_text())
    manifest_044 = json.loads(MANIFEST_044.read_text())
    if manifest_043.get("output_sha256") != PREDICTIONS_043_SHA256:
        raise ValueError("Experiment 043 manifest/output mismatch")
    if manifest_044.get("output_sha256") != PREDICTIONS_044_SHA256:
        raise ValueError("Experiment 044 manifest/output mismatch")
    if manifest_044.get("invalid_outputs") != 0:
        raise ValueError("Experiment 044 did not pass its validity gate")


def build_jobs(data: dict[str, list[dict]]) -> list[dict]:
    cases = select_repair_cases(data)
    natural = [
        row for row in make_jobs_043(cases)
        if row["condition"] == "opinion_masked"
    ]
    shuffled = make_shuffled_jobs_044(data)
    if len(natural) != 651 or len(shuffled) != 651:
        raise ValueError("Expected complete paired natural/shuffled prompt set")
    natural_keys = {(row["case_id"], row["lang"]) for row in natural}
    shuffled_keys = {(row["case_id"], row["lang"]) for row in shuffled}
    if natural_keys != shuffled_keys:
        raise ValueError("Natural and shuffled prompts have different item/language keys")
    jobs = []
    for condition in CONDITIONS:
        source_rows = natural if condition == "opinion_masked" else shuffled
        jobs.extend(
            {
                "case_id": row["case_id"],
                "lang": row["lang"],
                "condition": condition,
                "prompt": row["prompt"],
            }
            for row in source_rows
        )
    expected = {
        (case_id, lang, condition)
        for case_id in {row["case_id"] for row in natural}
        for lang in LANGS
        for condition in CONDITIONS
    }
    if {(row["case_id"], row["lang"], row["condition"]) for row in jobs} != expected:
        raise ValueError("Natural and shuffled jobs do not form identical item/language pairs")
    return sorted(jobs, key=lambda row: (row["condition"], row["case_id"], row["lang"]))


def run(
    source_dir: Path = Path(".context/dimabsa"),
    output: Path = OUT,
    manifest_path: Path = MANIFEST,
    device: str = "mps",
    batch_size: int = DEFAULT_BATCH,
    preflight_only: bool = False,
) -> dict:
    if digest(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 045 protocol hash mismatch")
    verify_parent_outputs()
    data, source_hashes = {}, {}
    for lang in LANGS:
        data[lang], source_hashes[lang] = read_source(
            source_dir / Path(FILES[lang]).name, lang
        )
    if source_hashes != SOURCE_SHA256:
        raise ValueError("DimABSA source hashes changed")
    jobs = build_jobs(data)
    options = candidates()
    model_config = dict(MODEL_SPECS["qwen-1.5b"])

    if preflight_only:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["model"], revision=model_config["revision"], local_files_only=True
        )
        tokenizer.padding_side = "left"
        sequences = candidate_sequences(tokenizer, options)
        verify_boundaries(tokenizer, jobs, options)
        return {
            "n_clusters": 217,
            "n_new_prompts": len(jobs),
            "condition_counts": {condition: sum(row["condition"] == condition for row in jobs) for condition in CONDITIONS},
            "n_candidate_values": len(options),
            "n_unique_candidate_token_sequences": len(sequences),
            "token_boundary_check": "passed for all candidate outputs on two prompts",
            "source_hashes": source_hashes,
            "parent_043_output_sha256": PREDICTIONS_043_SHA256,
            "parent_044_output_sha256": PREDICTIONS_044_SHA256,
            "protocol_sha256": digest(PROTOCOL.read_bytes()),
        }

    previous = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            key = (row["case_id"], row["lang"], row["condition"])
            if key in previous:
                raise ValueError(f"Duplicate 045 output: {key}")
            previous[key] = row
    all_keys = {(row["case_id"], row["lang"], row["condition"]) for row in jobs}
    if not set(previous).issubset(all_keys):
        raise ValueError("Private resume file contains an output outside the frozen sample")
    remaining = [
        row for row in jobs
        if (row["case_id"], row["lang"], row["condition"]) not in previous
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
            answers = generate_batch(model, tokenizer, batch, trie, actual_device)
            for job, raw in zip(batch, answers):
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
                f"045 {min(offset + len(batch), len(remaining))}/{len(remaining)}; "
                f"total {len(previous)}/{len(jobs)}",
                flush=True,
            )
    manifest = {
        "experiment": "045-order-control-model-size",
        "protocol_sha256": digest(PROTOCOL.read_bytes()),
        "parent_043_output_sha256": PREDICTIONS_043_SHA256,
        "parent_044_output_sha256": PREDICTIONS_044_SHA256,
        "source_revision": "bdc93be1224106ae7d3eb95739c02a76ed4ae8a1",
        "source_hashes": source_hashes,
        "model": model_config["model"],
        "model_revision": model_config["revision"],
        "device": actual_device,
        "batch_size": batch_size,
        "n_prompts": len(jobs),
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
