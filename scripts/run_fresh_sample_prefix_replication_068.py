"""Replicate forced-coordinate prefix coupling on fresh SIGHAN recipients."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import run_forced_coordinate_prefix_067 as exp067
import run_sighan_chinese_decoder_transfer_052 as exp052
import run_sighan_output_key_order_064 as exp064
import torch
from run_small_model_decoder_factorial_048 import (
    DEFAULT_BATCH,
    MAX_NEW_TOKENS,
    parse_free_va,
    sha256,
)
from task_ladder import MODEL_SPECS
from transformers import AutoTokenizer, set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/068-fresh-sample-prefix-coupling-replication.md")
PROTOCOL_SHA256 = "1f3e8ed96343c86ba9fa43d34c8c864d5588efd9f5c2f49a8fd5874ee1ea4d0f"
PARENT_064 = Path(".context/exp064-private-predictions.jsonl")
PARENT_064_MANIFEST = Path(".context/exp064-run-manifest.json")
PARENT_064_SHA256 = "48f7a8e2fe9d245d7742c7955b6349fe08a3935934f7ff0cd1db01e1eec9ace7"
OUT = Path(".context/exp068-private-predictions.jsonl")
MANIFEST = Path(".context/exp068-run-manifest.json")
ORDERS = exp067.ORDERS
FORCED_VALUES = exp067.FORCED_VALUES
DECODER = "free_greedy"
SAMPLE_QUOTAS = {"neg": {"食物#品质": 32}, "pos": {"食物#品质": 32}}
EXPECTED_IDS_SHA256 = "1e738712d6ec47bb90a5bdd47d676f3b79fcb958b9c0d945d3081428924132bd"


def select_cases(all_cases: list[dict], excluded_ids: set[str]) -> tuple[list[dict], dict]:
    selected = []
    stats = {}
    for pol, quotas in SAMPLE_QUOTAS.items():
        for cat, quota in quotas.items():
            group = [case for case in all_cases
                     if case["case_id"] not in excluded_ids
                     and exp067.exp066.polarity(case) == pol
                     and exp064.category(case) == cat]
            if len(group) < quota:
                raise ValueError(f"Insufficient fresh {pol}/{cat}: {len(group)} < {quota}")
            group.sort(key=lambda case: hashlib.sha256(
                f"exp068-fresh-prefix-replication-v1|{case['case_id']}".encode("utf-8")
            ).hexdigest())
            selected.extend(group[:quota])
            stats[f"{pol}|{cat}"] = quota
    selected.sort(key=lambda case: case["case_id"])
    ids = [case["case_id"] for case in selected]
    if len(ids) != 64 or len(set(ids)) != 64:
        raise ValueError("Fresh frozen Exp068 sample must contain 64 unique IDs")
    polarity_counts = dict(Counter(exp067.exp066.polarity(case) for case in selected))
    if polarity_counts != {"neg": 32, "pos": 32}:
        raise ValueError(f"Unexpected Exp068 polarity counts: {polarity_counts}")
    return selected, {
        "n_eligible_source_cases": len(all_cases),
        "n_exp064_ids_excluded": len(excluded_ids),
        "n_recipient_ids": len(selected),
        "polarity_counts": polarity_counts,
        "category_polarity_counts": stats,
        "recipient_ids_sha256": sha256("\n".join(ids).encode("utf-8")),
        "disjoint_from_exp064": not bool(set(ids) & excluded_ids),
    }


def jobs_for(cases: list[dict]) -> list[dict]:
    jobs = exp067.jobs_for(cases)
    if len(jobs) != len(cases) * 4:
        raise ValueError("Exp068 cells do not form the complete two-order/two-value grid")
    return jobs


def verify_parents() -> None:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 068 protocol hash mismatch")
    if sha256(exp067.PROTOCOL.read_bytes()) != exp067.PROTOCOL_SHA256:
        raise ValueError("Experiment 067 prefix protocol changed")
    if sha256(PARENT_064.read_bytes()) != PARENT_064_SHA256:
        raise ValueError("Experiment 064 frozen exclusion-cohort artifact mismatch")
    if json.loads(PARENT_064_MANIFEST.read_text()).get("output_sha256") != PARENT_064_SHA256:
        raise ValueError("Experiment 064 manifest does not match its frozen artifact")


def run(source_dir: Path = exp052.SOURCE_DIR, output: Path = OUT,
        manifest_path: Path = MANIFEST, device: str = "mps",
        batch_size: int = DEFAULT_BATCH, preflight_only: bool = False) -> dict:
    verify_parents()
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    source_rows, source_hashes = exp052.read_source(source_dir)
    all_cases = exp052.select_cases(source_rows)
    old_cohort, _ = exp064.select_cases(all_cases)
    excluded_ids = {case["case_id"] for case in old_cohort}
    cases, sample_stats = select_cases(all_cases, excluded_ids)
    if sample_stats["recipient_ids_sha256"] != EXPECTED_IDS_SHA256:
        raise ValueError("Experiment 068 IDs differ from the frozen protocol")
    if not sample_stats["disjoint_from_exp064"]:
        raise ValueError("Experiment 068 recipients overlap the Exp064 cohort")
    config = dict(MODEL_SPECS["qwen-3b"])
    if config["revision"] != "aa8e72537993ba99e69dfaafa59ed015b17504d1":
        raise ValueError("Unexpected target-model revision")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True
    )
    tokenizer.padding_side = "left"
    jobs = jobs_for(cases)
    for job in jobs:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": job["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        if not rendered.endswith("<|im_start|>assistant\n"):
            raise ValueError("Unexpected Qwen assistant-generation boundary")
        if not tokenizer.encode(rendered + job["assistant_prefix"], add_special_tokens=False):
            raise ValueError("Empty assistant-prefix prompt")
    if preflight_only:
        lengths = [len(tokenizer.encode(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": job["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            ) + job["assistant_prefix"], add_special_tokens=False,
        )) for job in jobs]
        return {
            "experiment": "068-fresh-sample-prefix-coupling-replication",
            "protocol_sha256": sha256(PROTOCOL.read_bytes()),
            "exp064_output_sha256": PARENT_064_SHA256,
            "source_hashes": source_hashes,
            **sample_stats,
            "n_jobs": len(jobs), "forced_first_values": FORCED_VALUES,
            "assistant_prefix_token_length_range": [min(lengths), max(lengths)],
            "model": config["model"], "model_revision": config["revision"],
        }

    previous = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            key = (row["case_id"], row["order"], row["forced_first_value"], row["decoder"])
            if key in previous:
                raise ValueError(f"Duplicate resumed Exp068 output: {key}")
            previous[key] = row
    allowed = {(job["case_id"], job["order"], job["forced_first_value"], job["decoder"])
               for job in jobs}
    if not set(previous).issubset(allowed):
        raise ValueError("Resume file contains rows outside frozen Experiment 068")
    remaining = [job for job in jobs if (
        job["case_id"], job["order"], job["forced_first_value"], job["decoder"]
    ) not in previous]
    del tokenizer
    model, tokenizer, actual_device = load_target(config, device, offline=True)
    tokenizer.padding_side = "left"
    set_seed(20260968)
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        offset = 0
        processed = 0
        while offset < len(remaining):
            order = remaining[offset]["order"]
            forced = remaining[offset]["forced_first_value"]
            end = offset
            while end < len(remaining) and end < offset + batch_size and (
                remaining[end]["order"], remaining[end]["forced_first_value"]
            ) == (order, forced):
                end += 1
            batch = remaining[offset:end]
            raw_outputs = exp067.generate_prefix_batch(model, tokenizer, batch, actual_device)
            for job, raw in zip(batch, raw_outputs):
                prediction = parse_free_va(job["assistant_prefix"] + raw)
                row = {key: job[key] for key in (
                    "case_id", "order", "forced_first_value", "decoder", "gold",
                    "assistant_prefix",
                )}
                row.update({"raw_continuation": raw, "prediction": prediction})
                key = (row["case_id"], row["order"], row["forced_first_value"], row["decoder"])
                previous[key] = row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            processed += len(batch)
            offset = end
            print(f"068 {processed}/{len(remaining)} new; total {len(previous)}/{len(jobs)}", flush=True)
    invalid = {
        order: {
            str(forced): sum(
                row["order"] == order and row["forced_first_value"] == forced
                and row["prediction"] is None for row in previous.values()
            ) for forced in FORCED_VALUES
        } for order in ORDERS
    }
    manifest = {
        "experiment": "068-fresh-sample-prefix-coupling-replication",
        "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        "exp064_output_sha256": PARENT_064_SHA256,
        "source_hashes": source_hashes,
        "model": config["model"], "model_revision": config["revision"],
        "device": actual_device, "batch_size": batch_size,
        "max_new_tokens_free": MAX_NEW_TOKENS,
        **sample_stats,
        "n_outputs": len(previous), "invalid_by_order_forced_value": invalid,
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
    parser.add_argument("--source-dir", type=Path, default=exp052.SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--manifest", dest="manifest_path", type=Path, default=MANIFEST)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--preflight-only", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
