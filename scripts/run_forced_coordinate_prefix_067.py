"""Run the forced-first-coordinate continuation test on SIGHAN reviews."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import run_counterfactual_context_swap_053 as exp053
import run_sighan_chinese_decoder_transfer_052 as exp052
import run_sighan_instruction_language_066 as exp066
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

PROTOCOL = Path("docs/experiments/067-forced-coordinate-prefix-coupling.md")
PROTOCOL_SHA256 = "5b9fa3cf3649b1c935dd4f7ed8b20f1c122bef2a7ca2060903a7928b7645f932"
PARENT = Path(".context/exp066-private-predictions.jsonl")
PARENT_SHA256 = "47f47317e1e609a5106d3024ebfad978c28c910ca545719a93dcdadfeb71041f"
PARENT_MANIFEST = Path(".context/exp066-run-manifest.json")
EXPECTED_IDS_SHA256 = "ddd1584d1f5151b4e29cf5de1dfbe5bfc6657f3c90599d494f919193b884a621"
OUT = Path(".context/exp067-private-predictions.jsonl")
MANIFEST = Path(".context/exp067-run-manifest.json")
ORDERS = exp066.ORDERS
FORCED_VALUES = (2.0, 8.0)
DECODER = "free_greedy"
SAMPLE_QUOTAS = {
    "neg": {
        "食物#品质": 16,
        "食物#份量与款式": 8,
        "食物#价格": 5,
        "饮料#品质": 3,
    },
    "pos": {"食物#品质": 16, "食物#份量与款式": 16},
}


def category(case: dict) -> str:
    return case["row"]["Triplet"][case["target_index"]]["Category"]


def select_cases(all_cases: list[dict]) -> tuple[list[dict], dict]:
    selected = []
    stats = {}
    for pol, quotas in SAMPLE_QUOTAS.items():
        for cat, quota in quotas.items():
            group = [case for case in all_cases
                     if exp066.polarity(case) == pol and category(case) == cat]
            if len(group) < quota:
                raise ValueError(f"Insufficient {pol}/{cat}: {len(group)} < {quota}")
            group.sort(key=lambda case: hashlib.sha256(
                f"exp067-forced-coordinate-prefix-v1|{case['case_id']}".encode("utf-8")
            ).hexdigest())
            selected.extend(group[:quota])
            stats[f"{pol}|{cat}"] = quota
    selected.sort(key=lambda case: case["case_id"])
    ids = [case["case_id"] for case in selected]
    if len(ids) != 64 or len(set(ids)) != 64:
        raise ValueError("Frozen Exp067 sample must contain 64 unique IDs")
    polarity_counts = dict(Counter(exp066.polarity(case) for case in selected))
    if polarity_counts != {"neg": 32, "pos": 32}:
        raise ValueError(f"Unexpected Exp067 polarity counts: {polarity_counts}")
    return selected, {
        "n_parent_066_ids": len(all_cases),
        "n_recipient_ids": len(selected),
        "polarity_counts": polarity_counts,
        "category_polarity_counts": stats,
        "recipient_ids_sha256": sha256("\n".join(ids).encode("utf-8")),
    }


def assistant_prefix(order: str, forced_value: float) -> str:
    if order not in ORDERS or forced_value not in FORCED_VALUES:
        raise ValueError(f"Unexpected prefix arm: {order}/{forced_value}")
    value = f"{forced_value:.1f}"
    if order == "valence_first":
        return f'{{"valence": {value}, "arousal": '
    return f'{{"arousal": {value}, "valence": '


def jobs_for(cases: list[dict]) -> list[dict]:
    jobs = []
    for case in cases:
        aspect = case["row"]["Triplet"][case["target_index"]]["Aspect"]
        masked = exp053.mask_opinions(case["row"])
        for order in ORDERS:
            prompt = exp066.chinese_prompt(masked, aspect, order)
            for forced in FORCED_VALUES:
                jobs.append({
                    "case_id": case["case_id"], "order": order,
                    "forced_first_value": forced, "decoder": DECODER,
                    "gold": case["gold"], "prompt": prompt,
                    "assistant_prefix": assistant_prefix(order, forced),
                })
    expected = {
        (case["case_id"], order, forced, DECODER)
        for case in cases for order in ORDERS for forced in FORCED_VALUES
    }
    keys = {(job["case_id"], job["order"], job["forced_first_value"], job["decoder"])
            for job in jobs}
    if len(jobs) != len(cases) * 4 or keys != expected:
        raise ValueError("Exp067 cells do not form the complete two-order/two-value grid")
    return sorted(jobs, key=lambda row: (
        row["order"], row["forced_first_value"], row["case_id"]
    ))


def validate_parent(path: Path) -> dict[tuple, dict]:
    if sha256(path.read_bytes()) != PARENT_SHA256:
        raise ValueError("Experiment 066 parent-output hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [(row["case_id"], row["order"], row["permutation"],
             row["condition"], row["decoder"]) for row in rows]
    if len(rows) != 2464 or len(set(keys)) != 2464:
        raise ValueError("Experiment 066 parent output is not the complete frozen artifact")
    return dict(zip(keys, rows))


def generate_prefix_batch(model, tokenizer, batch: list[dict], device: str) -> list[str]:
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": job["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        ) + job["assistant_prefix"]
        for job in batch
    ]
    tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    prompt_width = tokens.input_ids.shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **tokens,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)


def run(source_dir: Path = exp052.SOURCE_DIR, output: Path = OUT,
        manifest_path: Path = MANIFEST, device: str = "mps",
        batch_size: int = DEFAULT_BATCH, preflight_only: bool = False) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 067 protocol hash mismatch")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    parent = validate_parent(PARENT)
    source_rows, source_hashes = exp052.read_source(source_dir)
    base_cases, _ = exp064.select_cases(exp052.select_cases(source_rows))
    cases_88, _ = exp066.select_cases(base_cases)
    cases, sample_stats = select_cases(cases_88)
    if sample_stats["recipient_ids_sha256"] != EXPECTED_IDS_SHA256:
        raise ValueError("Experiment 067 IDs differ from the frozen protocol")
    if json.loads(PARENT_MANIFEST.read_text()).get("output_sha256") != PARENT_SHA256:
        raise ValueError("Experiment 066 manifest does not match its frozen parent output")
    ids = {case["case_id"] for case in cases}
    if not ids.issubset({key[0] for key in parent}):
        raise ValueError("Exp067 recipients are not present in Experiment 066")
    config = dict(MODEL_SPECS["qwen-3b"])
    if config["revision"] != "aa8e72537993ba99e69dfaafa59ed015b17504d1":
        raise ValueError("Unexpected target-model revision")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True
    )
    tokenizer.padding_side = "left"
    jobs = jobs_for(cases)
    if len(jobs) != 256:
        raise ValueError("Frozen Experiment 067 must contain 256 output cells")
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
        lens = [len(tokenizer.encode(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": job["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            ) + job["assistant_prefix"], add_special_tokens=False,
        )) for job in jobs]
        return {
            "experiment": "067-forced-coordinate-prefix-coupling",
            "protocol_sha256": sha256(PROTOCOL.read_bytes()),
            "parent_output_sha256": PARENT_SHA256,
            "source_hashes": source_hashes,
            **sample_stats,
            "n_jobs": len(jobs),
            "forced_first_values": FORCED_VALUES,
            "assistant_prefix_token_length_range": [min(lens), max(lens)],
            "model": config["model"], "model_revision": config["revision"],
        }

    previous = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            key = (row["case_id"], row["order"], row["forced_first_value"], row["decoder"])
            if key in previous:
                raise ValueError(f"Duplicate resumed Exp067 output: {key}")
            previous[key] = row
    allowed = {(job["case_id"], job["order"], job["forced_first_value"], job["decoder"])
               for job in jobs}
    if not set(previous).issubset(allowed):
        raise ValueError("Resume file contains rows outside frozen Experiment 067")
    remaining = [job for job in jobs if (
        job["case_id"], job["order"], job["forced_first_value"], job["decoder"]
    ) not in previous]
    del tokenizer
    model, tokenizer, actual_device = load_target(config, device, offline=True)
    tokenizer.padding_side = "left"
    set_seed(20260967)
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
            raw_outputs = generate_prefix_batch(model, tokenizer, batch, actual_device)
            for job, raw in zip(batch, raw_outputs):
                full_output = job["assistant_prefix"] + raw
                prediction = parse_free_va(full_output)
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
            print(f"067 {processed}/{len(remaining)} new; total {len(previous)}/{len(jobs)}", flush=True)
    invalid = {
        order: {
            str(forced): sum(
                row["order"] == order and row["forced_first_value"] == forced
                and row["prediction"] is None for row in previous.values()
            )
            for forced in FORCED_VALUES
        }
        for order in ORDERS
    }
    manifest = {
        "experiment": "067-forced-coordinate-prefix-coupling",
        "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        "parent_output_sha256": PARENT_SHA256,
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
