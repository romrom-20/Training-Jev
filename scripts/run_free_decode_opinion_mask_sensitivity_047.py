"""Run the free-greedy Qwen decoder sensitivity test."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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
    select_repair_cases,
)
from run_opinion_mask_crosslingual_dimabsa_041 import (
    FILES,
    LANGS,
    SOURCE_SHA256,
    read_source,
)
from run_opinion_mask_crosslingual_dimabsa_041 import (
    PROTOCOL as PROTOCOL_041,
)
from run_opinion_mask_crosslingual_dimabsa_041 import (
    make_jobs as make_jobs_041,
)
from task_ladder import MODEL_SPECS
from transformers import AutoTokenizer, set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/047-free-decode-opinion-mask-sensitivity.md")
PROTOCOL_SHA256 = "308a12a094e5d0de91228ae5ca51ade8d5d67e07e82d8dd8e76d0c18f42551f4"
PROTOCOL_043_SHA256 = "526a0de535a36deda9a2b0a4fb412967fad9720f5a39f492959ab3f71e0133a8"
PROTOCOL_041_SHA256 = "0e7d9991834a0bf0dd3d1010d0eefc4bd11b98c153ef8bbaa027af9a989f7e40"
PARENT_043_SHA256 = "1e93d359220c533a01f9cccca827f30611a713b70d857a898f59b6560303c7c8"
SOURCE_REVISION = "bdc93be1224106ae7d3eb95739c02a76ed4ae8a1"
SEED = 20260947
CONDITIONS = ("aspect_only", "opinion_masked")
MAX_NEW_TOKENS = 40
DEFAULT_BATCH = 4
OUT = Path(".context/exp047-private-predictions.jsonl")
MANIFEST = Path(".context/exp047-run-manifest.json")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_va(raw: str) -> list[float] | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in ("```", "```json"):
            return None
        if lines[-1].strip() != "```":
            return None
        text = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(
            text,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON constant: {constant}")
            ),
        )
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(value, dict) or set(value) != {"valence", "arousal"}:
        return None
    result = []
    for key in ("valence", "arousal"):
        score = value[key]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return None
        score = float(score)
        if not math.isfinite(score) or not 1 <= score <= 9:
            return None
        result.append(score)
    return result


def verify_parents() -> None:
    if sha256(PROTOCOL_043.read_bytes()) != PROTOCOL_043_SHA256:
        raise ValueError("Experiment 043 protocol changed")
    if sha256(PROTOCOL_041.read_bytes()) != PROTOCOL_041_SHA256:
        raise ValueError("Experiment 041 protocol changed")
    if sha256(OUT_043.read_bytes()) != PARENT_043_SHA256:
        raise ValueError("Experiment 043 private output changed")
    if json.loads(MANIFEST_043.read_text()).get("output_sha256") != PARENT_043_SHA256:
        raise ValueError("Experiment 043 manifest/output mismatch")


def build_jobs(source_dir: Path) -> tuple[list[dict], dict[str, str]]:
    data, hashes = {}, {}
    for lang in LANGS:
        data[lang], hashes[lang] = read_source(
            source_dir / Path(FILES[lang]).name, lang
        )
    if hashes != SOURCE_SHA256:
        raise ValueError("Pinned DimABSA test source hashes changed")
    cases = select_repair_cases(data)
    jobs = [row for row in make_jobs_041(cases) if row["condition"] in CONDITIONS]
    if len(cases) != 217 or len(jobs) != 1302:
        raise ValueError("Expected the frozen 217-ID / 1,302-prompt design")
    if any(sum(row["condition"] == condition for row in jobs) != 651 for condition in CONDITIONS):
        raise ValueError("Expected 651 prompts in each registered condition")
    expected = {(case["case_id"], lang) for case in cases for lang in LANGS}
    if {(row["case_id"], row["lang"]) for row in jobs if row["condition"] == "aspect_only"} != expected:
        raise ValueError("Aspect-only condition does not cover the frozen sample")
    keys = {(row["case_id"], row["lang"], row["condition"]) for row in jobs}
    if len(keys) != len(jobs):
        raise ValueError("Duplicate prompt in frozen sample")
    parent_gold = {
        (row["case_id"], row["lang"]): row["gold"]
        for line in OUT_043.read_text().splitlines()
        if (row := json.loads(line))["condition"] == "aspect_only"
    }
    if any(parent_gold.get((row["case_id"], row["lang"])) != row["gold"] for row in jobs):
        raise ValueError("Frozen source gold VA differs from Experiment 043")
    return jobs, hashes


def generate_batch(model, tokenizer, batch: list[dict], device: str) -> list[str]:
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in batch
    ]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    prompt_width = inputs.input_ids.shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.batch_decode(
        generated[:, prompt_width:], skip_special_tokens=True
    )


def run(
    source_dir: Path = Path(".context/dimabsa"),
    output: Path = OUT,
    manifest_path: Path = MANIFEST,
    device: str = "mps",
    batch_size: int = DEFAULT_BATCH,
    preflight_only: bool = False,
) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 047 protocol hash mismatch")
    verify_parents()
    jobs, source_hashes = build_jobs(source_dir)
    model_config = dict(MODEL_SPECS["qwen-3b"])
    if preflight_only:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["model"], revision=model_config["revision"], local_files_only=True
        )
        tokenizer.padding_side = "left"
        lengths = [
            len(tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            ))
            for row in jobs
        ]
        if max(lengths) >= model_config.get("max_length", 32768):
            raise ValueError("A frozen prompt exceeds the model context limit")
        return {
            "n_clusters": 217,
            "n_prompts": len(jobs),
            "condition_counts": {condition: sum(row["condition"] == condition for row in jobs) for condition in CONDITIONS},
            "max_prompt_tokens": max(lengths),
            "max_new_tokens": MAX_NEW_TOKENS,
            "source_hashes": source_hashes,
            "parent_043_output_sha256": PARENT_043_SHA256,
            "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        }
    previous = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            key = (row["case_id"], row["lang"], row["condition"])
            if key in previous:
                raise ValueError(f"Duplicate 047 output: {key}")
            previous[key] = row
    all_keys = {(row["case_id"], row["lang"], row["condition"]) for row in jobs}
    if not set(previous).issubset(all_keys):
        raise ValueError("Resume file contains output outside the frozen sample")
    remaining = [
        row for row in jobs
        if (row["case_id"], row["lang"], row["condition"]) not in previous
    ]
    model, tokenizer, actual_device = load_target(model_config, device, offline=True)
    tokenizer.padding_side = "left"
    set_seed(SEED)
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        for offset in range(0, len(remaining), batch_size):
            batch = remaining[offset : offset + batch_size]
            answers = generate_batch(model, tokenizer, batch, actual_device)
            for job, raw in zip(batch, answers):
                row = {
                    "case_id": job["case_id"],
                    "lang": job["lang"],
                    "condition": job["condition"],
                    "gold": job["gold"],
                    "raw": raw,
                    "prediction": parse_va(raw),
                }
                previous[(row["case_id"], row["lang"], row["condition"])] = row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                f"047 {min(offset + len(batch), len(remaining))}/{len(remaining)} new; "
                f"total {len(previous)}/{len(jobs)}",
                flush=True,
            )
    elapsed = time.monotonic() - started
    invalid = [row for row in previous.values() if row["prediction"] is None]
    manifest = {
        "experiment": "047-free-decode-opinion-mask-sensitivity",
        "protocol_sha256": sha256(PROTOCOL.read_bytes()),
        "parent_043_output_sha256": PARENT_043_SHA256,
        "source_revision": SOURCE_REVISION,
        "source_hashes": source_hashes,
        "model": model_config["model"],
        "model_revision": model_config["revision"],
        "device": actual_device,
        "batch_size": batch_size,
        "max_new_tokens": MAX_NEW_TOKENS,
        "n_prompts": len(jobs),
        "n_outputs": len(previous),
        "invalid_outputs": len(invalid),
        "invalid_by_condition": {
            condition: sum(row["prediction"] is None and row["condition"] == condition for row in previous.values())
            for condition in CONDITIONS
        },
        "generation_seconds_this_process_only": elapsed,
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
