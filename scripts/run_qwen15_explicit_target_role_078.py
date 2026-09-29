"""Run Experiment 078's 1.5B named-target role-control replication."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import run_explicit_target_role_control_077 as exp077
import run_qwen05_numeric_support_074 as exp074
import run_schema_corpus_controls_076 as exp076
import torch
from run_small_model_decoder_factorial_048 import sha256
from transformers import AutoTokenizer, set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/078-qwen15-explicit-target-role-replication.md")
PROTOCOL_SHA256 = "6050bf1e0fca5c431a3f23143fe3aa1292e11621ac0069ed42511893545c8f25"
OUT = Path(".context/exp078-private-qwen15-role-control.jsonl")
MANIFEST = Path(".context/exp078-run-manifest.json")
SEED = 20260978
MODEL_KEY = "qwen-1.5b"
MODEL_CONFIG = dict(exp076.MODEL_CONFIGS[MODEL_KEY])


def build_jobs() -> list[dict]:
    jobs = exp077.build_jobs()
    if len(jobs) != 192:
        raise ValueError("Experiment 078 must mirror the 192-cell Experiment 077 design")
    return jobs


def run(output: Path = OUT, manifest_path: Path = MANIFEST,
        device: str = "mps", preflight_only: bool = False) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 078 protocol hash mismatch")
    jobs = build_jobs()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG["model"], revision=MODEL_CONFIG["revision"], local_files_only=True
    )
    for job in jobs:
        text, ids = exp076.render(tokenizer, job)
        if not ids:
            raise ValueError("Empty rendered Experiment 078 prompt")
        exp074.verify_token_boundary(tokenizer, text, ids)
    meta = {
        "experiment": "078-qwen15-explicit-target-role-replication",
        "protocol_sha256": PROTOCOL_SHA256, "n_recipients": 24,
        "n_contexts": len(jobs), "model_key": MODEL_KEY,
        "model": MODEL_CONFIG, "recipient_ids_sha256": exp076.DIMABSA_ID_HASH,
        "candidate_path_validation": "all 171 numeric-plus-brace paths pass for every context",
    }
    if preflight_only:
        return meta
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    model, tokenizer, actual_device = load_target(MODEL_CONFIG, device, offline=True)
    set_seed(SEED)
    output.parent.mkdir(parents=True, exist_ok=True)
    invalid = {}
    started = time.monotonic()
    with output.open("w", encoding="utf-8") as stream:
        for index, job in enumerate(jobs, start=1):
            rendered_text, input_ids = exp076.render(tokenizer, job)
            forms = exp074.score_with_tokenizer(model, input_ids, actual_device, tokenizer)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=actual_device)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_tensor, attention_mask=torch.ones_like(input_tensor),
                    max_new_tokens=40, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )
            raw = tokenizer.decode(generated[0, input_tensor.shape[1]:], skip_special_tokens=True)
            parsed = exp076.parse_target(job, raw)
            key = (job["order"], job["anchor_location"], float(job["forced_value"]))
            invalid[key] = invalid.get(key, 0) + (parsed is None)
            row = {
                "model_key": MODEL_KEY, "model": MODEL_CONFIG["model"],
                "model_revision": MODEL_CONFIG["revision"], "experiment": job["study"],
                "case_id": job["case_id"], "order": job["order"],
                "anchor_axis": job["anchor_axis"], "target_axis": job["target_axis"],
                "anchor_location": job["anchor_location"],
                "forced_value": job["forced_value"], **forms,
                "greedy_raw_continuation": raw, "greedy_target_score": parsed,
                "rendered_prefix_sha256": hashlib.sha256(rendered_text.encode()).hexdigest(),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 8 == 0 or index == len(jobs):
                print(f"078 {index}/{len(jobs)}", flush=True)
    result = {
        **meta, "device": device,
        "invalid_by_order_location_anchor": {
            f"{order}/{location}/{value:.1f}": {"invalid": count, "n": 24,
                                                            "rate": count / 24}
            for (order, location, value), count in invalid.items()
        },
        "elapsed_seconds_this_process_only": time.monotonic() - started,
        "output_sha256": sha256(output.read_bytes()),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    del model, tokenizer
    gc.collect()
    if actual_device == "mps":
        torch.mps.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--manifest", dest="manifest_path", type=Path, default=MANIFEST)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--preflight-only", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
