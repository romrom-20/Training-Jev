"""Run Experiment 077's explicit target-name role-control follow-up."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import run_qwen05_numeric_support_074 as exp074
import run_schema_corpus_controls_076 as exp076
import torch
from run_small_model_decoder_factorial_048 import sha256
from transformers import AutoTokenizer, set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/077-explicit-target-role-control.md")
PROTOCOL_SHA256 = "8f69de2851960ec46de89e01fdc601c97104f376fcd1dabe3a4652f6573991e2"
OUT = Path(".context/exp077-private-explicit-target-role.jsonl")
MANIFEST = Path(".context/exp077-run-manifest.json")
SEED = 20260977
MODEL_KEY = "qwen-0.5b"
MODEL_CONFIG = dict(exp076.MODEL_CONFIGS[MODEL_KEY])


def build_jobs() -> list[dict]:
    role_cases, _ = exp076.select_role_cases()
    aste_cases, _ = exp076.select_aste_cases()
    jobs = [job for job in exp076.build_jobs(role_cases, aste_cases)
            if job["study"] == "role_control"]
    for job in jobs:
        job["study"] = "explicit_named_role"
        job["messages"][-1]["content"] = (
            f"The earlier {job['anchor_axis']} coordinate is fixed and must stay as supplied. "
            f"Estimate only the {job['target_axis']} coordinate. Return exactly one JSON object "
            f'with exactly one numeric key "{job["target_axis"]}" and a number from 1.0 to 9.0 '
            "in increments of 0.1, with exactly one decimal place."
        )
    if len(jobs) != 192:
        raise ValueError(f"Experiment 077 must contain 192 contexts, found {len(jobs)}")
    return jobs


def run(output: Path = OUT, manifest_path: Path = MANIFEST,
        device: str = "mps", preflight_only: bool = False) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 077 protocol hash mismatch")
    jobs = build_jobs()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIG["model"], revision=MODEL_CONFIG["revision"], local_files_only=True
    )
    rendered = [exp076.render(tokenizer, job) for job in jobs]
    if any(not ids for _text, ids in rendered):
        raise ValueError("Empty rendered Experiment 077 input")
    for text, ids in rendered:
        exp074.verify_token_boundary(tokenizer, text, ids)
    meta = {
        "experiment": "077-explicit-target-role-control",
        "protocol_sha256": PROTOCOL_SHA256,
        "n_recipients": 24, "n_contexts": len(jobs), "model_key": MODEL_KEY,
        "model": MODEL_CONFIG,
        "recipient_ids_sha256": exp076.DIMABSA_ID_HASH,
        "candidate_path_validation": "all 171 numeric-plus-brace paths pass for every context",
        "prefix_length_range": [min(len(ids) for _text, ids in rendered),
                                max(len(ids) for _text, ids in rendered)],
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
            text, input_ids = exp076.render(tokenizer, job)
            forms = exp074.score_with_tokenizer(model, input_ids, actual_device, tokenizer)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=actual_device)
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_tensor, attention_mask=torch.ones_like(input_tensor),
                    max_new_tokens=40, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )
            raw = tokenizer.decode(generated[0, input_tensor.shape[1]:], skip_special_tokens=True)
            row = {
                "model_key": MODEL_KEY, "model": MODEL_CONFIG["model"],
                "model_revision": MODEL_CONFIG["revision"], "experiment": job["study"],
                "case_id": job["case_id"], "order": job["order"],
                "anchor_axis": job["anchor_axis"], "target_axis": job["target_axis"],
                "anchor_location": job["anchor_location"],
                "forced_value": job["forced_value"],
                **forms, "greedy_raw_continuation": raw,
                "greedy_target_score": exp076.parse_target(job, raw),
                "rendered_prefix_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
            key = (job["order"], job["anchor_location"], float(job["forced_value"]))
            invalid[key] = invalid.get(key, 0) + (row["greedy_target_score"] is None)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 8 == 0 or index == len(jobs):
                print(f"077 {index}/{len(jobs)}", flush=True)
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
