"""Compare assistant-prefix and user-stated numeric anchors."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import run_cross_family_prefix_coupling_071 as exp071
import run_forced_coordinate_prefix_067 as exp067
import run_laptop_decoder_transfer_049 as exp049
import run_qwen05_numeric_support_074 as exp074
import torch
from run_small_model_decoder_factorial_048 import parse_free_va, sha256
from transformers import AutoTokenizer, set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/075-anchor-location-control.md")
PROTOCOL_SHA256 = "92435a3a0a9ba9907ac07599ebe04fd81ead89db634441d2f6435f1b2cf1f8ae"
OUT = Path(".context/exp075-private-anchor-location.jsonl")
MANIFEST = Path(".context/exp075-run-manifest.json")
SEED = 20260975
MODEL_CONFIGS = {
    "qwen-0.5b": {"model": "Qwen/Qwen2.5-0.5B-Instruct",
                  "revision": "7ae557604adf67be50417f59c2c2f167def9a775"},
    "qwen-1.5b": {"model": "Qwen/Qwen2.5-1.5B-Instruct",
                  "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"},
}
SYSTEM_PROMPT = exp071.SYSTEM_PROMPT
ORDERS = ("valence_first", "arousal_first")
FORCED_VALUES = (2.0, 8.0)
PRECISE_INSTRUCTION = (
    'Return exactly one JSON object with numeric keys "valence" and "arousal", '
    "both from 1.0 to 9.0 in increments of 0.1, with exactly one decimal place."
)


def select_cases(source_dir: Path) -> tuple[list[dict], dict]:
    cases, metadata = exp071.select_cases(source_dir)
    if metadata["recipient_ids_sha256"] != "a14b6300c88940b3ccb6ac6b1b78c78b1cb5e202c4ce945ea3fe8220d36c7ddf":
        raise ValueError("Experiment 075 recipient IDs differ from the frozen sample")
    return cases, metadata


def build_jobs(cases: list[dict]) -> list[dict]:
    jobs = []
    for case in cases:
        target = case["row"]["Triplet"][case["target_index"]]
        visible_review = exp049.mask_opinions(case["row"])
        base_prompt = exp049.prompt_with_registered_grid(visible_review, target["Aspect"])
        if PRECISE_INSTRUCTION not in base_prompt:
            raise ValueError("Experiment 049 numeric instruction changed")
        for order in ORDERS:
            anchor_axis = "valence" if order == "valence_first" else "arousal"
            target_axis = "arousal" if order == "valence_first" else "valence"
            for forced in FORCED_VALUES:
                assistant_prefix = exp067.assistant_prefix(order, forced)
                jobs.append({
                    "case_id": case["case_id"], "anchor_location": "assistant_prefix",
                    "order": order, "anchor_axis": anchor_axis, "target_axis": target_axis,
                    "forced_value": forced, "gold": case["gold"], "prompt": base_prompt,
                    "assistant_prefix": assistant_prefix,
                })
                user_instruction = (
                    f"The {anchor_axis} score is provided and fixed at {forced:.1f}. "
                    f"Estimate only the {target_axis} score, using this given {anchor_axis} "
                    "alongside the supplied review evidence. Return exactly one JSON object "
                    f'with the numeric key "{target_axis}" and a value from 1.0 to 9.0 '
                    "in increments of 0.1, with exactly one decimal place."
                )
                user_prompt = base_prompt.replace(PRECISE_INSTRUCTION, user_instruction, 1)
                jobs.append({
                    "case_id": case["case_id"], "anchor_location": "user_message",
                    "order": order, "anchor_axis": anchor_axis, "target_axis": target_axis,
                    "forced_value": forced, "gold": case["gold"], "prompt": user_prompt,
                    "assistant_prefix": f'{{"{target_axis}": ',
                })
    if len(jobs) != 512 or len({(j["case_id"], j["anchor_location"], j["order"],
                                j["forced_value"]) for j in jobs}) != 512:
        raise ValueError("Experiment 075 must contain 64 × 2 × 2 × 2 cells")
    for job in jobs:
        job["user_prompt_sha256"] = hashlib.sha256(job["prompt"].encode()).hexdigest()
    return sorted(jobs, key=lambda j: (j["anchor_location"], j["order"],
                                       j["forced_value"], j["case_id"]))


def rendered_input(tokenizer, job: dict) -> tuple[str, list[int]]:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": job["prompt"]}],
        tokenize=False, add_generation_prompt=True,
    ) + job["assistant_prefix"]
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text, ids


def parse_target(job: dict, raw: str) -> tuple[float | None, list[float] | None]:
    if job["anchor_location"] == "assistant_prefix":
        pair = parse_free_va(job["assistant_prefix"] + raw)
        if pair is None:
            return None, None
        first_axis = 0 if job["anchor_axis"] == "valence" else 1
        if pair[first_axis] != job["forced_value"]:
            raise ValueError("Assistant continuation changed its forced coordinate")
        target_index = 0 if job["target_axis"] == "valence" else 1
        return float(pair[target_index]), [float(x) for x in pair]
    try:
        value = json.loads(job["assistant_prefix"] + raw)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(value, dict) or set(value) != {job["target_axis"]}:
        return None, None
    score = value[job["target_axis"]]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 1 <= score <= 9:
        return None, None
    return float(score), None


def run(source_dir: Path = Path(".context/dimabsa"), output: Path = OUT,
        manifest_path: Path = MANIFEST, device: str = "mps", preflight_only: bool = False) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 075 protocol hash mismatch")
    cases, metadata = select_cases(source_dir)
    jobs = build_jobs(cases)
    token_metadata = {}
    for model_key, config in MODEL_CONFIGS.items():
        tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"],
                                                  local_files_only=True)
        meta = {}
        for location in ("assistant_prefix", "user_message"):
            candidates = []
            for job in [j for j in jobs if j["anchor_location"] == location]:
                text, ids = rendered_input(tokenizer, job)
                forms = exp074.verify_token_boundary(tokenizer, text, ids)
                candidates.append(forms)
            if not candidates or any({key: len(value) for key, value in forms.items()}
                                     != {"canonical": 81, "extended": 81, "integer": 9}
                                     for forms in candidates):
                raise ValueError(f"Numeric candidate boundary changed for {location}/{model_key}")
            meta[location] = {
                "n_contexts": len(candidates),
                "token_boundary": "passed for 171 numeric+brace paths per context",
            }
        token_metadata[model_key] = meta
    if preflight_only:
        return {
            "experiment": "075-anchor-location-control", "protocol_sha256": PROTOCOL_SHA256,
            **metadata, "n_contexts_per_model": len(jobs), "n_models": len(MODEL_CONFIGS),
            "n_total_model_contexts": len(jobs) * len(MODEL_CONFIGS),
            "tokenizer_metadata": token_metadata,
        }

    old = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            key = (row["model_key"], row["case_id"], row["anchor_location"], row["order"],
                   float(row["forced_value"]))
            if key in old:
                raise ValueError(f"Duplicate resumed Experiment 075 cell: {key}")
            old[key] = row
    expected = {(model, job["case_id"], job["anchor_location"], job["order"], job["forced_value"])
                for model in MODEL_CONFIGS for job in jobs}
    if not set(old).issubset(expected):
        raise ValueError("Resume artifact contains cells outside the frozen design")
    resumed = len(old)
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    invalid = {}
    for model_key, config in MODEL_CONFIGS.items():
        remaining = [job for job in jobs if (model_key, job["case_id"], job["anchor_location"],
                                              job["order"], job["forced_value"]) not in old]
        if not remaining:
            continue
        model, tokenizer, actual_device = load_target(config, device, offline=True)
        set_seed(SEED)
        with output.open("a", encoding="utf-8") as stream:
            for i, job in enumerate(remaining, start=1):
                text, input_ids = rendered_input(tokenizer, job)
                forms = exp074.score_with_tokenizer(model, input_ids, actual_device, tokenizer)
                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=actual_device)
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=input_tensor, attention_mask=torch.ones_like(input_tensor),
                        max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                raw = tokenizer.decode(generated[0, input_tensor.shape[1]:], skip_special_tokens=True)
                greedy, pair = parse_target(job, raw)
                row = {
                    "model_key": model_key, "model": config["model"],
                    "model_revision": config["revision"], "case_id": job["case_id"],
                    "anchor_location": job["anchor_location"], "order": job["order"],
                    "anchor_axis": job["anchor_axis"], "target_axis": job["target_axis"],
                    "forced_value": job["forced_value"], "gold": job["gold"],
                    "assistant_prefix": job["assistant_prefix"],
                    "user_prompt_sha256": job["user_prompt_sha256"],
                    **forms, "greedy_raw_continuation": raw,
                    "greedy_target_score": greedy, "greedy_prediction_pair": pair,
                    "rendered_prefix_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
                key = (model_key, job["case_id"], job["anchor_location"], job["order"],
                       job["forced_value"])
                old[key] = row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if i % 8 == 0 or i == len(remaining):
                    print(f"075 {model_key} {i}/{len(remaining)} new", flush=True)
        model_invalid = {}
        for location in ("assistant_prefix", "user_message"):
            model_invalid[location] = {}
            for order in ORDERS:
                model_invalid[location][order] = {
                    str(value): sum(
                        old[(model_key, job["case_id"], location, order, value)]
                        ["greedy_target_score"] is None
                        for job in jobs if job["anchor_location"] == location
                        and job["order"] == order and job["forced_value"] == value
                    ) for value in FORCED_VALUES
                }
        invalid[model_key] = model_invalid
        del model, tokenizer
        gc.collect()
        if actual_device == "mps":
            torch.mps.empty_cache()
    if len(old) != len(expected):
        raise ValueError("Experiment 075 output is incomplete")
    manifest = {
        "experiment": "075-anchor-location-control", "protocol_sha256": PROTOCOL_SHA256,
        **metadata, "models": MODEL_CONFIGS, "device": device,
        "n_contexts_per_model": len(jobs), "n_total_model_contexts": len(old),
        "invalid_by_model_location_order_value": invalid,
        "tokenizer_metadata": token_metadata,
        "elapsed_seconds_this_process_only": time.monotonic() - started,
        "resumed_contexts": resumed, "output_sha256": sha256(output.read_bytes()),
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
    parser.add_argument("--preflight-only", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
