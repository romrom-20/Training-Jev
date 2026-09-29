"""Measure the frozen prefix-coupling design on Qwen2.5-0.5B."""

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
import run_prefix_score_distribution_audit_069 as exp069
import run_small_model_decoder_factorial_048 as exp048
import torch
from run_small_model_decoder_factorial_048 import parse_free_va, sha256
from transformers import AutoTokenizer, set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/073-qwen05-prefix-size-continuation.md")
PROTOCOL_SHA256 = "44047797e01339858173aecfe3f8a4a9dd0f4d29083223bd83e2eafb3aa4d70e"
QWEN05_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN05_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SYSTEM_PROMPT = "You are an expert affective computing annotator. Follow the user's instructions exactly."
FORCED_VALUES = (2.0, 8.0)
ORDERS = ("valence_first", "arousal_first")
MODEL_CONFIGS = {
    "qwen-0.5b": {"model": QWEN05_MODEL, "revision": QWEN05_REVISION},
}
OUT = Path(".context/exp073-private-qwen05-prefix.jsonl")
MANIFEST = Path(".context/exp073-run-manifest.json")
SEED = 20260973


def select_cases(source_dir: Path) -> tuple[list[dict], dict]:
    cases, metadata = exp071.select_cases(source_dir)
    if metadata["recipient_ids_sha256"] != "a14b6300c88940b3ccb6ac6b1b78c78b1cb5e202c4ce945ea3fe8220d36c7ddf":
        raise ValueError("Experiment 073 recipient IDs differ from the frozen protocol")
    return cases, metadata


def build_jobs(cases: list[dict]) -> list[dict]:
    jobs = []
    for case in cases:
        target = case["row"]["Triplet"][case["target_index"]]
        visible_review = exp049.mask_opinions(case["row"])
        prompt = exp049.prompt_with_registered_grid(visible_review, target["Aspect"])
        user_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for order in ORDERS:
            for forced in FORCED_VALUES:
                prefix = exp067.assistant_prefix(order, forced)
                jobs.append({
                    "case_id": case["case_id"], "order": order,
                    "forced_first_value": float(forced), "decoder": "free_greedy",
                    "gold": case["gold"], "prompt": prompt,
                    "user_prompt_sha256": user_hash, "assistant_prefix": prefix,
                })
    expected = {(case["case_id"], order, forced)
                for case in cases for order in ORDERS for forced in FORCED_VALUES}
    actual = {(job["case_id"], job["order"], job["forced_first_value"]) for job in jobs}
    if len(jobs) != 256 or actual != expected:
        raise ValueError("Experiment 073 must include every two-order/two-prefix cell")
    return sorted(jobs, key=lambda row: (row["order"], row["forced_first_value"], row["case_id"]))


def rendered_input(tokenizer, job: dict) -> tuple[str, list[int]]:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": job["prompt"]}],
        tokenize=False, add_generation_prompt=True,
    ) + job["assistant_prefix"]
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError("Empty rendered prompt/prefix")
    return text, ids


def run(source_dir: Path = Path(".context/dimabsa"), output: Path = OUT,
        manifest_path: Path = MANIFEST, device: str = "mps",
        preflight_only: bool = False) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 073 protocol hash mismatch")
    cases, sample_metadata = select_cases(source_dir)
    jobs = build_jobs(cases)
    tokenizer_metadata = {}
    for model_key, config in MODEL_CONFIGS.items():
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"], revision=config["revision"], local_files_only=True
        )
        values, candidate_ids = exp069.candidate_token_ids(tokenizer)
        rendered = [rendered_input(tokenizer, job) for job in jobs]
        if any(not text.endswith(exp067.assistant_prefix(job["order"], job["forced_first_value"]))
               for text, job in zip((item[0] for item in rendered), jobs, strict=True)):
            raise ValueError(f"Assistant prefix is not at the generation boundary: {model_key}")
        tokenizer_metadata[model_key] = {
            "n_candidates": len(values), "candidate_token_ids_sha256": sha256(
                json.dumps(candidate_ids, separators=(",", ":")).encode("utf-8")),
            "prefix_length_range": [min(len(item[1]) for item in rendered),
                                    max(len(item[1]) for item in rendered)],
            "system_prompt": SYSTEM_PROMPT,
        }
    if sample_metadata["recipient_ids_sha256"] != "a14b6300c88940b3ccb6ac6b1b78c78b1cb5e202c4ce945ea3fe8220d36c7ddf":
        raise ValueError("Experiment 073 recipient IDs differ from the frozen protocol")
    if preflight_only:
        return {
            "experiment": "073-qwen05-prefix-size-continuation",
            "protocol_sha256": PROTOCOL_SHA256,
            **sample_metadata, "n_contexts_per_model": len(jobs),
            "n_models": len(MODEL_CONFIGS), "total_model_contexts": len(jobs) * len(MODEL_CONFIGS),
            "tokenizer_metadata": tokenizer_metadata,
            "models": {key: {k: v for k, v in config.items() if k in ("model", "revision")}
                       for key, config in MODEL_CONFIGS.items()},
        }

    previous = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            key = (row["model_key"], row["case_id"], row["order"], row["forced_first_value"])
            if key in previous:
                raise ValueError(f"Duplicate resumed Experiment 073 cell: {key}")
            previous[key] = row
    expected = {(model_key, job["case_id"], job["order"], job["forced_first_value"])
                for model_key in MODEL_CONFIGS for job in jobs}
    if not set(previous).issubset(expected):
        raise ValueError("Resume artifact contains cells outside the frozen design")
    resumed = len(previous)
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    invalid = {}
    for model_key, config in MODEL_CONFIGS.items():
        remaining = [job for job in jobs if (
            model_key, job["case_id"], job["order"], job["forced_first_value"]
        ) not in previous]
        if not remaining:
            continue
        model, tokenizer, actual_device = load_target(config, device, offline=True)
        values, _ = exp069.candidate_token_ids(tokenizer)
        digit_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 10)]
        decimal_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)]
        dot_id = tokenizer.encode(".", add_special_tokens=False)[0]
        set_seed(SEED)
        with output.open("a", encoding="utf-8") as stream:
            for i, job in enumerate(remaining, start=1):
                text, input_ids = rendered_input(tokenizer, job)
                logprobs = exp069.score_grid(
                    model, input_ids, digit_ids, dot_id, decimal_ids, actual_device
                )
                probabilities = exp069.normalize_score_logprobs(logprobs)
                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=actual_device)
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=input_tensor,
                        attention_mask=torch.ones_like(input_tensor),
                        max_new_tokens=exp048.MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                raw = tokenizer.decode(generated[0, input_tensor.shape[1]:],
                                       skip_special_tokens=True)
                prediction = parse_free_va(job["assistant_prefix"] + raw)
                first_axis = 0 if job["order"] == "valence_first" else 1
                second_axis = 1 if job["order"] == "valence_first" else 0
                if prediction is not None and float(prediction[first_axis]) != job["forced_first_value"]:
                    raise ValueError("Generated output changed the forced first coordinate")
                row = {
                    "model_key": model_key, "model": config["model"],
                    "model_revision": config["revision"], "case_id": job["case_id"],
                    "order": job["order"], "forced_first_value": job["forced_first_value"],
                    "gold": job["gold"], "assistant_prefix": job["assistant_prefix"],
                    "user_prompt_sha256": job["user_prompt_sha256"],
                    "rendered_prefix_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "score_logprobs": logprobs.tolist(),
                    "restricted_probabilities": probabilities.tolist(),
                    "greedy_raw_continuation": raw, "greedy_prediction": prediction,
                    "greedy_second_score": None if prediction is None else float(prediction[second_axis]),
                }
                key = (model_key, job["case_id"], job["order"], job["forced_first_value"])
                previous[key] = row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if i % 8 == 0 or i == len(remaining):
                    print(f"073 {model_key} {i}/{len(remaining)} new", flush=True)
        del model, tokenizer
        gc.collect()
        if actual_device == "mps":
            torch.mps.empty_cache()
    if len(previous) != len(expected):
        raise ValueError("Experiment 073 output is incomplete")
    for model_key in MODEL_CONFIGS:
        invalid[model_key] = {
            order: {str(value): sum(
                previous[(model_key, case["case_id"], order, value)]["greedy_prediction"] is None
                for case in cases
            ) for value in FORCED_VALUES}
            for order in ORDERS
        }
    manifest = {
        "experiment": "073-qwen05-prefix-size-continuation",
        "protocol_sha256": PROTOCOL_SHA256,
        **sample_metadata,
        "models": {key: {k: v for k, v in config.items() if k in ("model", "revision")}
                   for key, config in MODEL_CONFIGS.items()},
        "device": device, "n_contexts_per_model": len(jobs),
        "n_total_model_contexts": len(previous), "n_score_values": 81,
        "invalid_by_model_order_forced_value": invalid,
        "tokenizer_metadata": tokenizer_metadata,
        "elapsed_seconds_this_process_only": time.monotonic() - started,
        "resumed_contexts": resumed,
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
    parser.add_argument("--preflight-only", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
