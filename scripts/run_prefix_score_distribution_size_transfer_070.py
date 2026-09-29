"""Repeat the Exp069 score-distribution audit with Qwen2.5-1.5B."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import run_forced_coordinate_prefix_067 as exp067
import run_prefix_score_distribution_audit_069 as exp069
import run_sighan_chinese_decoder_transfer_052 as exp052
import torch
from run_small_model_decoder_factorial_048 import MAX_NEW_TOKENS, parse_free_va, sha256
from task_ladder import MODEL_SPECS
from transformers import AutoTokenizer

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/070-prefix-score-distribution-size-transfer.md")
PROTOCOL_SHA256 = "9c7604f0ef25f64875c568ccbe2966b1f3f3e0fc8f7c565276bf75de59300aeb"
OUT = Path(".context/exp070-private-score-distributions.jsonl")
MANIFEST = Path(".context/exp070-run-manifest.json")
EXPECTED_069_OUTPUT_SHA256 = "f14a610f30edfde5cfb3d3661051984b24f3d06fac9e8546162aaf94a781d94a"
EXPECTED_069_SUMMARY_SHA256 = "7cf1d182f5c9afe1430886f5af7c28231e22ed106b416f2b8bf2dbf96b7c49b0"


def run(source_dir: Path = exp052.SOURCE_DIR, output: Path = OUT,
        manifest_path: Path = MANIFEST, device: str = "mps",
        preflight_only: bool = False) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 070 protocol hash mismatch")
    if sha256(exp069.PROTOCOL.read_bytes()) != exp069.PROTOCOL_SHA256:
        raise ValueError("Experiment 069 protocol changed")
    if sha256(exp069.OUT.read_bytes()) != EXPECTED_069_OUTPUT_SHA256:
        raise ValueError("Experiment 069 private score artifact mismatch")
    summary_path = Path("results/prefix-score-distribution-audit-v1/summary.json")
    if sha256(summary_path.read_bytes()) != EXPECTED_069_SUMMARY_SHA256:
        raise ValueError("Experiment 069 aggregate summary mismatch")
    jobs_by_cohort, sample_metadata = exp069.frozen_jobs(source_dir)
    config = dict(MODEL_SPECS["qwen-1.5b"])
    if config["revision"] != "989aa7980e4cf806f80c7fef2b1adb7bc71aa306":
        raise ValueError("Unexpected 1.5B model revision")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True
    )
    values, _ = exp069.candidate_token_ids(tokenizer)
    digit_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 10)]
    decimal_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)]
    dot_id = tokenizer.encode(".", add_special_tokens=False)[0]
    contexts = []
    for cohort, jobs in jobs_by_cohort.items():
        for job in jobs:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": job["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            ) + job["assistant_prefix"]
            contexts.append({
                "cohort": cohort, "case_id": job["case_id"], "order": job["order"],
                "forced_first_value": float(job["forced_first_value"]),
                "decoder": job["decoder"], "input_ids": tokenizer.encode(
                    rendered, add_special_tokens=False),
                "rendered_prefix": rendered, "prompt_sha256": hashlib.sha256(
                    rendered.encode("utf-8")).hexdigest(),
                "job": job,
            })
    if len(contexts) != 512:
        raise ValueError(f"Expected 512 fixed contexts; got {len(contexts)}")
    if preflight_only:
        return {
            "experiment": "070-prefix-score-distribution-size-transfer",
            "protocol_sha256": PROTOCOL_SHA256,
            "model": config["model"], "model_revision": config["revision"],
            "source_hashes": sample_metadata["source_hashes"],
            "n_contexts": len(contexts), "n_greedy_generations": len(contexts),
            "n_recipients_per_cohort": 64, "score_grid_size": len(values),
            "prefix_tokens_range": [min(len(c["input_ids"]) for c in contexts),
                                     max(len(c["input_ids"]) for c in contexts)],
            "parent_069_output_sha256": EXPECTED_069_OUTPUT_SHA256,
            "parent_069_summary_sha256": EXPECTED_069_SUMMARY_SHA256,
        }

    model, tokenizer, actual_device = load_target(config, device, offline=True)
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            key = (row["cohort"], row["case_id"], row["order"], row["forced_first_value"])
            if key in previous:
                raise ValueError(f"Duplicate resumed Experiment 070 context: {key}")
            previous[key] = row
    expected = {(c["cohort"], c["case_id"], c["order"], c["forced_first_value"])
                for c in contexts}
    if not set(previous).issubset(expected):
        raise ValueError("Resume artifact contains contexts outside the frozen design")
    resumed_contexts = len(previous)
    with output.open("a", encoding="utf-8") as stream:
        for index, context in enumerate(contexts, start=1):
            key = (context["cohort"], context["case_id"], context["order"],
                   context["forced_first_value"])
            if key in previous:
                continue
            logprobs = exp069.score_grid(
                model, context["input_ids"], digit_ids, dot_id, decimal_ids, actual_device
            )
            probabilities = exp069.normalize_score_logprobs(logprobs)
            raw = exp067.generate_prefix_batch(model, tokenizer, [context["job"]], actual_device)[0]
            prediction = parse_free_va(context["job"]["assistant_prefix"] + raw)
            axis = 1 if context["order"] == "valence_first" else 0
            first_axis = 0 if context["order"] == "valence_first" else 1
            if prediction is not None and float(prediction[first_axis]) != context["forced_first_value"]:
                raise ValueError(f"Forced coordinate changed in greedy output: {key}")
            row = {k: context[k] for k in (
                "cohort", "case_id", "order", "forced_first_value", "decoder", "prompt_sha256")
            }
            row.update({
                "score_logprobs": logprobs.tolist(),
                "restricted_probabilities": probabilities.tolist(),
                "greedy_raw_continuation": raw,
                "greedy_prediction": prediction,
                "greedy_second_score": None if prediction is None else float(prediction[axis]),
            })
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            previous[key] = row
            if index % 8 == 0 or index == len(contexts):
                print(f"070 scored/generated {index}/{len(contexts)} contexts", flush=True)
    if len(previous) != len(contexts):
        raise ValueError("Experiment 070 raw output is incomplete")
    invalid = {}
    for cohort in ("067", "068"):
        invalid[cohort] = {}
        for order in exp067.ORDERS:
            invalid[cohort][order] = {}
            for forced in exp067.FORCED_VALUES:
                arm = [r for r in previous.values() if r["cohort"] == cohort
                       and r["order"] == order and r["forced_first_value"] == forced]
                invalid[cohort][order][str(forced)] = sum(
                    row["greedy_prediction"] is None for row in arm)
    manifest = {
        "experiment": "070-prefix-score-distribution-size-transfer",
        "protocol_sha256": PROTOCOL_SHA256,
        "parent_069_output_sha256": EXPECTED_069_OUTPUT_SHA256,
        "parent_069_summary_sha256": EXPECTED_069_SUMMARY_SHA256,
        "source_hashes": sample_metadata["source_hashes"],
        "model": config["model"], "model_revision": config["revision"],
        "device": actual_device, "n_contexts": len(previous),
        "n_greedy_generations": len(previous), "score_grid_size": len(values),
        "invalid_by_cohort_order_forced_value": invalid,
        "max_new_tokens_greedy": MAX_NEW_TOKENS,
        "elapsed_seconds_this_process_only": time.monotonic() - started,
        "resumed_contexts": resumed_contexts,
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
    parser.add_argument("--preflight-only", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
