"""Score valid numeric continuations after the frozen Exp067/068 prefixes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import run_forced_coordinate_prefix_067 as exp067
import run_fresh_sample_prefix_replication_068 as exp068
import run_sighan_chinese_decoder_transfer_052 as exp052
import run_sighan_instruction_language_066 as exp066
import run_sighan_output_key_order_064 as exp064
import torch
import torch.nn.functional as F
from run_small_model_decoder_factorial_048 import sha256
from task_ladder import MODEL_SPECS
from transformers import AutoTokenizer

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/069-prefix-score-distribution-audit.md")
PROTOCOL_SHA256 = "8002336bd83ecb39b8dabf819c74452c7e88c7623b237aa68c8b36f6b536bac1"
PREDICTIONS = {
    "067": Path(".context/exp067-private-predictions.jsonl"),
    "068": Path(".context/exp068-private-predictions.jsonl"),
}
MANIFESTS = {
    "067": Path(".context/exp067-run-manifest.json"),
    "068": Path(".context/exp068-run-manifest.json"),
}
EXPECTED_OUTPUT_HASHES = {
    "067": "0cfaffa2b5c0da553ddb952f6f969d43d966fc404cce4aed35aefe04f2fc0bf6",
    "068": "873f1ff5b63b5643adb94e480f1e1ee2272d013aea8c991116f4bea26488a2e7",
}
OUT = Path(".context/exp069-private-score-distributions.jsonl")
MANIFEST = Path(".context/exp069-run-manifest.json")
ORDERS = exp067.ORDERS
FORCED_VALUES = exp067.FORCED_VALUES
VALUES = np.arange(10, 91, dtype=np.float64) / 10.0
BOOTSTRAP_SEED = 20260969


def candidate_token_ids(tokenizer) -> tuple[list[float], list[list[int]]]:
    values = [float(f"{v:.1f}") for v in VALUES]
    encodings = [tokenizer.encode(f"{v:.1f}", add_special_tokens=False) for v in values]
    if any(len(ids) != 3 for ids in encodings):
        raise ValueError("Every canonical 1.0–9.0 score must use exactly three tokens")
    digits = {str(i): tokenizer.encode(str(i), add_special_tokens=False) for i in range(10)}
    dot = tokenizer.encode(".", add_special_tokens=False)
    if len(dot) != 1 or any(len(ids) != 1 for ids in digits.values()):
        raise ValueError("Expected one-token digits and decimal point")
    expected = [
        [digits[str(int(v))][0], dot[0], digits[f"{int(round((v % 1) * 10))}"][0]]
        for v in values
    ]
    if encodings != expected:
        raise ValueError("Numeric candidate tokenization differs from the frozen digit grammar")
    return values, encodings


def softmax_logprobs(logits: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.float(), dim=-1)


def normalize_score_logprobs(logprobs: np.ndarray) -> np.ndarray:
    values = np.asarray(logprobs, dtype=np.float64)
    if values.shape != (len(VALUES),) or not np.isfinite(values).all():
        raise ValueError(f"Expected {len(VALUES)} finite score-grid log probabilities")
    probabilities = np.exp(values - np.max(values))
    probabilities /= probabilities.sum()
    return probabilities


def score_grid(model, input_ids: list[int], digit_ids: list[int], dot_id: int,
               decimal_ids: list[int], device: str) -> np.ndarray:
    """Return log P(number string | prefix), before normalization over the grid."""
    prefix = torch.tensor([input_ids], dtype=torch.long, device=device)
    prefix_mask = torch.ones_like(prefix)
    with torch.inference_mode():
        prefill = model.model(input_ids=prefix, attention_mask=prefix_mask, use_cache=True)
        first_logits = model.lm_head(prefill.last_hidden_state[:, -1, :]).float()[0]
        first_logp = softmax_logprobs(first_logits)

        digit_batch = torch.tensor(digit_ids, dtype=torch.long, device=device)[:, None]
        first_cache = prefill.past_key_values
        first_cache.batch_repeat_interleave(len(digit_ids))
        after_digit = model.model(
            input_ids=digit_batch,
            attention_mask=torch.ones((len(digit_ids), len(input_ids) + 1),
                                      dtype=torch.long, device=device),
            past_key_values=first_cache,
            use_cache=True,
        )
        dot_logits = model.lm_head(after_digit.last_hidden_state[:, -1, :]).float()
        dot_logp = softmax_logprobs(dot_logits)[:, dot_id]

        dot_batch = torch.full((len(digit_ids), 1), dot_id, dtype=torch.long, device=device)
        after_dot = model.model(
            input_ids=dot_batch,
            attention_mask=torch.ones((len(digit_ids), len(input_ids) + 2),
                                      dtype=torch.long, device=device),
            past_key_values=after_digit.past_key_values,
            use_cache=True,
        )
        decimal_logits = model.lm_head(after_dot.last_hidden_state[:, -1, :]).float()
        decimal_logp = softmax_logprobs(decimal_logits)[:, decimal_ids]
        first_digit_logp = first_logp[digit_ids]
        joint = first_digit_logp[:, None] + dot_logp[:, None] + decimal_logp
        # Flattening yields 1.0–9.9. Keep only the frozen 1.0–9.0 support.
        return joint.reshape(-1)[:len(VALUES)].detach().cpu().numpy().astype(np.float64)


def load_prediction_rows() -> dict[str, dict[tuple, dict]]:
    out = {}
    for cohort, path in PREDICTIONS.items():
        manifest = json.loads(MANIFESTS[cohort].read_text())
        actual_hash = sha256(path.read_bytes())
        if actual_hash != EXPECTED_OUTPUT_HASHES[cohort] or manifest.get("output_sha256") != actual_hash:
            raise ValueError(f"Experiment {cohort} raw output hash mismatch")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        keys = [(r["case_id"], r["order"], r["forced_first_value"], r["decoder"]) for r in rows]
        if len(rows) != 256 or len(set(keys)) != 256:
            raise ValueError(f"Experiment {cohort} must have 256 unique frozen cells")
        if any(r["prediction"] is None for r in rows):
            raise ValueError(f"Experiment {cohort} contains invalid outputs")
        out[cohort] = dict(zip(keys, rows))
    return out


def frozen_jobs(source_dir: Path) -> tuple[dict[str, list[dict]], dict]:
    rows, source_hashes = exp052.read_source(source_dir)
    all_cases = exp052.select_cases(rows)
    cohort180, _ = exp064.select_cases(all_cases)
    cohort88, _ = exp066.select_cases(cohort180)
    cases67, stats67 = exp067.select_cases(cohort88)
    ids180 = {case["case_id"] for case in cohort180}
    cases68, stats68 = exp068.select_cases(all_cases, ids180)
    jobs = {"067": exp067.jobs_for(cases67), "068": exp068.jobs_for(cases68)}
    if stats67["recipient_ids_sha256"] != exp067.EXPECTED_IDS_SHA256:
        raise ValueError("Experiment 067 reconstructed IDs mismatch")
    if stats68["recipient_ids_sha256"] != exp068.EXPECTED_IDS_SHA256:
        raise ValueError("Experiment 068 reconstructed IDs mismatch")
    return jobs, {"source_hashes": source_hashes, "067": stats67, "068": stats68}


def run(source_dir: Path = exp052.SOURCE_DIR, output: Path = OUT,
        manifest_path: Path = MANIFEST, device: str = "mps",
        preflight_only: bool = False) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 069 protocol hash mismatch")
    source_jobs, sample_metadata = frozen_jobs(source_dir)
    predictions = load_prediction_rows()
    config = dict(MODEL_SPECS["qwen-3b"])
    if config["revision"] != "aa8e72537993ba99e69dfaafa59ed015b17504d1":
        raise ValueError("Unexpected model revision")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True
    )
    values, encodings = candidate_token_ids(tokenizer)
    digit_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 10)]
    decimal_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)]
    dot_id = tokenizer.encode(".", add_special_tokens=False)[0]
    contexts = []
    for cohort, jobs in source_jobs.items():
        rows = predictions[cohort]
        by_key = {(job["case_id"], job["order"], job["forced_first_value"], job["decoder"]): job
                  for job in jobs}
        if set(rows) != set(by_key):
            raise ValueError(f"Experiment {cohort} cells do not match the frozen design")
        for key, job in sorted(by_key.items()):
            saved = rows[key]
            if saved["assistant_prefix"] != job["assistant_prefix"]:
                raise ValueError(f"Experiment {cohort} assistant prefix mismatch: {key}")
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": job["prompt"]}],
                tokenize=False, add_generation_prompt=True,
            ) + job["assistant_prefix"]
            input_ids = tokenizer.encode(rendered, add_special_tokens=False)
            contexts.append({
                "cohort": cohort, "case_id": job["case_id"], "order": job["order"],
                "forced_first_value": float(job["forced_first_value"]),
                "decoder": job["decoder"], "input_ids": input_ids,
                "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "observed_second": float(saved["prediction"][0 if job["order"] == "arousal_first" else 1]),
                "candidate_encodings": encodings,
            })
    if len(contexts) != 512:
        raise ValueError(f"Expected 512 score contexts; got {len(contexts)}")
    if preflight_only:
        return {
            "experiment": "069-prefix-score-distribution-audit",
            "protocol_sha256": PROTOCOL_SHA256,
            "source_hashes": sample_metadata["source_hashes"],
            "n_contexts": len(contexts), "n_recipients_per_cohort": 64,
            "score_grid_size": len(values), "value_range": [values[0], values[-1]],
            "prefix_tokens_range": [min(len(c["input_ids"]) for c in contexts),
                                     max(len(c["input_ids"]) for c in contexts)],
            "prediction_hashes": EXPECTED_OUTPUT_HASHES,
            "model": config["model"], "model_revision": config["revision"],
        }

    model, tokenizer, actual_device = load_target(config, device, offline=True)
    model.config.use_cache = True
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            key = (row["cohort"], row["case_id"], row["order"], row["forced_first_value"])
            if key in previous:
                raise ValueError(f"Duplicate resumed Experiment 069 context: {key}")
            previous[key] = row
    expected = {(c["cohort"], c["case_id"], c["order"], c["forced_first_value"])
                for c in contexts}
    if not set(previous).issubset(expected):
        raise ValueError("Resume artifact contains rows outside frozen contexts")
    resumed_contexts = len(previous)
    with output.open("a", encoding="utf-8") as stream:
        for index, context in enumerate(contexts, start=1):
            key = (context["cohort"], context["case_id"], context["order"],
                   context["forced_first_value"])
            if key in previous:
                continue
            logprobs = score_grid(model, context["input_ids"], digit_ids,
                                  dot_id, decimal_ids, actual_device)
            probs = normalize_score_logprobs(logprobs)
            row = {k: context[k] for k in (
                "cohort", "case_id", "order", "forced_first_value", "decoder",
                "prompt_sha256", "observed_second",
            )}
            row["score_logprobs"] = logprobs.tolist()
            row["restricted_probabilities"] = probs.tolist()
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            previous[key] = row
            if index % 8 == 0 or index == len(contexts):
                print(f"069 scored {index}/{len(contexts)} contexts", flush=True)
    if len(previous) != len(contexts):
        raise ValueError("Experiment 069 output is incomplete")
    manifest = {
        "experiment": "069-prefix-score-distribution-audit",
        "protocol_sha256": PROTOCOL_SHA256,
        "prediction_hashes": EXPECTED_OUTPUT_HASHES,
        "source_hashes": sample_metadata["source_hashes"],
        "model": config["model"], "model_revision": config["revision"],
        "device": actual_device, "n_contexts": len(previous),
        "score_grid_size": len(values), "score_value_range": [values[0], values[-1]],
        "score_continuations_per_context": len(values),
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
