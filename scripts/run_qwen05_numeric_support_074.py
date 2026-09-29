"""Audit full JSON score likelihoods and common alternate numeric spellings."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import run_cross_family_prefix_coupling_071 as exp071
import run_forced_coordinate_prefix_067 as exp067
import run_prefix_score_distribution_audit_069 as exp069
import torch
from run_small_model_decoder_factorial_048 import parse_free_va, sha256
from task_ladder import MODEL_SPECS
from transformers import AutoTokenizer, set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/074-qwen05-numeric-support-audit.md")
PROTOCOL_SHA256 = "373aa9ecb3c4a7b47a831b1b7dc256a0014ab424e33a18fb03e1a340e62b3c77"
OUT = Path(".context/exp074-private-score-forms.jsonl")
MANIFEST = Path(".context/exp074-run-manifest.json")
SEED = 20260974
MODEL_CONFIGS = {
    "qwen-0.5b": {"model": "Qwen/Qwen2.5-0.5B-Instruct",
                  "revision": "7ae557604adf67be50417f59c2c2f167def9a775"},
    "qwen-1.5b": dict(MODEL_SPECS["qwen-1.5b"]),
}
ORDERS = ("valence_first", "arousal_first")
FORCED_VALUES = (2.0, 8.0)


def select_cases(source_dir: Path) -> tuple[list[dict], dict]:
    cases, metadata = exp071.select_cases(source_dir)
    chosen = []
    for predicate in (lambda value: value < 4.5, lambda value: value > 5.5):
        group = [case for case in cases if predicate(case["gold"][0])]
        group.sort(key=lambda case: hashlib.sha256(
            f"exp074-score-form-audit-v1|{case['case_id']}".encode()).hexdigest())
        if len(group) < 12:
            raise ValueError("Insufficient polarity cases for Experiment 074")
        chosen.extend(group[:12])
    chosen.sort(key=lambda case: case["case_id"])
    ids = [case["case_id"] for case in chosen]
    id_hash = sha256("\n".join(ids).encode())
    if id_hash != "4f03b671acd8aedfce6cef7eefae4c738a80d29fbcd7f30f042d4243fd9e001d":
        raise ValueError("Experiment 074 recipient IDs differ from the frozen protocol")
    metadata = {**metadata, "n_selected_recipients": 24,
                "polarity_counts": {"negative": 12, "positive": 12},
                "recipient_ids_sha256": id_hash}
    return chosen, metadata


def build_jobs(cases: list[dict]) -> list[dict]:
    jobs = []
    for case in cases:
        target = case["row"]["Triplet"][case["target_index"]]
        masked = exp071.exp049.mask_opinions(case["row"])
        prompt = exp071.exp049.prompt_with_registered_grid(masked, target["Aspect"])
        for order in ORDERS:
            for forced in FORCED_VALUES:
                jobs.append({
                    "case_id": case["case_id"], "order": order,
                    "forced_first_value": forced, "gold": case["gold"],
                    "prompt": prompt,
                    "user_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "assistant_prefix": exp067.assistant_prefix(order, forced),
                })
    if len(jobs) != 96:
        raise ValueError("Experiment 074 must include all four prefix cells per recipient")
    return sorted(jobs, key=lambda job: (job["order"], job["forced_first_value"], job["case_id"]))


def rendered_input(tokenizer, job: dict) -> tuple[str, list[int]]:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": exp071.SYSTEM_PROMPT},
         {"role": "user", "content": job["prompt"]}],
        tokenize=False, add_generation_prompt=True,
    ) + job["assistant_prefix"]
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text, ids


def canonical_strings() -> list[str]:
    return [f"{value:.1f}}}" for value in exp069.VALUES]


def extended_strings() -> list[str]:
    return [f"{value:.1f}0}}" for value in exp069.VALUES]


def integer_strings() -> list[str]:
    return [f"{value}}}" for value in range(1, 10)]


def verify_token_boundary(tokenizer, text: str, input_ids: list[int]) -> dict:
    tokenized = {}
    for family, strings in (("canonical", canonical_strings()),
                            ("extended", extended_strings()),
                            ("integer", integer_strings())):
        encodings = []
        for suffix in strings:
            full = tokenizer.encode(text + suffix, add_special_tokens=False)
            if full[:len(input_ids)] != input_ids:
                raise ValueError("Candidate crosses the open-prefix token boundary")
            ids = full[len(input_ids):]
            if not ids or tokenizer.decode(ids, skip_special_tokens=False) != suffix:
                raise ValueError(f"Candidate tokenization does not round-trip: {suffix}")
            encodings.append(ids)
        tokenized[family] = encodings
    return tokenized


def _log_softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.float(), dim=-1)


def score_with_tokenizer(model, input_ids: list[int], device: str, tokenizer) -> dict:
    prefix_len = len(input_ids)
    first_digits = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 10)]
    decimals = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)]
    dot = tokenizer.encode(".", add_special_tokens=False)
    zero = tokenizer.encode("0", add_special_tokens=False)
    brace = tokenizer.encode("}", add_special_tokens=False)
    if any(len(ids) != 1 for ids in [dot, zero, brace]
           ) or any(len(tokenizer.encode(str(i), add_special_tokens=False)) != 1 for i in range(10)):
        raise ValueError("Expected one-token digits, decimal point, and closing brace")
    dot_id, zero_id, brace_id = dot[0], zero[0], brace[0]
    with torch.inference_mode():
        prefix = torch.tensor([input_ids], dtype=torch.long, device=device)
        prefill = model.model(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                              use_cache=True)
        first_logp = _log_softmax(model.lm_head(prefill.last_hidden_state[:, -1, :]).float())[0]

        digit_batch = torch.tensor(first_digits, dtype=torch.long, device=device)[:, None]
        first_cache = prefill.past_key_values
        first_cache.batch_repeat_interleave(9)
        after_digit = model.model(
            input_ids=digit_batch,
            attention_mask=torch.ones((9, prefix_len + 1), dtype=torch.long, device=device),
            past_key_values=first_cache, use_cache=True,
        )
        after_digit_logp = _log_softmax(
            model.lm_head(after_digit.last_hidden_state[:, -1, :]).float())
        dot_logp = after_digit_logp[:, dot_id]
        int_close_logp = after_digit_logp[:, brace_id]

        dot_batch = torch.full((9, 1), dot_id, dtype=torch.long, device=device)
        digit_cache = after_digit.past_key_values
        after_dot = model.model(
            input_ids=dot_batch,
            attention_mask=torch.ones((9, prefix_len + 2), dtype=torch.long, device=device),
            past_key_values=digit_cache, use_cache=True,
        )
        after_dot_logp = _log_softmax(
            model.lm_head(after_dot.last_hidden_state[:, -1, :]).float())
        decimal_batch = torch.tensor(decimals * 9, dtype=torch.long, device=device)[:, None]
        dot_cache = after_dot.past_key_values
        dot_cache.batch_repeat_interleave(10)
        after_decimal = model.model(
            input_ids=decimal_batch,
            attention_mask=torch.ones((90, prefix_len + 3), dtype=torch.long, device=device),
            past_key_values=dot_cache, use_cache=True,
        )
        after_decimal_logp = _log_softmax(
            model.lm_head(after_decimal.last_hidden_state[:, -1, :]).float())
        canonical_close = after_decimal_logp[:, brace_id]
        extension_zero = after_decimal_logp[:, zero_id]

        zero_batch = torch.full((90, 1), zero_id, dtype=torch.long, device=device)
        decimal_cache = after_decimal.past_key_values
        after_zero = model.model(
            input_ids=zero_batch,
            attention_mask=torch.ones((90, prefix_len + 4), dtype=torch.long, device=device),
            past_key_values=decimal_cache, use_cache=True,
        )
        after_zero_logp = _log_softmax(
            model.lm_head(after_zero.last_hidden_state[:, -1, :]).float())
        extended_close = after_zero_logp[:, brace_id]

        first = first_logp[first_digits]
        canonical = (first[:, None] + dot_logp[:, None] +
                     after_dot_logp[:, decimals] + canonical_close.reshape(9, 10))
        extended = (first[:, None] + dot_logp[:, None] +
                    after_dot_logp[:, decimals] + extension_zero.reshape(9, 10) +
                    extended_close.reshape(9, 10))
        canonical = canonical.reshape(-1)[:81]
        extended = extended.reshape(-1)[:81]
        integer = first + int_close_logp
    return {
        "canonical_logprobs": canonical.detach().cpu().numpy().astype(float).tolist(),
        "extended_logprobs": extended.detach().cpu().numpy().astype(float).tolist(),
        "integer_logprobs": integer.detach().cpu().numpy().astype(float).tolist(),
    }


def sequence_logprob(model, prefix_ids: list[int], suffix_ids: list[int], device: str) -> float:
    all_ids = prefix_ids + suffix_ids
    tensor = torch.tensor([all_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model.model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                             use_cache=False)
        logits = model.lm_head(output.last_hidden_state[:, :-1, :]).float()[0]
        start = len(prefix_ids) - 1
        targets = torch.tensor(suffix_ids, dtype=torch.long, device=device)
        return float(_log_softmax(logits[start:start + len(suffix_ids)]).gather(
            1, targets[:, None]).sum().item())


def selected_validations(jobs: list[dict], model_key: str) -> list[tuple[tuple, str, str]]:
    contexts = [(job["case_id"], job["order"], job["forced_first_value"]) for job in jobs]
    contexts.sort(key=lambda key: hashlib.sha256(
        f"exp074-likelihood-check|{model_key}|{key}".encode()).hexdigest())
    chosen = contexts[:32]
    out = []
    for index, key in enumerate(chosen):
        family = ("canonical", "extended", "integer")[index % 3]
        value = (float(1 + index % 9) if family == "integer"
                 else float([2.0, 4.2, 9.0][index % 3]))
        if family == "canonical":
            suffix = f"{value:.1f}}}"
        elif family == "extended":
            suffix = f"{value:.1f}0}}"
        else:
            suffix = f"{int(value)}}}"
        out.append((key, family, suffix))
    return out


def _path_index(value: float) -> int:
    return int(round((value - 1.0) * 10))


def run(source_dir: Path = Path(".context/dimabsa"), output: Path = OUT,
        manifest_path: Path = MANIFEST, device: str = "mps",
        preflight_only: bool = False) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 074 protocol hash mismatch")
    cases, sample_meta = select_cases(source_dir)
    jobs = build_jobs(cases)
    token_meta = {}
    validation_plan = {}
    for model_key, config in MODEL_CONFIGS.items():
        tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"],
                                                  local_files_only=True)
        candidates = []
        for job in jobs:
            text, ids = rendered_input(tokenizer, job)
            candidates.append(verify_token_boundary(tokenizer, text, ids))
        token_meta[model_key] = {
            "candidate_families": {key: len(value) for key, value in candidates[0].items()},
            "token_boundary": "passed for all 171 continuations in all contexts",
        }
        validation_plan[model_key] = selected_validations(jobs, model_key)
    if preflight_only:
        return {"experiment": "074-qwen05-numeric-support-audit",
                "protocol_sha256": PROTOCOL_SHA256, **sample_meta,
                "n_contexts_per_model": len(jobs), "n_models": len(MODEL_CONFIGS),
                "n_total_model_contexts": len(jobs) * len(MODEL_CONFIGS),
                "n_candidate_continuations_per_context": 171,
                "n_sequence_validations_per_model": 32,
                "tokenizer_metadata": token_meta}

    old = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            key = (row["model_key"], row["case_id"], row["order"],
                   float(row["forced_first_value"]))
            if key in old:
                raise ValueError(f"Duplicate resumed cell: {key}")
            old[key] = row
    expected = {(model, job["case_id"], job["order"], job["forced_first_value"])
                for model in MODEL_CONFIGS for job in jobs}
    if not set(old).issubset(expected):
        raise ValueError("Resume artifact contains cells outside the frozen design")
    resumed = len(old)
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    validations = {}
    for model_key, config in MODEL_CONFIGS.items():
        remaining = [job for job in jobs if (model_key, job["case_id"], job["order"],
                                              job["forced_first_value"]) not in old]
        if not remaining:
            continue
        model, tokenizer, actual_device = load_target(config, device, offline=True)
        set_seed(SEED)
        checks = {(key, family): suffix for key, family, suffix in validation_plan[model_key]}
        errors = []
        with output.open("a", encoding="utf-8") as stream:
            for i, job in enumerate(remaining, start=1):
                text, input_ids = rendered_input(tokenizer, job)
                forms = score_with_tokenizer(model, input_ids, actual_device, tokenizer)
                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=actual_device)
                with torch.inference_mode():
                    generated = model.generate(input_ids=input_tensor,
                                               attention_mask=torch.ones_like(input_tensor),
                                               max_new_tokens=40, do_sample=False,
                                               pad_token_id=tokenizer.pad_token_id,
                                               eos_token_id=tokenizer.eos_token_id)
                raw = tokenizer.decode(generated[0, input_tensor.shape[1]:], skip_special_tokens=True)
                prediction = parse_free_va(job["assistant_prefix"] + raw)
                first_axis = 0 if job["order"] == "valence_first" else 1
                second_axis = 1 - first_axis
                if prediction is not None and prediction[first_axis] != job["forced_first_value"]:
                    raise ValueError("Greedy continuation changed the forced first coordinate")
                key = (job["case_id"], job["order"], job["forced_first_value"])
                checked = {}
                for family in ("canonical", "extended", "integer"):
                    suffix = checks.get((key, family))
                    if suffix is None:
                        continue
                    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
                    batched = forms[f"{family}_logprobs"]
                    if family == "integer":
                        score = int(suffix[:-1])
                        batched_lp = batched[score - 1]
                    else:
                        score = float(suffix[:-1]) if family == "canonical" else float(suffix[:-1][:-1])
                        batched_lp = batched[_path_index(score)]
                    direct = sequence_logprob(model, input_ids, suffix_ids, actual_device)
                    error = abs(float(batched_lp) - direct)
                    errors.append(error)
                    checked[family] = {"candidate": suffix, "absolute_logprob_error": error}
                row = {
                    "model_key": model_key, "model": config["model"],
                    "model_revision": config["revision"], "case_id": job["case_id"],
                    "order": job["order"], "forced_first_value": job["forced_first_value"],
                    "gold": job["gold"], "assistant_prefix": job["assistant_prefix"],
                    "user_prompt_sha256": job["user_prompt_sha256"],
                    **forms, "greedy_raw_continuation": raw,
                    "greedy_prediction": prediction,
                    "greedy_second_score": None if prediction is None else prediction[second_axis],
                    "sequence_likelihood_checks": checked,
                    "rendered_prefix_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
                old[(model_key, *key)] = row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if i % 4 == 0 or i == len(remaining):
                    print(f"074 {model_key} {i}/{len(remaining)} new", flush=True)
        validations[model_key] = {"n_checks": len(errors), "max_abs_logprob_error": max(errors)}
        del model, tokenizer
        gc.collect()
        if actual_device == "mps":
            torch.mps.empty_cache()
    if len(old) != len(expected):
        raise ValueError("Experiment 074 output is incomplete")
    invalid = {model: {order: {str(value): sum(
        old[(model, job["case_id"], order, value)]["greedy_prediction"] is None
        for job in jobs if job["order"] == order and job["forced_first_value"] == value
    ) for value in FORCED_VALUES} for order in ORDERS} for model in MODEL_CONFIGS}
    manifest = {
        "experiment": "074-qwen05-numeric-support-audit",
        "protocol_sha256": PROTOCOL_SHA256, **sample_meta,
        "models": MODEL_CONFIGS, "device": device,
        "n_contexts_per_model": len(jobs), "n_total_model_contexts": len(old),
        "n_score_surface_forms_per_context": 171,
        "invalid_by_model_order_forced_value": invalid,
        "sequence_likelihood_validation": validations,
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
