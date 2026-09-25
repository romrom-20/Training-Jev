"""Run a small constrained-output feasibility audit after Experiment 041."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from run_opinion_mask_crosslingual_dimabsa_041 import (
    FILES,
    LANGS,
    SOURCE_REVISION,
    SOURCE_SHA256,
    make_jobs,
    read_source,
    select_cases,
)
from run_opinion_mask_crosslingual_dimabsa_041 import (
    PROTOCOL as PROTOCOL_041,
)
from task_ladder import MODEL_SPECS
from transformers import set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/042-structured-output-feasibility.md")
PROTOCOL_SHA256 = "5b02d8b8a3771a19d1b8d14d0248f2235ea9e2ca0555b1d218b47a73aea8c069"
SOURCE_PREDICTIONS = Path(".context/exp041-private-predictions.jsonl")
SOURCE_PREDICTIONS_SHA256 = "8056a1fd91f56d0700e10968ec8c55cc0fc3ffb1de73a98375df5d21f428a9c8"
OUT = Path(".context/exp042-private-predictions.jsonl")
SEED = 20260942
CONTROL_PER_LANGUAGE = 16
DEVICE = "mps"
DEFAULT_BATCH = 4


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def output_candidates() -> list[tuple[str, dict]]:
    options = [
        (
            json.dumps(
                {"status": "estimate", "valence": valence, "arousal": arousal},
                separators=(",", ":"),
            ),
            {"status": "estimate", "valence": valence, "arousal": arousal},
        )
        for valence in range(1, 10)
        for arousal in range(1, 10)
    ]
    options.append(
        (json.dumps({"status": "insufficient"}, separators=(",", ":")), {"status": "insufficient"})
    )
    return options


def candidate_token_sequences(tokenizer) -> list[list[int]]:
    candidates = output_candidates()
    encoded = [tokenizer.encode(text, add_special_tokens=False) for text, _ in candidates]
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer has no EOS token for constrained completion")
    return [tokens + [eos] for tokens in encoded]


def allowed_next_tokens(generated: list[int], sequences: list[list[int]]) -> list[int]:
    terminal = sequences[0][-1]
    if any(
        len(sequence) <= len(generated) and generated[: len(sequence)] == sequence
        for sequence in sequences
    ):
        # Transformers pads finished rows with EOS while other batch members continue.
        return [terminal]
    allowed = sorted(
        {
            sequence[len(generated)]
            for sequence in sequences
            if len(sequence) > len(generated) and sequence[: len(generated)] == generated
        }
    )
    if not allowed:
        raise ValueError("Generated prefix is not a prefix of any registered candidate")
    return allowed


def parse_response(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in ("```", "```json"):
            return None
        if lines[-1].strip() != "```":
            return None
        text = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    if value == {"status": "insufficient"}:
        return {"status": "insufficient"}
    if set(value) != {"status", "valence", "arousal"} or value.get("status") != "estimate":
        return None
    try:
        valence, arousal = float(value["valence"]), float(value["arousal"])
    except (TypeError, ValueError):
        return None
    if not 1 <= valence <= 9 or not 1 <= arousal <= 9:
        return None
    return {"status": "estimate", "valence": valence, "arousal": arousal}


def select_audit_prompts(data: dict[str, list[dict]], baseline: list[dict]) -> list[dict]:
    cases = select_cases(data)
    all_jobs = make_jobs(cases)
    jobs_by_key = {
        (job["case_id"], job["lang"], job["condition"]): job
        for job in all_jobs
    }
    invalid = [row for row in baseline if row["prediction"] is None]
    if len(invalid) != 48:
        raise ValueError(f"Expected 48 preregistered-source invalid outputs; found {len(invalid)}")
    rows = []
    for row in invalid:
        key = (row["case_id"], row["lang"], row["condition"])
        job = jobs_by_key[key]
        rows.append(
            {
                "case_id": row["case_id"],
                "lang": row["lang"],
                "source_condition": row["condition"],
                "source_free_valid": False,
                "free_prediction": None,
                "group": "prior_invalid",
                "prompt": job["prompt"],
            }
        )

    for lang in LANGS:
        controls = [
            row
            for row in baseline
            if row["lang"] == lang
            and row["condition"] == "aspect_opinion"
            and row["prediction"] is not None
        ]
        controls = sorted(
            controls,
            key=lambda row: hashlib.sha256(
                f"exp042|{SEED}|{row['case_id']}|{lang}".encode()
            ).hexdigest(),
        )
        if len(controls) < CONTROL_PER_LANGUAGE:
            raise ValueError(f"Only {len(controls)} parseable {lang} controls")
        for row in controls[:CONTROL_PER_LANGUAGE]:
            key = (row["case_id"], lang, "aspect_opinion")
            job = jobs_by_key[key]
            rows.append(
                {
                    "case_id": row["case_id"],
                    "lang": lang,
                    "source_condition": "aspect_opinion",
                    "source_free_valid": True,
                    "free_prediction": row["prediction"],
                    "group": "prior_valid_control",
                    "prompt": job["prompt"],
                }
            )
    if len(rows) != 96 or len({(r["case_id"], r["lang"], r["source_condition"]) for r in rows}) != 96:
        raise ValueError("Expected 96 unique prompts in the audit sample")
    return sorted(rows, key=lambda row: (row["group"], row["lang"], row["case_id"], row["source_condition"]))


def revised_prompt(prompt: str) -> str:
    old = (
        'Return exactly one JSON object with numeric keys "valence" and "arousal", both from 1 to 9.'
    )
    new = (
        'Return exactly either {"status":"estimate","valence":N,"arousal":M}, '
        'where N and M are numbers from 1 to 9, or {"status":"insufficient"} if the supplied '
        "evidence does not support an estimate. Do not add other text."
    )
    if old not in prompt:
        raise ValueError("Experiment 041 output instruction changed")
    return prompt.replace(old, new, 1)


def _allowed_fn(sequences: list[list[int]], prompt_width: int):
    def callback(_batch_id, input_ids):
        generated = input_ids[prompt_width:].tolist()
        return allowed_next_tokens(generated, sequences)

    return callback


def generate_batch(model, tokenizer, batch: list[dict], decoder: str, device: str):
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": revised_prompt(row["prompt"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in batch
    ]
    tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    kwargs = {
        "max_new_tokens": 40,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if decoder == "constrained":
        sequences = candidate_token_sequences(tokenizer)
        kwargs["prefix_allowed_tokens_fn"] = _allowed_fn(sequences, tokens.input_ids.shape[1])
        kwargs["max_new_tokens"] = max(len(seq) for seq in sequences)
    elif decoder != "free":
        raise ValueError(f"Unknown decoder {decoder}")
    with torch.inference_mode():
        generated = model.generate(**tokens, **kwargs)
    decoded = tokenizer.batch_decode(
        generated[:, tokens.input_ids.shape[1] :], skip_special_tokens=True
    )
    return decoded


def _verify_token_boundary(tokenizer, prompts: list[dict], candidates: list[tuple[str, dict]]) -> None:
    for row in prompts[: min(12, len(prompts))]:
        messages = [{"role": "user", "content": revised_prompt(row["prompt"])}]
        prefix = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        for candidate, _ in candidates:
            full = tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": candidate}],
                tokenize=True,
                add_generation_prompt=False,
            )
            suffix = tokenizer.encode(candidate, add_special_tokens=False) + [tokenizer.eos_token_id]
            if full[: len(prefix)] != prefix or full[len(prefix) : len(prefix) + len(suffix)] != suffix:
                raise ValueError("Candidate token sequence is not stable at assistant boundary")


def run(
    source_dir: Path,
    output: Path = OUT,
    device: str = DEVICE,
    batch_size: int = DEFAULT_BATCH,
) -> dict:
    if digest(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 042 protocol hash mismatch")
    if digest(PROTOCOL_041.read_bytes()) != "0e7d9991834a0bf0dd3d1010d0eefc4bd11b98c153ef8bbaa027af9a989f7e40":
        raise ValueError("Experiment 041 protocol changed")
    baseline_bytes = SOURCE_PREDICTIONS.read_bytes()
    if digest(baseline_bytes) != SOURCE_PREDICTIONS_SHA256:
        raise ValueError("Experiment 041 private outputs changed")
    baseline = [json.loads(line) for line in baseline_bytes.decode().splitlines()]
    data, source_hashes = {}, {}
    for lang in LANGS:
        path = source_dir / Path(FILES[lang]).name
        data[lang], source_hashes[lang] = read_source(path, lang)
    if source_hashes != SOURCE_SHA256:
        raise ValueError("Source data hashes changed")
    audit = select_audit_prompts(data, baseline)
    candidates = output_candidates()

    model_config = dict(MODEL_SPECS["qwen-3b"])
    model, tokenizer, actual_device = load_target(model_config, device, offline=True)
    tokenizer.padding_side = "left"
    _verify_token_boundary(tokenizer, audit, candidates)
    candidate_sequences = candidate_token_sequences(tokenizer)
    if len(candidates) != 82 or len({tuple(seq) for seq in candidate_sequences}) != 82:
        raise ValueError("Finite candidate set must contain 82 unique token sequences")
    set_seed(SEED)

    output.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            previous[(row["case_id"], row["lang"], row["source_condition"], row["decoder"])] = row
    jobs = [
        {**row, "decoder": decoder}
        for row in audit
        for decoder in ("free", "constrained")
    ]
    remaining = [
        row
        for row in jobs
        if (row["case_id"], row["lang"], row["source_condition"], row["decoder"]) not in previous
    ]
    elapsed_start = time.monotonic()
    with output.open("a", encoding="utf-8") as stream:
        for decoder in ("free", "constrained"):
            selected = [row for row in remaining if row["decoder"] == decoder]
            for offset in range(0, len(selected), batch_size):
                batch = selected[offset : offset + batch_size]
                decoded = generate_batch(model, tokenizer, batch, decoder, actual_device)
                for row, answer in zip(batch, decoded):
                    result = {
                        key: row[key]
                        for key in (
                            "case_id",
                            "lang",
                            "source_condition",
                            "source_free_valid",
                            "free_prediction",
                            "group",
                            "decoder",
                        )
                    }
                    result["raw"] = answer
                    result["response"] = parse_response(answer)
                    previous[
                        (row["case_id"], row["lang"], row["source_condition"], decoder)
                    ] = result
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                print(
                    f"042 {decoder}: {min(offset + len(batch), len(selected))}/{len(selected)}; "
                    f"total {len(previous)}/192",
                    flush=True,
                )
    elapsed = time.monotonic() - elapsed_start
    invalid_by_decoder = {
        decoder: sum(row["response"] is None for row in previous.values() if row["decoder"] == decoder)
        for decoder in ("free", "constrained")
    }
    manifest = {
        "experiment": "042-structured-output-feasibility",
        "protocol_sha256": digest(PROTOCOL.read_bytes()),
        "source_prediction_sha256": SOURCE_PREDICTIONS_SHA256,
        "data_revision": SOURCE_REVISION,
        "source_hashes": source_hashes,
        "model": model_config["model"],
        "model_revision": model_config["revision"],
        "device": actual_device,
        "n_prompts": len(audit),
        "n_generations": len(previous),
        "candidate_outputs": len(candidates),
        "invalid_by_decoder": invalid_by_decoder,
        "elapsed_seconds": elapsed,
        "output_sha256": digest(output.read_bytes()),
    }
    print(json.dumps(manifest, indent=2), flush=True)
    del model, tokenizer
    gc.collect()
    if actual_device == "mps":
        torch.mps.empty_cache()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path(".context/dimabsa"))
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--device", choices=("mps", "cpu"), default=DEVICE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    args = parser.parse_args()
    print(json.dumps(run(args.source_dir, args.output, args.device, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
