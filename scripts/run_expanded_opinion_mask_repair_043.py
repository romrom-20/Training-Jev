"""Run a disjoint, constrained-decoding repair of Experiment 041's primary test."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from run_opinion_mask_crosslingual_dimabsa_041 import (
    FILES,
    LANGS,
    SOURCE_SHA256,
    _eligible_target,
    build_prompt,
    mask_opinions,
    read_source,
    select_cases,
)
from run_opinion_mask_crosslingual_dimabsa_041 import (
    OUT as OUT_041,
)
from run_opinion_mask_crosslingual_dimabsa_041 import (
    PROTOCOL as PROTOCOL_041,
)
from task_ladder import MODEL_SPECS
from transformers import AutoTokenizer, set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/043-expanded-opinion-mask-repair.md")
PROTOCOL_SHA256 = "526a0de535a36deda9a2b0a4fb412967fad9720f5a39f492959ab3f71e0133a8"
PROTOCOL_041_SHA256 = "0e7d9991834a0bf0dd3d1010d0eefc4bd11b98c153ef8bbaa027af9a989f7e40"
SOURCE_REVISION = "bdc93be1224106ae7d3eb95739c02a76ed4ae8a1"
SOURCE_PREDICTIONS_SHA256 = "8056a1fd91f56d0700e10968ec8c55cc0fc3ffb1de73a98375df5d21f428a9c8"
OUT = Path(".context/exp043-private-predictions.jsonl")
MANIFEST = Path(".context/exp043-run-manifest.json")
SEED = 20260943
QUOTAS = {"neg": 103, "neu": 11, "pos": 103}
CONDITIONS = ("aspect_only", "opinion_masked")
GRID = tuple(round(1.0 + 0.1 * i, 1) for i in range(81))
DEVICE = "mps"
DEFAULT_BATCH = 4


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def candidates() -> list[tuple[str, dict]]:
    values = [
        (
            f'{{"valence":{valence:.1f},"arousal":{arousal:.1f}}}',
            {"valence": valence, "arousal": arousal},
        )
        for valence in GRID
        for arousal in GRID
    ]
    if len(values) != 6561 or len({text for text, _ in values}) != 6561:
        raise ValueError("Expected 6,561 unique VA grid points")
    return values


def all_eligible_cases(data: dict[str, list[dict]]) -> list[dict]:
    indexed = {lang: {row["ID"]: row for row in data[lang]} for lang in LANGS}
    common_ids = sorted(set.intersection(*(set(indexed[lang]) for lang in LANGS)))
    cases = []
    for case_id in common_ids:
        rows = [indexed[lang][case_id] for lang in LANGS]
        if len({len(row["Triplet"]) for row in rows}) != 1:
            continue
        if not all(
            [item["VA"] for item in row["Triplet"]]
            == [item["VA"] for item in rows[0]["Triplet"]]
            for row in rows[1:]
        ):
            continue
        for target_index, target in enumerate(rows[0]["Triplet"]):
            if not _eligible_target(rows, target_index):
                continue
            valence, arousal = map(float, target["VA"].split("#"))
            bucket = "neg" if valence < 4.5 else "neu" if valence <= 5.5 else "pos"
            cases.append(
                {
                    "case_id": case_id,
                    "target_index": target_index,
                    "gold": [valence, arousal],
                    "bucket": bucket,
                    "rows": {lang: indexed[lang][case_id] for lang in LANGS},
                }
            )
            break
    return cases


def select_repair_cases(data: dict[str, list[dict]]) -> list[dict]:
    old_ids = {row["case_id"] for row in select_cases(data)}
    all_cases = all_eligible_cases(data)
    buckets = defaultdict(list)
    for case in all_cases:
        if case["case_id"] not in old_ids:
            buckets[case["bucket"]].append(case)
    selected = []
    for bucket, quota in QUOTAS.items():
        ordered = sorted(
            buckets[bucket],
            key=lambda row: hashlib.sha256(
                f"exp043|{SEED}|{row['case_id']}|{row['target_index']}".encode()
            ).hexdigest(),
        )
        if len(ordered) < quota:
            raise ValueError(f"Only {len(ordered)} unused eligible {bucket} cases; need {quota}")
        selected.extend(ordered[:quota])
    result = sorted(selected, key=lambda row: row["case_id"])
    if len(result) != 217 or old_ids.intersection(row["case_id"] for row in result):
        raise ValueError("Repair sample must contain 217 IDs disjoint from Experiment 041")
    return result


def make_jobs(cases: list[dict]) -> list[dict]:
    jobs = []
    for case in cases:
        for lang in LANGS:
            row = case["rows"][lang]
            target = row["Triplet"][case["target_index"]]
            values = {
                "aspect_only": (None, None),
                "opinion_masked": (mask_opinions(row), None),
            }
            for condition in CONDITIONS:
                visible_text, opinion = values[condition]
                prompt = build_prompt(visible_text, target["Aspect"], opinion)
                prompt = prompt.replace(
                    'Return exactly one JSON object with numeric keys "valence" and "arousal", both from 1 to 9.',
                    'Return exactly one JSON object with numeric keys "valence" and "arousal", '
                    "both from 1.0 to 9.0 in increments of 0.1, with exactly one decimal place.",
                    1,
                )
                jobs.append(
                    {
                        "case_id": case["case_id"],
                        "lang": lang,
                        "condition": condition,
                        "gold": case["gold"],
                        "prompt": prompt,
                    }
                )
    if len(jobs) != 1302:
        raise ValueError(f"Expected 1,302 prompts, found {len(jobs)}")
    return jobs


def parse_va(text: str) -> list[float] | None:
    match = re.fullmatch(
        r'\{"valence":([1-9]\.\d),"arousal":([1-9]\.\d)\}', text.strip()
    )
    if match is None:
        return None
    va = [float(match.group(1)), float(match.group(2))]
    if any(not 1 <= x <= 9 for x in va):
        return None
    return va


def candidate_sequences(tokenizer, outputs: list[tuple[str, dict]]) -> list[list[int]]:
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    sequences = [
        tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
        for text, _ in outputs
    ]
    if len({tuple(sequence) for sequence in sequences}) != len(outputs):
        raise ValueError("Two decimal VA outputs share a token sequence")
    return sequences


def build_trie(sequences: list[list[int]]) -> dict[tuple[int, ...], tuple[int, ...]]:
    trie: dict[tuple[int, ...], set[int]] = defaultdict(set)
    longest = max(map(len, sequences))
    for sequence in sequences:
        for offset, token in enumerate(sequence):
            trie[tuple(sequence[:offset])].add(token)
        # Transformers calls the token filter for already-finished batch rows
        # while other rows continue; permit EOS padding for those prefixes.
        for extra_eos in range(longest - len(sequence) + 2):
            trie[tuple(sequence + [sequence[-1]] * extra_eos)].add(sequence[-1])
    return {prefix: tuple(sorted(tokens)) for prefix, tokens in trie.items()}


def verify_boundaries(tokenizer, jobs: list[dict], outputs: list[tuple[str, dict]]) -> None:
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    for job in jobs[: min(2, len(jobs))]:
        messages = [{"role": "user", "content": job["prompt"]}]
        prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        for text, _ in outputs:
            full = tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": text}],
                tokenize=True,
                add_generation_prompt=False,
            )
            suffix = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
            tail = full[len(prefix) :]
            if full[: len(prefix)] != prefix or tail[: len(suffix)] != suffix:
                raise ValueError("Candidate is not stable at assistant token boundary")


def generate_batch(model, tokenizer, batch: list[dict], trie, device: str) -> list[str]:
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": job["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for job in batch
    ]
    tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    prompt_width = tokens.input_ids.shape[1]

    def allowed(_batch_index, input_ids):
        prefix = tuple(input_ids[prompt_width:].tolist())
        if prefix not in trie:
            raise ValueError("Generated prefix escaped the preregistered VA grammar")
        return list(trie[prefix])

    with torch.inference_mode():
        generated = model.generate(
            **tokens,
            max_new_tokens=max(len(token_ids) for token_ids in trie) + 1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=allowed,
        )
    return tokenizer.batch_decode(
        generated[:, prompt_width:], skip_special_tokens=True
    )


def run(
    source_dir: Path = Path(".context/dimabsa"),
    output: Path = OUT,
    manifest_path: Path = MANIFEST,
    device: str = DEVICE,
    batch_size: int = DEFAULT_BATCH,
    preflight_only: bool = False,
) -> dict:
    if digest(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 043 protocol hash mismatch")
    if digest(PROTOCOL_041.read_bytes()) != PROTOCOL_041_SHA256:
        raise ValueError("Experiment 041 protocol changed")
    if digest(OUT_041.read_bytes()) != SOURCE_PREDICTIONS_SHA256:
        raise ValueError("Experiment 041 private output changed")
    source_data, source_hashes = {}, {}
    for lang in LANGS:
        source_data[lang], source_hashes[lang] = read_source(
            source_dir / Path(FILES[lang]).name, lang
        )
    if source_hashes != SOURCE_SHA256:
        raise ValueError("DimABSA source hashes changed")
    cases = select_repair_cases(source_data)
    jobs = make_jobs(cases)
    outputs = candidates()
    previous = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            key = (row["case_id"], row["lang"], row["condition"])
            if key in previous:
                raise ValueError(f"Duplicate resumed output: {key}")
            previous[key] = row
    valid_keys = {
        (job["case_id"], job["lang"], job["condition"])
        for job in jobs
    }
    if not set(previous).issubset(valid_keys):
        raise ValueError("Private resume file contains a row outside the frozen sample")
    remaining = [
        job for job in jobs
        if (job["case_id"], job["lang"], job["condition"]) not in previous
    ]

    model_config = dict(MODEL_SPECS["qwen-3b"])
    if preflight_only:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["model"], revision=model_config["revision"], local_files_only=True
        )
        tokenizer.padding_side = "left"
        sequences = candidate_sequences(tokenizer, outputs)
        verify_boundaries(tokenizer, jobs, outputs)
        trie = build_trie(sequences)
        return {
            "n_eligible_aligned_ids": len(all_eligible_cases(source_data)),
            "n_selected_repair_ids": len(cases),
            "selected_bucket_counts": dict(Counter(case["bucket"] for case in cases)),
            "selected_disjoint_from_041": True,
            "n_jobs": len(jobs),
            "n_candidate_values": len(outputs),
            "n_unique_candidate_token_sequences": len(sequences),
            "n_trie_prefixes": len(trie),
            "max_candidate_token_length": max(map(len, sequences)),
            "token_boundary_check": "passed for every candidate on two prompts",
            "source_hashes": source_hashes,
            "protocol_sha256": digest(PROTOCOL.read_bytes()),
        }
    model, tokenizer, actual_device = load_target(model_config, device, offline=True)
    tokenizer.padding_side = "left"
    sequences = candidate_sequences(tokenizer, outputs)
    trie = build_trie(sequences)
    verify_boundaries(tokenizer, jobs, outputs)
    set_seed(SEED)
    generation_started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        for offset in range(0, len(remaining), batch_size):
            batch = remaining[offset : offset + batch_size]
            raw_outputs = generate_batch(model, tokenizer, batch, trie, actual_device)
            for job, raw in zip(batch, raw_outputs):
                row = {
                    "case_id": job["case_id"],
                    "lang": job["lang"],
                    "condition": job["condition"],
                    "gold": job["gold"],
                    "raw": raw,
                    "prediction": parse_va(raw),
                }
                key = (row["case_id"], row["lang"], row["condition"])
                previous[key] = row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                f"043 constrained {min(offset + len(batch), len(remaining))}/{len(remaining)}; "
                f"total {len(previous)}/{len(jobs)}",
                flush=True,
            )
    generation_seconds = time.monotonic() - generation_started
    invalid = sum(row["prediction"] is None for row in previous.values())
    manifest = {
        "experiment": "043-expanded-opinion-mask-repair",
        "protocol_sha256": digest(PROTOCOL.read_bytes()),
        "parent_experiment_041_protocol_sha256": PROTOCOL_041_SHA256,
        "parent_experiment_041_output_sha256": SOURCE_PREDICTIONS_SHA256,
        "source_revision": SOURCE_REVISION,
        "source_hashes": source_hashes,
        "eligible_aligned_ids": len(all_eligible_cases(source_data)),
        "selected_bucket_counts": dict(Counter(case["bucket"] for case in cases)),
        "sample_disjoint_from_041": True,
        "model": model_config["model"],
        "model_revision": model_config["revision"],
        "device": actual_device,
        "batch_size": batch_size,
        "n_prompts": len(jobs),
        "n_outputs": len(previous),
        "n_grid_candidates": len(outputs),
        "grid_resolution": 0.1,
        "invalid_outputs": invalid,
        "generation_seconds_this_process_only": generation_seconds,
        "resumed_rows": len(jobs) - len(remaining),
        "output_sha256": digest(output.read_bytes()),
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
    parser.add_argument("--device", choices=("mps", "cpu"), default=DEVICE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
