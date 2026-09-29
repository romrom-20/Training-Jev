"""Run Experiment 076's role-location and fresh-corpus controls."""

from __future__ import annotations

import argparse
import ast
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
from run_small_model_decoder_factorial_048 import sha256
from transformers import AutoTokenizer, set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/076-schema-and-corpus-controls.md")
PROTOCOL_SHA256 = "f7224589741ff44f5ec9fae0ce976f10770720ae8cab3e36b5761a5107878b62"
DIMABSA_DIR = Path(".context/dimabsa")
ASTE_FILE = Path(".context/aste14res/14lap_test_triplets.txt")
OUT = Path(".context/exp076-private-schema-corpus.jsonl")
MANIFEST = Path(".context/exp076-run-manifest.json")
SEED = 20260976
FORCED_VALUES = (2.0, 8.0)
ORDERS = ("valence_first", "arousal_first")
LOCATIONS = ("prior_assistant", "prior_user")
MODEL_CONFIGS = {
    "qwen-0.5b": {"model": "Qwen/Qwen2.5-0.5B-Instruct",
                  "revision": "7ae557604adf67be50417f59c2c2f167def9a775"},
    "qwen-1.5b": {"model": "Qwen/Qwen2.5-1.5B-Instruct",
                  "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"},
}
DIMABSA_ID_HASH = "92f9debeb17ba951aea6840b34d8f24992f3558358f98fb0083bc91ea487e983"
ASTE_ID_HASH = "3960ab23eeb21c749a205f459971dfc6e185e39851ba973ea8538262daa56020"
ASTE_SOURCE_HASH = "413a3f655409af25bcb03a9499709925349fff3edbc3a7ca95fc6cdf788eeb92"


def select_role_cases(source_dir: Path = DIMABSA_DIR) -> tuple[list[dict], dict]:
    cases, meta = exp071.select_cases(source_dir)
    selected = []
    for polarity, predicate in (
        ("negative", lambda row: row["gold"][0] < 4.5),
        ("positive", lambda row: row["gold"][0] > 5.5),
    ):
        group = [case for case in cases if predicate(case)]
        group.sort(key=lambda case: hashlib.sha256(
            f"exp076-role-control|{case['case_id']}".encode("utf-8")
        ).hexdigest())
        if len(group) < 12:
            raise ValueError(f"Insufficient {polarity} DimABSA recipients")
        selected.extend(group[:12])
    selected.sort(key=lambda case: case["case_id"])
    ids_hash = sha256("\n".join(case["case_id"] for case in selected).encode())
    if len(selected) != 24 or ids_hash != DIMABSA_ID_HASH:
        raise ValueError(f"Experiment 076A recipient selection changed: {ids_hash}")
    return selected, {**meta, "n_selected_recipients": len(selected),
                      "recipient_ids_sha256": ids_hash,
                      "polarity_counts": {"negative": 12, "positive": 12}}


def select_aste_cases(path: Path = ASTE_FILE) -> tuple[list[dict], dict]:
    raw = path.read_bytes()
    if sha256(raw) != ASTE_SOURCE_HASH:
        raise ValueError("SemEval-2014 laptop triplet source hash changed")
    grouped = {"NEG": [], "POS": []}
    lines = raw.decode("utf-8").splitlines()
    for line_index, line in enumerate(lines):
        if "####" not in line:
            raise ValueError(f"Malformed ASTE line {line_index}")
        sentence, packed = line.split("####", 1)
        tokens = sentence.split()
        triples = ast.literal_eval(packed)
        if not isinstance(triples, (tuple, list)):
            raise ValueError(f"Malformed ASTE triples on line {line_index}")
        for triple_index, (aspect_span, _opinion_span, polarity) in enumerate(triples):
            if polarity in grouped:
                grouped[polarity].append({
                    "case_id": f"aste14lap:{line_index}:{triple_index}",
                    "line_index": line_index,
                    "triple_index": triple_index,
                    "text": sentence,
                    "tokens": tokens,
                    "aspect": " ".join(tokens[aspect_span[0]:aspect_span[-1] + 1]),
                    "triples": triples,
                    "polarity": polarity,
                })
                break
    chosen = []
    for polarity in ("NEG", "POS"):
        group = grouped[polarity]
        group.sort(key=lambda case: hashlib.sha256(
            f"exp076-aste14-fresh|{case['line_index']}".encode("utf-8")
        ).hexdigest())
        if len(group) < 12:
            raise ValueError(f"Insufficient ASTE {polarity} sentences")
        chosen.extend(group[:12])
    chosen.sort(key=lambda case: case["line_index"])
    identity = "\n".join(
        f"14lap:{case['line_index']}:{case['triple_index']}" for case in chosen
    ).encode()
    ids_hash = sha256(identity)
    if len(chosen) != 24 or len({case['line_index'] for case in chosen}) != 24:
        raise ValueError("ASTE cohort must contain 24 unique sentences")
    if ids_hash != ASTE_ID_HASH:
        raise ValueError(f"Experiment 076B recipient selection changed: {ids_hash}")
    return chosen, {"source_file_sha256": sha256(raw), "n_source_lines": len(lines),
                    "n_selected_recipients": 24, "recipient_ids_sha256": ids_hash,
                    "polarity_counts": {"negative": 12, "positive": 12}}


def mask_aste_opinions(case: dict) -> str:
    tokens = list(case["tokens"])
    spans = set()
    for aspect_span, opinion_span, _polarity in case["triples"]:
        if not opinion_span:
            continue
        start, end = int(opinion_span[0]), int(opinion_span[-1])
        if start < 0 or end >= len(tokens) or end < start:
            raise ValueError(f"Invalid ASTE opinion span in {case['case_id']}")
        spans.add((start, end))
    occupied = set()
    for start, end in spans:
        indexes = set(range(start, end + 1))
        if occupied & indexes:
            raise ValueError(f"Overlapping ASTE opinion spans in {case['case_id']}")
        occupied.update(indexes)
    for start, end in sorted(spans, reverse=True):
        tokens[start:end + 1] = ["[MASKED]"]
    return " ".join(tokens)


def affect_context(text: str, aspect: str) -> str:
    return (
        "Estimate the author's expressed affect toward the named aspect. "
        "Valence runs from 1 (strongly negative) to 9 (strongly positive). "
        "Arousal runs from 1 (very calm) to 9 (very activated or intense). "
        "Use only the supplied evidence.\n"
        f"Review text: {text}\nTarget aspect: {aspect}"
    )


FINAL_REQUEST = (
    "Use the fixed coordinate supplied earlier in this conversation as given. "
    "Estimate only the other affect coordinate using the review evidence. "
    "Return exactly one JSON object with one numeric key for that other coordinate, "
    "from 1.0 to 9.0 in increments of 0.1, with exactly one decimal place."
)


def build_jobs(role_cases: list[dict], aste_cases: list[dict]) -> list[dict]:
    jobs = []
    for case in role_cases:
        target = case["row"]["Triplet"][case["target_index"]]
        visible = exp049.mask_opinions(case["row"])
        context = affect_context(visible, target["Aspect"])
        for order in ORDERS:
            anchor_axis = "valence" if order == "valence_first" else "arousal"
            target_axis = "arousal" if order == "valence_first" else "valence"
            for location in LOCATIONS:
                for forced in FORCED_VALUES:
                    anchor = f'{{"{anchor_axis}": {forced:.1f}}}'
                    if location == "prior_assistant":
                        messages = [
                            {"role": "system", "content": exp071.SYSTEM_PROMPT},
                            {"role": "user", "content": context},
                            {"role": "assistant", "content": anchor},
                            {"role": "user", "content": FINAL_REQUEST},
                        ]
                    else:
                        messages = [
                            {"role": "system", "content": exp071.SYSTEM_PROMPT},
                            {"role": "user", "content": context + "\nFixed coordinate: " + anchor},
                            {"role": "assistant", "content": "Acknowledged. I will use the fixed coordinate as supplied."},
                            {"role": "user", "content": FINAL_REQUEST},
                        ]
                    jobs.append({
                        "study": "role_control", "case_id": case["case_id"],
                        "order": order, "anchor_axis": anchor_axis,
                        "target_axis": target_axis, "anchor_location": location,
                        "forced_value": forced, "gold": case["gold"],
                        "messages": messages,
                        "assistant_prefix": f'{{"{target_axis}": ',
                    })

    for case in aste_cases:
        context = affect_context(mask_aste_opinions(case), case["aspect"])
        for order in ORDERS:
            for forced in FORCED_VALUES:
                prefix = exp067.assistant_prefix(order, forced)
                jobs.append({
                    "study": "fresh_aste_prefix", "case_id": case["case_id"],
                    "order": order, "anchor_axis": "valence" if order == "valence_first" else "arousal",
                    "target_axis": "arousal" if order == "valence_first" else "valence",
                    "anchor_location": "assistant_prefix", "forced_value": forced,
                    "messages": [
                        {"role": "system", "content": exp071.SYSTEM_PROMPT},
                        {"role": "user", "content": context + "\n" +
                         'Return exactly one JSON object with numeric keys "valence" and '
                         '"arousal", both from 1.0 to 9.0 in increments of 0.1, with exactly one decimal place.'},
                    ],
                    "assistant_prefix": prefix,
                })
    if len(jobs) != 288:
        raise ValueError(f"Expected 288 conditions per model, got {len(jobs)}")
    keys = [(job["study"], job["case_id"], job["order"], job["anchor_location"], job["forced_value"])
            for job in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate Experiment 076 job")
    return sorted(jobs, key=lambda job: (job["study"], job["order"],
                                         job["anchor_location"], job["forced_value"],
                                         job["case_id"]))


def render(tokenizer, job: dict) -> tuple[str, list[int]]:
    text = tokenizer.apply_chat_template(job["messages"], tokenize=False,
                                         add_generation_prompt=True)
    text += job["assistant_prefix"]
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text, ids


def parse_target(job: dict, raw: str) -> float | None:
    if job["study"] == "fresh_aste_prefix":
        def reject_duplicate_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate JSON key: {key}")
                result[key] = value
            return result

        try:
            pair = json.loads(job["assistant_prefix"] + raw,
                              object_pairs_hook=reject_duplicate_keys)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(pair, dict) or set(pair) != {"valence", "arousal"}:
            return None
        anchor = pair[job["anchor_axis"]]
        target = pair[job["target_axis"]]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not 1 <= value <= 9 for value in (anchor, target)):
            return None
        if float(anchor) != float(job["forced_value"]):
            return None
        return float(target)
    try:
        decoded = json.loads(job["assistant_prefix"] + raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict) or set(decoded) != {job["target_axis"]}:
        return None
    score = decoded[job["target_axis"]]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 1 <= score <= 9:
        return None
    return float(score)


def run(source_dir: Path = DIMABSA_DIR, aste_file: Path = ASTE_FILE,
        output: Path = OUT, manifest_path: Path = MANIFEST,
        device: str = "mps", preflight_only: bool = False) -> dict:
    if sha256(PROTOCOL.read_bytes()) != PROTOCOL_SHA256:
        raise ValueError("Experiment 076 protocol hash mismatch; freeze the hash before inference")
    role_cases, role_meta = select_role_cases(source_dir)
    aste_cases, aste_meta = select_aste_cases(aste_file)
    jobs = build_jobs(role_cases, aste_cases)
    token_metadata = {}
    for model_key, config in MODEL_CONFIGS.items():
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"], revision=config["revision"], local_files_only=True
        )
        rendered = [render(tokenizer, job) for job in jobs]
        if not all(ids for _text, ids in rendered):
            raise ValueError(f"Empty rendered input for {model_key}")
        for text, ids in rendered:
            exp074.verify_token_boundary(tokenizer, text, ids)
        token_metadata[model_key] = {
            "n_jobs_per_model": len(jobs),
            "candidate_path_validation": "all 171 numeric-plus-brace paths pass for every context",
            "prefix_length_range": [min(len(ids) for _text, ids in rendered),
                                    max(len(ids) for _text, ids in rendered)],
        }
    meta = {"experiment": "076-schema-corpus-controls", "protocol_sha256": PROTOCOL_SHA256,
            "role_control": role_meta, "fresh_aste": aste_meta,
            "n_contexts_per_model": len(jobs), "n_models": len(MODEL_CONFIGS),
            "n_total_model_contexts": len(jobs) * len(MODEL_CONFIGS),
            "models": MODEL_CONFIGS, "tokenizer_metadata": token_metadata}
    if preflight_only:
        return meta

    old = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            key = (row["model_key"], row["study"], row["case_id"], row["order"],
                   row["anchor_location"], float(row["forced_value"]))
            if key in old:
                raise ValueError(f"Duplicate Experiment 076 row {key}")
            old[key] = row
    expected = {(model, job["study"], job["case_id"], job["order"],
                 job["anchor_location"], float(job["forced_value"]))
                for model in MODEL_CONFIGS for job in jobs}
    if not set(old).issubset(expected):
        raise ValueError("Resume artifact has rows outside the registered design")
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    invalid = {}
    for model_key, config in MODEL_CONFIGS.items():
        remaining = [job for job in jobs if (model_key, job["study"], job["case_id"],
                     job["order"], job["anchor_location"], float(job["forced_value"])) not in old]
        if not remaining:
            continue
        model, tokenizer, actual_device = load_target(config, device, offline=True)
        set_seed(SEED)
        with output.open("a", encoding="utf-8") as stream:
            for index, job in enumerate(remaining, start=1):
                rendered_text, input_ids = render(tokenizer, job)
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
                    "model_key": model_key, "model": config["model"],
                    "model_revision": config["revision"], "study": job["study"],
                    "case_id": job["case_id"], "order": job["order"],
                    "anchor_axis": job["anchor_axis"], "target_axis": job["target_axis"],
                    "anchor_location": job["anchor_location"],
                    "forced_value": job["forced_value"],
                    "gold": job.get("gold"), **forms,
                    "greedy_raw_continuation": raw,
                    "greedy_target_score": parse_target(job, raw),
                    "rendered_prefix_sha256": hashlib.sha256(rendered_text.encode()).hexdigest(),
                }
                key = (model_key, job["study"], job["case_id"], job["order"],
                       job["anchor_location"], float(job["forced_value"]))
                old[key] = row
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if index % 8 == 0 or index == len(remaining):
                    print(f"076 {model_key} {index}/{len(remaining)} new", flush=True)
        invalid[model_key] = {}
        for study in ("role_control", "fresh_aste_prefix"):
            invalid[model_key][study] = {}
            for location in (LOCATIONS if study == "role_control" else ("assistant_prefix",)):
                invalid[model_key][study][location] = {}
                for order in ORDERS:
                    invalid[model_key][study][location][order] = {}
                    for value in FORCED_VALUES:
                        relevant = [old[(model_key, job["study"], job["case_id"], job["order"],
                                         job["anchor_location"], float(job["forced_value"]))]
                                    for job in jobs if job["study"] == study
                                    and job["anchor_location"] == location
                                    and job["order"] == order and job["forced_value"] == value]
                        invalid[model_key][study][location][order][str(value)] = sum(
                            row["greedy_target_score"] is None for row in relevant
                        )
        del model, tokenizer
        gc.collect()
        if actual_device == "mps":
            torch.mps.empty_cache()
    if len(old) != len(expected):
        raise ValueError(f"Incomplete Experiment 076 output: {len(old)} / {len(expected)}")
    result = {**meta, "device": device, "invalid_by_model": invalid,
              "elapsed_seconds_this_process_only": time.monotonic() - started,
              "output_sha256": sha256(output.read_bytes())}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DIMABSA_DIR)
    parser.add_argument("--aste-file", type=Path, default=ASTE_FILE)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--manifest", dest="manifest_path", type=Path, default=MANIFEST)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--preflight-only", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
