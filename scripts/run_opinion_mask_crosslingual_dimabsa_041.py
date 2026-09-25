"""Run the preregistered opinion-mask intervention on aligned DimABSA test items."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

import torch
from task_ladder import MODEL_SPECS
from transformers import set_seed

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/041-opinion-mask-crosslingual-dimabsa.md")
PROTOCOL_SHA256 = "0e7d9991834a0bf0dd3d1010d0eefc4bd11b98c153ef8bbaa027af9a989f7e40"
SOURCE_REVISION = "bdc93be1224106ae7d3eb95739c02a76ed4ae8a1"
SOURCE_BASE = f"https://raw.githubusercontent.com/DimABSA/DimABSA2026/{SOURCE_REVISION}/"
LANGS = ("rus", "ukr", "tat")
FILES = {
    "rus": "task-dataset/track_a/subtask_2/rus/rus_restaurant_test_task2.jsonl",
    "ukr": "task-dataset/track_a/subtask_2/ukr/ukr_restaurant_test_task2.jsonl",
    "tat": "task-dataset/track_a/subtask_2/tat/tat_restaurant_test_task2.jsonl",
}
SOURCE_SHA256 = {
    "rus": "912013b49db2bd387076f63ee9df016350180fbac621449121a27dc262d5459b",
    "ukr": "05fdf7e2dca235261060b785f969315385f962021f171abb85e45638bbadb036",
    "tat": "c4f1fb5c21f8f06f598c87e489e7adce1a953d4446ad14a60432ae2a13b85ce6",
}
SEED = 20260941
QUOTAS = {"neg": 55, "neu": 10, "pos": 55}
CONDITIONS = ("aspect_only", "aspect_opinion", "full_text", "opinion_masked")
OUT = Path(".context/exp041-private-predictions.jsonl")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_source(path: Path | None, lang: str) -> tuple[list[dict], str]:
    if path is not None:
        raw = path.read_bytes()
    else:
        request = urllib.request.Request(
            SOURCE_BASE + FILES[lang], headers={"User-Agent": "Training-Jev research runner"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
    digest = sha256(raw)
    if digest != SOURCE_SHA256[lang]:
        raise ValueError(f"{lang} source hash mismatch: {digest}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return rows, digest


def _eligible_target(rows: list[dict], target_index: int) -> bool:
    """Require exact unique spans and no overlap that would make opinion masking ambiguous."""
    triplets = rows[0]["Triplet"]
    target = triplets[target_index]
    aspect, opinion = target.get("Aspect"), target.get("Opinion")
    if aspect in (None, "NULL") or opinion in (None, "NULL"):
        return False
    for row in rows:
        text = row["Text"]
        item = row["Triplet"][target_index]
        aspect, opinion = item["Aspect"], item["Opinion"]
        if text.count(aspect) != 1 or text.count(opinion) != 1:
            return False
        spans: list[tuple[int, int]] = []
        for triplet in row["Triplet"]:
            phrase = triplet.get("Opinion")
            if phrase in (None, "NULL"):
                continue
            if text.count(phrase) != 1:
                return False
            span = (text.find(phrase), text.find(phrase) + len(phrase))
            if any(
                span[0] < end and start < span[1] and span != prior
                for prior in spans
                for start, end in [prior]
            ):
                return False
            if span not in spans:
                spans.append(span)
        aspect_span = (text.find(aspect), text.find(aspect) + len(aspect))
        if any(aspect_span[0] < end and start < aspect_span[1] for start, end in spans):
            return False
    return True


def select_cases(data: dict[str, list[dict]]) -> list[dict]:
    indexed = {
        lang: {row["ID"]: row for row in data[lang]}
        for lang in LANGS
    }
    common_ids = sorted(set.intersection(*(set(indexed[lang]) for lang in LANGS)))
    candidates = []
    for case_id in common_ids:
        rows = [indexed[lang][case_id] for lang in LANGS]
        lengths = {len(row["Triplet"]) for row in rows}
        if len(lengths) != 1:
            continue
        if not all(
            [item["VA"] for item in row["Triplet"]]
            == [item["VA"] for item in rows[0]["Triplet"]]
            for row in rows[1:]
        ):
            continue
        for target_index in range(len(rows[0]["Triplet"])):
            if not _eligible_target(rows, target_index):
                continue
            valence, arousal = map(float, rows[0]["Triplet"][target_index]["VA"].split("#"))
            bucket = "neg" if valence < 4.5 else "neu" if valence <= 5.5 else "pos"
            candidates.append(
                {
                    "case_id": case_id,
                    "target_index": target_index,
                    "gold": [valence, arousal],
                    "bucket": bucket,
                    "rows": {lang: indexed[lang][case_id] for lang in LANGS},
                }
            )
            break  # one target per aligned sentence cluster

    buckets: dict[str, list[dict]] = {key: [] for key in QUOTAS}
    for candidate in candidates:
        buckets[candidate["bucket"]].append(candidate)
    selected = []
    for bucket, quota in QUOTAS.items():
        ordered = sorted(
            buckets[bucket],
            key=lambda row: hashlib.sha256(
                f"exp041|{SEED}|{row['case_id']}|{row['target_index']}".encode()
            ).hexdigest(),
        )
        if len(ordered) < quota:
            raise ValueError(f"Only {len(ordered)} eligible {bucket} clusters; need {quota}")
        selected.extend(ordered[:quota])
    return sorted(selected, key=lambda row: row["case_id"])


def mask_opinions(row: dict) -> str:
    """Replace every distinct, non-NULL annotated opinion span in one sentence."""
    text = row["Text"]
    spans = set()
    for triplet in row["Triplet"]:
        phrase = triplet.get("Opinion")
        if phrase in (None, "NULL"):
            continue
        if text.count(phrase) != 1:
            raise ValueError(f"Opinion phrase is not unique in {row['ID']}")
        start = text.find(phrase)
        spans.add((start, start + len(phrase)))
    previous_end = -1
    for start, end in sorted(spans):
        if start < previous_end:
            raise ValueError(f"Overlapping opinion spans in {row['ID']}")
        previous_end = end
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + "[MASKED]" + text[end:]
    return text


def build_prompt(text: str, aspect: str, opinion: str | None) -> str:
    text_field = text if text is not None else "[NOT PROVIDED]"
    opinion_field = opinion if opinion is not None else "[NOT PROVIDED]"
    return (
        "Estimate the author's expressed affect toward the named aspect. "
        "Valence runs from 1 (strongly negative) to 9 (strongly positive). "
        "Arousal runs from 1 (very calm) to 9 (very activated or intense). "
        "Use only the supplied evidence. Return exactly one JSON object with numeric keys "
        '"valence" and "arousal", both from 1 to 9.\n'
        f"Review text: {text_field}\n"
        f"Target aspect: {aspect}\n"
        f"Target-associated opinion phrase: {opinion_field}"
    )


def make_jobs(cases: list[dict]) -> list[dict]:
    jobs = []
    for case in cases:
        for lang in LANGS:
            row = case["rows"][lang]
            target = row["Triplet"][case["target_index"]]
            text = row["Text"]
            aspect, opinion = target["Aspect"], target["Opinion"]
            values = {
                "aspect_only": (None, None),
                "aspect_opinion": (None, opinion),
                "full_text": (text, None),
                "opinion_masked": (mask_opinions(row), None),
            }
            for condition in CONDITIONS:
                visible_text, visible_opinion = values[condition]
                jobs.append(
                    {
                        "case_id": case["case_id"],
                        "lang": lang,
                        "condition": condition,
                        "gold": case["gold"],
                        "prompt": build_prompt(visible_text, aspect, visible_opinion),
                    }
                )
    return jobs


def parse_va(text: str) -> list[float] | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in ("```", "```json"):
            return None
        if lines[-1].strip() != "```":
            return None
        text = "\n".join(lines[1:-1]).strip()
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict) or not {"valence", "arousal"} <= decoded.keys():
        return None
    try:
        result = [float(decoded["valence"]), float(decoded["arousal"])]
    except (TypeError, ValueError):
        return None
    if any(value < 1 or value > 9 for value in result):
        return None
    return result


def run(jobs: list[dict], output: Path, device: str, batch_size: int) -> dict:
    config = dict(MODEL_SPECS["qwen-3b"])
    model, tokenizer, device = load_target(config, device, offline=True)
    tokenizer.padding_side = "left"
    set_seed(SEED)
    output.parent.mkdir(parents=True, exist_ok=True)
    prior = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            row["prediction"] = parse_va(row["raw"])
            prior[(row["case_id"], row["lang"], row["condition"])] = row
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prior.values())
        )
    remaining = [
        job
        for job in jobs
        if (job["case_id"], job["lang"], job["condition"]) not in prior
    ]
    started = time.monotonic()
    with output.open("a", encoding="utf-8") as stream, torch.inference_mode():
        for offset in range(0, len(remaining), batch_size):
            batch = remaining[offset : offset + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": job["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for job in batch
            ]
            inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            generated = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            decoded = tokenizer.batch_decode(
                generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            for job, answer in zip(batch, decoded):
                result = {
                    key: job[key]
                    for key in ("case_id", "lang", "condition", "gold")
                }
                result.update({"raw": answer, "prediction": parse_va(answer)})
                prior[(job["case_id"], job["lang"], job["condition"])] = result
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                f"041 Qwen: {min(offset + len(batch), len(remaining))}/{len(remaining)} new outputs; "
                f"total {len(prior)}/{len(jobs)}",
                flush=True,
            )
    elapsed = time.monotonic() - started
    del model, tokenizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    counts = Counter(row["condition"] for row in prior.values() if row["prediction"] is None)
    return {
        "device": device,
        "elapsed_seconds": elapsed,
        "output_count": len(prior),
        "invalid_by_condition": dict(counts),
        "output_sha256": sha256(output.read_bytes()),
        "model": config["model"],
        "model_revision": config["revision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, help="Use exact-hash local JSONL files")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    protocol_hash = sha256(PROTOCOL.read_bytes())
    if protocol_hash != PROTOCOL_SHA256:
        raise ValueError("Protocol hash is not frozen to the registered version")
    data = {}
    source_hashes = {}
    for lang in LANGS:
        local = args.source_dir / Path(FILES[lang]).name if args.source_dir else None
        data[lang], source_hashes[lang] = read_source(local, lang)
    cases = select_cases(data)
    jobs = make_jobs(cases)
    if len(cases) != sum(QUOTAS.values()) or len(jobs) != 1_440:
        raise ValueError(f"Unexpected sample/jobs: {len(cases)} cases, {len(jobs)} jobs")
    print(
        json.dumps(
            {
                "protocol_sha256": protocol_hash,
                "source_revision": SOURCE_REVISION,
                "source_sha256": source_hashes,
                "selected_clusters": len(cases),
                "sample_by_valence": dict(Counter(case["bucket"] for case in cases)),
                "judgments": len(jobs),
                "model": MODEL_SPECS["qwen-3b"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(json.dumps(run(jobs, args.output, args.device, args.batch_size), indent=2), flush=True)


if __name__ == "__main__":
    main()
