"""Run Experiment 037 on naturally annotated opinion-span prefixes."""

import argparse
import ast
import gc
import hashlib
import json
import random
import time
import warnings
from collections import Counter
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from run_laya_decision_audit_030 import (
    MODEL_ID as LAYA_ID,
)
from run_laya_decision_audit_030 import (
    MODEL_REVISION as LAYA_REVISION,
)
from run_laya_decision_audit_030 import (
    build_question,
    choice_key_maps,
)
from run_tripr_prefix_threshold_034 import DEVICE, PHI_ID, PHI_REVISION
from task_ladder import MODEL_SPECS
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/037-natural-opinion-span-abstention.md")
PRIVATE = Path(".context/natural-opinion-span-abstention-037")
OUT = Path("results/natural-opinion-span-abstention-v1")
STIMULI_INDEX = Path("docs/experiments/037-stimuli.json")
SOURCE_DIR = Path(".context/aste14res")
SOURCE_REVISION = "d0df6600b259b6114de23cc5047c7e776cd89750"
SEED = 20260937
STIMULI_INDEX_SHA256 = "22dea039b2b235f8cfeb09b3f104d811f7097176f4a640d37b6fc3b9ef27d2d6"
BATCH_SIZE = 8
DATASETS = ("14res", "14lap", "15res", "16res")
SOURCE_FILES = {
    "14res": "test_triplets.txt",
    "14lap": "14lap_test_triplets.txt",
    "15res": "15res_test_triplets.txt",
    "16res": "16res_test_triplets.txt",
}
SOURCE_SHA256 = {
    "14res": "e3dcc31f520bc2e654fc6b2c588bb3e0c7f0f24b93a56dc585a3079480d871a5",
    "14lap": "413a3f655409af25bcb03a9499709925349fff3edbc3a7ca95fc6cdf788eeb92",
    "15res": "1079183e82e91cb2e6c93e79826c47230bca9152631fff5c37bac450062c0153",
    "16res": "de842428dc8bc11a4eee87b0ba382b0a59ef37b973a92829759caa8664208630",
}
CONDITIONS = ("before_opinion", "opinion_visible")
WRAPPERS = ("forced_binary", "abstention")
JUDGES = ("laya", "qwen2.5-3b", "phi3-mini")
PUBLIC_STIMULUS_KEYS = (
    "id",
    "cluster_id",
    "dataset",
    "line_index",
    "aspect_token_indices",
    "opinion_token_indices",
    "polarity",
)
OUTCOME_KEYS = (
    "id",
    "cluster_id",
    "dataset",
    "condition",
    "opinion_span_visible",
    "expected_decision",
    "polarity",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_contiguous(indices):
    values = sorted(indices)
    return bool(values) and values == list(range(values[0], values[-1] + 1))


def load_source_items():
    """Read local ignored source files, verify pins, and select balanced items."""
    by_dataset = {}
    for dataset in DATASETS:
        path = SOURCE_DIR / SOURCE_FILES[dataset]
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; download the pinned ASTE-Data-V2 test file locally"
            )
        if sha256(path) != SOURCE_SHA256[dataset]:
            raise ValueError(f"Source data hash mismatch: {dataset}")
        candidates = []
        for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if "####" not in line:
                raise ValueError(f"Malformed ASTE row: {dataset}:{line_index}")
            sentence, triplets_text = line.split("####", maxsplit=1)
            tokens = sentence.split()
            triplets = ast.literal_eval(triplets_text)
            if len(triplets) != 1:
                continue
            aspect_indices, opinion_indices, sentiment = triplets[0]
            if sentiment not in {"POS", "NEG"}:
                continue
            if not is_contiguous(aspect_indices) or not is_contiguous(opinion_indices):
                continue
            aspect_end = max(aspect_indices)
            opinion_start = min(opinion_indices)
            opinion_end = max(opinion_indices)
            if opinion_start < 2 or aspect_end >= opinion_start:
                continue
            if opinion_end >= len(tokens) or aspect_end >= len(tokens):
                raise ValueError(f"Out-of-bounds ASTE span: {dataset}:{line_index}")
            public = {
                "id": f"{dataset}-{line_index:04d}",
                "cluster_id": f"{dataset}-{line_index:04d}",
                "dataset": dataset,
                "line_index": line_index,
                "aspect_token_indices": list(aspect_indices),
                "opinion_token_indices": list(opinion_indices),
                "polarity": {"POS": "positive", "NEG": "negative"}[sentiment],
            }
            candidates.append(
                {
                    **public,
                    "_tokens": tokens,
                    "_aspect": " ".join(tokens[min(aspect_indices) : max(aspect_indices) + 1]),
                }
            )
        by_dataset[dataset] = candidates

    rng = random.Random(SEED)
    selected = []
    for dataset in DATASETS:
        candidates = by_dataset[dataset]
        negatives = [row for row in candidates if row["polarity"] == "negative"]
        positives = [row for row in candidates if row["polarity"] == "positive"]
        if len(positives) < len(negatives):
            raise ValueError(f"Cannot balance positive and negative items for {dataset}")
        picked_positives = rng.sample(positives, len(negatives))
        selected.extend(sorted(negatives + picked_positives, key=lambda row: row["line_index"]))

    if len(selected) != 234:
        raise ValueError(f"Frozen selection changed: expected 234 items, got {len(selected)}")
    if Counter(row["polarity"] for row in selected) != {"positive": 117, "negative": 117}:
        raise ValueError("Frozen polarity balance changed")
    return selected


def make_jobs(selected):
    jobs = []
    for item in selected:
        tokens = item["_tokens"]
        opinion_start = min(item["opinion_token_indices"])
        opinion_end = max(item["opinion_token_indices"]) + 1
        for condition, stop, visible in (
            ("before_opinion", opinion_start, False),
            ("opinion_visible", opinion_end, True),
        ):
            jobs.append(
                {
                    **{key: item[key] for key in PUBLIC_STIMULUS_KEYS},
                    "condition": condition,
                    "aspect": item["_aspect"],
                    "visible_text": " ".join(tokens[:stop]),
                    "opinion_span_visible": visible,
                    "expected_decision": item["polarity"] if visible else "insufficient",
                }
            )
    return jobs


def laya_map_key(row):
    return f"{row['aspect']}\t{row['visible_text']}"


def decode_laya_choice(choice, mapping):
    if choice not in mapping:
        raise ValueError(f"Laya returned unknown choice key: {choice}")
    return mapping[choice]


def run_laya(jobs):
    import laya

    checkpoint = snapshot_download(
        repo_id=LAYA_ID,
        revision=LAYA_REVISION,
        allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/**", "tokenizer/**"],
    )
    mappings = choice_key_maps((laya_map_key(row) for row in jobs), seed=SEED)
    groups = {}
    for row in jobs:
        signature = tuple(mappings[laya_map_key(row)][key] for key in "ABCD")
        groups.setdefault(signature, []).append(row)
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = laya.load(str(checkpoint), device=DEVICE)
    outcomes = []
    for signature, group in groups.items():
        mapping = dict(zip("ABCD", signature))
        results = agent.predict_batch(
            [{"aspect": row["aspect"], "text": row["visible_text"]} for row in group],
            build_question(mapping),
            batch_size=BATCH_SIZE,
            sort_by_length=True,
        )
        outcomes.extend(
            {key: row[key] for key in OUTCOME_KEYS}
            | {
                "judge": "laya",
                "wrapper": "four_way",
                "label": decode_laya_choice(result["answers"]["polarity"]["choice"], mapping),
            }
            for row, result in zip(group, results)
        )
        print(f"037 Laya: {len(outcomes)}/{len(jobs)}", flush=True)
    del agent
    gc.collect()
    torch.mps.empty_cache()
    return (
        outcomes,
        hashlib.sha256((Path(checkpoint) / "model.safetensors").read_bytes()).hexdigest(),
        [str(item.message) for item in caught],
        time.monotonic() - started,
    )


def prompt_content(row, wrapper):
    if wrapper == "forced_binary":
        policy = "Reply with exactly one word: positive or negative."
    elif wrapper == "abstention":
        policy = (
            "If the visible excerpt contains no explicit opinion expression supporting "
            "sentiment toward the named aspect, reply exactly insufficient. Otherwise "
            "reply with exactly one word, positive or negative."
        )
    else:
        raise ValueError(f"Unknown wrapper: {wrapper}")
    return (
        "Classify the sentiment expressed toward the named aspect in this review excerpt.\n"
        f"Aspect: {row['aspect']}\n"
        f"Visible review excerpt: {row['visible_text']}\n"
        f"{policy}"
    )


def parse_label(text, wrapper):
    value = text.strip().lower().strip(".,!?;:\"'` ")
    allowed = {"positive", "negative"}
    if wrapper == "abstention":
        allowed.add("insufficient")
    return value if value in allowed else None


def run_binary_judge(judge, jobs):
    if judge == "qwen2.5-3b":
        model, tokenizer, _ = load_target(dict(MODEL_SPECS["qwen-3b"]), DEVICE, offline=True)
        tokenizer.padding_side = "left"
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            PHI_ID, revision=PHI_REVISION, local_files_only=True, padding_side="left"
        )
        model = AutoModelForCausalLM.from_pretrained(
            PHI_ID,
            revision=PHI_REVISION,
            local_files_only=True,
            dtype=torch.float16,
            attn_implementation="eager",
        ).to(DEVICE).eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.requires_grad_(False)

    outcomes = []
    started = time.monotonic()
    for wrapper in WRAPPERS:
        wrapped_jobs = [(row, wrapper) for row in jobs]
        for start in range(0, len(wrapped_jobs), BATCH_SIZE):
            batch = wrapped_jobs[start : start + BATCH_SIZE]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_content(row, wrapper)}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row, _ in batch
            ]
            tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                generated = model.generate(
                    **tokens,
                    max_new_tokens=4,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            labels = tokenizer.batch_decode(
                generated[:, tokens.input_ids.shape[1] :], skip_special_tokens=True
            )
            outcomes.extend(
                {key: row[key] for key in OUTCOME_KEYS}
                | {"judge": judge, "wrapper": wrapper, "label": parse_label(label, wrapper)}
                for (row, _), label in zip(batch, labels)
            )
            print(
                f"037 {judge}/{wrapper}: {min(start + len(batch), len(wrapped_jobs))}/"
                f"{len(wrapped_jobs)}",
                flush=True,
            )
    del model, tokenizer
    gc.collect()
    torch.mps.empty_cache()
    return outcomes, time.monotonic() - started


def run(output=OUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Experiment 037 requires local MPS")
    if sha256(PROTOCOL) != expected_protocol_hash():
        raise ValueError("Protocol was edited after frozen stimulus selection")
    selected = load_source_items()
    jobs = make_jobs(selected)
    public_stimuli = [{key: row[key] for key in PUBLIC_STIMULUS_KEYS} for row in selected]
    if sha256(STIMULI_INDEX) != STIMULI_INDEX_SHA256:
        raise ValueError("Frozen selected-ID/span file changed")
    frozen_index = json.loads(STIMULI_INDEX.read_text())
    if frozen_index != public_stimuli:
        raise ValueError("Regenerated selection does not match the frozen ID/span file")
    if len(jobs) != 468:
        raise ValueError(f"Expected 468 paired-prefix jobs, got {len(jobs)}")

    laya_rows, laya_hash, laya_warnings, laya_seconds = run_laya(jobs)
    qwen_rows, qwen_seconds = run_binary_judge("qwen2.5-3b", jobs)
    phi_rows, phi_seconds = run_binary_judge("phi3-mini", jobs)
    outcomes = laya_rows + qwen_rows + phi_rows
    expected_outcomes = len(selected) * len(CONDITIONS) * (1 + 2 * len(WRAPPERS))
    if len(outcomes) != expected_outcomes or expected_outcomes != 2340:
        raise ValueError(f"Expected 2,340 judgments; got {len(outcomes)}")

    output.mkdir(parents=True)
    stimulus_path = output / "stimuli.json"
    outcome_path = output / "predictions.json"
    stimulus_path.write_text(json.dumps(public_stimuli, indent=2) + "\n")
    outcome_path.write_text(json.dumps(outcomes, indent=2) + "\n")
    manifest = {
        "experiment": "037",
        "protocol_sha256": sha256(PROTOCOL),
        "runner_sha256": sha256(Path(__file__)),
        "source_revision": SOURCE_REVISION,
        "source_files_sha256": SOURCE_SHA256,
        "stimuli_sha256": sha256(stimulus_path),
        "predictions_sha256": sha256(outcome_path),
        "stimuli": len(selected),
        "clusters": len({row["cluster_id"] for row in selected}),
        "outcomes": len(outcomes),
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "models": {
            "laya": {
                "id": LAYA_ID,
                "revision": LAYA_REVISION,
                "package": "laya==0.3.20",
                "weights_sha256": laya_hash,
                "seconds": laya_seconds,
            },
            "qwen2.5-3b": {
                "revision": MODEL_SPECS["qwen-3b"]["revision"],
                "seconds": qwen_seconds,
            },
            "phi3-mini": {"id": PHI_ID, "revision": PHI_REVISION, "seconds": phi_seconds},
        },
        "laya_warnings": laya_warnings,
        "label_only_outputs": True,
        "raw_review_text_published": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def expected_protocol_hash():
    return "820fa15fb757d75d2191e0dfea022364e371fe6a4eeed6c52317e3d33b6c346f"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
