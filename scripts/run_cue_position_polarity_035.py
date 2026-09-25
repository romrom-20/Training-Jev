"""Run the frozen controlled cue-position polarity test on local judges."""

import argparse
import gc
import hashlib
import json
import random
import time
import warnings
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from run_cross_judge_length_robustness_032 import (
    DEVICE,
    PHI_ID,
    PHI_REVISION,
    judge_content,
    parse_label,
)
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
from task_ladder import MODEL_SPECS
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/035-cue-position-polarity.md")
STIMULI_FILE = Path("docs/experiments/035-cue-position-stimuli.json")
STIMULI_SHA256 = "5b68f5d4dff58953536a41612b1933a9bc9639a13f242a93c35de72b42aa5054"
OUT = Path(".context/cue-position-polarity-035")
BUDGETS = (8, 12)
JUDGES = ("laya", "qwen2.5-3b", "phi3-mini")
ASPECTS = ("food", "service", "atmosphere")
FRAMES = (
    "We arrived on Tuesday with two friends.",
    "After lunch we spent another hour nearby.",
    "Before our meal the taxi stopped outside.",
    "Our group met downtown early on Saturday.",
    "We walked through the neighborhood before dinner.",
    "The visit began after our train arrived.",
    "We traveled here with family last weekend.",
    "Our visit was on a weekday afternoon.",
    "It took twenty minutes to get here.",
    "The first visit happened in early spring.",
    "Our reservation was made several days earlier.",
    "A friend joined us at the end.",
    "We arrived in the city that morning.",
    "It was our first visit to town.",
    "We returned here during our short vacation.",
    "The group came together after the conference.",
)
SEED = 20260935
GENERATION_BATCH_SIZE = 8


def make_stimuli():
    stimuli = []
    for frame_index, frame in enumerate(FRAMES):
        if len(frame.split()) != 7:
            raise ValueError(f"Frame {frame_index} is not exactly seven words: {frame!r}")
        for aspect in ASPECTS:
            for polarity in ("negative", "positive"):
                for form in ("direct", "negated"):
                    if form == "direct":
                        adjective = "good" if polarity == "positive" else "bad"
                        clause = f"The {aspect} was {adjective}."
                    else:
                        adjective = "bad" if polarity == "positive" else "good"
                        clause = f"The {aspect} was not {adjective}."
                    for position in ("early", "late"):
                        answer = (
                            f"{clause} {frame}"
                            if position == "early"
                            else f"{frame} {clause}"
                        )
                        stimulus_id = (
                            f"f{frame_index:02d}-{aspect}-{polarity}-{form}-{position}"
                        )
                        stimuli.append(
                            {
                                "id": stimulus_id,
                                "cluster_id": f"f{frame_index:02d}-{aspect}",
                                "frame_id": f"f{frame_index:02d}",
                                "aspect": aspect,
                                "polarity": polarity,
                                "gold": int(polarity == "positive"),
                                "form": form,
                                "position": position,
                                "clause": clause,
                                "frame": frame,
                                "answer": answer,
                            }
                        )
    if len(stimuli) != 384 or len({row["id"] for row in stimuli}) != 384:
        raise ValueError("Cue-position factorial is incomplete or has duplicate IDs")
    rng = random.Random(SEED)
    rng.shuffle(stimuli)
    return stimuli


def make_jobs(stimuli):
    jobs = []
    for row in stimuli:
        words = row["answer"].split()
        for budget in BUDGETS:
            jobs.append(
                {
                    key: row[key]
                    for key in (
                        "id",
                        "cluster_id",
                        "frame_id",
                        "aspect",
                        "polarity",
                        "gold",
                        "form",
                        "position",
                        "clause",
                        "frame",
                        "answer",
                    )
                }
                | {"budget": budget, "visible_text": " ".join(words[:budget])}
            )
    return jobs


def run_laya(jobs):
    import laya

    checkpoint = snapshot_download(
        repo_id=LAYA_ID,
        revision=LAYA_REVISION,
        allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/**", "tokenizer/**"],
    )
    mappings = choice_key_maps((row["id"] for row in jobs), seed=SEED)
    groups = {}
    for row in jobs:
        signature = tuple(mappings[row["id"]][key] for key in ("A", "B", "C", "D"))
        groups.setdefault(signature, []).append(row)
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = laya.load(str(checkpoint), device=DEVICE)
    outcomes = []
    for signature, group in groups.items():
        mapping = dict(zip(("A", "B", "C", "D"), signature))
        results = agent.predict_batch(
            [{"aspect": row["aspect"], "text": row["visible_text"]} for row in group],
            build_question(mapping),
            batch_size=8,
            sort_by_length=True,
        )
        reverse = {key: label for key, label in mapping.items()}
        outcomes.extend(
            {key: row[key] for key in ROW_KEYS}
            | {"judge": "laya", "label": reverse[result["answers"]["polarity"]["choice"]]}
            for row, result in zip(group, results)
        )
        print(f"035 Laya decisions: {len(outcomes)}/{len(jobs)}", flush=True)
    del agent
    gc.collect()
    torch.mps.empty_cache()
    return outcomes, hashlib.sha256((Path(checkpoint) / "model.safetensors").read_bytes()).hexdigest(), [
        str(item.message) for item in caught
    ], time.monotonic() - started


ROW_KEYS = (
    "id",
    "cluster_id",
    "frame_id",
    "aspect",
    "polarity",
    "gold",
    "form",
    "position",
    "budget",
    "visible_text",
)


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
    for start in range(0, len(jobs), GENERATION_BATCH_SIZE):
        batch = jobs[start : start + GENERATION_BATCH_SIZE]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": judge_content(row | {"category": row["aspect"], "answer": row["visible_text"]})}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
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
            {key: row[key] for key in ROW_KEYS}
            | {"judge": judge, "label": parse_label(label)}
            for row, label in zip(batch, labels)
        )
        print(f"035 {judge}: {min(start + len(batch), len(jobs))}/{len(jobs)}", flush=True)
    del model, tokenizer
    gc.collect()
    torch.mps.empty_cache()
    return outcomes, time.monotonic() - started


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output=OUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Experiment 035 requires local MPS")
    frozen_stimuli_bytes = STIMULI_FILE.read_bytes()
    if hashlib.sha256(frozen_stimuli_bytes).hexdigest() != STIMULI_SHA256:
        raise ValueError("Frozen Experiment 035 stimuli do not match the preregistered hash")
    stimuli = json.loads(frozen_stimuli_bytes)
    if stimuli != make_stimuli():
        raise ValueError("Frozen stimulus file and deterministic generator disagree")
    jobs = make_jobs(stimuli)
    protocol_hash = sha256(PROTOCOL)
    laya_rows, laya_hash, warnings_found, laya_seconds = run_laya(jobs)
    qwen_rows, qwen_seconds = run_binary_judge("qwen2.5-3b", jobs)
    phi_rows, phi_seconds = run_binary_judge("phi3-mini", jobs)
    outcomes = laya_rows + qwen_rows + phi_rows
    expected = len(stimuli) * len(BUDGETS) * len(JUDGES)
    if len(outcomes) != expected:
        raise ValueError(f"Expected {expected} judgments, got {len(outcomes)}")

    output.mkdir(parents=True)
    stimuli_path = output / "stimuli.json"
    outcomes_path = output / "outcomes.json"
    stimuli_path.write_text(json.dumps(stimuli, indent=2, ensure_ascii=False) + "\n")
    outcomes_path.write_text(json.dumps(outcomes, indent=2, ensure_ascii=False) + "\n")
    manifest = {
        "experiment": "035",
        "protocol_sha256": protocol_hash,
        "stimuli_sha256": sha256(stimuli_path),
        "outcomes_sha256": sha256(outcomes_path),
        "stimuli": len(stimuli),
        "scaffold_clusters": len({row["cluster_id"] for row in stimuli}),
        "judgments": expected,
        "device": DEVICE,
        "batch_size": GENERATION_BATCH_SIZE,
        "seed": SEED,
        "budgets_whitespace_words": list(BUDGETS),
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
            "phi3-mini": {
                "id": PHI_ID,
                "revision": PHI_REVISION,
                "seconds": phi_seconds,
            },
        },
        "warnings": warnings_found,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
