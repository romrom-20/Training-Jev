"""Paired 8-vs-32 token generation intervention with local Laya labels."""

import argparse
import gc
import hashlib
import json
import os
import warnings
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import torch
from analyze_prompt_format_score_generation import MODELS
from huggingface_hub import snapshot_download
from natural_aspect_selectivity import DATA, load_stimuli
from prompt_format_score_generation import CONFIGS, prompt_for
from run_laya_decision_audit_030 import (
    CHOICES,
    MAP_SEED,
    build_question,
    choice_key_maps,
)
from run_laya_decision_audit_030 import (
    MODEL_ID as LAYA_ID,
)
from run_laya_decision_audit_030 import (
    MODEL_REVISION as LAYA_REVISION,
)

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/031-answer-length-censoring.md")
OUT = Path(".context/length-censoring-031")
BUDGETS = (8, 32)
BATCH_SIZE = 4


def eos_content(token_ids, eos_id):
    values = [int(token) for token in token_ids]
    if eos_id in values:
        eos_position = values.index(eos_id)
        return values[:eos_position], True
    return values, False


def generate_pair(model_name, stimuli, device):
    model, tokenizer, actual_device = load_target(CONFIGS[model_name], device, offline=True)
    if str(actual_device) != device:
        raise RuntimeError(f"Expected requested device {device}, received {actual_device}")
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_for(item, "open_question")}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for item in stimuli
    ]
    records = {}
    short_ids = {}
    for budget in BUDGETS:
        for start in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[start : start + BATCH_SIZE]
            batch_items = stimuli[start : start + BATCH_SIZE]
            tokens = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **tokens,
                    max_new_tokens=budget,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[:, tokens.input_ids.shape[1] :].cpu().tolist()
            for item, sequence in zip(batch_items, new_tokens):
                content_ids, ended = eos_content(sequence, tokenizer.eos_token_id)
                key = item["id"]
                if budget == 32 and key in short_ids and short_ids[key]["capped"]:
                    if content_ids[:8] != short_ids[key]["token_ids"]:
                        raise RuntimeError(f"Capped answer was not an exact prefix for {model_name}/{key}")
                record = {
                    "model": model_name,
                    "id": key,
                    "sentence_id": item["sentence_id"],
                    "category": item["category"],
                    "gold": item["label"],
                    "budget": budget,
                    "token_count": len(content_ids),
                    "ended_with_eos": ended,
                    "capped": not ended and len(content_ids) == budget,
                    "token_ids": content_ids,
                    "answer": tokenizer.decode(content_ids, skip_special_tokens=True).strip(),
                }
                records[(key, budget)] = record
                if budget == 8:
                    short_ids[key] = {"token_ids": content_ids, "capped": record["capped"]}
            print(
                f"{model_name} budget={budget}: {min(start + len(batch_items), len(stimuli))}/{len(stimuli)}",
                flush=True,
            )
    del model, tokenizer
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return records


def label_with_laya(stimuli, generated, device):
    if version("laya") != "0.3.20":
        raise RuntimeError("Expected laya==0.3.20")
    path = snapshot_download(
        repo_id=LAYA_ID,
        revision=LAYA_REVISION,
        allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/**", "tokenizer/**"],
    )
    import laya

    mappings = choice_key_maps(item["id"] for item in stimuli)
    jobs = [
        {
            "model": model,
            "id": item["id"],
            "sentence_id": item["sentence_id"],
            "category": item["category"],
            "gold": item["label"],
            "budget": budget,
            "aspect": item["category"],
            "text": generated[(model, item["id"], budget)]["answer"],
        }
        for model in MODELS
        for budget in BUDGETS
        for item in stimuli
    ]
    groups = defaultdict(list)
    for job in jobs:
        mapping = mappings[job["id"]]
        groups[tuple(mapping[key] for key in ("A", "B", "C", "D"))].append(job)
    if not torch.backends.mps.is_available() and device == "mps":
        raise RuntimeError("MPS unavailable; explicitly select CPU")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = laya.load(str(path), device=device)
    runtime_warnings = [str(item.message) for item in caught]
    outcomes = []
    for signature, group in groups.items():
        mapping = dict(zip(("A", "B", "C", "D"), signature))
        result = agent.predict_batch(
            [{"aspect": job["aspect"], "text": job["text"]} for job in group],
            build_question(mapping),
            batch_size=8,
            sort_by_length=True,
        )
        for job, prediction in zip(group, result):
            answer = prediction["answers"]["polarity"]
            reverse = {key: label for key, label in mappings[job["id"]].items()}
            probabilities = {
                reverse[key]: value for key, value in answer["probabilities"].items()
            }
            outcomes.append(
                {
                    key: job[key]
                    for key in ("model", "id", "sentence_id", "category", "gold", "budget")
                }
                | {
                    "laya_label": reverse[answer["choice"]],
                    "laya_probabilities": probabilities,
                    "laya_confidence": answer.get("answer_confidence"),
                }
            )
        print(f"Laya decisions: {len(outcomes)}/{len(jobs)}", flush=True)
    del agent
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return outcomes, runtime_warnings, hashlib.sha256(
        (Path(path) / "model.safetensors").read_bytes()
    ).hexdigest()


def run(output=OUT, device="mps"):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    stimuli = load_stimuli(DATA)
    if len(stimuli) != 233:
        raise ValueError("Frozen SemEval item filter changed")
    generated = {}
    for model_name in MODELS:
        records = generate_pair(model_name, stimuli, device)
        generated.update(
            {(model_name, item_id, budget): value for (item_id, budget), value in records.items()}
        )
    expected = len(MODELS) * len(stimuli) * len(BUDGETS)
    if len(generated) != expected:
        raise RuntimeError(f"Expected {expected} generations, got {len(generated)}")
    outcomes, runtime_warnings, weights_sha = label_with_laya(stimuli, generated, device)
    if len(outcomes) != expected:
        raise RuntimeError(f"Expected {expected} Laya decisions, got {len(outcomes)}")
    output.mkdir(parents=True)
    private_path = output / "private-generations.json"
    private_path.write_text(json.dumps(list(generated.values()), indent=2, ensure_ascii=False) + "\n")
    predictions_path = output / "predictions.json"
    predictions_path.write_text(json.dumps(outcomes, indent=2, sort_keys=True) + "\n")
    manifest = {
        "experiment": "031",
        "target_models": CONFIGS,
        "device": device,
        "batch_size": BATCH_SIZE,
        "generation_budgets": BUDGETS,
        "decoding": {"do_sample": False},
        "laya": {
            "package": "0.3.20",
            "model_id": LAYA_ID,
            "revision": LAYA_REVISION,
            "weights_sha256": weights_sha,
            "choice_map_seed": MAP_SEED,
            "choices": list(CHOICES),
            "runtime_warnings": runtime_warnings,
        },
        "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "private_generation_sha256": hashlib.sha256(private_path.read_bytes()).hexdigest(),
        "n_items": len(stimuli),
        "n_generations": len(generated),
        "n_decisions": len(outcomes),
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote private generations and label-only results to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    run(args.output, device=args.device)


if __name__ == "__main__":
    main()
