"""Independent TripR corpus test of short-prefix judge sensitivity."""

import argparse
import gc
import hashlib
import json
import warnings
from collections import defaultdict
from pathlib import Path

import torch
from analyze_prompt_format_score_generation import MODELS
from huggingface_hub import snapshot_download
from prompt_format_score_generation import CONFIGS, prompt_for
from run_cross_judge_length_robustness_032 import (
    BATCH_SIZE as JUDGE_BATCH_SIZE,
)
from run_cross_judge_length_robustness_032 import (
    DEVICE,
    PHI_ID,
    PHI_REVISION,
    judge_content,
    parse_label,
)
from run_laya_decision_audit_030 import (
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
from task_ladder import MODEL_SPECS
from transformers import AutoModelForCausalLM, AutoTokenizer
from tripr_layer_confirmation import DATA, SOURCE_REPOSITORY_REVISION, load_tripr

from latent_decisions.target import load_target

OUT = Path(".context/tripr-prefix-threshold-034")
PROTOCOL = Path("docs/experiments/034-tripr-prefix-threshold.md")
BUDGETS = (8, 12, 32)
GENERATION_BATCH_SIZE = 4


def generated_content(token_ids, eos_id):
    values = [int(token) for token in token_ids]
    if eos_id in values:
        index = values.index(eos_id)
        return values[:index], True
    return values, False


def generate_answers(model_name, stimuli):
    model, tokenizer, device = load_target(CONFIGS[model_name], "auto", offline=True)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_for(item, "open_question")}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for item in stimuli
    ]
    rows = []
    for start in range(0, len(prompts), GENERATION_BATCH_SIZE):
        batch = stimuli[start : start + GENERATION_BATCH_SIZE]
        encoded = tokenizer(
            prompts[start : start + GENERATION_BATCH_SIZE], padding=True, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for item, sequence in zip(batch, output[:, encoded.input_ids.shape[1] :].cpu().tolist()):
            ids, ended = generated_content(sequence, tokenizer.eos_token_id)
            rows.append(
                {
                    "model": model_name,
                    "id": item["id"],
                    "sentence_id": item["sentence_id"],
                    "category": item["category"],
                    "gold": item["label"],
                    "token_ids": ids,
                    "token_count": len(ids),
                    "ended_with_eos": ended,
                    "capped_at_32": not ended and len(ids) == 32,
                    "answer": tokenizer.decode(ids, skip_special_tokens=True).strip(),
                }
            )
        print(
            f"{model_name} TripR generations: {min(start + len(batch), len(stimuli))}/{len(stimuli)}",
            flush=True,
        )
    del model, tokenizer
    gc.collect()
    if str(device).startswith("mps"):
        torch.mps.empty_cache()
    return rows


def make_prefix_jobs(generations):
    tokenizers = {
        model_name: AutoTokenizer.from_pretrained(
            CONFIGS[model_name]["model"],
            revision=CONFIGS[model_name]["revision"],
            local_files_only=True,
        )
        for model_name in MODELS
    }
    jobs = []
    for row in generations:
        tokenizer = tokenizers[row["model"]]
        for budget in BUDGETS:
            ids = row["token_ids"][:budget]
            jobs.append(
                {
                    key: row[key]
                    for key in ("model", "id", "sentence_id", "category", "gold")
                }
                | {
                    "budget": budget,
                    "actual_tokens": len(ids),
                    "answer": tokenizer.decode(ids, skip_special_tokens=True).strip(),
                }
            )
    del tokenizers
    if len(jobs) != len(MODELS) * 385 * len(BUDGETS):
        raise ValueError("TripR prefix factorial is incomplete")
    return jobs


def run_laya(jobs):
    import laya

    path = snapshot_download(
        repo_id=LAYA_ID,
        revision=LAYA_REVISION,
        allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/**", "tokenizer/**"],
    )
    mappings = choice_key_maps(job["id"] for job in jobs)
    groups = defaultdict(list)
    for job in jobs:
        mapping = mappings[job["id"]]
        groups[tuple(mapping[key] for key in ("A", "B", "C", "D"))].append(job)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = laya.load(str(path), device=DEVICE)
    outcomes = []
    for signature, group in groups.items():
        mapping = dict(zip(("A", "B", "C", "D"), signature))
        results = agent.predict_batch(
            [{"aspect": row["category"], "text": row["answer"]} for row in group],
            build_question(mapping),
            batch_size=8,
            sort_by_length=True,
        )
        for row, result in zip(group, results):
            answer = result["answers"]["polarity"]
            reverse = {key: label for key, label in mappings[row["id"]].items()}
            outcomes.append(
                {
                    key: row[key]
                    for key in ("model", "id", "sentence_id", "category", "gold", "budget", "actual_tokens")
                }
                | {
                    "judge": "laya",
                    "label": reverse[answer["choice"]],
                    "probabilities": {
                        reverse[key]: probability
                        for key, probability in answer["probabilities"].items()
                    },
                }
            )
        print(f"TripR Laya decisions: {len(outcomes)}/{len(jobs)}", flush=True)
    del agent
    gc.collect()
    torch.mps.empty_cache()
    return outcomes, [str(item.message) for item in caught], hashlib.sha256(
        (Path(path) / "model.safetensors").read_bytes()
    ).hexdigest()


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
    for start in range(0, len(jobs), JUDGE_BATCH_SIZE):
        batch = jobs[start : start + JUDGE_BATCH_SIZE]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": judge_content(row)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        labels = tokenizer.batch_decode(
            generated[:, encoded.input_ids.shape[1] :], skip_special_tokens=True
        )
        outcomes.extend(
            {
                key: row[key]
                for key in ("model", "id", "sentence_id", "category", "gold", "budget", "actual_tokens")
            }
            | {"judge": judge, "label": parse_label(label)}
            for row, label in zip(batch, labels)
        )
        print(
            f"TripR {judge}: {min(start + len(batch), len(jobs))}/{len(jobs)}", flush=True
        )
    del model, tokenizer
    gc.collect()
    torch.mps.empty_cache()
    return outcomes


def run(output=OUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Registered experiment 034 requires local MPS")
    stimuli, _sentence_labels, conflicts = load_tripr(DATA)
    if (len(stimuli), len({row["sentence_id"] for row in stimuli}), conflicts) != (385, 187, 29):
        raise ValueError("Frozen TripR source filter changed")
    generations = []
    for model_name in MODELS:
        generations.extend(generate_answers(model_name, stimuli))
    if len(generations) != len(MODELS) * 385:
        raise ValueError("Incomplete TripR generation factorial")
    jobs = make_prefix_jobs(generations)
    laya_rows, laya_warnings, laya_hash = run_laya(jobs)
    qwen_rows = run_binary_judge("qwen2.5-3b", jobs)
    phi_rows = run_binary_judge("phi3-mini", jobs)
    outcomes = laya_rows + qwen_rows + phi_rows
    expected = len(MODELS) * 385 * len(BUDGETS) * 3
    if len(outcomes) != expected:
        raise ValueError(f"Expected {expected} judge decisions, got {len(outcomes)}")
    output.mkdir(parents=True)
    generation_path = output / "private-generations.json"
    generation_path.write_text(json.dumps(generations, indent=2, ensure_ascii=False) + "\n")
    predictions_path = output / "predictions.json"
    predictions_path.write_text(json.dumps(outcomes, indent=2, sort_keys=True) + "\n")
    manifest = {
        "experiment": "034",
        "dataset": "OD-TripR-2020Large manually annotated restaurant reviews",
        "dataset_repository": "https://github.com/ari-dasci/OD-TripR-2020Large",
        "dataset_repository_revision": SOURCE_REPOSITORY_REVISION,
        "dataset_license": "CC BY-SA 4.0",
        "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "source_filter": {"sentences": 187, "queries": 385, "polarity_conflicts": 29},
        "target_models": CONFIGS,
        "generation_budget": 32,
        "judged_prefix_budgets": BUDGETS,
        "device": DEVICE,
        "generation_batch_size": GENERATION_BATCH_SIZE,
        "judge_batch_size": JUDGE_BATCH_SIZE,
        "decoding": {"do_sample": False},
        "laya": {
            "package": "0.3.20",
            "model_id": LAYA_ID,
            "revision": LAYA_REVISION,
            "weights_sha256": laya_hash,
            "choice_map_seed": MAP_SEED,
            "runtime_warnings": laya_warnings,
        },
        "qwen_model": MODEL_SPECS["qwen-3b"],
        "phi_model": {"model": PHI_ID, "revision": PHI_REVISION},
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "private_generations_sha256": hashlib.sha256(generation_path.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "n_generations": len(generations),
        "n_prefixes_per_engine": len(jobs),
        "n_decisions": len(outcomes),
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote private TripR text and label-only outcomes to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
