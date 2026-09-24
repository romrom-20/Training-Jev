"""Run experiment 028's independent-family judge check."""

import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from analyze_prompt_format_score_generation import MODELS
from natural_aspect_selectivity import DATA, load_stimuli
from prompt_effect_forecast import sha
from prompt_format_score_generation import CONFIGS, prompt_for
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_decisions.experiment import provenance
from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/028-independent-judge-check.md")
REFERENCE = Path("results/prompt-format-score-generation-v1")
QWEN_RESULTS = Path("results/local-open-judge-v1/analysis.json")
OUT = Path("results/independent-judge-check-v1")
JUDGE = "microsoft/Phi-3-mini-4k-instruct"
REVISION = "f39ac1d28e925b323eae81227eaba4464caced4e"
LABELS = {"positive": 1, "negative": 0}
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261028


def parse_label(text):
    return LABELS.get(text.strip().lower().strip(".,!?;:\"'` "))


def load_judge():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Experiment 028 requires the local MPS device")
    tokenizer = AutoTokenizer.from_pretrained(
        JUDGE, revision=REVISION, local_files_only=True, padding_side="left"
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            JUDGE,
            revision=REVISION,
            local_files_only=True,
            dtype=torch.float16,
            attn_implementation="eager",
        )
        .to("mps")
        .eval()
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.requires_grad_(False)
    return model, tokenizer


def judge_batch(model, tokenizer, contents):
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for content in contents
    ]
    tokens = tokenizer(prompts, padding=True, return_tensors="pt").to("mps")
    with torch.inference_mode():
        generated = model.generate(
            **tokens,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.batch_decode(
        generated[:, tokens.input_ids.shape[1] :], skip_special_tokens=True
    )


def screen(model, tokenizer, stimuli):
    rows = []
    for offset in range(0, len(stimuli), 4):
        batch = stimuli[offset : offset + 4]
        contents = [item["user"] for item in batch]
        answers = judge_batch(model, tokenizer, contents)
        for item, answer in zip(batch, answers):
            rows.append(
                {
                    "id": item["id"],
                    "sentence_id": item["sentence_id"],
                    "category": item["category"],
                    "gold": item["label"],
                    "judge_label": parse_label(answer),
                }
            )
        print(f"Phi-3 source screen: {len(rows)}/{len(stimuli)}", flush=True)
    n = len(rows)
    accuracy = sum(row["judge_label"] == row["gold"] for row in rows) / n
    negative_rows = [row for row in rows if row["gold"] == 0]
    negative_recall = sum(row["judge_label"] == 0 for row in negative_rows) / len(negative_rows)
    return rows, {
        "n": n,
        "parseability": sum(row["judge_label"] is not None for row in rows) / n,
        "accuracy": accuracy,
        "negative_recall": negative_recall,
        "gate_passed": accuracy >= 0.90 and negative_recall >= 0.80,
    }


def generate_open(model_name, stimuli):
    model, tokenizer, device = load_target(CONFIGS[model_name], "auto", offline=True)
    results = {}
    for offset in range(0, len(stimuli), 8):
        batch = stimuli[offset : offset + 8]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_for(item, "open_question")}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in batch
        ]
        tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(
                **tokens,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answers = tokenizer.batch_decode(
            generated[:, tokens.input_ids.shape[1] :], skip_special_tokens=True
        )
        results.update({item["id"]: answer for item, answer in zip(batch, answers)})
        print(
            f"{model_name} open answers: {min(offset + len(batch), len(stimuli))}/{len(stimuli)}",
            flush=True,
        )
    del model, tokenizer
    gc.collect()
    if str(device).startswith("mps"):
        torch.mps.empty_cache()
    return results


def load_reference_rows(name):
    with gzip.open(REFERENCE / f"{name}-outcomes.json.gz", "rt") as stream:
        rows = json.load(stream)
    return {row["id"]: row for row in rows if row["format"] == "open_question"}


def bootstrap_agreement(rows):
    valid = [row for row in rows if row["phi_label"] is not None and row["qwen_label"] is not None]
    groups = {}
    for row in valid:
        groups.setdefault(row["sentence_id"], []).append(row)
    sentence_ids = np.asarray(sorted(groups))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPS)
    for index in range(BOOTSTRAP_REPS):
        chosen = rng.choice(sentence_ids, size=len(sentence_ids), replace=True)
        sample = [row for sid in chosen for row in groups[sid]]
        estimates[index] = np.mean([row["phi_label"] == row["qwen_label"] for row in sample])
    point = np.mean([row["phi_label"] == row["qwen_label"] for row in valid])
    return {
        "n_both_parseable": len(valid),
        "agreement": float(point),
        "sentence_cluster_bootstrap_95_ci": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
    }


def run(output=OUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    stimuli = load_stimuli(DATA)
    if len(stimuli) != 233:
        raise ValueError("Frozen SemEval item filter changed")
    model, tokenizer = load_judge()
    screen_rows, screen_metrics = screen(model, tokenizer, stimuli)
    del model, tokenizer
    gc.collect()
    torch.mps.empty_cache()
    if screen_metrics["gate_passed"]:
        qwen = json.loads(QWEN_RESULTS.read_text())
        qwen_rows = {(row["model"], row["id"]): row["judge_label"] for row in qwen["outcomes"]}
        all_answers = {model_name: generate_open(model_name, stimuli) for model_name in MODELS}
        model, tokenizer = load_judge()
        outcomes = []
        for model_name in MODELS:
            answers = all_answers[model_name]
            contents = []
            for item in stimuli:
                contents.append(
                    f"A model was asked about the sentiment of the {item['category']} in a restaurant review. "
                    f"Its answer was: {answers[item['id']]}\n"
                    "Based only on that answer, which sentiment did it communicate toward the named aspect? "
                    "Reply with exactly one word: positive or negative."
                )
            labels = []
            for offset in range(0, len(contents), 8):
                labels.extend(judge_batch(model, tokenizer, contents[offset : offset + 8]))
                print(
                    f"Phi-3 judge {model_name}: {min(offset + 8, len(contents))}/{len(contents)}",
                    flush=True,
                )
            reference = load_reference_rows(model_name)
            for item, answer in zip(stimuli, labels):
                frozen = reference[item["id"]]
                outcomes.append(
                    {
                        "model": model_name,
                        "id": item["id"],
                        "sentence_id": item["sentence_id"],
                        "category": item["category"],
                        "gold": item["label"],
                        "phi_label": parse_label(answer),
                        "qwen_label": qwen_rows[(model_name, item["id"])],
                        "candidate_pair_prediction": frozen["candidate_pair_prediction"],
                        "candidate_margin": frozen["candidate_margin"],
                    }
                )
        del model, tokenizer
        per_model = {}
        for model_name in MODELS:
            rows = [row for row in outcomes if row["model"] == model_name]
            valid = [row for row in rows if row["phi_label"] is not None]
            per_model[model_name] = {
                "n": len(rows),
                "phi_parseability": len(valid) / len(rows),
                "phi_accuracy_vs_review_gold": sum(row["phi_label"] == row["gold"] for row in valid)
                / len(rows),
                "qwen_accuracy_vs_review_gold": sum(
                    row["qwen_label"] == row["gold"] for row in rows
                )
                / len(rows),
                "phi_qwen_agreement": bootstrap_agreement(rows),
            }
    else:
        outcomes = []
        per_model = {}
    gc.collect()
    torch.mps.empty_cache()
    report = {
        "experiment": 28,
        "phase": "conditional_answer_judging"
        if screen_metrics["gate_passed"]
        else "source_gate_failed",
        "judge": {"model": JUDGE, "revision": REVISION, "dtype": "float16", "device": "mps"},
        "source_label_screen": screen_metrics,
        "target_models": CONFIGS,
        "models": per_model,
        "dataset_sha256": sha(DATA),
        "protocol_sha256": sha(PROTOCOL),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "provenance": provenance(),
        "source_screen_outcomes": screen_rows,
        "outcomes": outcomes,
    }
    output.mkdir(parents=True)
    (output / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"source_label_screen": screen_metrics, "models": per_model}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    run(parser.parse_args().output)
