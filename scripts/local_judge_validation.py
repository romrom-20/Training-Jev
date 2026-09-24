"""Run the frozen source-label screen for experiment 027."""

import gc
import hashlib
import json
from pathlib import Path

import torch
from natural_aspect_selectivity import DATA, load_stimuli
from prompt_effect_forecast import sha
from task_ladder import MODEL_SPECS

from latent_decisions.experiment import provenance
from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/027-local-judge-validation.md")
OUT = Path("results/local-judge-validation-v1")
MODEL = dict(MODEL_SPECS["qwen-3b"])
LABELS = {"positive": 1, "negative": 0}


def parse_label(text):
    value = text.strip().lower().strip(".,!?;:\"'` ")
    return LABELS.get(value)


def capture():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    stimuli = load_stimuli(DATA)
    if len(stimuli) != 233:
        raise ValueError("Frozen SemEval item filter changed")
    model, tokenizer, device = load_target(MODEL, "auto", offline=True)
    outcomes = []
    for offset in range(0, len(stimuli), 4):
        batch = stimuli[offset : offset + 4]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": item["user"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in batch
        ]
        tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(
                **tokens,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answers = tokenizer.batch_decode(
            generated[:, tokens.input_ids.shape[1] :], skip_special_tokens=True
        )
        for item, answer in zip(batch, answers):
            outcomes.append(
                {
                    "id": item["id"],
                    "sentence_id": item["sentence_id"],
                    "category": item["category"],
                    "gold": item["label"],
                    "judge_label": parse_label(answer),
                }
            )
        print(f"source-label screen: {len(outcomes)}/{len(stimuli)}", flush=True)

    valid = [row for row in outcomes if row["judge_label"] is not None]
    correct = [row for row in valid if row["judge_label"] == row["gold"]]
    negatives = [row for row in valid if row["gold"] == 0]
    negative_correct = [row for row in negatives if row["judge_label"] == 0]
    accuracy = len(correct) / len(outcomes)
    negative_recall = len(negative_correct) / sum(row["gold"] == 0 for row in outcomes)
    gate = accuracy >= 0.90 and negative_recall >= 0.80
    report = {
        "experiment": 27,
        "phase": "source_label_screen",
        "n": len(outcomes),
        "parseability": len(valid) / len(outcomes),
        "accuracy": accuracy,
        "negative_recall": negative_recall,
        "confusion_matrix_negative_positive": [
            [
                sum(row["gold"] == gold and row["judge_label"] == pred for row in outcomes)
                for pred in (0, 1)
            ]
            for gold in (0, 1)
        ],
        "frozen_gate_passed": gate,
        "gate": {"accuracy_at_least": 0.90, "negative_recall_at_least": 0.80},
        "model": MODEL,
        "device": str(device),
        "dtype": "float32",
        "dataset_sha256": sha(DATA),
        "protocol_sha256": sha(PROTOCOL),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "provenance": provenance(),
        "outcomes": outcomes,
    }
    if len(outcomes) != 233 or len({row["id"] for row in outcomes}) != 233:
        raise ValueError("Incomplete source-label screen")
    OUT.mkdir(parents=True)
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    del model
    gc.collect()
    if str(device).startswith("mps"):
        torch.mps.empty_cache()
    print(
        f"gate={gate} accuracy={accuracy:.3%} negative_recall={negative_recall:.3%} "
        f"parseability={len(valid) / len(outcomes):.3%}",
        flush=True,
    )


if __name__ == "__main__":
    capture()
