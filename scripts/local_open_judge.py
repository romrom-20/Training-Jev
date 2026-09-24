"""Run experiment 027's conditional open-answer judge phase."""

import gc
import hashlib
import json
from pathlib import Path

import torch
from analyze_prompt_format_score_generation import MODELS
from natural_aspect_selectivity import DATA, load_stimuli
from prompt_effect_forecast import sha
from prompt_format_score_generation import CONFIGS, prompt_for
from task_ladder import MODEL_SPECS

from latent_decisions.experiment import provenance
from latent_decisions.target import forward_last, load_target

PROTOCOL = Path("docs/experiments/027-local-judge-validation.md")
SCREEN = Path("results/local-judge-validation-v1/analysis.json")
FORMAT_BUNDLE = Path("results/prompt-format-score-generation-v1")
OUT = Path("results/local-open-judge-v1")
JUDGE = dict(MODEL_SPECS["qwen-3b"])


def parse_label(text):
    value = text.strip().lower().strip(".,!?;:\"'` ")
    return {"positive": 1, "negative": 0}.get(value)


def generate_open(model_name, stimuli):
    model, tokenizer, device = load_target(CONFIGS[model_name], "auto", offline=True)
    output = {}
    for offset in range(0, len(stimuli), 8):
        batch = stimuli[offset : offset + 8]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_for(row, "open_question")}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = forward_last(model, tokens)
            generated = model.generate(
                **tokens,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        pos = tokenizer.encode("positive", add_special_tokens=False)
        neg = tokenizer.encode("negative", add_special_tokens=False)
        if len(pos) != 1 or len(neg) != 1:
            raise ValueError("Frozen candidate labels must be single tokens")
        margins = (logits[:, pos[0]] - logits[:, neg[0]]).float().cpu().tolist()
        answers = tokenizer.batch_decode(
            generated[:, tokens.input_ids.shape[1] :], skip_special_tokens=True
        )
        for row, margin, answer in zip(batch, margins, answers):
            output[row["id"]] = {"answer": answer, "candidate_margin": float(margin)}
        print(
            f"{model_name} free completions: {min(offset + len(batch), len(stimuli))}/{len(stimuli)}",
            flush=True,
        )
    del model, tokenizer
    gc.collect()
    if str(device).startswith("mps"):
        torch.mps.empty_cache()
    return output


def summarize(rows):
    valid = [row for row in rows if row["judge_label"] is not None]
    n = len(rows)
    return {
        "n": n,
        "judge_parseability": len(valid) / n,
        "judge_accuracy_vs_review_gold": sum(row["judge_label"] == row["gold"] for row in valid)
        / n,
        "candidate_judge_agreement_among_parseable": (
            sum(row["candidate_pair_prediction"] == row["judge_label"] for row in valid)
            / len(valid)
            if valid
            else None
        ),
        "candidate_pair_accuracy": sum(
            row["candidate_pair_prediction"] == row["gold"] for row in rows
        )
        / n,
        "n_sentences": len({row["sentence_id"] for row in rows}),
    }


def run():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    screen = json.loads(SCREEN.read_text())
    if not screen["frozen_gate_passed"]:
        raise RuntimeError("Conditional phase forbidden: source-label gate failed")
    stimuli = load_stimuli(DATA)
    all_generated = {}
    for model_name in MODELS:
        all_generated[model_name] = generate_open(model_name, stimuli)
    # Reuse the judge once for all 699 items; answers remain in process memory.
    rows = []
    model, tokenizer, device = load_target(JUDGE, "auto", offline=True)
    todo = [
        (model_name, item, all_generated[model_name][item["id"]])
        for model_name in MODELS
        for item in stimuli
    ]
    for offset in range(0, len(todo), 8):
        batch = todo[offset : offset + 8]
        prompts = []
        for _model_name, item, generation in batch:
            content = (
                f"A model was asked about the sentiment of the {item['category']} in a restaurant review. "
                f"Its answer was: {generation['answer']}\n"
                "Based only on that answer, which sentiment did it communicate toward the named aspect? "
                "Reply with exactly one word: positive or negative."
            )
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            labels = model.generate(
                **tokens,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answers = tokenizer.batch_decode(
            labels[:, tokens.input_ids.shape[1] :], skip_special_tokens=True
        )
        for (model_name, item, generation), answer in zip(batch, answers):
            rows.append(
                {
                    "model": model_name,
                    "id": item["id"],
                    "sentence_id": item["sentence_id"],
                    "category": item["category"],
                    "gold": item["label"],
                    "judge_label": parse_label(answer),
                    "candidate_margin": generation["candidate_margin"],
                    "candidate_pair_prediction": int(generation["candidate_margin"] > 0),
                }
            )
        print(f"Qwen2.5-3B judge: {offset + len(batch)}/{len(todo)}", flush=True)

    summary = {name: summarize([row for row in rows if row["model"] == name]) for name in MODELS}
    report = {
        "experiment": 27,
        "phase": "conditional_open_answer_evaluation",
        "source_label_screen": {
            "accuracy": screen["accuracy"],
            "negative_recall": screen["negative_recall"],
            "gate_passed": screen["frozen_gate_passed"],
        },
        "models": summary,
        "judge": JUDGE,
        "target_models": CONFIGS,
        "target_device": "mps",
        "judge_device": str(device),
        "dataset_sha256": sha(DATA),
        "protocol_sha256": sha(PROTOCOL),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "provenance": provenance(),
        "outcomes": rows,
    }
    if len(rows) != 699 or len({f"{row['model']}|{row['id']}" for row in rows}) != 699:
        raise ValueError("Incomplete conditional judge factorial")
    if any(key in row for row in rows for key in ("answer", "text", "user", "review")):
        raise ValueError("Generated or source text leaked to result rows")
    OUT.mkdir(parents=True)
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    audit = {
        "checks": {
            "frozen_source_label_gate_passed": bool(screen["frozen_gate_passed"]),
            "699_unique_model_item_outcomes": len(rows) == 699
            and len({f"{row['model']}|{row['id']}" for row in rows}) == 699,
            "no_raw_source_or_generation_text": all(
                not ({"answer", "text", "user", "review"} & row.keys()) for row in rows
            ),
            "all_target_revisions_match_experiment_025": all(
                report["target_models"][name]["revision"]
                == json.loads((FORMAT_BUNDLE / f"{name}-manifest.json").read_text())["model"][
                    "revision"
                ]
                for name in MODELS
            ),
        }
    }
    audit["checks_passed"] = all(audit["checks"].values())
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()
