"""Paired output-format ablation for candidate-score/generated-label agreement."""

import argparse
import gc
import hashlib
import re
from pathlib import Path

import natural_aspect_selectivity as natural
import torch
from granite_tripr_residual_transfer import MODEL as GRANITE_MODEL
from prompt_effect_forecast import sha
from task_ladder import MODEL_SPECS

from latent_decisions.experiment import provenance, write_json
from latent_decisions.target import forward_last, load_target

PROTOCOL = Path("docs/experiments/025-prompt-format-score-generation.md")
ROOT = Path("runs/prompt-format-score-generation-v1")
MODELS = ("granite-3.1-2b", "qwen-1.5b", "smollm2-1.7b")
FORMATS = ("forced_choice", "open_question")
CONFIGS = {
    "granite-3.1-2b": GRANITE_MODEL,
    "qwen-1.5b": dict(MODEL_SPECS["qwen-1.5b"], name="qwen-1.5b"),
    "smollm2-1.7b": dict(MODEL_SPECS["smollm2-1.7b"], name="smollm2-1.7b"),
}


def prompt_for(item, prompt_format):
    if prompt_format == "forced_choice":
        return item["user"]
    review, question = item["user"].split("\n", maxsplit=1)
    question = question.split(" Reply with exactly one word:", maxsplit=1)[0]
    return f"{review}\n{question}"


def collect_model(name, stimuli, offline):
    output = ROOT / name
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    spec = CONFIGS[name]
    model, tokenizer, device = load_target(spec, "auto", offline)
    pos_id, neg_id = natural.candidate_ids(tokenizer)
    records = []
    for prompt_format in FORMATS:
        for offset in range(0, len(stimuli), 8):
            batch = stimuli[offset : offset + 8]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_for(item, prompt_format)}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for item in batch
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
            margins = (logits[:, pos_id] - logits[:, neg_id]).float().cpu().tolist()
            answers = tokenizer.batch_decode(
                generated[:, tokens.input_ids.shape[1] :], skip_special_tokens=True
            )
            for item, margin, answer_text in zip(batch, margins, answers):
                answer = answer_text.strip().lower().strip(".,!?;:\"'` ")
                polarity_words = set(re.findall(r"\b(?:positive|negative)\b", answer))
                parsed = (
                    1 if polarity_words == {"positive"} else
                    0 if polarity_words == {"negative"} else None
                )
                exact = answer in {"positive", "negative"}
                records.append(
                    {
                        "id": item["id"],
                        "sentence_id": item["sentence_id"],
                        "category": item["category"],
                        "target": item["target"],
                        "label": item["label"],
                        "format": prompt_format,
                        "candidate_margin": margin,
                        "candidate_pair_prediction": int(margin > 0),
                        "exact_one_word": int(exact),
                        "parseable_polarity": int(parsed is not None),
                        "generated_label": parsed,
                        "generated_strict_correct": int(parsed == item["label"]),
                    }
                )
        print(f"{name} {prompt_format}: {offset + len(batch)}/{len(stimuli)}", flush=True)
    if len(records) != len(stimuli) * len(FORMATS):
        raise ValueError(f"Incomplete format factorial for {name}")
    write_json(output / "outcomes.json", records)
    write_json(
        output / "manifest.json",
        {
            "model": spec,
            "device": str(device),
            "dtype": "float32",
            "dataset_xml_sha256": sha(natural.DATA),
            "n_prompts_per_format": len(stimuli),
            "n_sentences": len({item["sentence_id"] for item in stimuli}),
            "formats": FORMATS,
            "candidate_token_ids": {"positive": pos_id, "negative": neg_id},
            "protocol_sha256": sha(PROTOCOL),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "provenance": provenance(),
        },
    )
    del model
    gc.collect()
    if str(device).startswith("mps"):
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()
    stimuli = natural.load_stimuli(natural.DATA)
    if len(stimuli) != 233:
        raise ValueError("Frozen SemEval stimulus filter changed")
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        collect_model(name, stimuli, args.offline)


if __name__ == "__main__":
    main()
