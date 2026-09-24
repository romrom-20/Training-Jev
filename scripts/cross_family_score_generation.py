"""Replicate candidate-score/generation alignment on Qwen and SmolLM2."""

import argparse
import gc
import hashlib
from pathlib import Path

import natural_aspect_selectivity as natural
import torch
from prompt_effect_forecast import sha
from task_ladder import MODEL_SPECS

from latent_decisions.experiment import provenance, write_json
from latent_decisions.target import forward_last, load_target

PROTOCOL = Path("docs/experiments/024-cross-family-score-generation.md")
ROOT = Path("runs/cross-family-score-generation-v1")
MODELS = ("qwen-1.5b", "smollm2-1.7b")


def run_model(name, stimuli, offline):
    spec = dict(MODEL_SPECS[name], name=name)
    output = ROOT / name
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device = load_target(spec, "auto", offline)
    pos_id, neg_id = natural.candidate_ids(tokenizer)
    records = []
    for offset in range(0, len(stimuli), 8):
        batch = stimuli[offset : offset + 8]
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
            logits = forward_last(model, tokens)
            generated = model.generate(
                **tokens,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        margins = (logits[:, pos_id] - logits[:, neg_id]).float().cpu().tolist()
        top_ids = logits.argmax(-1).cpu().tolist()
        answers = tokenizer.batch_decode(
            generated[:, tokens.input_ids.shape[1] :], skip_special_tokens=True
        )
        for item, margin, top_id, answer_text in zip(batch, margins, top_ids, answers):
            answer = answer_text.strip().strip(".,!?;:").lower()
            generated_label = 1 if answer == "positive" else 0 if answer == "negative" else None
            records.append(
                {
                    "id": item["id"],
                    "sentence_id": item["sentence_id"],
                    "category": item["category"],
                    "target": item["target"],
                    "label": item["label"],
                    "candidate_margin": margin,
                    "candidate_pair_prediction": int(margin > 0),
                    "all_vocabulary_top1_correct": int(
                        top_id == (pos_id if item["label"] else neg_id)
                    ),
                    "generated_label": generated_label,
                    "generated_valid_exact_one_word": int(generated_label is not None),
                    "generated_strict_correct": int(generated_label == item["label"]),
                }
            )
        print(f"{name}: {len(records)}/{len(stimuli)}", flush=True)
    if len(records) != 233 or len({row["id"] for row in records}) != 233:
        raise ValueError(f"Incomplete or duplicate prompts for {name}")
    write_json(output / "outcomes.json", records)
    write_json(
        output / "manifest.json",
        {
            "model": spec,
            "device": str(device),
            "dtype": "float32",
            "dataset": "SemEval-2014 Restaurants test gold subset",
            "dataset_xml_sha256": sha(natural.DATA),
            "n_prompts": len(stimuli),
            "n_sentences": len({row["sentence_id"] for row in stimuli}),
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
        raise ValueError("Frozen SemEval prompt filter changed")
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        run_model(name, stimuli, args.offline)


if __name__ == "__main__":
    main()
