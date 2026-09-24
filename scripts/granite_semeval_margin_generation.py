"""Measure candidate-margin/generation alignment on SemEval restaurant reviews."""

import argparse
import gc
import hashlib
from pathlib import Path

import natural_aspect_selectivity as natural
import torch
from granite_tripr_residual_transfer import MODEL
from prompt_effect_forecast import sha

from latent_decisions.experiment import provenance, write_json
from latent_decisions.target import forward_last, load_target

PROTOCOL = Path("docs/experiments/023-granite-score-generation-alignment.md")
ROOT = Path("runs/granite-semeval-margin-generation-v1")
DATA = natural.DATA


def collect(offline=False):
    if (ROOT / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite {ROOT}")
    stimuli = natural.load_stimuli(DATA)
    if (len(stimuli), len({item["sentence_id"] for item in stimuli})) != (233, 112):
        raise ValueError("Frozen SemEval aspect-query filter changed")
    model, tokenizer, device = load_target(MODEL, "auto", offline)
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
        tops = logits.argmax(-1).cpu().tolist()
        answers = tokenizer.batch_decode(
            generated[:, tokens.input_ids.shape[1] :], skip_special_tokens=True
        )
        for stimulus, margin, top, answer_text in zip(batch, margins, tops, answers):
            answer = answer_text.strip().strip(".,!?;:").lower()
            generated_label = (
                1 if answer == "positive" else 0 if answer == "negative" else None
            )
            records.append(
                {
                    "id": stimulus["id"],
                    "sentence_id": stimulus["sentence_id"],
                    "category": stimulus["category"],
                    "target": stimulus["target"],
                    "label": stimulus["label"],
                    "candidate_margin": margin,
                    "candidate_pair_prediction": int(margin > 0),
                    "all_vocabulary_top1_correct": int(
                        top == (pos_id if stimulus["label"] else neg_id)
                    ),
                    "generated_label": generated_label,
                    "generated_valid_exact_one_word": int(generated_label is not None),
                    "generated_strict_correct": int(generated_label == stimulus["label"]),
                }
            )
        print(f"Granite SemEval score/generation: {len(records)}/{len(stimuli)}", flush=True)
    if len({row["id"] for row in records}) != len(stimuli):
        raise ValueError("Duplicate or missing stimulus IDs")

    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / "outcomes.json", records)
    write_json(
        ROOT / "manifest.json",
        {
            "model": MODEL,
            "device": str(device),
            "dtype": "float32",
            "dataset": "SemEval-2014 Restaurants test gold subset",
            "dataset_repository": "https://github.com/HSLCY/ABSA-BERT-pair",
            "dataset_xml_sha256": sha(DATA),
            "n_prompts": len(stimuli),
            "n_sentences": len({item["sentence_id"] for item in stimuli}),
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
    args = parser.parse_args()
    collect(args.offline)


if __name__ == "__main__":
    main()
