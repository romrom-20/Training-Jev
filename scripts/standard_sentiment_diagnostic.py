"""Frozen standard-label sentiment capability diagnostic (protocol 006)."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from capability_diagnostic import REVIEWS

from latent_decisions.target import load_target

PROMPTS = (
    "Classify the sentiment of this review. Reply with exactly one word: positive or negative.\nReview: {review}",
    'Read this review and label its sentiment. Use only the word positive or negative.\n"{review}"',
    "Determine whether this review expresses positive or negative sentiment. Answer with one word only.\nText: {review}",
    "Review: {review}\nSentiment (answer exactly positive or negative):",
)


def run(model_config, output_root, offline):
    output = output_root / model_config["name"]
    output.mkdir(parents=True, exist_ok=False)
    rows = [
        {
            "id": f"review-{i}-format-{fmt}",
            "review": review,
            "label": label,
            "format": fmt,
            "answer": "positive" if label else "negative",
            "user": PROMPTS[fmt].format(review=review),
        }
        for i, (review, label) in enumerate(REVIEWS)
        for fmt in range(4)
    ]
    started = time.perf_counter()
    model, tokenizer, device = load_target(model_config, "auto", offline)
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": r["user"]}], tokenize=False, add_generation_prompt=True
        )
        for r in rows
    ]
    label_ids = [
        tokenizer.encode(label, add_special_tokens=False) for label in ("positive", "negative")
    ]
    if any(len(ids) != 1 for ids in label_ids):
        raise ValueError("Sentiment labels must be single tokens")
    label_ids = [ids[0] for ids in label_ids]
    predictions = []
    for offset in range(0, len(rows), 8):
        batch = rows[offset : offset + 8]
        tokens = tokenizer(
            texts[offset : offset + 8], padding=True, return_tensors="pt", truncation=False
        ).to(device)
        if tokens.input_ids.shape[1] > 256:
            raise ValueError("Prompt exceeded 256 tokens")
        with torch.inference_mode():
            logits = model(**tokens, use_cache=False).logits[:, -1, :].float()
            pair = logits[:, label_ids]
            conditional = pair.softmax(-1)
            mass = logits.softmax(-1)[:, label_ids].sum(-1)
            generated = model.generate(
                **tokens, do_sample=False, max_new_tokens=4, pad_token_id=tokenizer.pad_token_id
            )
        decoded = tokenizer.batch_decode(
            generated[:, tokens.input_ids.shape[1] :], skip_special_tokens=True
        )
        for i, row in enumerate(batch):
            response = decoded[i]
            first_word = (
                response.strip().split(maxsplit=1)[0].strip(".,:;!?\"'()[]{}").lower()
                if response.strip()
                else ""
            )
            predictions.append(
                {
                    **row,
                    "response": response,
                    "first_word": first_word,
                    "strict_correct": first_word == row["answer"],
                    "conditional_correct": (
                        "positive" if conditional[i, 0] >= conditional[i, 1] else "negative"
                    )
                    == row["answer"],
                    "conditional_p_positive": float(conditional[i, 0]),
                    "label_mass": float(mass[i]),
                }
            )
        print(
            f"{model_config['name']}: {min(offset + len(batch), len(rows))}/{len(rows)}", flush=True
        )
    cells = []
    for fmt in range(4):
        selected = [r for r in predictions if r["format"] == fmt]
        cells.append(
            {
                "format": fmt,
                "n": len(selected),
                "strict_accuracy": float(np.mean([r["strict_correct"] for r in selected])),
                "conditional_accuracy": float(
                    np.mean([r["conditional_correct"] for r in selected])
                ),
                "mean_label_mass": float(np.mean([r["label_mass"] for r in selected])),
            }
        )
    result = {
        "model": model_config,
        "device": device,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seconds": time.perf_counter() - started,
        "n": len(predictions),
        "cells": cells,
        "pooled_strict_accuracy": float(np.mean([r["strict_correct"] for r in predictions])),
        "pooled_conditional_accuracy": float(
            np.mean([r["conditional_correct"] for r in predictions])
        ),
        "continuation_gate": all(c["strict_accuracy"] >= 0.9 for c in cells),
        "protocol_sha256": hashlib.sha256(
            Path("docs/experiments/006-standard-sentiment-capability.md").read_bytes()
        ).hexdigest(),
        "rows": predictions,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/standard-sentiment-v1"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("qwen-0.5b", "qwen-1.5b"),
        default=("qwen-0.5b", "qwen-1.5b"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    models = {
        "qwen-0.5b": {
            "name": "qwen-0.5b",
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        },
        "qwen-1.5b": {
            "name": "qwen-1.5b",
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        },
    }
    for name in args.models:
        run(models[name], args.output, args.offline)


if __name__ == "__main__":
    main()
