"""Run the frozen, small sentiment capability diagnostic in protocol 005."""

import argparse
import hashlib
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from latent_decisions.target import load_target

REVIEWS = (
    ("The meal was delicious and the service was wonderful.", 1),
    ("I loved the comfortable room and friendly staff.", 1),
    ("The book was delightful, clever, and beautifully written.", 1),
    ("This phone is fast, reliable, and easy to use.", 1),
    ("The concert was exciting and the musicians were fantastic.", 1),
    ("A lovely experience from start to finish; I highly recommend it.", 1),
    ("The jacket fits perfectly and feels excellent quality.", 1),
    ("The garden looks beautiful and smells wonderful.", 1),
    ("The meal was cold, bland, and poorly prepared.", 0),
    ("I hated the noisy room and rude staff.", 0),
    ("The book was dull, confusing, and badly written.", 0),
    ("This phone is slow, unreliable, and difficult to use.", 0),
    ("The concert was boring and the musicians sounded awful.", 0),
    ("A terrible experience from start to finish; I cannot recommend it.", 0),
    ("The jacket fits badly and feels very cheap.", 0),
    ("The garden looks ugly and smells unpleasant.", 0),
)

PROMPTS = (
    "Classify the sentiment of this review as positive or negative. Reply with only A or B.\n"
    "A means positive; B means negative.\nReview: {review}",
    "Read the review and decide whether its sentiment is positive or negative. Output one letter only.\n"
    'Positive = A. Negative = B.\n"{review}"',
    "Choose the sentiment label for the following text. Answer only A or B.\n"
    "A: positive sentiment\nB: negative sentiment\nText: {review}",
    "Is the writer expressing a positive or negative opinion? Give just the matching code.\n"
    "Code A = positive; code B = negative. Review: {review}",
)


def rows():
    result = []
    for review_id, (review, label) in enumerate(REVIEWS):
        for phrasing, prompt in enumerate(PROMPTS):
            for mapping in (1, -1):
                positive, negative = ("A", "B") if mapping == 1 else ("B", "A")
                text = prompt.format(review=review)
                text = text.replace(
                    "A means positive; B means negative.",
                    f"{positive} means positive; {negative} means negative.",
                )
                text = text.replace(
                    "Positive = A. Negative = B.", f"Positive = {positive}. Negative = {negative}."
                )
                text = text.replace(
                    "A: positive sentiment", f"{positive}: positive sentiment"
                ).replace("B: negative sentiment", f"{negative}: negative sentiment")
                text = text.replace(
                    "Code A = positive; code B = negative.",
                    f"Code {positive} = positive; code {negative} = negative.",
                )
                result.append(
                    {
                        "id": f"r{review_id}-p{phrasing}-m{mapping}",
                        "review": review,
                        "label": label,
                        "phrasing": phrasing,
                        "mapping": mapping,
                        "answer": positive if label else negative,
                        "user": text,
                    }
                )
    return result


def run(model_config, output_root, offline):
    output = output_root / model_config["name"]
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model, tokenizer, device = load_target(model_config, "auto", offline)
    examples = rows()
    # Include any tokenizer-required assistant prefix, but no system instruction.
    chats = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["user"]}], tokenize=False, add_generation_prompt=True
        )
        for row in examples
    ]
    answer_ids = [tokenizer.encode(x, add_special_tokens=False) for x in ("A", "B")]
    if any(len(x) != 1 for x in answer_ids):
        raise ValueError("A/B choices must each be a single token")
    answer_ids = [x[0] for x in answer_ids]
    predictions = []
    for offset in range(0, len(examples), 8):
        batch_rows = examples[offset : offset + 8]
        batch_text = chats[offset : offset + 8]
        tokens = tokenizer(batch_text, padding=True, return_tensors="pt", truncation=False).to(
            device
        )
        if tokens.input_ids.shape[1] > 256:
            raise ValueError("Prompt exceeded the frozen 256-token maximum")
        with torch.inference_mode():
            logits = model(**tokens, use_cache=False).logits[:, -1, :].float()
            pair = logits[:, answer_ids]
            conditional = pair.softmax(-1)
            mass = logits.softmax(-1)[:, answer_ids].sum(-1)
            generated = model.generate(
                **tokens, do_sample=False, max_new_tokens=4, pad_token_id=tokenizer.pad_token_id
            )
        suffix = generated[:, tokens.input_ids.shape[1] :]
        decoded = tokenizer.batch_decode(suffix, skip_special_tokens=True)
        for i, row in enumerate(batch_rows):
            response = decoded[i]
            first = response.strip().split(maxsplit=1)[0] if response.strip() else ""
            predictions.append(
                {
                    **row,
                    "response": response,
                    "first_token_text": first,
                    "strict_correct": first == row["answer"],
                    "conditional_correct": ("A" if conditional[i, 0] >= conditional[i, 1] else "B")
                    == row["answer"],
                    "conditional_p_a": float(conditional[i, 0]),
                    "ab_mass": float(mass[i]),
                }
            )
        print(
            f"{model_config['name']}: {min(offset + len(batch_rows), len(examples))}/{len(examples)}",
            flush=True,
        )
    cells = []
    for phrasing, mapping in itertools.product(range(4), (1, -1)):
        selected = [r for r in predictions if r["phrasing"] == phrasing and r["mapping"] == mapping]
        cells.append(
            {
                "phrasing": phrasing,
                "mapping": mapping,
                "n": len(selected),
                "strict_accuracy": float(np.mean([r["strict_correct"] for r in selected])),
                "conditional_accuracy": float(
                    np.mean([r["conditional_correct"] for r in selected])
                ),
                "mean_ab_mass": float(np.mean([r["ab_mass"] for r in selected])),
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
        "capability_gate": all(c["strict_accuracy"] >= 0.9 for c in cells),
        "protocol_sha256": hashlib.sha256(
            Path("docs/experiments/005-capability-diagnostic.md").read_bytes()
        ).hexdigest(),
        "rows": predictions,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/capability-diagnostic-v1"))
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
    all_models = {
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
    for model_name in args.models:
        run(all_models[model_name], args.output, args.offline)


if __name__ == "__main__":
    main()
