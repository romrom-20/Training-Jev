"""Frozen Qwen2/Llama targets; explicit block hooks avoid hidden-state index ambiguity."""

import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def block_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def replace_block_tensor(output, value):
    return (value, *output[1:]) if isinstance(output, tuple) else value


def load_target(config, device="auto", offline=False):
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device not in ("cpu", "mps"):
        raise ValueError("This bounded prototype supports cpu or mps")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=offline, padding_side="left"
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            config["model"],
            revision=config["revision"],
            local_files_only=offline,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    if model.config.model_type not in ("qwen2", "llama") or not hasattr(model.model, "layers"):
        raise ValueError("Only Qwen2 and Llama decoder blocks are validated by this adapter")
    model.requires_grad_(False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def encode_rows(tokenizer, rows, device, max_length):
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in rows
    ]
    tokens = tokenizer(texts, padding=True, return_tensors="pt")
    if tokens.input_ids.shape[1] > max_length:
        raise ValueError("Prompt exceeds token budget; truncation could erase the treatment")
    return tokens.to(device)


def forward_last(model, tokens):
    # Apply the expensive vocabulary head only at the final token.
    output = model.model(**tokens, use_cache=False)
    return model.lm_head(output.last_hidden_state[:, -1, :]).float()


def collect(rows, config, device="auto", offline=False):
    start = time.perf_counter()
    model, tokenizer, device = load_target(config, device, offline)
    captures, handles = {}, []
    layers = config["layers"]
    if any(layer < 1 or layer > len(model.model.layers) for layer in layers):
        raise ValueError("Layers are one-indexed transformer block outputs")
    for layer in layers:

        def hook(module, inputs, output, layer=layer):
            captures[layer] = block_tensor(output)[:, -1, :].detach().float().cpu().numpy()

        handles.append(model.model.layers[layer - 1].register_forward_hook(hook))
    all_h, all_probs, all_top = [], [], []
    color_ids = [tokenizer.encode(word, add_special_tokens=False) for word in ("blue", "red")]
    if any(len(ids) != 1 for ids in color_ids):
        raise ValueError("Color readout requires single-token options")
    color_ids = [ids[0] for ids in color_ids]
    lengths = []
    try:
        with torch.inference_mode():
            for offset in range(0, len(rows), config["batch_size"]):
                batch = rows[offset : offset + config["batch_size"]]
                tokens = encode_rows(tokenizer, batch, device, config["max_length"])
                lengths.extend(tokens.attention_mask.sum(1).cpu().tolist())
                logits = forward_last(model, tokens)
                probs = logits.softmax(-1)
                all_h.append(np.stack([captures[layer] for layer in layers], axis=1))
                all_probs.append(probs[:, color_ids].cpu().numpy())
                all_top.extend(tokenizer.batch_decode(logits.argmax(-1)[:, None]))
                if offset % 64 == 0:
                    print(
                        f"Captured {min(offset + len(batch), len(rows))}/{len(rows)} prompts",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()
    from .data import QUESTIONS

    query_vectors = []
    # Frozen semantic encoder: mean final-normalized hidden states on the question alone.
    # No scenario or answer is present in a query embedding.
    with torch.inference_mode():
        for questions in QUESTIONS:
            tokens = tokenizer(list(questions), padding=True, return_tensors="pt").to(device)
            hidden = model.model(**tokens, use_cache=False).last_hidden_state
            mask = tokens.attention_mask[..., None]
            query_vectors.append(((hidden * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
    metadata = dict(
        device=device,
        dtype="float32",
        seconds=time.perf_counter() - start,
        max_tokens=max(lengths),
        mean_tokens=float(np.mean(lengths)),
        model_parameters=sum(p.numel() for p in model.parameters()),
        color_token_ids=color_ids,
        top_tokens=all_top,
    )
    return np.concatenate(all_h), np.stack(query_vectors), np.concatenate(all_probs), metadata
