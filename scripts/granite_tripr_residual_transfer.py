"""Cross-family residual steering test on Granite 3.1 2B and TripR reviews."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import tripr_layer_confirmation as tripr
from prompt_effect_forecast import sha
from task_ladder import make_rows

from latent_decisions.experiment import provenance, write_json
from latent_decisions.steering import decompose_shared_residual
from latent_decisions.target import (
    block_tensor,
    forward_last,
    load_target,
    replace_block_tensor,
)

PROTOCOL = Path("docs/experiments/021-granite-tripr-residual-transfer.md")
MODEL = {
    "name": "granite-3.1-2b",
    "model": "ibm-granite/granite-3.1-2b-instruct",
    "revision": "bbc2aed595bd38bd770263dc3ab831db9794441d",
    "family": "Granite3.1",
    "layers": [23],
}
LAYER = 23
DOSE_FRACTION = 0.05
RANDOM_SEED = 20261001
CONTROL_SEEDS = tuple(range(20261101, 20261121))
CONDITIONS = ("native", "shared", "residual", "random_residual")


def training_rows():
    return [
        row
        for row in make_rows()
        if row["split"] == "train"
        and row["task"] == "mixed_review"
        and row["format"] == 0
    ]


def capture_training(root, offline):
    folder = root / "training"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        return
    rows = training_rows()
    model, tokenizer, device = load_target(MODEL, "auto", offline)
    cache = {}

    def make_hook(layer):
        def hook(_module, _inputs, output):
            cache[layer] = block_tensor(output)[:, -1].detach().float().cpu().numpy()

        return hook

    handles = [
        model.model.layers[layer - 1].register_forward_hook(make_hook(layer))
        for layer in MODEL["layers"]
    ]
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["user"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    h = []
    try:
        with torch.inference_mode():
            for offset in range(0, len(rows), 8):
                tokens = tokenizer(
                    texts[offset : offset + 8], padding=True, return_tensors="pt"
                ).to(device)
                forward_last(model, tokens)
                h.append(np.stack([cache[layer] for layer in MODEL["layers"]], axis=1))
                if offset % 128 == 0:
                    print(f"Granite train activations: {min(offset+8, len(rows))}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    np.savez_compressed(folder / "activations.npz", h=np.concatenate(h))
    write_json(folder / "dataset.json", rows)
    write_json(
        manifest_path,
        {
            "model": MODEL,
            "device": str(device),
            "dtype": "float32",
            "n_training_rows": len(rows),
            "layer": LAYER,
            "dataset_sha256": sha(folder / "dataset.json"),
            "activation_sha256": sha(folder / "activations.npz"),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "provenance": provenance(),
        },
    )
    del model
    gc.collect()
    if str(device).startswith("mps"):
        torch.mps.empty_cache()


def training_directions(root):
    source = root / "training"
    rows = json.loads((source / "dataset.json").read_text())
    with np.load(source / "activations.npz", allow_pickle=False) as saved:
        activations = saved["h"][:, 0]
    directions, norms = [], []
    for aspect in range(3):
        indices = [i for i, row in enumerate(rows) if row["active"] == aspect]
        positive = activations[[i for i in indices if rows[i]["label"] == 1]].mean(0)
        negative = activations[[i for i in indices if rows[i]["label"] == 0]].mean(0)
        vector = torch.from_numpy(positive - negative).float()
        directions.append(vector / vector.norm().clamp_min(1e-12))
        norms.extend(np.linalg.norm(activations[indices], axis=-1).tolist())
    return torch.stack(directions), DOSE_FRACTION * float(np.median(norms))


def infer_logits(
    model, tokenizer, device, stimuli, vectors, dose, layer, pos_id, neg_id, baseline_by_id
):
    cases = [
        (stimulus, condition, source, vectors[condition][source])
        for stimulus in stimuli
        for condition in CONDITIONS
        for source in range(3)
    ]
    block = model.model.layers[layer - 1]
    output = []
    for offset in range(0, len(cases), 8):
        batch = cases[offset : offset + 8]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": item[0]["user"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in batch
        ]
        tokens = tokenizer(texts, padding=True, return_tensors="pt").to(device)
        delta = torch.stack([item[3] for item in batch]).to(device) * dose

        def patch(_module, _inputs, result):
            value = block_tensor(result).clone()
            value[:, -1, :] += delta
            return replace_block_tensor(result, value)

        handle = block.register_forward_hook(patch)
        try:
            with torch.inference_mode():
                logits = forward_last(model, tokens)
        finally:
            handle.remove()
        margins = (logits[:, pos_id] - logits[:, neg_id]).float().cpu().tolist()
        top_ids = logits.argmax(-1).cpu().tolist()
        for (stimulus, condition, source, _vector), margin, top in zip(batch, margins, top_ids):
            output.append(
                {
                    "effect_id": f"{stimulus['id']}|{condition}|source={source}",
                    "id": stimulus["id"],
                    "sentence_id": stimulus["sentence_id"],
                    "target": stimulus["target"],
                    "source": source,
                    "condition": condition,
                    "label": stimulus["label"],
                    "margin_delta": margin - baseline_by_id[stimulus["id"]]["margin"],
                    "top_token_id": top,
                }
            )
        if (offset + len(batch)) % 256 == 0 or offset + len(batch) == len(cases):
            print(f"Granite score effects: {offset+len(batch)}/{len(cases)}", flush=True)
    return output


def infer_random_seed_controls(
    model, tokenizer, device, stimuli, native, dose, layer, seeds, pos_id, neg_id, baseline_by_id
):
    cases = []
    for seed in seeds:
        random = decompose_shared_residual(native, seed=seed)["random_residual"]
        for stimulus in stimuli:
            for source in range(3):
                cases.append((stimulus, seed, source, random[source]))
    block = model.model.layers[layer - 1]
    output = []
    for offset in range(0, len(cases), 8):
        batch = cases[offset : offset + 8]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": item[0]["user"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in batch
        ]
        tokens = tokenizer(texts, padding=True, return_tensors="pt").to(device)
        delta = torch.stack([item[3] for item in batch]).to(device) * dose

        def patch(_module, _inputs, result):
            value = block_tensor(result).clone()
            value[:, -1, :] += delta
            return replace_block_tensor(result, value)

        handle = block.register_forward_hook(patch)
        try:
            with torch.inference_mode():
                logits = forward_last(model, tokens)
        finally:
            handle.remove()
        margins = (logits[:, pos_id] - logits[:, neg_id]).float().cpu().tolist()
        for (stimulus, seed, source, _vector), margin in zip(batch, margins):
            output.append(
                {
                    "effect_id": f"{stimulus['id']}|seed={seed}|source={source}",
                    "id": stimulus["id"],
                    "sentence_id": stimulus["sentence_id"],
                    "target": stimulus["target"],
                    "source": source,
                    "seed": seed,
                    "condition": "random_seed_residual",
                    "label": stimulus["label"],
                    "margin_delta": margin - baseline_by_id[stimulus["id"]]["margin"],
                }
            )
        if (offset + len(batch)) % 256 == 0 or offset + len(batch) == len(cases):
            print(f"Granite random controls: {offset+len(batch)}/{len(cases)}", flush=True)
    return output


def generate_answers(model, tokenizer, device, stimuli, residual, random_residual, dose, layer, pos_id, neg_id):
    block = model.model.layers[layer - 1]
    cases = [(stim, "baseline", None) for stim in stimuli]
    for stim in stimuli:
        cases.extend(
            [
                (stim, "trained_residual", residual[stim["target"]]),
                (stim, "random_residual", random_residual[stim["target"]]),
            ]
        )
    outcomes = []
    for offset in range(0, len(cases), 4):
        batch = cases[offset : offset + 4]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": stim["user"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for stim, _condition, _vector in batch
        ]
        tokens = tokenizer(texts, padding=True, return_tensors="pt").to(device)
        active = [item[2] is not None for item in batch]
        delta = torch.stack(
            [item[2] if item[2] is not None else torch.zeros_like(residual[0]) for item in batch]
        ).to(device) * dose

        def patch(_module, _inputs, result):
            value = block_tensor(result).clone()
            value[:, -1, :] += delta
            return replace_block_tensor(result, value)

        handle = block.register_forward_hook(patch)
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **tokens,
                    max_new_tokens=4,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        finally:
            handle.remove()
        texts_out = tokenizer.batch_decode(
            generated[:, tokens.input_ids.shape[1] :], skip_special_tokens=True
        )
        for (stimulus, condition, _vector), text, _is_active in zip(batch, texts_out, active):
            answer = text.strip().strip(".,!?;:").lower()
            gold = "positive" if stimulus["label"] else "negative"
            outcomes.append(
                {
                    "effect_id": f"{stimulus['id']}|{condition}",
                    "id": stimulus["id"],
                    "sentence_id": stimulus["sentence_id"],
                    "target": stimulus["target"],
                    "label": stimulus["label"],
                    "condition": condition,
                    "valid_exact_one_word": int(answer in {"positive", "negative"}),
                    "strict_correct": int(answer == gold),
                }
            )
        print(f"Granite generated answers: {min(offset+len(batch), len(cases))}/{len(cases)}", flush=True)
    return outcomes


def collect(root, data_path, offline, resume):
    output = root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        if resume:
            print("Granite evaluation already complete", flush=True)
            return
        raise FileExistsError(f"Refusing to overwrite {output}; use --resume")
    stimuli, labels, _all_conflicts = tripr.load_tripr(data_path)
    conflict_ids = {
        sid for sid, aspects in labels.items() if len(set(aspects.values())) > 1
    }
    conflict_stimuli = [row for row in stimuli if row["sentence_id"] in conflict_ids]
    if (len(stimuli), len(labels), len(conflict_ids), len(conflict_stimuli)) != (385, 187, 29, 63):
        raise ValueError("Frozen TripR filter or conflict subset changed")

    native, dose = training_directions(root)
    arms = decompose_shared_residual(native, seed=RANDOM_SEED)
    residual_norms = [float(value) for value in arms["residual"].norm(dim=1)]
    model, tokenizer, device = load_target(MODEL, "auto", offline)
    if len(model.model.layers) != 40:
        raise ValueError(f"Expected pinned 40-layer Granite model, found {len(model.model.layers)}")
    pos, neg = [tokenizer.encode(word, add_special_tokens=False) for word in ("positive", "negative")]
    if len(pos) != 1 or len(neg) != 1 or pos[0] == neg[0]:
        raise ValueError("Positive and negative labels must be distinct single tokens")
    pos_id, neg_id = pos[0], neg[0]

    baseline = []
    for offset in range(0, len(stimuli), 8):
        batch = stimuli[offset : offset + 8]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": item["user"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in batch
        ]
        tokens = tokenizer(texts, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = forward_last(model, tokens)
        margins = (logits[:, pos_id] - logits[:, neg_id]).float().cpu().tolist()
        top_ids = logits.argmax(-1).cpu().tolist()
        for stimulus, margin, top in zip(batch, margins, top_ids):
            correct_id = pos_id if stimulus["label"] else neg_id
            baseline.append(
                {
                    "id": stimulus["id"],
                    "sentence_id": stimulus["sentence_id"],
                    "target": stimulus["target"],
                    "label": stimulus["label"],
                    "margin": margin,
                    "strict_correct": int(top == correct_id),
                    "valid_polarity_token": int(top in (pos_id, neg_id)),
                }
            )
        print(f"Granite baseline: {min(offset+len(batch),len(stimuli))}/{len(stimuli)}", flush=True)

    baseline_by_id = {row["id"]: row for row in baseline}
    effects = infer_logits(
        model, tokenizer, device, stimuli, arms, dose, LAYER, pos_id, neg_id, baseline_by_id
    )
    random_controls = infer_random_seed_controls(
        model,
        tokenizer,
        device,
        conflict_stimuli,
        native,
        dose,
        LAYER,
        CONTROL_SEEDS,
        pos_id,
        neg_id,
        baseline_by_id,
    )
    gen_baseline = generate_answers(
        model,
        tokenizer,
        device,
        conflict_stimuli,
        arms["residual"],
        arms["random_residual"],
        dose,
        LAYER,
        pos_id,
        neg_id,
    )
    write_json(output / "baseline.json", baseline)
    write_json(output / "effects.json", effects)
    write_json(output / "random_seed_effects.json", random_controls)
    write_json(output / "generation.json", gen_baseline)
    source_manifest = json.loads((root / "training" / "manifest.json").read_text())
    write_json(
        output / "manifest.json",
        {
            "model": MODEL,
            "device": str(device),
            "dtype": "float32",
            "model_layers": len(model.model.layers),
            "layer": LAYER,
            "relative_depth": LAYER / len(model.model.layers),
            "dose_fraction_of_training_activation_norm": DOSE_FRACTION,
            "dose_l2": dose,
            "residual_l2_norms": residual_norms,
            "n_prompts": len(stimuli),
            "n_sentences": len(labels),
            "n_conflict_sentences": len(conflict_ids),
            "n_conflict_queries": len(conflict_stimuli),
            "control_seeds": CONTROL_SEEDS,
            "training_activation_sha256": source_manifest["activation_sha256"],
            "training_dataset_sha256": source_manifest["dataset_sha256"],
            "dataset_xml_sha256": sha(data_path),
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
    parser.add_argument("stage", choices=("train", "collect", "all"))
    parser.add_argument("--root", type=Path, default=Path("runs/granite-tripr-residual-transfer-v1"))
    parser.add_argument("--data", type=Path, default=tripr.DATA)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage in ("train", "all"):
        capture_training(args.root, args.offline)
    if args.stage in ("collect", "all"):
        collect(args.root, args.data, args.offline, args.resume)


if __name__ == "__main__":
    main()
