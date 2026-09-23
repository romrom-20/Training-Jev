"""Frozen aspect-sentiment activation readout study (protocol 007)."""

import argparse
import hashlib
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

from latent_decisions.experiment import provenance, write_json
from latent_decisions.metrics import fit_temperature, metrics, sigmoid
from latent_decisions.probes import (
    BilinearProbe,
    IndependentProbe,
    Standardizer,
    fit_probe,
    seed_all,
)
from latent_decisions.target import block_tensor, load_target

ASPECTS = ("food", "service", "value")
LAYERS = (8, 16, 24)
MODEL = {
    "name": "qwen-1.5b",
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "layers": list(LAYERS),
}
SPLITS = {"train": 32, "validation": 8, "calibration": 8, "source_test": 24, "shift_test": 24}
FORMATS = (
    "Classify the sentiment of this review. Reply with exactly one word: positive or negative.\n"
    "Review: {review}\nQuestion: Does the reviewer feel positive about the {aspect}?",
    "Review text:\n{review}\n\nWas the {aspect} evaluated favorably or unfavorably? Use only positive or negative.",
    "The reviewer wrote: {review}\nWhat is their opinion of the {aspect}? Answer exactly positive or negative.",
    "{review}\nRate the sentiment about {aspect} with one label: positive or negative.",
)
TEXT_VARIANTS = {
    "food": (("The food was delicious and fresh.", "The meal tasted excellent and hearty."),
             ("The food was bland and stale.", "The meal tasted awful and cold.")),
    "service": (("The staff were friendly and attentive.", "The service was warm and helpful."),
                 ("The staff were rude and inattentive.", "The service was cold and unhelpful.")),
    "value": (("The price felt fair and reasonable.", "It was good value for the money."),
              ("The price felt unfair and excessive.", "It was poor value for the money.")),
}
QUESTION_TEXT = (
    "Does the reviewer feel positive about the food?",
    "Does the reviewer feel positive about the service?",
    "Does the reviewer feel positive about the value?",
)


def make_rows():
    rng = random.Random(20260923)
    rows = []
    for split, count in SPLITS.items():
        for group_no in range(count):
            group = f"{split}-{group_no:02d}"
            # Each group's clause phrasing and ordering are fixed across all eight
            # counterfactual settings, while disjoint groups vary wording.
            variant = group_no % 2
            order = list(range(3))
            rng.shuffle(order)
            for bits in itertools.product((0, 1), repeat=3):
                clauses = [TEXT_VARIANTS[aspect][bits[j]][variant] for j, aspect in enumerate(ASPECTS)]
                for fmt in (range(4) if split == "shift_test" else (0,)):
                    ordered = [clauses[j] for j in (order if fmt in (0, 1) else order[::-1])]
                    if fmt == 1:
                        review = "\n".join(ordered)
                    elif fmt == 2:
                        review = " ".join(ordered)
                    elif fmt == 3:
                        review = " | ".join(ordered)
                    else:
                        review = " ".join(ordered)
                    for active, aspect in enumerate(ASPECTS):
                        user = FORMATS[fmt].format(review=review, aspect=aspect)
                        rows.append({"id": f"{group}-b{''.join(map(str,bits))}-q{active}-f{fmt}",
                                     "group": group, "split": split, "group_no": group_no,
                                     "bits": list(bits), "active": active, "aspect": aspect,
                                     "label": bits[active], "format": fmt, "review": review, "user": user})
    return rows


def query_vectors(model, tokenizer, device):
    with torch.inference_mode():
        batch = tokenizer(list(QUESTION_TEXT), padding=True, return_tensors="pt").to(device)
        hidden = model.model(**batch, use_cache=False).last_hidden_state
        mask = batch.attention_mask[..., None]
        q = (hidden * mask).sum(1) / mask.sum(1)
    return q.float().cpu()


def collect(run, offline):
    rows = make_rows()
    write_json(run / "dataset.json", rows)
    model, tokenizer, device = load_target(MODEL, "auto", offline)
    q = query_vectors(model, tokenizer, device)
    captures, handles = {}, []
    for layer in LAYERS:
        def hook(module, inputs, output, layer=layer):
            captures[layer] = block_tensor(output)[:, -1, :].detach().float().cpu().numpy()
        handles.append(model.model.layers[layer - 1].register_forward_hook(hook))
    label_ids = [tokenizer.encode(x, add_special_tokens=False) for x in ("positive", "negative")]
    if any(len(x) != 1 for x in label_ids):
        raise ValueError("positive/negative output labels must each be one token")
    label_ids = [x[0] for x in label_ids]
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": r["user"]}], tokenize=False, add_generation_prompt=True) for r in rows]
    hs, pair_probs, label_masses, generated = [], [], [], []
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for offset in range(0, len(rows), 8):
                batch_rows = rows[offset:offset + 8]
                tokens = tokenizer(texts[offset:offset + 8], padding=True, return_tensors="pt", truncation=False).to(device)
                if tokens.input_ids.shape[1] > 256:
                    raise ValueError("Prompt exceeded 256 token cap")
                logits = model(**tokens, use_cache=False).logits[:, -1, :].float()
                pair = logits[:, label_ids]
                hs.append(np.stack([captures[layer] for layer in LAYERS], axis=1))
                pair_probs.append(pair.softmax(-1).cpu().numpy())
                label_masses.append(logits.softmax(-1)[:, label_ids].sum(-1).cpu().numpy())
                output = model.generate(**tokens, do_sample=False, max_new_tokens=4, pad_token_id=tokenizer.pad_token_id)
                decoded = tokenizer.batch_decode(output[:, tokens.input_ids.shape[1]:], skip_special_tokens=True)
                generated.extend(decoded)
                if offset % 192 == 0:
                    print(f"captured {min(offset + len(batch_rows), len(rows))}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    probs = np.concatenate(pair_probs)
    masses = np.concatenate(label_masses)
    for i, row in enumerate(rows):
        word = generated[i].strip().split(maxsplit=1)[0].strip(".,:;!?\"'()[]{}").lower() if generated[i].strip() else ""
        row["generated"] = generated[i]
        row["strict_correct"] = word == ("positive" if row["label"] else "negative")
        row["conditional_correct"] = (int(probs[i, 0] >= probs[i, 1]) == row["label"])
        row["conditional_p_positive"] = float(probs[i, 0])
        row["label_mass"] = float(masses[i])
    np.savez_compressed(run / "activations.npz", h=np.concatenate(hs), q=q.numpy(),
                        conditional_p_positive=probs[:, 0], label_mass=masses)
    write_json(run / "dataset.json", rows)
    capability = []
    for active, aspect in enumerate(ASPECTS):
        selected = [r for r in rows if r["split"] == "source_test" and r["active"] == active]
        capability.append({"aspect": aspect, "n": len(selected),
                           "strict_accuracy": float(np.mean([r["strict_correct"] for r in selected])),
                           "conditional_accuracy": float(np.mean([r["conditional_correct"] for r in selected]))})
    gate = all(c["strict_accuracy"] >= 0.9 for c in capability)
    write_json(run / "manifest.json", {"model": MODEL, "device": device, "capability": capability,
        "capability_gate_passed": gate, "capture_seconds": time.perf_counter() - started,
        "protocol_sha256": hashlib.sha256(Path("docs/experiments/007-aspect-sentiment-readout.md").read_bytes()).hexdigest(),
        "provenance": provenance()})
    del model
    if device == "mps":
        torch.mps.empty_cache()
    print(json.dumps({"capability": capability, "gate_passed": gate}, indent=2), flush=True)


def pack(h, q, rows, indices):
    return (h[indices], q[[rows[i]["active"] for i in indices]],
            torch.tensor([rows[i]["active"] for i in indices], dtype=torch.long),
            torch.tensor([rows[i]["label"] for i in indices], dtype=torch.float32))


def group_bootstrap_delta(y, p_base, p_centered, group_ids, reps=3000, seed=20260923):
    unique = np.unique(group_ids)
    rng = np.random.default_rng(seed)
    by_group = {g: np.where(group_ids == g)[0] for g in unique}
    deltas = []
    for _ in range(reps):
        sampled = rng.choice(unique, len(unique), replace=True)
        ix = np.concatenate([by_group[g] for g in sampled])
        deltas.append(np.mean((p_centered[ix] - y[ix]) ** 2) - np.mean((p_base[ix] - y[ix]) ** 2))
    return [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))]


def train(run):
    manifest = json.loads((run / "manifest.json").read_text())
    if not manifest["capability_gate_passed"]:
        print("Capability gate failed; stopping before probe training.", flush=True)
        return
    torch.set_num_threads(4)
    rows = json.loads((run / "dataset.json").read_text())
    with np.load(run / "activations.npz") as cache:
        h_all = torch.from_numpy(cache["h"].copy())
        q = torch.from_numpy(cache["q"].copy())
    q = q / q.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    test_ix = np.array([i for i, r in enumerate(rows) if r["split"] == "shift_test"])
    test_rows = [rows[i] for i in test_ix]
    y = np.asarray([r["label"] for r in test_rows])
    groups = np.asarray([r["group"] for r in test_rows])
    formats = np.asarray([r["format"] for r in test_rows])
    active = np.asarray([r["active"] for r in test_rows])
    logits_saved, keys, summaries = [], [], []
    for li, layer in enumerate(LAYERS):
        train_ix = np.array([i for i, r in enumerate(rows) if r["split"] == "train" and r["format"] == 0])
        val_ix = np.array([i for i, r in enumerate(rows) if r["split"] == "validation" and r["format"] == 0])
        cal_ix = np.array([i for i, r in enumerate(rows) if r["split"] == "calibration" and r["format"] == 0])
        norm = Standardizer().fit(h_all[train_ix, li])
        h = norm(h_all[:, li])
        train_data, val_data, cal_data = (pack(h, q, rows, ix) for ix in (train_ix, val_ix, cal_ix))
        eval_data = pack(h, q, rows, test_ix)
        for seed, method in itertools.product((0, 1, 2), ("bilinear", "independent")):
            seed_all(seed)
            probe = BilinearProbe(h.shape[-1], q.shape[-1], 4) if method == "bilinear" else IndependentProbe(h.shape[-1], tasks=3)
            fitted = fit_probe(probe, train_data, val_data, steps=300, lr=.01, weight_decay=.01, brier_weight=.1)
            with torch.no_grad():
                cal_logits = probe(*cal_data[:3]).numpy()
                eval_logits = probe(*eval_data[:3]).numpy()
            temperature = fit_temperature(cal_logits, cal_data[3].numpy())
            key = {"layer": layer, "seed": seed, "method": method, "temperature": temperature,
                   "n_train": len(train_ix), **fitted}
            keys.append(key)
            logits_saved.append(eval_logits)
            for fmt in (0, 1, 2, 3):
                for aspect in range(3):
                    mask = (formats == fmt) & (active == aspect)
                    summaries.append({**key, "format": fmt, "aspect": ASPECTS[aspect],
                                      "raw": metrics(y[mask], sigmoid(eval_logits[mask])),
                                      "calibrated": metrics(y[mask], sigmoid(eval_logits[mask] / temperature))})
            print(f"fit layer {layer}, {method}, seed {seed}", flush=True)
    logits_saved = np.stack(logits_saved)
    np.savez_compressed(run / "probe-scores.npz", logits=logits_saved, indices=test_ix)
    # TF-IDF text-only confound, trained on the same source prompts and labels.
    train_text_ix = [i for i, r in enumerate(rows) if r["split"] == "train" and r["format"] == 0]
    text_model = make_pipeline(FeatureUnion([("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                                             ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, sublinear_tf=True))]),
                               LogisticRegression(C=1.0, max_iter=1000, random_state=0))
    text_model.fit([rows[i]["user"] for i in train_text_ix], [rows[i]["label"] for i in train_text_ix])
    text_p = text_model.predict_proba([r["user"] for r in test_rows])[:, list(text_model.classes_).index(1)]
    text_metrics = []
    for fmt in (0, 1, 2, 3):
        for aspect in range(3):
            mask = (formats == fmt) & (active == aspect)
            text_metrics.append({"format": fmt, "aspect": ASPECTS[aspect], "metrics": metrics(y[mask], text_p[mask])})
    # Preselected primary: block 16, bilinear. Compare source-scale output with a
    # per-query mean-logit transport offset fitted only from unlabeled anchors.
    primary_index = [i for i, k in enumerate(keys) if k["layer"] == 16 and k["method"] == "bilinear"]
    primary = logits_saved[primary_index] / np.asarray([keys[i]["temperature"] for i in primary_index])[:, None]
    primary = primary.mean(0)
    corrections = []
    for fmt in (1, 2, 3):
        for aspect in range(3):
            anchor = (formats == fmt) & (active == aspect) & np.isin([r["group_no"] for r in test_rows], range(8))
            source_anchor = (formats == 0) & (active == aspect) & np.isin([r["group_no"] for r in test_rows], range(8))
            eval_mask = (formats == fmt) & (active == aspect) & np.isin([r["group_no"] for r in test_rows], range(8, 24))
            if not anchor.any() or not source_anchor.any() or not eval_mask.any():
                raise ValueError("Missing disjoint source/target anchor or eval rows")
            offset = float(primary[anchor].mean() - primary[source_anchor].mean())
            p_base, p_shift = sigmoid(primary[eval_mask]), sigmoid(primary[eval_mask] - offset)
            yy, gg = y[eval_mask], groups[eval_mask]
            delta = float(np.mean((p_shift - yy) ** 2) - np.mean((p_base - yy) ** 2))
            corrections.append({"format": fmt, "aspect": ASPECTS[aspect], "n_eval": int(eval_mask.sum()),
                                "offset": offset, "baseline": metrics(yy, p_base),
                                "centered": metrics(yy, p_shift), "delta_brier_centered_minus_base": delta,
                                "group_bootstrap_95_ci": group_bootstrap_delta(yy, p_base, p_shift, gg)})
    write_json(run / "analysis.json", {"protocol_sha256": manifest["protocol_sha256"], "model": MODEL,
        "probe_records": summaries, "probe_keys": keys, "text_baseline": text_metrics,
        "primary_transport": corrections,
        "gate_passed": manifest["capability_gate_passed"],
        "provenance": provenance()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "train", "all"))
    parser.add_argument("--run", type=Path, default=Path("runs/aspect-sentiment-v1/qwen-1.5b"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.stage in ("collect", "all"):
        if args.run.exists():
            raise FileExistsError(f"Refusing to overwrite {args.run}")
        args.run.mkdir(parents=True)
        collect(args.run, args.offline)
    if args.stage in ("train", "all"):
        train(args.run)


if __name__ == "__main__":
    main()
