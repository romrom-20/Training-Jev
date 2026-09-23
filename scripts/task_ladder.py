"""Controlled task ladder: capability screening followed by gated readouts."""

import argparse
import gc
import hashlib
import itertools
import json
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
from latent_decisions.target import block_tensor, forward_last, load_target

ASPECTS = ("food", "service", "value")
TASKS = ("isolated_clause", "neutral_distractors", "mixed_review", "keyed_record")
PREFERRED = ("mixed_review", "neutral_distractors", "isolated_clause", "keyed_record")
FORMATS = (
    "Review: {context}\nWhat is the sentiment about the {aspect}? Reply with exactly one word: positive or negative.",
    "Look at this text: {context}\nTarget aspect = {aspect}. Choose just its sentiment: positive or negative.",
)
SPLITS = (
    ("selector", 12),
    ("train", 16),
    ("validation", 6),
    ("calibration", 6),
    ("final_test", 24),
)
MODEL_SPECS = {
    "qwen-1.5b": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "layers": [8, 16, 24],
        "family": "Qwen2.5",
    },
    "smollm2-1.7b": {
        "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "revision": "31b70e2e869a7173562077fd711b654946d38674",
        "layers": [8, 16, 24],
        "family": "SmolLM2",
    },
    "qwen-3b": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "layers": [12, 24, 36],
        "family": "Qwen2.5",
    },
}
CUES = {
    "food": {
        1: (
            "The meal tasted delicious and fresh.",
            "The food was excellent and well prepared.",
            "I enjoyed the flavorful dish.",
            "It was a wonderful meal.",
            "Dinner was delightful.",
            "The dish tasted fresh and satisfying.",
            "The meal was a pleasure to eat.",
            "Food quality was outstanding.",
        ),
        0: (
            "The meal tasted awful and stale.",
            "The food was terrible and poorly prepared.",
            "I disliked the bland dish.",
            "It was a dreadful meal.",
            "Dinner was disappointing.",
            "The dish tasted old and unsatisfying.",
            "The meal was a chore to eat.",
            "Food quality was awful.",
        ),
    },
    "service": {
        1: (
            "The staff were friendly and attentive.",
            "The service was warm and helpful.",
            "Our server was kind and quick.",
            "The team treated us wonderfully.",
            "The staff handled everything well.",
            "Service felt thoughtful and prompt.",
            "The employees were pleasant and capable.",
            "We received excellent assistance.",
        ),
        0: (
            "The staff were rude and inattentive.",
            "The service was cold and unhelpful.",
            "Our server was hostile and slow.",
            "The team treated us poorly.",
            "The staff mishandled everything.",
            "Service felt careless and delayed.",
            "The employees were unpleasant and inept.",
            "We received terrible assistance.",
        ),
    },
    "value": {
        1: (
            "The price felt fair and reasonable.",
            "It was good value for the money.",
            "The cost was worthwhile.",
            "The purchase felt like a bargain.",
            "The price was modest for the quality.",
            "It was worth every dollar.",
            "The cost seemed very fair.",
            "This was excellent value.",
        ),
        0: (
            "The price felt unfair and excessive.",
            "It was poor value for the money.",
            "The cost was not worthwhile.",
            "The purchase felt like a rip-off.",
            "The price was steep for the quality.",
            "It was not worth the money.",
            "The cost seemed unreasonable.",
            "This was terrible value.",
        ),
    },
}
NEUTRAL = ("The visit lasted about an hour.", "The building had three windows.")
QUESTIONS = tuple(f"What is the sentiment about the {aspect}?" for aspect in ASPECTS)


def group_split(group):
    for name, count in SPLITS:
        if group < sum(size for _, size in SPLITS[: SPLITS.index((name, count)) + 1]):
            return name
    raise ValueError(group)


def lexical_variant(group):
    return 4 + group % 4 if group >= 40 else group % 4


def make_rows():
    rows = []
    permutations = tuple(itertools.permutations(range(3)))
    for group in range(sum(n for _, n in SPLITS)):
        split = group_split(group)
        variant = lexical_variant(group)
        order = permutations[group % len(permutations)]
        for task in TASKS:
            for active, aspect in enumerate(ASPECTS):
                if task in ("isolated_clause", "neutral_distractors"):
                    settings = [None] * 3
                    for label in (0, 1):
                        settings[active] = label
                        if task == "isolated_clause":
                            context = CUES[aspect][label][variant]
                        else:
                            context = " ".join((CUES[aspect][label][variant], *NEUTRAL))
                        rows.append(_row(group, split, task, 0, active, settings, context))
                else:
                    for bits in itertools.product((0, 1), repeat=3):
                        if task == "mixed_review":
                            clauses = [CUES[ASPECTS[j]][bits[j]][variant] for j in order]
                            context = " ".join(clauses)
                        else:
                            fields = [
                                f"{ASPECTS[j]}={('positive' if bits[j] else 'negative')}"
                                for j in order
                            ]
                            context = "; ".join(fields)
                        rows.append(_row(group, split, task, 0, active, list(bits), context))
            # Copy the same scenario into a second, paraphrased user instruction.
        current = [r for r in rows if r["group_no"] == group]
        for row in current:
            shifted = dict(row)
            shifted["format"] = 1
            shifted["id"] += "-f1"
            shifted["user"] = FORMATS[1].format(context=row["context"], aspect=row["aspect"])
            rows.append(shifted)
        # Source rows are constructed after the loop to keep one source record per condition.
    # Rebuild source-format messages for rows created in the inner loop.
    for row in rows:
        if row["format"] == 0:
            row["user"] = FORMATS[0].format(context=row["context"], aspect=row["aspect"])
    return rows


def _row(group, split, task, fmt, active, bits, context):
    bits = list(bits)
    return {
        "id": f"g{group:02d}-{task}-q{active}-b{''.join('x' if b is None else str(b) for b in bits)}-f{fmt}",
        "group": f"g{group:02d}",
        "group_no": group,
        "split": split,
        "task": task,
        "format": fmt,
        "active": active,
        "aspect": ASPECTS[active],
        "bits": bits,
        "label": bits[active],
        "context": context,
        "user": FORMATS[fmt].format(context=context, aspect=ASPECTS[active]),
    }


def capture(model_name, run, offline):
    spec = dict(MODEL_SPECS[model_name], name=model_name)
    rows = make_rows()
    write_json(run / "dataset.json", rows)
    model, tokenizer, device = load_target(spec, "auto", offline)
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in ("positive", "negative")]
    if any(len(x) != 1 for x in ids):
        raise ValueError("The standard output labels must each be a single token")
    ids = [x[0] for x in ids]
    queries = tokenizer(list(QUESTIONS), padding=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        q_hidden = model.model(**queries, use_cache=False).last_hidden_state
        qmask = queries.attention_mask[..., None]
        q = ((q_hidden * qmask).sum(1) / qmask.sum(1)).float().cpu()
    capture_cache, handles = {}, []
    for layer in spec["layers"]:

        def hook(module, inputs, output, layer=layer):
            capture_cache[layer] = block_tensor(output)[:, -1].detach().float().cpu().numpy()

        handles.append(model.model.layers[layer - 1].register_forward_hook(hook))
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["user"]}], tokenize=False, add_generation_prompt=True
        )
        for row in rows
    ]
    hs, probs, masses, generated = [], [], [], []
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            for offset in range(0, len(rows), 8):
                batch_rows = rows[offset : offset + 8]
                tokens = tokenizer(
                    texts[offset : offset + 8], padding=True, return_tensors="pt", truncation=False
                ).to(device)
                if tokens.input_ids.shape[1] > 256:
                    raise ValueError("Prompt exceeded the predeclared 256 token limit")
                logits = forward_last(model, tokens)
                pair = logits[:, ids]
                top = logits.argmax(-1)
                hs.append(np.stack([capture_cache[layer] for layer in spec["layers"]], axis=1))
                probs.append(pair.softmax(-1).cpu().numpy())
                masses.append(logits.softmax(-1)[:, ids].sum(-1).cpu().numpy())
                generated.extend(tokenizer.batch_decode(top[:, None], skip_special_tokens=True))
                for local, row in enumerate(batch_rows):
                    word = (
                        generated[-len(batch_rows) + local]
                        .strip()
                        .split(maxsplit=1)[0]
                        .strip(".,:;!?\"'()[]{}")
                        .lower()
                        if generated[-len(batch_rows) + local].strip()
                        else ""
                    )
                    row["generated_first_token"] = word
                    row["strict_correct"] = word == ("positive" if row["label"] else "negative")
                    row["conditional_p_positive"] = float(pair[local].softmax(-1)[0])
                    row["conditional_correct"] = (
                        int(pair[local, 0] >= pair[local, 1]) == row["label"]
                    )
                    row["label_mass"] = float(logits[local].softmax(-1)[ids].sum())
                if offset % 384 == 0:
                    print(
                        f"{model_name}: {min(offset + len(batch_rows), len(rows))}/{len(rows)}",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()
    np.savez_compressed(run / "activations.npz", h=np.concatenate(hs), q=q.numpy())
    write_json(run / "dataset.json", rows)
    write_json(
        run / "manifest.json",
        {
            "model": spec,
            "device": device,
            "n": len(rows),
            "capture_seconds": time.perf_counter() - start,
            "protocol_sha256": hashlib.sha256(
                Path("docs/experiments/008-controlled-task-ladder.md").read_bytes()
            ).hexdigest(),
            "dataset_sha256": hashlib.sha256((run / "dataset.json").read_bytes()).hexdigest(),
            "activation_sha256": hashlib.sha256((run / "activations.npz").read_bytes()).hexdigest(),
            "provenance": provenance(),
        },
    )
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def group_bootstrap_brier_delta(y, base, adjusted, group_ids, reps=3000, seed=20260924):
    unique = np.unique(group_ids)
    by_group = {group: np.flatnonzero(group_ids == group) for group in unique}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(reps):
        sampled_groups = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([by_group[group] for group in sampled_groups])
        samples.append(np.mean((adjusted[ix] - y[ix]) ** 2) - np.mean((base[ix] - y[ix]) ** 2))
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def fit_selected(run, model_name, task, test_gate):
    if not test_gate:
        return {"skipped": "final behavior gate failed"}
    torch.set_num_threads(4)
    spec = MODEL_SPECS[model_name]
    rows = json.loads((run / "dataset.json").read_text())
    with np.load(run / "activations.npz") as f:
        h_all = torch.from_numpy(f["h"].copy())
        q = torch.from_numpy(f["q"].copy())
    q = q / q.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)

    def selected(split, fmt=None):
        return np.asarray(
            [
                i
                for i, row in enumerate(rows)
                if row["split"] == split
                and row["task"] == task
                and (fmt is None or row["format"] == fmt)
            ]
        )

    train_ix, val_ix, cal_ix = (selected(s, 0) for s in ("train", "validation", "calibration"))
    test_ix = selected("final_test")
    test_rows = [rows[i] for i in test_ix]
    y = np.asarray([r["label"] for r in test_rows])
    formats = np.asarray([r["format"] for r in test_rows])
    active = np.asarray([r["active"] for r in test_rows])
    group_no = np.asarray([r["group_no"] for r in test_rows])
    summaries, all_logits, keys = [], [], []

    def pack(h, indices):
        ix_rows = [rows[i] for i in indices]
        return (
            h[indices],
            q[[r["active"] for r in ix_rows]],
            torch.tensor([r["active"] for r in ix_rows]),
            torch.tensor([r["label"] for r in ix_rows], dtype=torch.float32),
        )

    for li, layer in enumerate(spec["layers"]):
        norm = Standardizer().fit(h_all[train_ix, li])
        h = norm(h_all[:, li])
        tr, va, ca = pack(h, train_ix), pack(h, val_ix), pack(h, cal_ix)
        te = pack(h, test_ix)
        for seed, method in itertools.product((0, 1, 2), ("bilinear", "independent")):
            seed_all(seed)
            probe = (
                BilinearProbe(h.shape[-1], q.shape[-1], 4)
                if method == "bilinear"
                else IndependentProbe(h.shape[-1], 3)
            )
            fit_info = fit_probe(
                probe, tr, va, steps=300, lr=0.01, weight_decay=0.01, brier_weight=0.1
            )
            with torch.no_grad():
                cal_logits = probe(*ca[:3]).numpy()
                pred_logits = probe(*te[:3]).numpy()
            temperatures = []
            for j in range(3):
                mask = np.asarray([rows[i]["active"] == j for i in cal_ix])
                temperatures.append(fit_temperature(cal_logits[mask], ca[3].numpy()[mask]))
            key = {
                "layer": layer,
                "seed": seed,
                "method": method,
                "temperatures": temperatures,
                **fit_info,
            }
            keys.append(key)
            all_logits.append(pred_logits)
            for fmt, aspect in itertools.product((0, 1), range(3)):
                mask = (formats == fmt) & (active == aspect)
                summaries.append(
                    {
                        **key,
                        "format": fmt,
                        "aspect": ASPECTS[aspect],
                        "raw": metrics(y[mask], sigmoid(pred_logits[mask])),
                        "calibrated": metrics(
                            y[mask], sigmoid(pred_logits[mask] / temperatures[aspect])
                        ),
                    }
                )
    all_logits = np.stack(all_logits)
    # The word/character text baseline sees the same source prompt strings as probes.
    text_ix = train_ix
    text_model = make_pipeline(
        FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char", ngram_range=(2, 5), min_df=1, sublinear_tf=True
                    ),
                ),
            ]
        ),
        LogisticRegression(C=1.0, max_iter=1000, random_state=0),
    )
    text_model.fit([rows[i]["user"] for i in text_ix], [rows[i]["label"] for i in text_ix])
    classes = list(text_model.classes_)
    text_probs = text_model.predict_proba([r["user"] for r in test_rows])[:, classes.index(1)]
    text_metrics = [
        {
            "format": fmt,
            "aspect": ASPECTS[j],
            "metrics": metrics(
                y[(formats == fmt) & (active == j)], text_probs[(formats == fmt) & (active == j)]
            ),
        }
        for fmt, j in itertools.product((0, 1), range(3))
    ]
    mid = len(spec["layers"]) // 2
    pri = [
        i
        for i, key in enumerate(keys)
        if key["layer"] == spec["layers"][mid] and key["method"] == "bilinear"
    ]
    calibrated_logits = np.mean(
        [
            all_logits[i]
            / np.asarray([keys[i]["temperatures"][a] for a in [r["active"] for r in test_rows]])
            for i in pri
        ],
        axis=0,
    )
    transport = []
    for j, aspect in enumerate(ASPECTS):
        anchor = (formats == 1) & (active == j) & (group_no < 48)
        source_anchor = (formats == 0) & (active == j) & (group_no < 48)
        eval_mask = (formats == 1) & (active == j) & (group_no >= 48)
        offset = float(calibrated_logits[anchor].mean() - calibrated_logits[source_anchor].mean())
        base = sigmoid(calibrated_logits[eval_mask])
        shifted = sigmoid(calibrated_logits[eval_mask] - offset)
        eval_groups = group_no[eval_mask]
        transport.append(
            {
                "aspect": aspect,
                "offset": offset,
                "baseline": metrics(y[eval_mask], base),
                "centered": metrics(y[eval_mask], shifted),
                "delta_brier_centered_minus_baseline": float(
                    np.mean((shifted - y[eval_mask]) ** 2) - np.mean((base - y[eval_mask]) ** 2)
                ),
                "group_bootstrap_95_ci": group_bootstrap_brier_delta(
                    y[eval_mask], base, shifted, eval_groups
                ),
            }
        )
    np.savez_compressed(run / f"probe-scores-{task}.npz", logits=all_logits, indices=test_ix)
    output = {
        "model": spec,
        "task": task,
        "probe_records": summaries,
        "probe_keys": keys,
        "text_baseline": text_metrics,
        "transport": transport,
        "n_test": len(test_ix),
    }
    write_json(run / f"probe-analysis-{task}.json", output)
    return {"task": task, "n_test": len(test_ix), "analysis": f"probe-analysis-{task}.json"}


def summarize(root):
    all_data = {}
    selector_gate = {}
    final_gate = {}
    paired_specificity = {}
    for name in MODEL_SPECS:
        run = root / name
        rows = json.loads((run / "dataset.json").read_text())
        by_cell = []
        for split in ("selector", "final_test"):
            for task, fmt, aspect in itertools.product(TASKS, (0, 1), range(3)):
                cell = [
                    r
                    for r in rows
                    if r["split"] == split
                    and r["task"] == task
                    and r["format"] == fmt
                    and r["active"] == aspect
                ]
                by_cell.append(
                    {
                        "split": split,
                        "task": task,
                        "format": fmt,
                        "aspect": ASPECTS[aspect],
                        "n": len(cell),
                        "strict_accuracy": float(np.mean([r["strict_correct"] for r in cell])),
                        "conditional_accuracy": float(
                            np.mean([r["conditional_correct"] for r in cell])
                        ),
                    }
                )
        all_data[name] = by_cell
        for split, dest in (("selector", selector_gate), ("final_test", final_gate)):
            dest[name] = {
                task: all(
                    c["strict_accuracy"] >= 0.9
                    for c in by_cell
                    if c["split"] == split and c["task"] == task
                )
                for task in TASKS
            }
        row_lookup = {
            (r["group_no"], r["task"], r["format"], r["active"], tuple(r["bits"])): r for r in rows
        }
        specificity = []
        for split in ("selector", "final_test"):
            for task in TASKS:
                for fmt in (0, 1):
                    for active in range(3):
                        active_pairs, irrelevant_pairs = [], []
                        for group in range(64):
                            if not any(
                                r["group_no"] == group
                                and r["split"] == split
                                and r["task"] == task
                                and r["format"] == fmt
                                for r in rows
                            ):
                                continue
                            candidate = [
                                r
                                for r in rows
                                if r["group_no"] == group
                                and r["split"] == split
                                and r["task"] == task
                                and r["format"] == fmt
                                and r["active"] == active
                            ]
                            for row in candidate:
                                bits = list(row["bits"])
                                if bits[active] == 0:
                                    flip = bits.copy()
                                    flip[active] = 1
                                    other = row_lookup.get((group, task, fmt, active, tuple(flip)))
                                    if other is not None:
                                        active_pairs.append((row, other))
                                    for field in range(3):
                                        if (
                                            field == active
                                            or bits[field] is None
                                            or bits[field] != 0
                                        ):
                                            continue
                                        flip_irrelevant = bits.copy()
                                        flip_irrelevant[field] = 1
                                        other_irrelevant = row_lookup.get(
                                            (group, task, fmt, active, tuple(flip_irrelevant))
                                        )
                                        if other_irrelevant is not None:
                                            irrelevant_pairs.append((row, other_irrelevant))

                        def prediction(row):
                            if row["generated_first_token"] == "positive":
                                return 1
                            if row["generated_first_token"] == "negative":
                                return 0
                            return None

                        active_success = sum(
                            prediction(a) == a["label"] and prediction(b) == b["label"]
                            for a, b in active_pairs
                        )
                        active_flip = sum(
                            prediction(a) is not None
                            and prediction(b) is not None
                            and prediction(a) != prediction(b)
                            for a, b in active_pairs
                        )
                        irrelevant_stable = sum(
                            prediction(a) is not None and prediction(a) == prediction(b)
                            for a, b in irrelevant_pairs
                        )
                        specificity.append(
                            {
                                "split": split,
                                "task": task,
                                "format": fmt,
                                "aspect": ASPECTS[active],
                                "queried_flip_pairs": len(active_pairs),
                                "queried_flip_both_correct": active_success / len(active_pairs)
                                if active_pairs
                                else None,
                                "queried_flip_any_response_change": active_flip / len(active_pairs)
                                if active_pairs
                                else None,
                                "irrelevant_flip_pairs": len(irrelevant_pairs),
                                "irrelevant_flip_response_invariance": irrelevant_stable
                                / len(irrelevant_pairs)
                                if irrelevant_pairs
                                else None,
                            }
                        )
        for fmt in (0, 1):
            for active in range(3):
                mixed = [
                    r
                    for r in rows
                    if r["split"] == "final_test"
                    and r["task"] == "mixed_review"
                    and r["format"] == fmt
                    and r["active"] == active
                ]
                conflict = [r for r in mixed if int(sum(r["bits"]) >= 2) != r["label"]]
                matches_majority = sum(
                    r["generated_first_token"]
                    == ("positive" if sum(r["bits"]) >= 2 else "negative")
                    for r in mixed
                )
                matches_target_in_conflict = sum(r["strict_correct"] for r in conflict)
                specificity.append(
                    {
                        "split": "final_test",
                        "task": "mixed_review_global_majority",
                        "format": fmt,
                        "aspect": ASPECTS[active],
                        "n": len(mixed),
                        "matches_overall_majority": matches_majority / len(mixed)
                        if mixed
                        else None,
                        "n_when_overall_majority_conflicts_with_query": len(conflict),
                        "queried_accuracy_when_conflicting": matches_target_in_conflict
                        / len(conflict)
                        if conflict
                        else None,
                    }
                )
        paired_specificity[name] = specificity
    core = ("qwen-1.5b", "smollm2-1.7b")
    qualified = [task for task in PREFERRED if all(selector_gate[name][task] for name in core)]
    chosen = qualified[0] if qualified else None
    probes = {}
    if chosen:
        for name in MODEL_SPECS:
            probes[name] = fit_selected(root / name, name, chosen, final_gate[name][chosen])
    result = {
        "protocol": "008",
        "selector_gates": selector_gate,
        "final_test_gates": final_gate,
        "qualified_tasks": qualified,
        "selected_task": chosen,
        "probes": probes,
        "cells": all_data,
        "paired_specificity": paired_specificity,
    }
    write_json(root / "task-ladder-analysis.json", result)
    print(
        json.dumps(
            {
                "selector_gates": selector_gate,
                "final_test_gates": final_gate,
                "qualified_tasks": qualified,
                "selected_task": chosen,
                "probes": probes,
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "analyze", "all"))
    parser.add_argument("--root", type=Path, default=Path("runs/task-ladder-v1"))
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_SPECS), default=tuple(MODEL_SPECS)
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.stage in ("collect", "all"):
        if args.root.exists():
            raise FileExistsError(f"Refusing to overwrite {args.root}")
        args.root.mkdir(parents=True)
        for name in args.models:
            path = args.root / name
            path.mkdir()
            capture(name, path, args.offline)
    if args.stage in ("analyze", "all"):
        summarize(args.root)


if __name__ == "__main__":
    main()
