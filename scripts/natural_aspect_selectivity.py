"""Test frozen synthetic aspect directions on multi-aspect SemEval reviews."""

import argparse
import gc
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from answer_encoding_control import setup
from prompt_effect_forecast import LAYER, sha
from shared_residual_steering import MODELS
from task_ladder import MODEL_SPECS

from latent_decisions.experiment import provenance, write_json
from latent_decisions.target import block_tensor, forward_last, load_target, replace_block_tensor

PROTOCOL = Path("docs/experiments/014-natural-aspect-selectivity.md")
DATA = Path(".context/datasets/semeval2014/Restaurants_Test_Gold.xml")
CATEGORIES = ("food", "service", "price")
SOURCE_ASPECT = {"food": 0, "service": 1, "price": 2}
QUESTIONS = {
    "food": "What is the sentiment about the food?",
    "service": "What is the sentiment about the service?",
    "price": "What is the sentiment about the price or value?",
}
CONDITIONS = ("native", "shared", "random")
SEED = 20260924


def load_stimuli(path):
    stimuli = []
    for sentence in ET.parse(path).getroot().findall(".//sentence"):
        text_node = sentence.find("text")
        if text_node is None or not text_node.text:
            continue
        section = sentence.find("aspectCategories")
        if section is None:
            continue
        labels = {}
        for category in section.findall("aspectCategory"):
            name, polarity = category.get("category"), category.get("polarity")
            if name in CATEGORIES and polarity in ("positive", "negative"):
                if name in labels and labels[name] != polarity:
                    raise ValueError("Repeated category with inconsistent polarity")
                labels[name] = polarity
        if len(labels) < 2:
            continue
        sid = sentence.get("id")
        for category, polarity in labels.items():
            user = (
                f"Review: {text_node.text.strip()}\n{QUESTIONS[category]} "
                "Reply with exactly one word: positive or negative."
            )
            stimuli.append(
                {
                    "id": f"{sid}|{category}",
                    "sentence_id": sid,
                    "category": category,
                    "target": SOURCE_ASPECT[category],
                    "label": int(polarity == "positive"),
                    "user": user,
                }
            )
    return stimuli


def candidate_ids(tokenizer):
    pos, neg = tokenizer.encode("positive", add_special_tokens=False), tokenizer.encode(
        "negative", add_special_tokens=False
    )
    if len(pos) != 1 or len(neg) != 1 or pos[0] == neg[0]:
        raise ValueError("positive and negative must be distinct single-token candidates")
    return pos[0], neg[0]


def capture_logits(model, tokenizer, device, stimuli, vectors, dose, output):
    pos_id, neg_id = candidate_ids(tokenizer)
    expected = {item["id"] for item in stimuli}
    baseline_path = output / "baseline.partial.jsonl"
    baseline = {}
    if baseline_path.exists():
        for line in baseline_path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                baseline[item["id"]] = item
    if not set(baseline).issubset(expected):
        raise ValueError("Baseline checkpoint contains unknown stimulus IDs")
    pending = [item for item in stimuli if item["id"] not in baseline]
    with baseline_path.open("a") as stream:
        for offset in range(0, len(pending), 8):
            batch = pending[offset : offset + 8]
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
            lp = logits[:, pos_id].float().cpu().tolist()
            ln = logits[:, neg_id].float().cpu().tolist()
            top = logits.argmax(-1).cpu().tolist()
            for stimulus, positive, negative, top_id in zip(batch, lp, ln, top):
                record = {
                    "id": stimulus["id"],
                    "sentence_id": stimulus["sentence_id"],
                    "category": stimulus["category"],
                    "target": stimulus["target"],
                    "label": stimulus["label"],
                    "positive_logit": positive,
                    "negative_logit": negative,
                    "top_token_id": int(top_id),
                    "correct_token_id": int(pos_id if stimulus["label"] else neg_id),
                    "strict_correct": int(top_id) == int(pos_id if stimulus["label"] else neg_id),
                    "valid_polarity_token": int(top_id) in (pos_id, neg_id),
                }
                baseline[stimulus["id"]] = record
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
            print(f"baseline: {len(baseline)}/{len(expected)}", flush=True)
    if set(baseline) != expected:
        raise ValueError("Baseline capture incomplete")
    write_json(output / "baseline.json", [baseline[item["id"]] for item in stimuli])

    cases = []
    for item in stimuli:
        for source in range(3):
            cases.append((item, "native", source, vectors["native"][source]))
            for condition in ("shared", "random"):
                # Repeating the same control under each source label gives it
                # the same diagonal/off-diagonal query structure as native arms.
                cases.append((item, condition, source, vectors[condition]))
    effects_path = output / "effects.partial.jsonl"
    completed = {}
    if effects_path.exists():
        for line in effects_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["effect_id"]] = row
    expected_effects = {
        f"{item['id']}|{condition}|{source}"
        for item, condition, source, _vector in cases
    }
    if not set(completed).issubset(expected_effects):
        raise ValueError("Effect checkpoint contains unknown IDs")
    pending_cases = [
        case
        for case in cases
        if f"{case[0]['id']}|{case[1]}|{case[2]}"
        not in completed
    ]
    block = model.model.layers[LAYER - 1]
    with effects_path.open("a") as stream:
        for offset in range(0, len(pending_cases), 8):
            batch = pending_cases[offset : offset + 8]
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": item["user"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for item, _condition, _source, _vector in batch
            ]
            tokens = tokenizer(texts, padding=True, return_tensors="pt").to(device)
            delta = torch.stack([vector for _item, _condition, _source, vector in batch]).to(device)
            delta *= dose

            def patch_block(_module, _inputs, output_value):
                value = block_tensor(output_value).clone()
                value[:, -1, :] += delta
                return replace_block_tensor(output_value, value)

            handle = block.register_forward_hook(patch_block)
            try:
                with torch.inference_mode():
                    logits = forward_last(model, tokens)
            finally:
                handle.remove()
            lp = logits[:, pos_id].float().cpu().tolist()
            ln = logits[:, neg_id].float().cpu().tolist()
            top = logits.argmax(-1).cpu().tolist()
            for (item, condition, source, _vector), positive, negative, top_id in zip(
                batch, lp, ln, top
            ):
                base = baseline[item["id"]]
                effect_id = f"{item['id']}|{condition}|{source}"
                row = {
                    "effect_id": effect_id,
                    "id": item["id"],
                    "sentence_id": item["sentence_id"],
                    "category": item["category"],
                    "target": item["target"],
                    "label": item["label"],
                    "condition": condition,
                    "source": source,
                    "baseline_margin": base["positive_logit"] - base["negative_logit"],
                    "steered_margin": positive - negative,
                    "margin_delta": (positive - negative)
                    - (base["positive_logit"] - base["negative_logit"]),
                    "steered_top_token_id": int(top_id),
                    "steered_strict_correct": int(top_id)
                    == int(pos_id if item["label"] else neg_id),
                }
                completed[effect_id] = row
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            stream.flush()
            print(f"effects: {len(completed)}/{len(expected_effects)}", flush=True)
    if set(completed) != expected_effects:
        raise ValueError("Treatment capture incomplete")
    write_json(
        output / "effects.json",
        [
                completed[
                f"{item['id']}|{condition}|{source}"
            ]
            for item, condition, source, _vector in cases
        ],
    )
    return pos_id, neg_id


def bootstrap_ci(values, groups, reps=5000, seed=SEED):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    by_group = {group: values[groups == group] for group in unique}
    rng = np.random.default_rng(seed)
    estimates = np.empty(reps)
    for i in range(reps):
        selected = rng.choice(unique, size=len(unique), replace=True)
        estimates[i] = np.concatenate([by_group[group] for group in selected]).mean()
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def sentence_contrasts(effects, condition):
    by_sentence = {}
    for row in effects:
        if row["condition"] == condition:
            by_sentence.setdefault(row["sentence_id"], []).append(row)
    values = {}
    for sentence, rows in by_sentence.items():
        diagonal, off_diagonal = [], []
        for source in range(3):
            source_rows = [row for row in rows if row["source"] == source]
            match = [row["margin_delta"] for row in source_rows if row["target"] == source]
            other = [row["margin_delta"] for row in source_rows if row["target"] != source]
            if match and other:
                diagonal.extend(match)
                off_diagonal.extend(other)
        if diagonal and off_diagonal:
            values[sentence] = float(np.mean(diagonal) - np.mean(off_diagonal))
    return values


def analyze_model(root, model):
    base = json.loads((root / model / "baseline.json").read_text())
    effects = json.loads((root / model / "effects.json").read_text())
    correct = np.array([x["strict_correct"] for x in base], dtype=float)
    valid = np.array([x["valid_polarity_token"] for x in base], dtype=float)
    summaries = {}
    for condition in ("native", "shared", "random"):
        for source in range(3):
            rows = [
                x
                for x in effects
                if x["condition"] == condition and x["source"] == source
            ]
            key = f"{condition}_{source}"
            summaries[key] = {
                "mean_margin_delta": float(np.mean([x["margin_delta"] for x in rows])),
                "mean_strict_accuracy_under_steering": float(
                    np.mean([x["steered_strict_correct"] for x in rows])
                ),
                "n": len(rows),
            }
    contrast_by_condition = {}
    contrast = {}
    for condition in ("native", "shared", "random"):
        vals = sentence_contrasts(effects, condition)
        contrast_by_condition[condition] = {
            "mean_diagonal_minus_off_diagonal": float(np.mean(list(vals.values())))
            if vals
            else None,
            "sentence_bootstrap_95_ci": bootstrap_ci(
                [vals[g] for g in sorted(vals)], sorted(vals)
            )
            if vals
            else None,
            "n_sentences": len(vals),
        }
        if vals:
            contrast[condition] = vals
    common = sorted(set(contrast.get("native", {})) & set(contrast.get("random", {})))
    native_minus_random = [
        contrast["native"][sid] - contrast["random"][sid] for sid in common
    ]
    specificity = {
        "mean_native_minus_random_specificity": float(np.mean(native_minus_random)),
        "sentence_bootstrap_95_ci": bootstrap_ci(native_minus_random, common),
        "n_sentences": len(common),
    }
    conflict_ids = {
        sentence
        for sentence in {x["sentence_id"] for x in effects}
        if len(
            {
                x["label"]
                for x in base
                if x["sentence_id"] == sentence
            }
        ) > 1
    }
    conflict_rows = [x for x in effects if x["sentence_id"] in conflict_ids]
    conflict_native = sentence_contrasts(conflict_rows, "native")
    return {
        "model": model,
        "baseline_strict_accuracy": float(correct.mean()),
        "baseline_valid_positive_or_negative_rate": float(valid.mean()),
        "baseline_behavior_gate_pass_over_chance": bool(correct.mean() > 0.5),
        "n_baseline_prompts": len(base),
        "effects": summaries,
        "specificity_contrasts": contrast_by_condition,
        "primary_native_minus_random_specificity": specificity,
        "polarity_disagreement_subset": {
            "n_sentences": len(conflict_ids),
            "native_diagonal_minus_off_diagonal_mean": float(
                np.mean(list(conflict_native.values()))
            )
            if conflict_native
            else None,
            "interpretation": "descriptive only; subset may be small",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "analyze", "all"))
    parser.add_argument("--root", type=Path, default=Path("runs/natural-aspect-selectivity-v1"))
    parser.add_argument("--source-root", type=Path, default=Path("runs/task-ladder-v1"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage in ("collect", "all"):
        if args.root.exists() and not args.resume:
            raise FileExistsError(f"Refusing to overwrite {args.root}; use --resume")
        args.root.mkdir(parents=True, exist_ok=True)
        stimuli = load_stimuli(args.data)
        if not stimuli:
            raise ValueError("No eligible multi-aspect gold stimuli found")
        sentence_labels = {}
        for item in stimuli:
            sentence_labels.setdefault(item["sentence_id"], {})[item["category"]] = item["label"]
        conflict_count = sum(len(set(labels.values())) > 1 for labels in sentence_labels.values())
        print(
            f"Eligible gold prompts={len(stimuli)}; sentences={len(sentence_labels)}; "
            f"polarity-disagreement sentences={conflict_count}",
            flush=True,
        )
        for model_name in args.models:
            output = args.root / model_name
            output.mkdir(exist_ok=True)
            if (output / "manifest.json").exists():
                continue
            rows, dose, vectors = setup(model_name, args.source_root)
            model_spec = dict(MODEL_SPECS[model_name], name=model_name)
            model, tokenizer, device = load_target(model_spec, "auto", args.offline)
            pos_id, neg_id = capture_logits(model, tokenizer, device, stimuli, vectors, dose, output)
            source = args.source_root / model_name
            source_manifest = json.loads((source / "manifest.json").read_text())
            manifest = {
                "model": model_spec,
                "device": str(device),
                "dtype": "float32",
                "layer": LAYER,
                "dose_fraction": 0.05,
                "dose_l2": dose,
                "candidate_token_ids": {"positive": pos_id, "negative": neg_id},
                "n_gold_prompts": len(stimuli),
                "n_sentences": len(sentence_labels),
                "n_polarity_disagreement_sentences": conflict_count,
                "category_source_mapping": {**SOURCE_ASPECT},
                "source_dataset_sha256": sha(source / "dataset.json"),
                "source_activation_sha256": source_manifest["activation_sha256"],
                "gold_xml_sha256": sha(args.data),
                "protocol_sha256": sha(PROTOCOL),
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "baseline_sha256": sha(output / "baseline.json"),
                "effects_sha256": sha(output / "effects.json"),
                "provenance": provenance(),
            }
            write_json(output / "manifest.json", manifest)
            del model
            gc.collect()
            if str(device).startswith("mps"):
                torch.mps.empty_cache()
    if args.stage in ("analyze", "all"):
        result = {model: analyze_model(args.root, model) for model in args.models}
        write_json(args.root / "analysis.json", result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
