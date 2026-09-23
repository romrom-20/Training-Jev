"""Paired target-only answer-format control for experiment 009."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from task_ladder import MODEL_SPECS, make_rows

from latent_decisions.experiment import provenance, write_json
from latent_decisions.target import forward_last, load_target

FORMAT_C = (
    "Look at this text: {context}\nTarget aspect = {aspect}. Choose just its sentiment: "
    "positive or negative. Reply with exactly one word: positive or negative."
)


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_control_rows():
    base = make_rows()
    rows = []
    for source in base:
        if source["format"] != 0:
            continue
        for fmt in range(3):
            row = dict(source)
            row["id"] = source["id"].rsplit("-f", 1)[0] + f"-f{fmt}"
            row["format"] = fmt
            if fmt == 0:
                row["user"] = (
                    f"Review: {row['context']}\nWhat is the sentiment about the "
                    f"{row['aspect']}? Reply with exactly one word: positive or negative."
                )
            elif fmt == 1:
                row["user"] = (
                    f"Look at this text: {row['context']}\nTarget aspect = {row['aspect']}. "
                    "Choose just its sentiment: positive or negative."
                )
            else:
                row["user"] = FORMAT_C.format(context=row["context"], aspect=row["aspect"])
            rows.append(row)
    return rows


def capture(name, root, offline):
    spec = dict(MODEL_SPECS[name], name=name)
    rows = make_control_rows()
    write_json(root / "dataset.json", rows)
    model, tokenizer, device = load_target(spec, "auto", offline)
    label_ids = [
        tokenizer.encode(label, add_special_tokens=False) for label in ("positive", "negative")
    ]
    if any(len(ids) != 1 for ids in label_ids):
        raise ValueError("The standard output labels must each be a single token")
    label_ids = [ids[0] for ids in label_ids]
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["user"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    partial_path = root / "predictions.partial.json"
    output = json.loads(partial_path.read_text()) if partial_path.exists() else []
    if len(output) > len(rows) or any(output[i]["id"] != rows[i]["id"] for i in range(len(output))):
        raise ValueError("Checkpoint does not match the frozen prompt order")
    start_offset = len(output)
    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(start_offset, len(rows), 8):
            tokens = tokenizer(
                texts[offset : offset + 8], padding=True, return_tensors="pt", truncation=False
            ).to(device)
            logits = forward_last(model, tokens)
            selected = logits[:, label_ids]
            predictions = logits.argmax(-1)
            p_positive = selected.softmax(-1)[:, 0].cpu().numpy()
            vocab_prob = logits.softmax(-1)
            label_mass = vocab_prob[:, label_ids].sum(-1).cpu().numpy()
            words = tokenizer.batch_decode(predictions[:, None], skip_special_tokens=True)
            for j, word in enumerate(words):
                row = rows[offset + j]
                token = word.strip().split(maxsplit=1)[0].strip(".,:;!?\"'()[]{}").lower()
                row.update(
                    generated_first_token=token,
                    strict_correct=token == ("positive" if row["label"] else "negative"),
                    compliant=token in ("positive", "negative"),
                    conditional_p_positive=float(p_positive[j]),
                    conditional_correct=bool((p_positive[j] >= 0.5) == bool(row["label"])),
                    label_mass=float(label_mass[j]),
                )
                output.append(row)
            completed = offset + len(tokens.input_ids)
            if completed % 512 == 0 or completed == len(rows):
                write_json(partial_path, output)
                print(f"{name}: {completed}/{len(rows)}", flush=True)
    write_json(root / "predictions.json", output)
    partial_path.unlink(missing_ok=True)
    manifest = {
        "model": spec,
        "device": str(device),
        "n": len(output),
        "capture_seconds": time.perf_counter() - start,
        "protocol_sha256": checksum(
            Path(__file__).resolve().parents[1]
            / "docs/experiments/009-output-constraint-control.md"
        ),
        "dataset_sha256": checksum(root / "dataset.json"),
        "predictions_sha256": checksum(root / "predictions.json"),
        "provenance": provenance(),
    }
    write_json(root / "manifest.json", manifest)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()


def summarize(root):
    names = [n for n in MODEL_SPECS if (root / n / "manifest.json").exists()]
    results = {}
    for name in names:
        rows = json.loads((root / name / "predictions.json").read_text())
        cells = []
        for split in ("selector", "final_test"):
            for task in ("isolated_clause", "neutral_distractors", "mixed_review", "keyed_record"):
                for fmt in (0, 1, 2):
                    for aspect in ("food", "service", "value"):
                        cell = [
                            r
                            for r in rows
                            if r["split"] == split
                            and r["task"] == task
                            and r["format"] == fmt
                            and r["aspect"] == aspect
                        ]
                        if cell:
                            cells.append(
                                {
                                    "split": split,
                                    "task": task,
                                    "format": fmt,
                                    "aspect": aspect,
                                    "n": len(cell),
                                    "strict_accuracy": float(
                                        np.mean([r["strict_correct"] for r in cell])
                                    ),
                                    "compliance": float(np.mean([r["compliant"] for r in cell])),
                                    "conditional_accuracy": float(
                                        np.mean([r["conditional_correct"] for r in cell])
                                    ),
                                    "mean_label_mass": float(
                                        np.mean([r["label_mass"] for r in cell])
                                    ),
                                }
                            )
        results[name] = cells
    comparisons = {}
    for name in names:
        rows = json.loads((root / name / "predictions.json").read_text())
        comparisons[name] = []
        for task in ("isolated_clause", "neutral_distractors", "mixed_review", "keyed_record"):
            for aspect in ("food", "service", "value"):
                paired = {}
                for row in rows:
                    if (
                        row["split"] == "final_test"
                        and row["task"] == task
                        and row["aspect"] == aspect
                        and row["format"] in (1, 2)
                    ):
                        paired.setdefault((row["group_no"], row["id"].rsplit("-f", 1)[0]), {})[
                            row["format"]
                        ] = row
                diffs = []
                for by_condition in paired.values():
                    if 1 in by_condition and 2 in by_condition:
                        diffs.append(
                            (
                                float(by_condition[2]["strict_correct"])
                                - float(by_condition[1]["strict_correct"]),
                                float(by_condition[2]["conditional_correct"])
                                - float(by_condition[1]["conditional_correct"]),
                            )
                        )
                group_values = {}
                for (group, _), by_condition in paired.items():
                    if 1 in by_condition and 2 in by_condition:
                        group_values.setdefault(group, [[], []])
                        group_values[group][0].append(
                            float(by_condition[2]["strict_correct"])
                            - float(by_condition[1]["strict_correct"])
                        )
                        group_values[group][1].append(
                            float(by_condition[2]["conditional_correct"])
                            - float(by_condition[1]["conditional_correct"])
                        )
                rng = np.random.default_rng(20260923)
                groups = sorted(group_values)
                bootstrap = []
                for _ in range(5000):
                    sampled = rng.choice(groups, len(groups), replace=True)
                    bootstrap.append(
                        [
                            float(np.mean([np.mean(group_values[g][i]) for g in sampled]))
                            for i in range(2)
                        ]
                    )
                comparisons[name].append(
                    {
                        "task": task,
                        "aspect": aspect,
                        "n_paired": len(diffs),
                        "n_groups": len(groups),
                        "format2_minus_format1_strict_accuracy": float(
                            np.mean([x[0] for x in diffs])
                        )
                        if diffs
                        else None,
                        "strict_group_bootstrap_95_ci": np.quantile(
                            np.asarray(bootstrap)[:, 0], [0.025, 0.975]
                        ).tolist()
                        if bootstrap
                        else None,
                        "format2_minus_format1_conditional_accuracy": float(
                            np.mean([x[1] for x in diffs])
                        )
                        if diffs
                        else None,
                        "conditional_group_bootstrap_95_ci": np.quantile(
                            np.asarray(bootstrap)[:, 1], [0.025, 0.975]
                        ).tolist()
                        if bootstrap
                        else None,
                    }
                )
    simple_rule = None
    if "smollm2-1.7b" in results:
        cells = results["smollm2-1.7b"]
        simple = [
            c
            for c in cells
            if c["split"] == "final_test"
            and c["task"] in ("isolated_clause", "neutral_distractors")
        ]
        c_cells = [c for c in simple if c["format"] == 2]
        b_cells = [c for c in simple if c["format"] == 1]
        simple_rule = {
            "all_constraint_cells_at_least_90_percent": bool(
                c_cells and all(c["strict_accuracy"] >= 0.9 for c in c_cells)
            ),
            "pooled_strict_accuracy_gain": float(
                np.mean([c["strict_accuracy"] for c in c_cells])
                - np.mean([c["strict_accuracy"] for c in b_cells])
            ),
            "pooled_conditional_accuracy_change": float(
                np.mean([c["conditional_accuracy"] for c in c_cells])
                - np.mean([c["conditional_accuracy"] for c in b_cells])
            ),
        }
        simple_rule["supported"] = (
            simple_rule["all_constraint_cells_at_least_90_percent"]
            and simple_rule["pooled_strict_accuracy_gain"] >= 0.5
            and simple_rule["pooled_conditional_accuracy_change"] >= -0.03
        )
    analysis = {
        "cells": results,
        "paired_format_comparisons": comparisons,
        "smollm_simple_task_rule": simple_rule,
    }
    write_json(root / "analysis.json", analysis)
    print(
        json.dumps(
            {
                "completed_models": names,
                "smollm_simple_task_rule": simple_rule,
                "paired_format_comparisons": comparisons,
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "analyze", "all"))
    parser.add_argument("--root", type=Path, default=Path("runs/response-policy-control-v1"))
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_SPECS), default=("qwen-1.5b", "smollm2-1.7b")
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from saved batch checkpoints")
    args = parser.parse_args()
    if args.stage in ("collect", "all"):
        if args.root.exists() and not args.resume:
            raise FileExistsError(f"Refusing to overwrite {args.root}; pass --resume to continue")
        args.root.mkdir(parents=True, exist_ok=True)
        for name in args.models:
            path = args.root / name
            path.mkdir(exist_ok=args.resume)
            if (path / "manifest.json").exists():
                continue
            capture(name, path, args.offline)
    if args.stage in ("analyze", "all"):
        summarize(args.root)


if __name__ == "__main__":
    main()
