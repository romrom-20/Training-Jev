"""Test if the shared positive-score intervention transfers across task structures."""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from prompt_effect_forecast import LAYER, sha
from shared_residual_steering import MODELS, load_directions
from task_ladder import ASPECTS, MODEL_SPECS

from latent_decisions.experiment import provenance, write_json
from latent_decisions.target import block_tensor, forward_last, load_target, replace_block_tensor

PROTOCOL = Path("docs/experiments/012-cross-task-shared-shift.md")
RANDOM_SEED = 20260927
BOOTSTRAP_SEED = 20260928
TASKS = ("mixed_review", "isolated_clause", "neutral_distractors", "keyed_record")
PRIMARY_TASKS = ("isolated_clause", "neutral_distractors", "keyed_record")


def interventions(model_name, source_root):
    rows, dose, directions = load_directions(model_name, source_root)
    native = directions["original"]
    axis = native.mean(0)
    axis = axis / axis.norm().clamp_min(1e-12)
    shared = directions["shared"].mean(0)
    gen = torch.Generator(device="cpu").manual_seed(RANDOM_SEED)
    random = torch.randn(shared.shape, generator=gen)
    random = random - torch.dot(random, axis) * axis
    random = random / random.norm().clamp_min(1e-12) * shared.norm()
    return rows, dose, native, shared, random


def capture(model_name, output, source_root, offline):
    rows, dose, native, shared, random = interventions(model_name, source_root)
    prompts = [
        row
        for row in rows
        if row["split"] == "final_test" and row["format"] == 0 and row["task"] in TASKS
    ]
    cases = []
    for row in prompts:
        for source_ix in range(3):
            cases.append({"condition": "native", "source": source_ix, "vector": native[source_ix]})
        cases.extend(
            (
                {"condition": "shared", "source": None, "vector": shared},
                {"condition": "random", "source": None, "vector": random},
            )
        )
    examples = []
    for row_ix, row in enumerate(prompts):
        for condition in cases[row_ix * 5 : (row_ix + 1) * 5]:
            source = condition["source"]
            effect_id = f"{row['id']}|{condition['condition']}|source={source if source is not None else 'na'}"
            examples.append(
                {
                    "effect_id": effect_id,
                    "base_id": row["id"],
                    "group": row["group"],
                    "group_no": row["group_no"],
                    "task": row["task"],
                    "target": row["active"],
                    "condition": condition["condition"],
                    "source": source,
                    "vector": condition["vector"],
                    "conditional_p_positive": row["conditional_p_positive"],
                    "user": row["user"],
                }
            )
    expected_counts = {"mixed_review": 576, "isolated_clause": 144, "neutral_distractors": 144, "keyed_record": 576}
    counts = {task: sum(row["task"] == task for row in prompts) for task in TASKS}
    if counts != expected_counts or len(examples) != 7200:
        raise ValueError(f"Unexpected test design: {counts}, {len(examples)} effects")

    partial = output / "outcomes.partial.jsonl"
    completed = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                completed[record["effect_id"]] = record
    expected = {case["effect_id"] for case in examples}
    if not set(completed).issubset(expected):
        raise ValueError("Checkpoint contains IDs outside the frozen test set")
    pending = [case for case in examples if case["effect_id"] not in completed]

    spec = dict(MODEL_SPECS[model_name], name=model_name)
    model, tokenizer, device = load_target(spec, "auto", offline)
    label_ids = [tokenizer.encode(token, add_special_tokens=False) for token in ("positive", "negative")]
    if any(len(token) != 1 for token in label_ids):
        raise ValueError("Positive and negative labels must be one token each")
    pos_id, neg_id = [token[0] for token in label_ids]
    block = model.model.layers[LAYER - 1]
    with partial.open("a") as stream:
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
            delta = torch.stack([item["vector"] for item in batch]).to(device) * dose

            def patch(module, inputs, output):
                value = block_tensor(output).clone()
                value[:, -1, :] += delta
                return replace_block_tensor(output, value)

            hook = block.register_forward_hook(patch)
            try:
                with torch.inference_mode():
                    logits = forward_last(model, tokens)
            finally:
                hook.remove()
            steered = (logits[:, pos_id] - logits[:, neg_id]).float().cpu().tolist()
            for case, score in zip(batch, steered):
                p = min(max(float(case["conditional_p_positive"]), 1e-8), 1 - 1e-8)
                baseline = math.log(p / (1 - p))
                record = {
                    **{key: case[key] for key in ("effect_id", "base_id", "group", "group_no", "task", "target", "condition", "source")},
                    "unmodified_log_odds": baseline,
                    "steered_log_odds": score,
                    "observed_delta": score - baseline,
                }
                completed[record["effect_id"]] = record
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
            if (offset + len(batch)) % 256 == 0 or offset + len(batch) == len(pending):
                print(f"{model_name}: captured {len(completed)}/{len(examples)}", flush=True)
    if set(completed) != expected:
        raise ValueError("Capture is incomplete")
    outcomes = [completed[case["effect_id"]] for case in examples]
    write_json(output / "outcomes.json", outcomes)
    source = source_root / model_name
    source_manifest = json.loads((source / "manifest.json").read_text())
    manifest = {
        "model": spec,
        "device": str(device),
        "layer": LAYER,
        "dose_fraction": 0.05,
        "dose_l2": dose,
        "n_prompts": len(prompts),
        "n_effects": len(outcomes),
        "task_prompt_counts": counts,
        "random_seed": RANDOM_SEED,
        "source_dataset_sha256": sha(source / "dataset.json"),
        "source_activation_sha256": source_manifest["activation_sha256"],
        "protocol_sha256": sha(PROTOCOL),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outcomes_sha256": sha(output / "outcomes.json"),
        "provenance": provenance(),
    }
    write_json(output / "manifest.json", manifest)
    del model
    gc.collect()
    if str(device).startswith("mps"):
        torch.mps.empty_cache()


def bootstrap_ci(differences, groups, reps=5000, seed=BOOTSTRAP_SEED):
    unique = np.unique(groups)
    by_group = {group: differences[groups == group] for group in unique}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(reps):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        means.append(np.concatenate([by_group[group] for group in sampled]).mean())
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def analyze_model(root, model_name):
    outcomes = json.loads((root / model_name / "outcomes.json").read_text())
    task_results = {}
    for task in TASKS:
        records = [x for x in outcomes if x["task"] == task]
        n_groups = len({x["group_no"] for x in records})
        if n_groups != 24:
            raise ValueError(f"{model_name} {task} has {n_groups} groups")
        per_prompt = {}
        for record in records:
            per_prompt.setdefault(record["base_id"], {})[(record["condition"], record["source"])] = record
        shared_deltas = np.array([x["observed_delta"] for x in records if x["condition"] == "shared"])
        random_deltas = np.array([x["observed_delta"] for x in records if x["condition"] == "random"])
        shared_map = {x["base_id"]: x for x in records if x["condition"] == "shared"}
        random_map = {x["base_id"]: x for x in records if x["condition"] == "random"}
        ids = sorted(shared_map)
        diff = np.array([shared_map[i]["observed_delta"] - random_map[i]["observed_delta"] for i in ids])
        group_for_id = {i: shared_map[i]["group_no"] for i in ids}
        native = [x for x in records if x["condition"] == "native"]
        native_selectivity_by_source = {}
        for source in range(3):
            on = [x["observed_delta"] for x in native if x["source"] == source and x["target"] == source]
            off = [x["observed_delta"] for x in native if x["source"] == source and x["target"] != source]
            native_selectivity_by_source[ASPECTS[source]] = float(np.mean(on) - np.mean(off))
        task_results[task] = {
            "n_prompts": len(ids),
            "n_groups": n_groups,
            "mean_effect": {
                "shared": float(shared_deltas.mean()),
                "random": float(random_deltas.mean()),
                "native_mean_over_sources": float(np.mean([x["observed_delta"] for x in native])),
                "shared_minus_random": float(diff.mean()),
            },
            "shared_minus_random_group_bootstrap_95_ci": bootstrap_ci(diff, np.array([group_for_id[i] for i in ids])),
            "native_target_selectivity_by_source": native_selectivity_by_source,
            "shared_mean_by_target": {
                ASPECTS[t]: float(np.mean([x["observed_delta"] for x in records if x["condition"] == "shared" and x["target"] == t]))
                for t in range(3)
            },
            "all_shared_effects_positive": bool(np.all(shared_deltas > 0)),
        }
    rule_pass = all(
        task_results[task]["shared_minus_random_group_bootstrap_95_ci"][0] > 0
        for task in PRIMARY_TASKS
    )
    return {"model": model_name, "all_three_primary_tasks_pass": rule_pass, "tasks": task_results}


def analyze(root):
    results = {model: analyze_model(root, model) for model in MODELS}
    write_json(root / "analysis.json", results)
    print(json.dumps({m: {"all_three_primary_tasks_pass": x["all_three_primary_tasks_pass"], "primary_tasks": {t: x["tasks"][t] for t in PRIMARY_TASKS}} for m,x in results.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "analyze", "all"))
    parser.add_argument("--root", type=Path, default=Path("runs/cross-task-shared-shift-v1"))
    parser.add_argument("--source-root", type=Path, default=Path("runs/task-ladder-v1"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage in ("collect", "all"):
        if args.root.exists() and not args.resume:
            raise FileExistsError(f"Refusing to overwrite {args.root}; use --resume")
        args.root.mkdir(parents=True, exist_ok=True)
        for model in args.models:
            output = args.root / model
            output.mkdir(exist_ok=args.resume)
            if not (output / "manifest.json").exists():
                capture(model, output, args.source_root, args.offline)
    if args.stage in ("analyze", "all"):
        analyze(args.root)


if __name__ == "__main__":
    main()
