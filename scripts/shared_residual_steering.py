"""Test shared and aspect-residual components of synthetic review steering vectors."""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from prompt_effect_forecast import DOSE_FRACTION, LAYER, sha
from task_ladder import ASPECTS, MODEL_SPECS

from latent_decisions.experiment import provenance, write_json
from latent_decisions.target import block_tensor, forward_last, load_target, replace_block_tensor

PROTOCOL = Path("docs/experiments/011-shared-vs-residual-steering.md")
SEED = 20260925
MODELS = ("qwen-1.5b", "smollm2-1.7b")


def load_directions(model_name, source_root):
    source = source_root / model_name
    rows = json.loads((source / "dataset.json").read_text())
    with np.load(source / "activations.npz", allow_pickle=False) as cache:
        h = cache["h"]
    layer_ix = MODEL_SPECS[model_name]["layers"].index(LAYER)
    directions = []
    norms = []
    for aspect in range(3):
        chosen = [
            i
            for i, row in enumerate(rows)
            if row["split"] == "train"
            and row["task"] == "mixed_review"
            and row["format"] == 0
            and row["active"] == aspect
        ]
        pos = h[[i for i in chosen if rows[i]["label"] == 1], layer_ix]
        neg = h[[i for i in chosen if rows[i]["label"] == 0], layer_ix]
        vector = torch.from_numpy(pos.mean(0) - neg.mean(0)).float()
        directions.append(vector / vector.norm().clamp_min(1e-12))
        norms.extend(np.linalg.norm(h[chosen, layer_ix], axis=-1).tolist())
    dose = DOSE_FRACTION * float(np.median(norms))
    original = torch.stack(directions)
    shared_axis = original.mean(0)
    shared_axis = shared_axis / shared_axis.norm().clamp_min(1e-12)
    projected, residual, random_residual = [], [], []
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    for direction in original:
        component = torch.dot(direction, shared_axis) * shared_axis
        remainder = direction - component
        rand = torch.randn(direction.shape, generator=generator)
        rand = rand - torch.dot(rand, shared_axis) * shared_axis
        rand = rand / rand.norm().clamp_min(1e-12) * remainder.norm()
        projected.append(component)
        residual.append(remainder)
        random_residual.append(rand)
    return rows, dose, {
        "original": original,
        "shared": torch.stack(projected),
        "residual": torch.stack(residual),
        "random_residual": torch.stack(random_residual),
    }


def capture(model_name, root, source_root, offline):
    rows, dose, arms = load_directions(model_name, source_root)
    source = source_root / model_name
    prompts = [
        row
        for row in rows
        if row["split"] == "final_test"
        and row["task"] == "mixed_review"
        and row["format"] == 0
    ]
    examples = []
    for row in prompts:
        for arm, vectors in arms.items():
            for source_aspect in range(3):
                examples.append(
                    {
                        "effect_id": f"{row['id']}|arm={arm}|source={source_aspect}",
                        "base_id": row["id"],
                        "group": row["group"],
                        "group_no": row["group_no"],
                        "source": source_aspect,
                        "target": row["active"],
                        "arm": arm,
                        "vector": vectors[source_aspect],
                        "conditional_p_positive": row["conditional_p_positive"],
                        "user": row["user"],
                    }
                )
    if (len(prompts), len(examples)) != (576, 6912):
        raise ValueError(f"Unexpected final-test design size: {len(prompts)} prompts, {len(examples)} effects")
    partial = root / "outcomes.partial.jsonl"
    completed = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                completed[record["effect_id"]] = record
    expected = {item["effect_id"] for item in examples}
    if not set(completed).issubset(expected):
        raise ValueError("Checkpoint contains effects outside the frozen dataset")
    pending = [item for item in examples if item["effect_id"] not in completed]
    spec = dict(MODEL_SPECS[model_name], name=model_name)
    model, tokenizer, device = load_target(spec, "auto", offline)
    label_ids = [tokenizer.encode(w, add_special_tokens=False) for w in ("positive", "negative")]
    if any(len(ids) != 1 for ids in label_ids):
        raise ValueError("Positive and negative must each be one token")
    pos_id, neg_id = [ids[0] for ids in label_ids]
    block = model.model.layers[LAYER - 1]
    batch_size = 8
    with partial.open("a") as stream:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset : offset + batch_size]
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
            log_odds = (logits[:, pos_id] - logits[:, neg_id]).float().cpu().tolist()
            for item, steered in zip(batch, log_odds):
                p = min(max(float(item["conditional_p_positive"]), 1e-8), 1 - 1e-8)
                baseline = math.log(p / (1 - p))
                record = {
                    **{key: item[key] for key in ("effect_id", "base_id", "group", "group_no", "source", "target", "arm")},
                    "unmodified_log_odds": baseline,
                    "steered_log_odds": steered,
                    "observed_delta": steered - baseline,
                }
                completed[record["effect_id"]] = record
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
            if (offset + len(batch)) % 256 == 0 or offset + len(batch) == len(pending):
                print(f"{model_name}: captured {len(completed)}/{len(examples)}", flush=True)
    if set(completed) != expected:
        raise ValueError("Capture is incomplete")
    outcomes = [completed[item["effect_id"]] for item in examples]
    write_json(root / "outcomes.json", outcomes)
    manifest = {
        "model": spec,
        "device": str(device),
        "layer": LAYER,
        "dose_fraction": DOSE_FRACTION,
        "dose_l2": dose,
        "n_prompts": len(prompts),
        "n_effects": len(outcomes),
        "direction_seed": SEED,
        "source_dataset_sha256": sha(source / "dataset.json"),
        "source_activation_sha256": json.loads((source / "manifest.json").read_text())["activation_sha256"],
        "protocol_sha256": sha(PROTOCOL),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outcomes_sha256": sha(root / "outcomes.json"),
        "provenance": provenance(),
    }
    write_json(root / "manifest.json", manifest)
    del model
    gc.collect()
    if str(device).startswith("mps"):
        torch.mps.empty_cache()


def analyze_model(root, model_name):
    outcomes = json.loads((root / model_name / "outcomes.json").read_text())
    groups = sorted({x["group_no"] for x in outcomes})
    if len(groups) != 24:
        raise ValueError(f"Expected 24 independent scenario groups, got {len(groups)}")

    def summarize(records):
        means = {}
        for arm in ("original", "shared", "residual", "random_residual"):
            for source in range(3):
                for target in range(3):
                    vals = [
                        x["observed_delta"]
                        for x in records
                        if x["arm"] == arm and x["source"] == source and x["target"] == target
                    ]
                    means[(arm, source, target)] = float(np.mean(vals))
        selectivity = {}
        for arm in ("original", "shared", "residual", "random_residual"):
            per_source = []
            for source in range(3):
                on = means[(arm, source, source)]
                off = np.mean([means[(arm, source, target)] for target in range(3) if target != source])
                per_source.append(on - off)
            selectivity[arm] = float(np.mean(per_source))
        return means, selectivity

    means, selectivity = summarize(outcomes)
    rng = np.random.default_rng(20260926)
    grouped = {g: [x for x in outcomes if x["group_no"] == g] for g in groups}
    draws = {key: [] for key in (("residual", "shared"), ("residual", "random_residual"), ("original", "shared"))}
    for _ in range(5000):
        selected = rng.choice(groups, size=len(groups), replace=True)
        sample = [x for group in selected for x in grouped[group]]
        _, s = summarize(sample)
        for a, b in draws:
            draws[(a, b)].append(s[a] - s[b])
    intervals = {f"{a}_minus_{b}": [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for (a,b),v in draws.items()}
    cells = []
    for arm in ("original", "shared", "residual", "random_residual"):
        for source in range(3):
            for target in range(3):
                cells.append({"arm": arm, "source": ASPECTS[source], "target": ASPECTS[target], "mean_effect": means[(arm,source,target)]})
    return {
        "model": model_name,
        "n_groups": len(groups),
        "overall_mean_effect_by_arm": {arm: float(np.mean([x["observed_delta"] for x in outcomes if x["arm"] == arm])) for arm in ("original", "shared", "residual", "random_residual")},
        "target_selectivity_by_arm": selectivity,
        "paired_group_bootstrap_95_ci": intervals,
        "all_source_target_cell_means": cells,
        "all_original_effects_positive": all(x["observed_delta"] > 0 for x in outcomes if x["arm"] == "original"),
    }


def analyze(root):
    results = {model: analyze_model(root, model) for model in MODELS}
    write_json(root / "analysis.json", results)
    print(json.dumps({m: {"overall_mean_effect_by_arm": x["overall_mean_effect_by_arm"], "target_selectivity_by_arm": x["target_selectivity_by_arm"], "paired_group_bootstrap_95_ci": x["paired_group_bootstrap_95_ci"]} for m,x in results.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "analyze", "all"))
    parser.add_argument("--root", type=Path, default=Path("runs/shared-residual-steering-v1"))
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
            dest = args.root / model
            dest.mkdir(exist_ok=args.resume)
            if not (dest / "manifest.json").exists():
                capture(model, dest, args.source_root, args.offline)
    if args.stage in ("analyze", "all"):
        analyze(args.root)


if __name__ == "__main__":
    main()
