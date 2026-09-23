"""Cross-encode a fixed small-model sentiment steering intervention."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from prompt_effect_forecast import LAYER, sha
from shared_residual_steering import MODELS, load_directions
from task_ladder import MODEL_SPECS

from latent_decisions.experiment import provenance, write_json
from latent_decisions.target import block_tensor, forward_last, load_target, replace_block_tensor

PROTOCOL = Path("docs/experiments/013-answer-encoding-control.md")
MAPPINGS = ("positive_is_A", "positive_is_B")
CONDITIONS = ("native", "shared", "random")
BOOTSTRAP_SEED = 20260929


def mapped_prompt(user, mapping):
    old = "Reply with exactly one word: positive or negative."
    new = (
        "Reply with exactly one letter. A means positive and B means negative."
        if mapping == "positive_is_A"
        else "Reply with exactly one letter. A means negative and B means positive."
    )
    if user.count(old) != 1:
        raise ValueError("Expected exactly one frozen answer instruction to replace")
    return user.replace(old, new)


def setup(model_name, source_root):
    rows, dose, directions = load_directions(model_name, source_root)
    native = directions["original"]
    axis = native.mean(0)
    axis = axis / axis.norm().clamp_min(1e-12)
    shared = directions["shared"].mean(0)
    generator = torch.Generator(device="cpu").manual_seed(20260930)
    random = torch.randn(shared.shape, generator=generator)
    random -= torch.dot(random, axis) * axis
    random = random / random.norm().clamp_min(1e-12) * shared.norm()
    return rows, dose, {"native": native, "shared": shared, "random": random}


def baseline_capture(model, tokenizer, device, prompts, output):
    id_a = tokenizer.encode("A", add_special_tokens=False)[0]
    id_b = tokenizer.encode("B", add_special_tokens=False)[0]
    partial = output / "baseline.partial.jsonl"
    records = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                records[(row["base_id"], row["mapping"])] = row
    expected = {(row["id"], mapping) for row in prompts for mapping in MAPPINGS}
    if not set(records).issubset(expected):
        raise ValueError("Baseline checkpoint contains unknown prompts")
    pending = [(row, mapping) for row in prompts for mapping in MAPPINGS if (row["id"], mapping) not in records]
    with partial.open("a") as stream:
        for offset in range(0, len(pending), 8):
            batch = pending[offset : offset + 8]
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": mapped_prompt(row["user"], mapping)}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row, mapping in batch
            ]
            tokens = tokenizer(texts, padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                logits = forward_last(model, tokens)
            a = logits[:, id_a].float().cpu().tolist()
            b = logits[:, id_b].float().cpu().tolist()
            top = logits.argmax(-1).cpu().tolist()
            for (row, mapping), la, lb, top_id in zip(batch, a, b, top):
                positive_id, negative_id = (id_a, id_b) if mapping == "positive_is_A" else (id_b, id_a)
                correct = positive_id if row["label"] == 1 else negative_id
                record = {
                    "base_id": row["id"],
                    "group": row["group"],
                    "group_no": row["group_no"],
                    "target": row["active"],
                    "label": row["label"],
                    "mapping": mapping,
                    "logit_a": la,
                    "logit_b": lb,
                    "top_token_id": int(top_id),
                    "correct_token_id": int(correct),
                    "mapped_next_token_correct": int(top_id) == int(correct),
                    "valid_a_or_b": int(top_id) in {id_a, id_b},
                }
                records[(record["base_id"], mapping)] = record
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
            if offset % 64 == 0 or offset + len(batch) == len(pending):
                print(f"baseline: {len(records)}/{len(expected)}", flush=True)
    if set(records) != expected:
        raise ValueError("Baseline capture is incomplete")
    ordered = [records[(row["id"], mapping)] for row in prompts for mapping in MAPPINGS]
    write_json(output / "baseline.json", ordered)
    return records


def treatment_capture(model_name, output, source_root, offline):
    rows, dose, vectors = setup(model_name, source_root)
    prompts = [
        row for row in rows
        if row["split"] == "final_test" and row["format"] == 0 and row["task"] == "isolated_clause"
    ]
    if len(prompts) != 144 or len({row["group_no"] for row in prompts}) != 24:
        raise ValueError("Expected 144 isolated-clause test prompts in 24 groups")
    spec = dict(MODEL_SPECS[model_name], name=model_name)
    model, tokenizer, device = load_target(spec, "auto", offline)
    a_ids = tokenizer.encode("A", add_special_tokens=False)
    b_ids = tokenizer.encode("B", add_special_tokens=False)
    if len(a_ids) != 1 or len(b_ids) != 1 or a_ids[0] == b_ids[0]:
        raise ValueError("A and B must be distinct single-token candidates")
    pos_ids = tokenizer.encode("positive", add_special_tokens=False)
    neg_ids = tokenizer.encode("negative", add_special_tokens=False)
    if len(pos_ids) != 1 or len(neg_ids) != 1:
        raise ValueError("Positive and negative must be single tokens in source direction setup")
    baseline = baseline_capture(model, tokenizer, device, prompts, output)

    cases = []
    for row in prompts:
        for mapping in MAPPINGS:
            for source in range(3):
                cases.append({"row": row, "mapping": mapping, "condition": "native", "source": source, "vector": vectors["native"][source]})
            for condition in ("shared", "random"):
                cases.append({"row": row, "mapping": mapping, "condition": condition, "source": None, "vector": vectors[condition]})
    if len(cases) != 1440:
        raise ValueError(f"Unexpected treatment count: {len(cases)}")
    partial = output / "effects.partial.jsonl"
    completed = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                completed[item["effect_id"]] = item
    expected = {
        f"{case['row']['id']}|{case['mapping']}|{case['condition']}|{case['source'] if case['source'] is not None else 'na'}"
        for case in cases
    }
    if not set(completed).issubset(expected):
        raise ValueError("Treatment checkpoint contains unknown IDs")
    pending = [case for case in cases if f"{case['row']['id']}|{case['mapping']}|{case['condition']}|{case['source'] if case['source'] is not None else 'na'}" not in completed]
    block = model.model.layers[LAYER - 1]
    with partial.open("a") as stream:
        for offset in range(0, len(pending), 8):
            batch = pending[offset : offset + 8]
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": mapped_prompt(case["row"]["user"], case["mapping"])}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for case in batch
            ]
            tokens = tokenizer(texts, padding=True, return_tensors="pt").to(device)
            delta = torch.stack([case["vector"] for case in batch]).to(device) * dose

            def patch(module, inputs, output_value):
                value = block_tensor(output_value).clone()
                value[:, -1, :] += delta
                return replace_block_tensor(output_value, value)

            hook = block.register_forward_hook(patch)
            try:
                with torch.inference_mode():
                    logits = forward_last(model, tokens)
            finally:
                hook.remove()
            la = logits[:, a_ids[0]].float().cpu().tolist()
            lb = logits[:, b_ids[0]].float().cpu().tolist()
            top = logits.argmax(-1).cpu().tolist()
            for case, steered_a, steered_b, top_id in zip(batch, la, lb, top):
                row, mapping = case["row"], case["mapping"]
                base = baseline[(row["id"], mapping)]
                pos_is_a = mapping == "positive_is_A"
                base_sem = base["logit_a"] - base["logit_b"] if pos_is_a else base["logit_b"] - base["logit_a"]
                steer_sem = steered_a - steered_b if pos_is_a else steered_b - steered_a
                identifier_delta = (steered_a - steered_b) - (base["logit_a"] - base["logit_b"])
                effect_id = f"{row['id']}|{mapping}|{case['condition']}|{case['source'] if case['source'] is not None else 'na'}"
                record = {
                    "effect_id": effect_id,
                    "base_id": row["id"],
                    "group": row["group"],
                    "group_no": row["group_no"],
                    "target": row["active"],
                    "label": row["label"],
                    "mapping": mapping,
                    "condition": case["condition"],
                    "source": case["source"],
                    "baseline_semantic_margin": base_sem,
                    "steered_semantic_margin": steer_sem,
                    "semantic_margin_delta": steer_sem - base_sem,
                    "identifier_ab_delta": identifier_delta,
                    "steered_logit_a": steered_a,
                    "steered_logit_b": steered_b,
                    "steered_top_token_id": int(top_id),
                    "steered_valid_a_or_b": int(top_id) in {a_ids[0], b_ids[0]},
                    "baseline_mapped_next_token_correct": base["mapped_next_token_correct"],
                    "baseline_valid_a_or_b": base["valid_a_or_b"],
                }
                completed[effect_id] = record
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
            if (offset + len(batch)) % 256 == 0 or offset + len(batch) == len(pending):
                print(f"{model_name}: interventions {len(completed)}/{len(cases)}", flush=True)
    if set(completed) != expected:
        raise ValueError("Treatment capture incomplete")
    outcomes = [completed[f"{case['row']['id']}|{case['mapping']}|{case['condition']}|{case['source'] if case['source'] is not None else 'na'}"] for case in cases]
    write_json(output / "effects.json", outcomes)
    baseline_ordered = [baseline[(row["id"], mapping)] for row in prompts for mapping in MAPPINGS]
    source = source_root / model_name
    source_manifest = json.loads((source / "manifest.json").read_text())
    manifest = {
        "model": spec,
        "device": str(device),
        "layer": LAYER,
        "dose_fraction": 0.05,
        "dose_l2": dose,
        "n_prompts": len(prompts),
        "n_baseline_rows": len(baseline_ordered),
        "n_effects": len(outcomes),
        "mapping_counts": {mapping: 720 for mapping in MAPPINGS},
        "source_dataset_sha256": sha(source / "dataset.json"),
        "source_activation_sha256": source_manifest["activation_sha256"],
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


def bootstrap_ci(values, groups, reps=5000, seed=BOOTSTRAP_SEED):
    unique = np.unique(groups)
    by_group = {group: values[groups == group] for group in unique}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(reps):
        selected = rng.choice(unique, size=len(unique), replace=True)
        estimates.append(np.concatenate([by_group[group] for group in selected]).mean())
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def analyze_model(root, model_name):
    baseline = json.loads((root / model_name / "baseline.json").read_text())
    effects = json.loads((root / model_name / "effects.json").read_text())
    mappings = {}
    for mapping in MAPPINGS:
        base_rows = [x for x in baseline if x["mapping"] == mapping]
        competence = float(np.mean([x["mapped_next_token_correct"] for x in base_rows]))
        valid_rate = float(np.mean([x["valid_a_or_b"] for x in base_rows]))
        by_condition = {}
        for condition in CONDITIONS:
            records = [x for x in effects if x["mapping"] == mapping and x["condition"] == condition]
            by_condition[condition] = {
                "mean_semantic_margin_delta": float(np.mean([x["semantic_margin_delta"] for x in records])),
                "mean_identifier_ab_delta": float(np.mean([x["identifier_ab_delta"] for x in records])),
                "n": len(records),
            }
        shared = {x["base_id"]: x for x in effects if x["mapping"] == mapping and x["condition"] == "shared"}
        random = {x["base_id"]: x for x in effects if x["mapping"] == mapping and x["condition"] == "random"}
        ids = sorted(shared)
        diffs = np.array([shared[i]["semantic_margin_delta"] - random[i]["semantic_margin_delta"] for i in ids])
        groups = np.array([shared[i]["group_no"] for i in ids])
        by_condition["shared_minus_random"] = {
            "mean_semantic_margin_delta": float(diffs.mean()),
            "group_bootstrap_95_ci": bootstrap_ci(diffs, groups),
        }
        mappings[mapping] = {
            "baseline_mapped_next_token_accuracy": competence,
            "baseline_valid_a_or_b_rate": valid_rate,
            "competence_gate_pass": competence >= 0.90,
            "condition_effects": by_condition,
        }
    gate = all(mappings[m]["competence_gate_pass"] for m in MAPPINGS) and all(
        mappings[m]["condition_effects"]["shared_minus_random"]["group_bootstrap_95_ci"][0] > 0
        for m in MAPPINGS
    )
    return {"model": model_name, "semantic_mapping_robustness_gate_pass": gate, "mappings": mappings}


def analyze(root):
    results = {model: analyze_model(root, model) for model in MODELS}
    write_json(root / "analysis.json", results)
    print(json.dumps(results, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "analyze", "all"))
    parser.add_argument("--root", type=Path, default=Path("runs/answer-encoding-control-v1"))
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
                treatment_capture(model, output, args.source_root, args.offline)
    if args.stage in ("analyze", "all"):
        analyze(args.root)


if __name__ == "__main__":
    main()
