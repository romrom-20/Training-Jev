"""Decompose layer-16 TripR aspect directions into shared and residual parts."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
import tripr_layer_confirmation as tripr
from mams_layer_selectivity import directions_at_layer
from prompt_effect_forecast import sha
from task_ladder import MODEL_SPECS

from latent_decisions.experiment import provenance, write_json
from latent_decisions.steering import decompose_shared_residual
from latent_decisions.target import (
    block_tensor,
    forward_last,
    load_target,
    replace_block_tensor,
)

PROTOCOL = Path("docs/experiments/019-tripr-shared-residual-layer16.md")
BASELINE_ROOT = Path("runs/tripr-layer-confirmation-v1")
MODELS = ("qwen-1.5b", "smollm2-1.7b")
LAYER = 16
CONDITIONS = ("native", "shared", "residual", "random_residual")
RANDOM_SEED = 20261001


def collect_model(model_name, root, source_root, data_path, offline, resume):
    folder = root / model_name
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        if resume:
            print(f"{model_name}: already complete", flush=True)
            return
        raise FileExistsError(f"Refusing to overwrite {folder}; use --resume")

    stimuli, sentence_labels, conflicts = tripr.load_tripr(data_path)
    vectors, dose, activation_hash, dataset_hash = directions_at_layer(
        model_name, LAYER, source_root
    )
    arms = decompose_shared_residual(vectors["native"], RANDOM_SEED)
    source = source_root / model_name
    expected_baseline = BASELINE_ROOT / f"layer-{LAYER:02d}" / model_name / "baseline.json"
    baseline_path = folder / "baseline.json"
    if not baseline_path.exists():
        if not expected_baseline.exists():
            raise FileNotFoundError(f"Missing frozen 018 baseline: {expected_baseline}")
        baseline_path.write_bytes(expected_baseline.read_bytes())
    baseline = json.loads(baseline_path.read_text())
    if {row["id"] for row in baseline} != {row["id"] for row in stimuli}:
        raise ValueError("Copied baseline IDs differ from the frozen TripR prompt filter")
    baseline_by_id = {row["id"]: row for row in baseline}

    cases = []
    for stimulus in stimuli:
        for condition in CONDITIONS:
            for source_aspect in range(3):
                cases.append(
                    (
                        stimulus,
                        condition,
                        source_aspect,
                        arms[condition][source_aspect],
                    )
                )
    partial_path = folder / "effects.partial.jsonl"
    completed = {}
    if partial_path.exists():
        for line in partial_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["effect_id"]] = row
    expected_ids = {
        f"{stim['id']}|{condition}|source={source}"
        for stim, condition, source, _vector in cases
    }
    if not set(completed).issubset(expected_ids):
        raise ValueError("Checkpoint contains unknown effect IDs")
    pending = [
        case
        for case in cases
        if f"{case[0]['id']}|{case[1]}|source={case[2]}" not in completed
    ]

    spec = dict(MODEL_SPECS[model_name], name=model_name)
    model, tokenizer, device = load_target(spec, "auto", offline)
    label_ids = [tokenizer.encode(word, add_special_tokens=False) for word in ("positive", "negative")]
    if any(len(ids) != 1 for ids in label_ids) or label_ids[0][0] == label_ids[1][0]:
        raise ValueError("Positive and negative must be distinct single tokens")
    pos_id, neg_id = label_ids[0][0], label_ids[1][0]
    block = model.model.layers[LAYER - 1]
    with partial_path.open("a") as stream:
        for offset in range(0, len(pending), 8):
            batch = pending[offset : offset + 8]
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

            def patch(_module, _inputs, output):
                value = block_tensor(output).clone()
                value[:, -1, :] += delta
                return replace_block_tensor(output, value)

            hook = block.register_forward_hook(patch)
            try:
                with torch.inference_mode():
                    logits = forward_last(model, tokens)
            finally:
                hook.remove()
            margins = (logits[:, pos_id] - logits[:, neg_id]).float().cpu().tolist()
            top_ids = logits.argmax(-1).cpu().tolist()
            for (stim, condition, source_aspect, _vector), margin, top_id in zip(
                batch, margins, top_ids
            ):
                base = baseline_by_id[stim["id"]]
                row = {
                    "effect_id": f"{stim['id']}|{condition}|source={source_aspect}",
                    "id": stim["id"],
                    "sentence_id": stim["sentence_id"],
                    "target": stim["target"],
                    "source": source_aspect,
                    "condition": condition,
                    "label": stim["label"],
                    "baseline_margin": base["positive_logit"] - base["negative_logit"],
                    "steered_margin": margin,
                    "margin_delta": margin - (base["positive_logit"] - base["negative_logit"]),
                    "steered_strict_correct": int(
                        top_id == (pos_id if stim["label"] else neg_id)
                    ),
                }
                completed[row["effect_id"]] = row
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            stream.flush()
            if (offset + len(batch)) % 256 == 0 or offset + len(batch) == len(pending):
                print(f"{model_name}: {len(completed)}/{len(cases)} effects", flush=True)
    if set(completed) != expected_ids:
        raise ValueError("Effect capture incomplete")
    write_json(folder / "effects.json", [completed[key] for key in sorted(expected_ids)])
    source_manifest = json.loads((source / "manifest.json").read_text())
    geometry = {}
    for condition, matrix in arms.items():
        geometry[condition] = [float(v.norm()) for v in matrix]
    write_json(
        folder / "manifest.json",
        {
            "model": spec,
            "device": str(device),
            "dtype": "float32",
            "layer": LAYER,
            "dose_fraction_of_training_activation_norm": 0.05,
            "dose_l2": dose,
            "n_prompts": len(stimuli),
            "n_sentences": len(sentence_labels),
            "n_conflict_sentences": conflicts,
            "conditions": CONDITIONS,
            "random_residual_seed": RANDOM_SEED,
            "component_l2_norms": geometry,
            "decomposition_code_sha256": sha(Path("src/latent_decisions/steering.py")),
            "source_activation_sha256": activation_hash,
            "source_manifest_activation_sha256": source_manifest["activation_sha256"],
            "source_dataset_sha256": dataset_hash,
            "dataset_xml_sha256": sha(data_path),
            "protocol_sha256": sha(PROTOCOL),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "direction_helper_sha256": sha(Path("scripts/mams_layer_selectivity.py")),
            "dataset_helper_sha256": sha(Path("scripts/tripr_layer_confirmation.py")),
            "baseline_sha256": sha(baseline_path),
            "effects_sha256": sha(folder / "effects.json"),
            "provenance": provenance(),
        },
    )
    del model
    gc.collect()
    if str(device).startswith("mps"):
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "all"))
    parser.add_argument("--root", type=Path, default=Path("runs/tripr-component-decomposition-v1"))
    parser.add_argument("--source-root", type=Path, default=Path("runs/task-ladder-v1"))
    parser.add_argument("--data", type=Path, default=tripr.DATA)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        collect_model(model, args.root, args.source_root, args.data, args.offline, args.resume)


if __name__ == "__main__":
    main()
