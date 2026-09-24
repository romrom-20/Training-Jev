"""Test the learned TripR residual against 20 seeded random-residual controls."""

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

PROTOCOL = Path("docs/experiments/020-tripr-residual-control-seeds.md")
BASELINE_ROOT = Path("runs/tripr-layer-confirmation-v1")
MODELS = ("qwen-1.5b", "smollm2-1.7b")
SEEDS = tuple(range(20261101, 20261121))
CONDITIONS = ("trained_residual",) + tuple(f"random_{seed}" for seed in SEEDS)
LAYER = 16
DOSE_FRACTION = 0.05


def collect_model(model_name, root, source_root, data_path, offline, resume):
    folder = root / model_name
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        if resume:
            print(f"{model_name}: already complete", flush=True)
            return
        raise FileExistsError(f"Refusing to overwrite {folder}; use --resume")

    all_stimuli, sentence_labels, conflict_count = tripr.load_tripr(data_path)
    conflict_ids = {
        sid
        for sid, labels in sentence_labels.items()
        if len(set(labels.values())) > 1
    }
    stimuli = [row for row in all_stimuli if row["sentence_id"] in conflict_ids]
    if (len(conflict_ids), len(stimuli), conflict_count) != (29, 63, 29):
        raise ValueError("Frozen conflict subset changed")
    vectors, dose, activation_hash, dataset_hash = directions_at_layer(
        model_name, LAYER, source_root
    )
    native_parts = decompose_shared_residual(vectors["native"], seed=20261001)
    trained = native_parts["residual"]
    random_by_seed = {
        seed: decompose_shared_residual(vectors["native"], seed=seed)["random_residual"]
        for seed in SEEDS
    }

    baseline_source = BASELINE_ROOT / f"layer-{LAYER:02d}" / model_name / "baseline.json"
    baseline_path = folder / "baseline.json"
    baseline_all = json.loads(baseline_source.read_text())
    stimulus_ids = {row["id"] for row in stimuli}
    baseline = [row for row in baseline_all if row["id"] in stimulus_ids]
    if len(baseline) != len(stimuli):
        raise ValueError("Frozen baseline missing conflict-query IDs")
    if not baseline_path.exists():
        write_json(baseline_path, baseline)
    elif json.loads(baseline_path.read_text()) != baseline:
        raise ValueError("Existing baseline differs from frozen conflict slice")
    baseline_by_id = {row["id"]: row for row in baseline}

    cases = []
    for stimulus in stimuli:
        for source_aspect in range(3):
            cases.append((stimulus, None, source_aspect, trained[source_aspect]))
            for seed in SEEDS:
                cases.append(
                    (stimulus, seed, source_aspect, random_by_seed[seed][source_aspect])
                )
    expected = {
        f"{stim['id']}|seed={seed if seed is not None else 'trained'}|source={source}"
        for stim, seed, source, _vector in cases
    }
    partial_path = folder / "effects.partial.jsonl"
    completed = {}
    if partial_path.exists():
        for line in partial_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["effect_id"]] = row
    if not set(completed).issubset(expected):
        raise ValueError("Checkpoint contains effect IDs outside the frozen design")
    pending = [
        case
        for case in cases
        if f"{case[0]['id']}|seed={case[1] if case[1] is not None else 'trained'}|source={case[2]}"
        not in completed
    ]

    spec = dict(MODEL_SPECS[model_name], name=model_name)
    model, tokenizer, device = load_target(spec, "auto", offline)
    pos, neg = [tokenizer.encode(word, add_special_tokens=False) for word in ("positive", "negative")]
    if len(pos) != 1 or len(neg) != 1 or pos[0] == neg[0]:
        raise ValueError("Positive and negative labels must be distinct single tokens")
    pos_id, neg_id = pos[0], neg[0]
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
            for (stim, seed, source_aspect, _vector), margin, top_id in zip(
                batch, margins, top_ids
            ):
                base = baseline_by_id[stim["id"]]
                row = {
                    "effect_id": f"{stim['id']}|seed={seed if seed is not None else 'trained'}|source={source_aspect}",
                    "id": stim["id"],
                    "sentence_id": stim["sentence_id"],
                    "target": stim["target"],
                    "source": source_aspect,
                    "seed": seed,
                    "condition": "trained_residual" if seed is None else "random_residual",
                    "label": stim["label"],
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
    if set(completed) != expected:
        raise ValueError("Effect capture incomplete")
    write_json(folder / "effects.json", [completed[key] for key in sorted(expected)])
    source_manifest = json.loads((source_root / model_name / "manifest.json").read_text())
    residual_norms = [float(value) for value in trained.norm(dim=1)]
    write_json(
        manifest_path,
        {
            "model": spec,
            "device": str(device),
            "dtype": "float32",
            "layer": LAYER,
            "dose_fraction_of_training_activation_norm": DOSE_FRACTION,
            "dose_l2": dose,
            "n_prompts": len(stimuli),
            "n_conflict_sentences": len(conflict_ids),
            "n_random_control_seeds": len(SEEDS),
            "random_control_seeds": SEEDS,
            "residual_l2_norms": residual_norms,
            "source_activation_sha256": activation_hash,
            "source_manifest_activation_sha256": source_manifest["activation_sha256"],
            "source_dataset_sha256": dataset_hash,
            "dataset_xml_sha256": sha(data_path),
            "protocol_sha256": sha(PROTOCOL),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
    parser.add_argument("--root", type=Path, default=Path("runs/tripr-residual-control-seeds-v1"))
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
