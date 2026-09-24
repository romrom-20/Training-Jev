"""Compare frozen aspect-steering selectivity across cached training layers."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import natural_aspect_selectivity as natural
import numpy as np
import torch
from mams_aspect_selectivity import (
    DATA,
    MODELS,
    REPOSITORY_REVISION,
    load_mams,
)
from prompt_effect_forecast import sha
from task_ladder import MODEL_SPECS

from latent_decisions.experiment import provenance, write_json
from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/017-mams-layer-selectivity.md")
LAYERS = (8, 16, 24)
DOSE_FRACTION = 0.05
CONTROL_SEED = 20260930


def directions_at_layer(model_name, layer, source_root):
    source = source_root / model_name
    rows = json.loads((source / "dataset.json").read_text())
    with np.load(source / "activations.npz", allow_pickle=False) as cache:
        activations = cache["h"]
    layer_index = MODEL_SPECS[model_name]["layers"].index(layer)
    native, training_norms = [], []
    for aspect in range(3):
        indices = [
            i
            for i, row in enumerate(rows)
            if row["split"] == "train"
            and row["task"] == "mixed_review"
            and row["format"] == 0
            and row["active"] == aspect
        ]
        positive = activations[
            [i for i in indices if rows[i]["label"] == 1], layer_index
        ].mean(axis=0)
        negative = activations[
            [i for i in indices if rows[i]["label"] == 0], layer_index
        ].mean(axis=0)
        vector = torch.from_numpy(positive - negative).float()
        native.append(vector / vector.norm().clamp_min(1e-12))
        training_norms.extend(
            np.linalg.norm(activations[indices, layer_index], axis=-1).tolist()
        )
    native = torch.stack(native)
    axis = native.mean(0)
    axis = axis / axis.norm().clamp_min(1e-12)
    shared = (native * (native @ axis).unsqueeze(-1)).mean(0)
    generator = torch.Generator(device="cpu").manual_seed(CONTROL_SEED)
    random = torch.randn(shared.shape, generator=generator)
    random -= torch.dot(random, axis) * axis
    random = random / random.norm().clamp_min(1e-12) * shared.norm()
    dose = DOSE_FRACTION * float(np.median(training_norms))
    return {"native": native, "shared": shared, "random": random}, dose, sha(
        source / "activations.npz"
    ), sha(source / "dataset.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "all"))
    parser.add_argument("--root", type=Path, default=Path("runs/mams-layer-selectivity-v1"))
    parser.add_argument("--source-root", type=Path, default=Path("runs/task-ladder-v1"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.root.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {args.root}; use --resume")
    args.root.mkdir(parents=True, exist_ok=True)
    stimuli, sentence_labels, conflicts = load_mams(args.data)
    if (len(stimuli), len(sentence_labels), conflicts) != (71, 35, 18):
        raise ValueError("017 must use the frozen 71-prompt MAMS-ACSA test filter")
    print(
        f"Prompts={len(stimuli)}, sentences={len(sentence_labels)}, conflicts={conflicts}, "
        f"layers={LAYERS}",
        flush=True,
    )
    for model_name in args.models:
        spec = dict(MODEL_SPECS[model_name], name=model_name)
        model, tokenizer, device = load_target(spec, "auto", args.offline)
        for layer in LAYERS:
            output = args.root / f"layer-{layer:02d}" / model_name
            output.mkdir(parents=True, exist_ok=True)
            if (output / "manifest.json").exists():
                continue
            vectors, dose, activation_hash, dataset_hash = directions_at_layer(
                model_name, layer, args.source_root
            )
            natural.LAYER = layer
            pos_id, neg_id = natural.candidate_ids(tokenizer)
            natural.capture_logits(
                model, tokenizer, device, stimuli, vectors, dose, output
            )
            source_manifest = json.loads(
                (args.source_root / model_name / "manifest.json").read_text()
            )
            manifest = {
                "model": spec,
                "device": str(device),
                "dtype": "float32",
                "layer": layer,
                "layer_index_in_cached_activations": MODEL_SPECS[model_name]["layers"].index(
                    layer
                ),
                "dose_fraction_of_training_activation_norm": DOSE_FRACTION,
                "dose_l2": dose,
                "candidate_token_ids": {"positive": pos_id, "negative": neg_id},
                "dataset": "MAMS-ACSA official test split",
                "dataset_repository_revision": REPOSITORY_REVISION,
                "mams_xml_sha256": sha(args.data),
                "n_gold_prompts": len(stimuli),
                "n_sentences": len(sentence_labels),
                "n_polarity_conflict_sentences": conflicts,
                "category_source_mapping": {
                    "food": 0,
                    "menu": 0,
                    "service": 1,
                    "staff": 1,
                    "price": 2,
                },
                "control_seed": CONTROL_SEED,
                "source_dataset_sha256": dataset_hash,
                "source_activation_sha256": activation_hash,
                "source_manifest_activation_sha256": source_manifest[
                    "activation_sha256"
                ],
                "protocol_sha256": sha(PROTOCOL),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "capture_helper_sha256": sha(Path("scripts/natural_aspect_selectivity.py")),
                "baseline_sha256": sha(output / "baseline.json"),
                "effects_sha256": sha(output / "effects.json"),
                "provenance": provenance(),
            }
            write_json(output / "manifest.json", manifest)
            print(f"{model_name} layer {layer}: complete", flush=True)
        del model
        gc.collect()
        if str(device).startswith("mps"):
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
