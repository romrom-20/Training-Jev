"""Test the layer-16 specificity lead on independent TripR restaurant reviews."""

import argparse
import gc
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import natural_aspect_selectivity as natural
import torch
from mams_layer_selectivity import directions_at_layer
from prompt_effect_forecast import sha
from task_ladder import MODEL_SPECS

from latent_decisions.experiment import provenance, write_json
from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/018-independent-tripadvisor-layer16.md")
DATA = Path(
    ".context/datasets/OD-TripR-2020Large/TripR-2020Large_AnnotatedReviews.xml"
)
SOURCE_REPOSITORY_REVISION = "2e3f56f8f3691019139bdec36db0b6115ed7b191"
SOURCE_REPOSITORY = "https://github.com/ari-dasci/OD-TripR-2020Large"
MODELS = ("qwen-1.5b", "smollm2-1.7b")
LAYERS = (16, 24)
GROUPS = {"food/menu": 0, "service/staff": 1, "price": 2}
GROUP_QUESTIONS = {
    0: "food/menu",
    1: "service/staff",
    2: "price",
}


def mapped_group(category):
    category = category.upper()
    if category.endswith("#PRICES"):
        return 2
    if category in {"FOOD#QUALITY", "FOOD#STYLE_OPTIONS"}:
        return 0
    if category == "SERVICE#GENERAL":
        return 1
    return None


def load_tripr(path):
    stimuli = []
    labels_by_sentence = {}
    for review in ET.parse(path).getroot().findall("Review"):
        review_id = review.get("rid")
        for sentence in review.findall(".//sentence"):
            text_node = sentence.find("text")
            opinions = sentence.find("Opinions")
            if text_node is None or not text_node.text or opinions is None:
                continue
            sentence_id = f"{review_id}|{sentence.get('id')}"
            mapped = {}
            for opinion in opinions.findall("Opinion"):
                polarity = opinion.get("polarity")
                group = mapped_group(opinion.get("category", ""))
                if group is not None and polarity in {"positive", "negative"}:
                    mapped.setdefault(group, set()).add(polarity)
            clean = {
                group: next(iter(polarities))
                for group, polarities in mapped.items()
                if len(polarities) == 1
            }
            if len(clean) < 2:
                continue
            labels_by_sentence[sentence_id] = clean
            for group, polarity in sorted(clean.items()):
                category = GROUP_QUESTIONS[group]
                user = (
                    f"Review: {text_node.text.strip()}\n"
                    f"What is the sentiment about {category}? "
                    "Reply with exactly one word: positive or negative."
                )
                stimuli.append(
                    {
                        "id": f"{sentence_id}|group={group}",
                        "sentence_id": sentence_id,
                        "category": category,
                        "target": group,
                        "label": int(polarity == "positive"),
                        "user": user,
                    }
                )
    conflict_count = sum(
        len(set(labels.values())) > 1 for labels in labels_by_sentence.values()
    )
    if (len(labels_by_sentence), len(stimuli), conflict_count) != (187, 385, 29):
        raise ValueError(
            "Frozen TripR filter changed; expected 187 sentences, 385 queries and "
            f"29 polarity conflicts, got {len(labels_by_sentence)}, {len(stimuli)}, "
            f"{conflict_count}"
        )
    return stimuli, labels_by_sentence, conflict_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "all"))
    parser.add_argument("--root", type=Path, default=Path("runs/tripr-layer-confirmation-v1"))
    parser.add_argument("--source-root", type=Path, default=Path("runs/task-ladder-v1"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.root.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {args.root}; use --resume")
    args.root.mkdir(parents=True, exist_ok=True)
    stimuli, sentence_labels, conflicts = load_tripr(args.data)
    print(
        f"TripR prompts={len(stimuli)}, sentences={len(sentence_labels)}, "
        f"conflicts={conflicts}, layers={LAYERS}",
        flush=True,
    )
    for model_name in args.models:
        spec = dict(MODEL_SPECS[model_name], name=model_name)
        model, tokenizer, device = load_target(spec, "auto", args.offline)
        source = args.source_root / model_name
        source_manifest = json.loads((source / "manifest.json").read_text())
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
            manifest = {
                "model": spec,
                "device": str(device),
                "dtype": "float32",
                "layer": layer,
                "dose_fraction_of_training_activation_norm": 0.05,
                "dose_l2": dose,
                "candidate_token_ids": {"positive": pos_id, "negative": neg_id},
                "dataset": "TripR-2020Large manually annotated restaurant reviews",
                "dataset_repository": SOURCE_REPOSITORY,
                "dataset_repository_revision": SOURCE_REPOSITORY_REVISION,
                "dataset_license": "CC BY-SA 4.0",
                "dataset_xml_sha256": sha(args.data),
                "n_gold_prompts": len(stimuli),
                "n_sentences": len(sentence_labels),
                "n_polarity_conflict_sentences": conflicts,
                "category_crosswalk": {
                    "FOOD#QUALITY": 0,
                    "FOOD#STYLE_OPTIONS": 0,
                    "SERVICE#GENERAL": 1,
                    "*#PRICES": 2,
                },
                "prompt_category_names": GROUP_QUESTIONS,
                "control_seed": 20260930,
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
