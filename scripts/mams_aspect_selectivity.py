"""Run the frozen aspect-selectivity test on the MAMS-ACSA gold test split."""

import argparse
import gc
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from answer_encoding_control import setup
from natural_aspect_selectivity import MODELS, candidate_ids, capture_logits
from prompt_effect_forecast import LAYER, sha
from task_ladder import MODEL_SPECS

from latent_decisions.experiment import provenance, write_json
from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/015-mams-conflict-selectivity.md")
DATA = Path(".context/datasets/MAMS-for-ABSA/data/MAMS-ACSA/raw/test.xml")
REPOSITORY_REVISION = "cddcdb0f423b3fdb2c76b70744737abed3a00d17"
GROUPS = {"food": 0, "menu": 0, "service": 1, "staff": 1, "price": 2}
GROUP_NAMES = {0: "food/menu", 1: "service/staff", 2: "price"}


def load_mams(path):
    stimuli = []
    sentence_labels = {}
    for index, sentence in enumerate(ET.parse(path).getroot().findall("sentence")):
        text_node = sentence.find("text")
        section = sentence.find("aspectCategories")
        if text_node is None or not text_node.text or section is None:
            continue
        mapped = {}
        for annotation in section.findall("aspectCategory"):
            category = annotation.get("category")
            polarity = annotation.get("polarity")
            if category not in GROUPS or polarity not in ("positive", "negative"):
                continue
            target = GROUPS[category]
            mapped.setdefault(target, set()).add(polarity)
        mapped = {target: next(iter(labels)) for target, labels in mapped.items() if len(labels) == 1}
        if len(mapped) < 2:
            continue
        sentence_id = f"test-{index:04d}"
        sentence_labels[sentence_id] = mapped
        for target, polarity in sorted(mapped.items()):
            group_name = GROUP_NAMES[target]
            user = (
                f"Review: {text_node.text.strip()}\n"
                f"What is the sentiment about {group_name}? "
                "Reply with exactly one word: positive or negative."
            )
            stimuli.append(
                {
                    "id": f"{sentence_id}|group={target}",
                    "sentence_id": sentence_id,
                    "category": group_name,
                    "target": target,
                    "label": int(polarity == "positive"),
                    "user": user,
                }
            )
    conflicts = sum(len(set(labels.values())) > 1 for labels in sentence_labels.values())
    if (len(sentence_labels), len(stimuli), conflicts) != (35, 71, 18):
        raise ValueError(
            "Frozen MAMS test filter changed; expected 35 sentences, 71 queries, "
            f"and 18 mapped-polarity conflicts; got {len(sentence_labels)}, "
            f"{len(stimuli)}, {conflicts}"
        )
    return stimuli, sentence_labels, conflicts


def collect_model(model_name, output, source_root, stimuli, data_path, offline):
    rows, dose, vectors = setup(model_name, source_root)
    model_spec = dict(MODEL_SPECS[model_name], name=model_name)
    model, tokenizer, device = load_target(model_spec, "auto", offline)
    pos_id, neg_id = candidate_ids(tokenizer)
    capture_logits(model, tokenizer, device, stimuli, vectors, dose, output)
    source = source_root / model_name
    source_manifest = json.loads((source / "manifest.json").read_text())
    manifest = {
        "model": model_spec,
        "device": str(device),
        "dtype": "float32",
        "layer": LAYER,
        "dose_fraction": 0.05,
        "dose_l2": dose,
        "candidate_token_ids": {"positive": pos_id, "negative": neg_id},
        "dataset": "MAMS-ACSA official test split",
        "dataset_repository_revision": REPOSITORY_REVISION,
        "mams_xml_sha256": sha(data_path),
        "n_gold_prompts": len(stimuli),
        "n_sentences": len({row["sentence_id"] for row in stimuli}),
        "n_polarity_conflict_sentences": 18,
        "category_source_mapping": {"food": 0, "menu": 0, "service": 1, "staff": 1, "price": 2},
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "all"))
    parser.add_argument("--root", type=Path, default=Path("runs/mams-aspect-selectivity-v1"))
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
    print(
        f"MAMS eligible prompts={len(stimuli)}; sentences={len(sentence_labels)}; "
        f"mapped-polarity conflicts={conflicts}",
        flush=True,
    )
    for model in args.models:
        output = args.root / model
        output.mkdir(exist_ok=True)
        if (output / "manifest.json").exists():
            continue
        collect_model(model, output, args.source_root, stimuli, args.data, args.offline)


if __name__ == "__main__":
    main()
