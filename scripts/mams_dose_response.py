"""Measure whether MAMS aspect selectivity changes relative to valence bias by dose."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from answer_encoding_control import setup
from mams_aspect_selectivity import DATA, MODELS, REPOSITORY_REVISION, load_mams
from natural_aspect_selectivity import candidate_ids, capture_logits
from prompt_effect_forecast import LAYER, sha
from task_ladder import MODEL_SPECS

from latent_decisions.experiment import provenance, write_json
from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/016-mams-dose-response.md")
DOSES = (0.0125, 0.025, 0.05, 0.10, 0.20)


def dose_name(dose):
    return f"dose-{dose:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "all"))
    parser.add_argument("--root", type=Path, default=Path("runs/mams-dose-response-v1"))
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
        raise ValueError("016 must use the frozen 71-prompt MAMS-ACSA test filter")
    print(
        f"Prompts={len(stimuli)}, sentences={len(sentence_labels)}, conflicts={conflicts}, "
        f"dose grid={DOSES}",
        flush=True,
    )
    for model_name in args.models:
        source_rows, base_dose, vectors = setup(model_name, args.source_root)
        model_spec = dict(MODEL_SPECS[model_name], name=model_name)
        model, tokenizer, device = load_target(model_spec, "auto", args.offline)
        pos_id, neg_id = candidate_ids(tokenizer)
        source_folder = args.source_root / model_name
        source_manifest = json.loads((source_folder / "manifest.json").read_text())
        for fraction in DOSES:
            output = args.root / dose_name(fraction) / model_name
            output.mkdir(parents=True, exist_ok=True)
            if (output / "manifest.json").exists():
                continue
            absolute_dose = base_dose * fraction / 0.05
            capture_logits(
                model,
                tokenizer,
                device,
                stimuli,
                vectors,
                absolute_dose,
                output,
            )
            manifest = {
                "model": model_spec,
                "device": str(device),
                "dtype": "float32",
                "layer": LAYER,
                "dose_fraction_of_training_activation_norm": fraction,
                "dose_l2": absolute_dose,
                "candidate_token_ids": {"positive": pos_id, "negative": neg_id},
                "dataset": "MAMS-ACSA official test split",
                "dataset_repository_revision": REPOSITORY_REVISION,
                "mams_xml_sha256": sha(args.data),
                "n_gold_prompts": len(stimuli),
                "n_sentences": len(sentence_labels),
                "n_polarity_conflict_sentences": conflicts,
                "source_dataset_sha256": sha(source_folder / "dataset.json"),
                "source_activation_sha256": source_manifest["activation_sha256"],
                "protocol_sha256": sha(PROTOCOL),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "capture_helper_sha256": sha(Path("scripts/natural_aspect_selectivity.py")),
                "baseline_sha256": sha(output / "baseline.json"),
                "effects_sha256": sha(output / "effects.json"),
                "provenance": provenance(),
            }
            write_json(output / "manifest.json", manifest)
            print(f"{model_name} {fraction:.4f}: complete", flush=True)
        del model
        gc.collect()
        if str(device).startswith("mps"):
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
