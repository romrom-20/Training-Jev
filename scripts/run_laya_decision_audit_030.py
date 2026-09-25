"""Run experiment 030 with the pinned local Laya decision checkpoint."""

import argparse
import gc
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import torch
from huggingface_hub import snapshot_download
from natural_aspect_selectivity import DATA, load_stimuli

PROTOCOL = Path("docs/experiments/030-laya-decision-audit.md")
PRIVATE = Path(".context/human-coding-029/private-key.json")
OUT = Path(".context/laya-decision-audit-030")
MODEL_ID = "convaiinnovations/laya"
MODEL_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
PKG_VERSION = "0.3.20"
MAP_SEED = 20260930
CHOICES = {
    "positive": "The text clearly communicates favorable sentiment toward the named aspect.",
    "negative": "The text clearly communicates unfavorable sentiment toward the named aspect.",
    "mixed": "The text communicates both favorable and unfavorable sentiment toward the named aspect.",
    "unclear": "The text communicates no clear sentiment toward the named aspect, is irrelevant, or is too incomplete to tell.",
}


def choice_key_maps(stimulus_ids, seed=MAP_SEED):
    rng = random.Random(seed)
    base = list(CHOICES)
    rng.shuffle(base)
    ordered_ids = list(sorted(set(stimulus_ids)))
    rng.shuffle(ordered_ids)
    keys = ["A", "B", "C", "D"]
    return {
        stimulus_id: dict(zip(keys, base[offset % len(base) :] + base[: offset % len(base)]))
        for offset, stimulus_id in enumerate(ordered_ids)
    }


def build_question(mapping):
    return {
        "polarity": {
            "type": "choice",
            "instructions": (
                "Classify the sentiment expressed by the text toward the named aspect. "
                "Distinguish mixed sentiment from no clear sentiment. Choose the closest description."
            ),
            "criteria": {key: CHOICES[label] for key, label in mapping.items()},
        }
    }


def parse_jobs(stimuli):
    generated = json.loads(PRIVATE.read_text())
    if len(generated) != 60 or len({row["stimulus_id"] for row in generated}) != 20:
        raise ValueError("Experiment 029 private sample must contain 60 paired answers / 20 items")
    sample_ids = {row["stimulus_id"] for row in generated}
    source = {item["id"]: item for item in stimuli}
    if not sample_ids.issubset(source):
        raise ValueError("Private answer key includes an item outside the frozen SemEval set")
    source_jobs = []
    for item in stimuli:
        review = item["user"].split("\n", maxsplit=1)[0].removeprefix("Review: ")
        source_jobs.append(
            {
                "phase": "source",
                "id": item["id"],
                "sentence_id": item["sentence_id"],
                "category": item["category"],
                "gold": item["label"],
                "model": None,
                "state": {"aspect": item["category"], "text": review},
            }
        )
    answer_jobs = []
    for row in generated:
        answer_jobs.append(
            {
                "phase": "generated",
                "id": row["stimulus_id"],
                "sentence_id": row["sentence_id"],
                "category": row["category"],
                "gold": row["gold"],
                "model": row["model"],
                "state": {"aspect": row["category"], "text": row["answer"]},
            }
        )
    if len(source_jobs) != 233:
        raise ValueError("Frozen source screen changed")
    return source_jobs, answer_jobs


def infer_jobs(agent, jobs, mappings, batch_size=8):
    by_mapping = defaultdict(list)
    for job in jobs:
        by_mapping[tuple(mappings[job["id"]][key] for key in ("A", "B", "C", "D"))].append(job)
    outcomes = []
    for signature, group in by_mapping.items():
        mapping = dict(zip(("A", "B", "C", "D"), signature))
        question = build_question(mapping)
        results = agent.predict_batch(
            [job["state"] for job in group],
            question,
            batch_size=batch_size,
            sort_by_length=True,
        )
        for job, result in zip(group, results):
            answer = result["answers"]["polarity"]
            key_to_label = mappings[job["id"]]
            probs = {
                key_to_label[key]: probability
                for key, probability in answer["probabilities"].items()
            }
            outcomes.append(
                {
                    key: job[key]
                    for key in ("phase", "id", "sentence_id", "category", "gold", "model")
                }
                | {
                    "laya_label": key_to_label[answer["choice"]],
                    "laya_probabilities": probs,
                    "laya_confidence": answer.get("answer_confidence"),
                    "laya_choice_entropy_confidence": answer.get("confidence"),
                }
            )
        print(f"Laya decisions: {len(outcomes)}/{len(jobs)}", flush=True)
    return outcomes


def run(output=OUT, batch_size=8, device="mps"):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if importlib_metadata_version("laya") != PKG_VERSION:
        raise RuntimeError(f"Expected laya=={PKG_VERSION}")
    stimuli = load_stimuli(DATA)
    source_jobs, answer_jobs = parse_jobs(stimuli)
    all_jobs = source_jobs + answer_jobs
    mappings = choice_key_maps(job["id"] for job in all_jobs)
    model_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/**", "tokenizer/**"],
    )
    import laya

    if not torch.backends.mps.is_available() and device == "mps":
        raise RuntimeError("MPS is unavailable; explicitly rerun with --device cpu")
    print(f"Loading {MODEL_ID}@{MODEL_REVISION} on {device}", flush=True)
    agent = laya.load(str(model_path), device=device)
    outcomes = infer_jobs(agent, all_jobs, mappings, batch_size=batch_size)
    if len(outcomes) != len(all_jobs):
        raise ValueError("Incomplete Laya outcomes")
    output.mkdir(parents=True)
    (output / "predictions.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    manifest = {
        "experiment": "030",
        "engine": "Laya local typed choice",
        "package": {"name": "laya", "version": PKG_VERSION},
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": hashlib.sha256(
            (Path(model_path) / "model.safetensors").read_bytes()
        ).hexdigest(),
        "device": device,
        "batch_size": batch_size,
        "choice_map_seed": MAP_SEED,
        "choice_categories": list(CHOICES),
        "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "private_answer_key_sha256": hashlib.sha256(PRIVATE.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_source_reviews": len(source_jobs),
        "n_generated_answers": len(answer_jobs),
        "n_unique_generated_items": len({job["id"] for job in answer_jobs}),
        "n_map_permutations": len(
            set(tuple(mappings[job["id"]][k] for k in "ABCD") for job in all_jobs)
        ),
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    del agent
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    print(f"Wrote local label-only outputs to {output}", flush=True)


def importlib_metadata_version(name):
    from importlib.metadata import version

    return version(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    run(args.output, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
