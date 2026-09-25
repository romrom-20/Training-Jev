"""Run the fixed abstention-choice wrapper on the clear subset of Exp 035."""

import argparse
import gc
import hashlib
import json
import time
import warnings
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from run_cue_position_polarity_035 import (
    BUDGETS,
    STIMULI_FILE,
    STIMULI_SHA256,
)
from run_cue_position_polarity_035 import (
    make_jobs as make_035_jobs,
)
from run_laya_decision_audit_030 import (
    MODEL_ID as LAYA_ID,
)
from run_laya_decision_audit_030 import (
    MODEL_REVISION as LAYA_REVISION,
)
from run_laya_decision_audit_030 import (
    build_question,
    choice_key_maps,
)
from run_tripr_prefix_threshold_034 import DEVICE, PHI_ID, PHI_REVISION
from task_ladder import MODEL_SPECS
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/036-evidence-aware-abstention.md")
OUT = Path(".context/evidence-aware-abstention-036")
BASE_PRIVATE = Path(".context/cue-position-polarity-035")
SEED = 20260936
BATCH_SIZE = 8
JUDGES = ("laya", "qwen2.5-3b", "phi3-mini")

ROW_KEYS = (
    "id",
    "cluster_id",
    "aspect",
    "gold",
    "polarity",
    "position",
    "form",
    "budget",
    "cue_visible",
    "expected_decision",
    "visible_text",
)


def load_jobs():
    stimulus_bytes = STIMULI_FILE.read_bytes()
    if hashlib.sha256(stimulus_bytes).hexdigest() != STIMULI_SHA256:
        raise ValueError("Experiment 035 frozen stimulus hash changed")
    stimuli = json.loads(stimulus_bytes)
    selected = [row for row in stimuli if row["form"] == "direct"]
    if len(selected) != 192 or sum(row["gold"] for row in selected) != 96:
        raise ValueError("Experiment 036 clear, balanced subset changed")
    jobs = make_035_jobs(selected)
    for row in jobs:
        row["cue_visible"] = row["position"] == "early" or row["budget"] == 12
        row["expected_decision"] = (
            row["polarity"] if row["cue_visible"] else "insufficient"
        )
    return selected, jobs


def map_key(row):
    return f"{row['aspect']}\t{row['visible_text']}"


def run_laya(jobs):
    import laya

    checkpoint = snapshot_download(
        repo_id=LAYA_ID,
        revision=LAYA_REVISION,
        allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/**", "tokenizer/**"],
    )
    mappings = choice_key_maps((map_key(row) for row in jobs), seed=SEED)
    groups = {}
    for row in jobs:
        signature = tuple(mappings[map_key(row)][key] for key in ("A", "B", "C", "D"))
        groups.setdefault(signature, []).append(row)
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = laya.load(str(checkpoint), device=DEVICE)
    outcomes = []
    for signature, group in groups.items():
        mapping = dict(zip(("A", "B", "C", "D"), signature))
        results = agent.predict_batch(
            [{"aspect": row["aspect"], "text": row["visible_text"]} for row in group],
            build_question(mapping),
            batch_size=BATCH_SIZE,
            sort_by_length=True,
        )
        reverse = {key: label for key, label in mapping.items()}
        outcomes.extend(
            {key: row[key] for key in ROW_KEYS}
            | {"judge": "laya", "label": reverse[result["answers"]["polarity"]["choice"]]}
            for row, result in zip(group, results)
        )
        print(f"036 Laya: {len(outcomes)}/{len(jobs)}", flush=True)
    del agent
    gc.collect()
    torch.mps.empty_cache()
    return outcomes, hashlib.sha256((Path(checkpoint) / "model.safetensors").read_bytes()).hexdigest(), [
        str(item.message) for item in caught
    ], time.monotonic() - started


def abstention_content(row):
    return (
        "A model was asked about the sentiment of the "
        f"{row['aspect']} in a restaurant review. Only this answer prefix is visible: "
        f"{row['visible_text']}\n"
        "If the visible text gives no sentiment evidence about the named aspect, "
        "reply exactly: insufficient. Otherwise reply with exactly one word, "
        "positive or negative, for the sentiment expressed toward that aspect."
    )


def parse_decision(text):
    value = text.strip().lower().strip(".,!?;:\"'` ")
    if value in {"positive", "negative", "insufficient"}:
        return value
    return None


def run_binary_judge(judge, jobs):
    if judge == "qwen2.5-3b":
        model, tokenizer, _ = load_target(dict(MODEL_SPECS["qwen-3b"]), DEVICE, offline=True)
        tokenizer.padding_side = "left"
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            PHI_ID, revision=PHI_REVISION, local_files_only=True, padding_side="left"
        )
        model = AutoModelForCausalLM.from_pretrained(
            PHI_ID,
            revision=PHI_REVISION,
            local_files_only=True,
            dtype=torch.float16,
            attn_implementation="eager",
        ).to(DEVICE).eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.requires_grad_(False)

    outcomes = []
    started = time.monotonic()
    for start in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[start : start + BATCH_SIZE]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": abstention_content(row)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            generated = model.generate(
                **tokens,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        labels = tokenizer.batch_decode(
            generated[:, tokens.input_ids.shape[1] :], skip_special_tokens=True
        )
        outcomes.extend(
            {key: row[key] for key in ROW_KEYS}
            | {"judge": judge, "label": parse_decision(label)}
            for row, label in zip(batch, labels)
        )
        print(f"036 {judge}: {min(start + len(batch), len(jobs))}/{len(jobs)}", flush=True)
    del model, tokenizer
    gc.collect()
    torch.mps.empty_cache()
    return outcomes, time.monotonic() - started


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output=OUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Experiment 036 requires local MPS")
    base_manifest_path = BASE_PRIVATE / "manifest.json"
    base_outcomes_path = BASE_PRIVATE / "outcomes.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    if sha256(base_outcomes_path) != base_manifest["outcomes_sha256"]:
        raise ValueError("Experiment 035 baseline outcomes failed their hash check")
    stimuli, jobs = load_jobs()
    laya_rows, laya_hash, laya_warnings, laya_seconds = run_laya(jobs)
    qwen_rows, qwen_seconds = run_binary_judge("qwen2.5-3b", jobs)
    phi_rows, phi_seconds = run_binary_judge("phi3-mini", jobs)
    outcomes = laya_rows + qwen_rows + phi_rows
    expected = len(stimuli) * len(BUDGETS) * len(JUDGES)
    if len(outcomes) != expected or expected != 1152:
        raise ValueError(f"Expected 1,152 outcomes, got {len(outcomes)}")
    output.mkdir(parents=True)
    stimuli_path = output / "stimuli.json"
    outcomes_path = output / "outcomes.json"
    stimuli_path.write_text(json.dumps(stimuli, indent=2, ensure_ascii=False) + "\n")
    outcomes_path.write_text(json.dumps(outcomes, indent=2, ensure_ascii=False) + "\n")
    manifest = {
        "experiment": "036",
        "protocol_sha256": sha256(PROTOCOL),
        "source_035_outcomes_sha256": base_manifest["outcomes_sha256"],
        "source_035_stimuli_sha256": STIMULI_SHA256,
        "stimuli_sha256": sha256(stimuli_path),
        "outcomes_sha256": sha256(outcomes_path),
        "stimuli": len(stimuli),
        "clusters": len({row["cluster_id"] for row in stimuli}),
        "outcomes": expected,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "budgets_words": list(BUDGETS),
        "models": {
            "laya": {
                "id": LAYA_ID,
                "revision": LAYA_REVISION,
                "package": "laya==0.3.20",
                "weights_sha256": laya_hash,
                "seconds": laya_seconds,
            },
            "qwen2.5-3b": {"revision": MODEL_SPECS["qwen-3b"]["revision"], "seconds": qwen_seconds},
            "phi3-mini": {"id": PHI_ID, "revision": PHI_REVISION, "seconds": phi_seconds},
        },
        "laya_warnings": laya_warnings,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
