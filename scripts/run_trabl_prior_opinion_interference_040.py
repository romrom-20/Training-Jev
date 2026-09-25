"""Run the preregistered cross-aspect prior-opinion intervention (Experiment 040)."""

import argparse
import gc
import hashlib
import json
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
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
from run_natural_opinion_span_abstention_037 import (
    DEVICE,
    PHI_ID,
    PHI_REVISION,
    decode_laya_choice,
    laya_map_key,
    parse_label,
)
from task_ladder import MODEL_SPECS
from trabl_prior_opinion_interference_040 import (
    SOURCE,
    SOURCE_REVISION,
    SOURCE_SHA256,
    STIMULI,
    make_jobs,
    parse_source,
    public_stimuli,
    select_pairs,
    sha256,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_decisions.target import load_target

PROTOCOL = Path("docs/experiments/040-trabl-prior-opinion-interference.md")
OUT = Path("results/trabl-prior-opinion-interference-v1")
PRIVATE_TRIAGE = Path(".context/laya-research-triage-040.json")
EXPECTED_PROTOCOL_SHA256 = "491a366a8bac8b3b43f65980b33c03017777b4ef2a21247f134c41c6572735e5"
EXPECTED_STIMULI_SHA256 = "7abd6a1ab5d4f86190cfd727f452042953d4e030f39d335a88cec17f37db997b"
EXPECTED_TRIAGE_SHA256 = "f6728865fd43d3b55ad3825c1a3b54733093389236d84130bb3401b2d1c9fb51"
SEED = 20260940
BATCH_SIZE = 8
WRAPPERS = ("forced_binary", "abstention")


def question_text(row, wrapper):
    if wrapper == "forced_binary":
        policy = "Reply with exactly one word: positive or negative."
    elif wrapper == "abstention":
        policy = (
            "If the visible excerpt contains no explicit opinion expression supporting "
            "sentiment toward the named aspect, reply exactly insufficient. Otherwise "
            "reply with exactly one word, positive or negative."
        )
    else:
        raise ValueError(f"Unknown wrapper: {wrapper}")
    return (
        "Classify the sentiment expressed toward the named aspect in this review excerpt.\n"
        f"Aspect: {row['aspect']}\n"
        f"Visible review excerpt: {row['visible_text']}\n"
        f"{policy}"
    )


def run_laya(jobs):
    import laya

    checkpoint = snapshot_download(
        repo_id=LAYA_ID,
        revision=LAYA_REVISION,
        allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/**", "tokenizer/**"],
    )
    mappings = choice_key_maps((laya_map_key(row) for row in jobs), seed=SEED)
    groups = {}
    for row in jobs:
        signature = tuple(mappings[laya_map_key(row)][key] for key in "ABCD")
        groups.setdefault(signature, []).append(row)
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = laya.load(str(checkpoint), device=DEVICE)
    outcomes = []
    for signature, group in groups.items():
        mapping = dict(zip("ABCD", signature))
        answers = agent.predict_batch(
            [{"aspect": row["aspect"], "text": row["visible_text"]} for row in group],
            build_question(mapping),
            batch_size=BATCH_SIZE,
            sort_by_length=True,
        )
        outcomes.extend(
            {
                key: row[key]
                for key in ("id", "cluster_id", "role", "gold", "prior_gold", "condition")
            }
            | {
                "judge": "laya",
                "wrapper": "four_way",
                "label": decode_laya_choice(result["answers"]["polarity"]["choice"], mapping),
            }
            for row, result in zip(group, answers)
        )
        print(f"040 Laya: {len(outcomes)}/{len(jobs)}", flush=True)
    del agent
    gc.collect()
    torch.mps.empty_cache()
    return (
        outcomes,
        hashlib.sha256((Path(checkpoint) / "model.safetensors").read_bytes()).hexdigest(),
        [str(item.message) for item in caught],
        time.monotonic() - started,
    )


def run_binary_judge(judge, jobs):
    if judge == "qwen2.5-3b":
        model, tokenizer, _ = load_target(dict(MODEL_SPECS["qwen-3b"]), DEVICE, offline=True)
        tokenizer.padding_side = "left"
    elif judge == "phi3-mini":
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
    else:
        raise ValueError(f"Unknown binary judge: {judge}")

    outcomes = []
    started = time.monotonic()
    for wrapper in WRAPPERS:
        for start in range(0, len(jobs), BATCH_SIZE):
            batch = jobs[start : start + BATCH_SIZE]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": question_text(row, wrapper)}],
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
                {
                    key: row[key]
                    for key in ("id", "cluster_id", "role", "gold", "prior_gold", "condition")
                }
                | {
                    "judge": judge,
                    "wrapper": wrapper,
                    "label": parse_label(label, wrapper),
                }
                for row, label in zip(batch, labels)
            )
            print(
                f"040 {judge}/{wrapper}: {min(start + len(batch), len(jobs))}/{len(jobs)}",
                flush=True,
            )
    del model, tokenizer
    gc.collect()
    torch.mps.empty_cache()
    return outcomes, time.monotonic() - started


def run(output=OUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Experiment 040 requires local MPS")
    if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Experiment 040 protocol changed after preregistration")
    if sha256(STIMULI) != EXPECTED_STIMULI_SHA256:
        raise ValueError("Frozen Experiment 040 stimuli changed")
    if sha256(PRIVATE_TRIAGE) != EXPECTED_TRIAGE_SHA256:
        raise ValueError("Laya research-triage trace hash mismatch")

    items = select_pairs(parse_source(SOURCE))
    stimuli = public_stimuli(items)
    frozen = json.loads(STIMULI.read_text())
    if stimuli != frozen or len(items) != 96 or len({row["cluster_id"] for row in items}) != 48:
        raise ValueError("Regenerated trials differ from the frozen 48-review selection")
    jobs = make_jobs(items)
    if len(jobs) != 336:
        raise ValueError(f"Expected 336 paired condition prompts, got {len(jobs)}")

    laya_rows, laya_hash, laya_warnings, laya_seconds = run_laya(jobs)
    qwen_rows, qwen_seconds = run_binary_judge("qwen2.5-3b", jobs)
    phi_rows, phi_seconds = run_binary_judge("phi3-mini", jobs)
    outcomes = laya_rows + qwen_rows + phi_rows
    expected = len(jobs) * (1 + 2 * len(WRAPPERS))
    if len(outcomes) != expected or expected != 1680:
        raise ValueError(f"Expected 1,680 judgments, got {len(outcomes)}")

    output.mkdir(parents=True)
    stimulus_path = output / "stimuli.json"
    prediction_path = output / "predictions.json"
    stimulus_path.write_text(json.dumps(stimuli, indent=2) + "\n")
    prediction_path.write_text(json.dumps(outcomes, indent=2) + "\n")
    manifest = {
        "experiment": "040",
        "dataset": "Booking-com/absa-dataset",
        "dataset_url": "https://huggingface.co/datasets/Booking-com/absa-dataset",
        "dataset_revision": SOURCE_REVISION,
        "source_file_sha256": SOURCE_SHA256,
        "protocol_sha256": sha256(PROTOCOL),
        "runner_sha256": sha256(Path(__file__)),
        "preparation_sha256": sha256(Path("scripts/trabl_prior_opinion_interference_040.py")),
        "stimuli_sha256": sha256(stimulus_path),
        "predictions_sha256": sha256(prediction_path),
        "laya_triage_trace_sha256": sha256(PRIVATE_TRIAGE),
        "stimuli": len(stimuli),
        "review_clusters": len({row["cluster_id"] for row in stimuli}),
        "jobs_per_condition_set": len(jobs),
        "outcomes": len(outcomes),
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "models": {
            "laya": {
                "id": LAYA_ID,
                "revision": LAYA_REVISION,
                "package": "laya==" + version("laya"),
                "weights_sha256": laya_hash,
                "seconds": laya_seconds,
            },
            "qwen2.5-3b": {
                "revision": MODEL_SPECS["qwen-3b"]["revision"],
                "seconds": qwen_seconds,
            },
            "phi3-mini": {
                "id": PHI_ID,
                "revision": PHI_REVISION,
                "seconds": phi_seconds,
            },
        },
        "laya_warnings": laya_warnings,
        "raw_review_text_published": False,
        "data_license": "CC BY-SA 4.0; non-commercial research use per dataset card",
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
