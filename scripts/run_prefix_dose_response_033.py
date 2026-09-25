"""Judge nested prefixes of the same 32-token continuations at several budgets."""

import argparse
import gc
import hashlib
import json
import warnings
from collections import defaultdict
from pathlib import Path

import torch
from analyze_prompt_format_score_generation import MODELS
from huggingface_hub import snapshot_download
from prompt_format_score_generation import CONFIGS
from run_cross_judge_length_robustness_032 import (
    BATCH_SIZE as JUDGE_BATCH_SIZE,
)
from run_cross_judge_length_robustness_032 import (
    DEVICE,
    PHI_ID,
    PHI_REVISION,
    judge_content,
    parse_label,
)
from run_laya_decision_audit_030 import (
    MAP_SEED,
    build_question,
    choice_key_maps,
)
from run_laya_decision_audit_030 import (
    MODEL_ID as LAYA_ID,
)
from run_laya_decision_audit_030 import (
    MODEL_REVISION as LAYA_REVISION,
)
from task_ladder import MODEL_SPECS
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_decisions.target import load_target

PRIVATE_031 = Path(".context/length-censoring-031")
PRIVATE_032 = Path(".context/answer-length-crossjudge-032")
OUT = Path(".context/prefix-dose-response-033")
PROTOCOL = Path("docs/experiments/033-prefix-dose-response.md")
BUDGETS = (4, 12, 16, 24)


def prefix_records(generations, tokenizer):
    jobs = []
    for full in generations:
        token_ids = full["token_ids"]
        for budget in BUDGETS:
            used = token_ids[:budget]
            jobs.append(
                {
                    key: full[key]
                    for key in ("model", "id", "sentence_id", "category", "gold")
                }
                | {
                    "budget": budget,
                    "actual_tokens": len(used),
                    "answer": tokenizer.decode(used, skip_special_tokens=True).strip(),
                }
            )
    return jobs


def source_material():
    manifest = json.loads((PRIVATE_031 / "manifest.json").read_text())
    private_path = PRIVATE_031 / "private-generations.json"
    if hashlib.sha256(private_path.read_bytes()).hexdigest() != manifest[
        "private_generation_sha256"
    ]:
        raise ValueError("Experiment 031 generation hash mismatch")
    generation_rows = json.loads(private_path.read_text())
    full = {
        (row["model"], row["id"]): row
        for row in generation_rows
        if row["budget"] == 32
    }
    if len(full) != len(MODELS) * 233:
        raise ValueError("Expected 699 full 32-token generations")
    tokenizers = {
        model: AutoTokenizer.from_pretrained(
            CONFIGS[model]["model"],
            revision=CONFIGS[model]["revision"],
            local_files_only=True,
        )
        for model in MODELS
    }
    jobs = []
    for model in MODELS:
        rows = [full[(model, item_id)] for item_id in sorted(key[1] for key in full if key[0] == model)]
        jobs.extend(prefix_records(rows, tokenizers[model]))
        del tokenizers[model]
        gc.collect()
    del tokenizers
    if len(jobs) != len(MODELS) * 233 * len(BUDGETS):
        raise ValueError("Incomplete nested-prefix factorial")
    return jobs, manifest


def run_laya(jobs):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Registered experiment 033 requires local MPS")
    import laya

    path = snapshot_download(
        repo_id=LAYA_ID,
        revision=LAYA_REVISION,
        allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/**", "tokenizer/**"],
    )
    mappings = choice_key_maps(job["id"] for job in jobs)
    groups = defaultdict(list)
    for job in jobs:
        mapping = mappings[job["id"]]
        groups[tuple(mapping[key] for key in ("A", "B", "C", "D"))].append(job)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = laya.load(str(path), device=DEVICE)
    outcomes = []
    for signature, group in groups.items():
        mapping = dict(zip(("A", "B", "C", "D"), signature))
        batch_results = agent.predict_batch(
            [{"aspect": row["category"], "text": row["answer"]} for row in group],
            build_question(mapping),
            batch_size=8,
            sort_by_length=True,
        )
        for row, result in zip(group, batch_results):
            answer = result["answers"]["polarity"]
            reverse = {key: label for key, label in mappings[row["id"]].items()}
            outcomes.append(
                {
                    key: row[key]
                    for key in ("model", "id", "sentence_id", "category", "gold", "budget", "actual_tokens")
                }
                | {
                    "judge": "laya",
                    "label": reverse[answer["choice"]],
                    "probabilities": {
                        reverse[key]: probability
                        for key, probability in answer["probabilities"].items()
                    },
                }
            )
        print(f"Laya prefix decisions: {len(outcomes)}/{len(jobs)}", flush=True)
    del agent
    gc.collect()
    torch.mps.empty_cache()
    return outcomes, [str(item.message) for item in caught], hashlib.sha256(
        (Path(path) / "model.safetensors").read_bytes()
    ).hexdigest()


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
    for start in range(0, len(jobs), JUDGE_BATCH_SIZE):
        batch = jobs[start : start + JUDGE_BATCH_SIZE]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": judge_content(row)}],
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
                for key in ("model", "id", "sentence_id", "category", "gold", "budget", "actual_tokens")
            }
            | {"judge": judge, "label": parse_label(text)}
            for row, text in zip(batch, labels)
        )
        print(f"{judge} prefix decisions: {min(start + len(batch), len(jobs))}/{len(jobs)}", flush=True)
    del model, tokenizer
    gc.collect()
    torch.mps.empty_cache()
    return outcomes


def run(output=OUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Registered experiment 033 requires local MPS")
    jobs, generation_manifest = source_material()
    laya_rows, laya_warnings, laya_weights_sha = run_laya(jobs)
    qwen_rows = run_binary_judge("qwen2.5-3b", jobs)
    phi_rows = run_binary_judge("phi3-mini", jobs)
    outcomes = laya_rows + qwen_rows + phi_rows
    expected = len(jobs) * 3
    if len(outcomes) != expected:
        raise ValueError(f"Expected {expected} decisions, received {len(outcomes)}")
    output.mkdir(parents=True)
    predictions_path = output / "predictions.json"
    predictions_path.write_text(json.dumps(outcomes, indent=2, sort_keys=True) + "\n")
    manifest = {
        "experiment": "033",
        "status": "post-result dose-response follow-up",
        "device": DEVICE,
        "budgets": list(BUDGETS),
        "target_models": CONFIGS,
        "laya": {
            "package": "0.3.20",
            "model_id": LAYA_ID,
            "revision": LAYA_REVISION,
            "weights_sha256": laya_weights_sha,
            "choice_map_seed": MAP_SEED,
            "runtime_warnings": laya_warnings,
        },
        "qwen_model": MODEL_SPECS["qwen-3b"],
        "phi_model": {"model": PHI_ID, "revision": PHI_REVISION},
        "source_generation_sha256": generation_manifest["private_generation_sha256"],
        "source_031_manifest_sha256": hashlib.sha256(
            (PRIVATE_031 / "manifest.json").read_bytes()
        ).hexdigest(),
        "source_032_manifest_sha256": hashlib.sha256(
            (PRIVATE_032 / "manifest.json").read_bytes()
        ).hexdigest(),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "n_prefixes_per_judge": len(jobs),
        "n_decisions": len(outcomes),
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote private label-only prefix judgments to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
