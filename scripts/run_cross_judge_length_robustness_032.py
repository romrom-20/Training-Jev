"""Judge both answer lengths with the fixed local Qwen and Phi evaluators."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from analyze_prompt_format_score_generation import MODELS
from task_ladder import MODEL_SPECS
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_decisions.target import load_target

PRIVATE_031 = Path(".context/length-censoring-031")
OUT = Path(".context/answer-length-crossjudge-032")
PROTOCOL = Path("docs/experiments/032-cross-judge-length-robustness.md")
QWEN_RESULTS = Path("results/local-open-judge-v1/analysis.json")
PHI_RESULTS = Path("results/independent-judge-check-v1/analysis.json")
PHI_ID = "microsoft/Phi-3-mini-4k-instruct"
PHI_REVISION = "f39ac1d28e925b323eae81227eaba4464caced4e"
LABELS = {"positive": 1, "negative": 0}
DEVICE = "mps"
MAX_NEW_TOKENS = 4
BATCH_SIZE = 8


def parse_label(text):
    return LABELS.get(text.strip().lower().strip(".,!?;:\"'` "))


def judge_content(record):
    return (
        "A model was asked about the sentiment of the "
        f"{record['category']} in a restaurant review. Its answer was: {record['answer']}\n"
        "Based only on that answer, which sentiment did it communicate toward the named aspect? "
        "Reply with exactly one word: positive or negative."
    )


def records_to_judge():
    manifest = json.loads((PRIVATE_031 / "manifest.json").read_text())
    raw_path = PRIVATE_031 / "private-generations.json"
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != manifest["private_generation_sha256"]:
        raise ValueError("Experiment 031 private answer file does not match its manifest")
    records = json.loads(raw_path.read_text())
    if len(records) != len(MODELS) * 233 * 2:
        raise ValueError("Expected all 1,398 paired answer texts from experiment 031")
    return records, manifest


def build_outcomes(judge, records, decoded):
    if len(records) != len(decoded):
        raise ValueError(f"{judge} output count mismatch")
    return [
        {
            "judge": judge,
            "model": row["model"],
            "id": row["id"],
            "sentence_id": row["sentence_id"],
            "category": row["category"],
            "gold": row["gold"],
            "budget": row["budget"],
            "judge_label": parse_label(answer),
        }
        for row, answer in zip(records, decoded)
    ]


def run_qwen(records):
    config = dict(MODEL_SPECS["qwen-3b"])
    model, tokenizer, _ = load_target(config, DEVICE, offline=True)
    tokenizer.padding_side = "left"
    decoded = []
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
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
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded.extend(
            tokenizer.batch_decode(
                generated[:, tokens.input_ids.shape[1] :], skip_special_tokens=True
            )
        )
        print(f"Qwen2.5-3B: {min(start + len(batch), len(records))}/{len(records)}", flush=True)
    del model, tokenizer
    gc.collect()
    torch.mps.empty_cache()
    return build_outcomes("qwen2.5-3b", records, decoded)


def run_phi(records):
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
    decoded = []
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
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
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded.extend(
            tokenizer.batch_decode(
                generated[:, tokens.input_ids.shape[1] :], skip_special_tokens=True
            )
        )
        print(f"Phi-3 Mini: {min(start + len(batch), len(records))}/{len(records)}", flush=True)
    del model, tokenizer
    gc.collect()
    torch.mps.empty_cache()
    return build_outcomes("phi3-mini", records, decoded)


def run(output=OUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("This registered run requires local MPS")
    records, source_manifest = records_to_judge()
    outcomes = run_qwen(records)
    outcomes.extend(run_phi(records))
    if len(outcomes) != 2 * len(MODELS) * 233 * 2:
        raise ValueError("Incomplete independent-judge factorial")
    output.mkdir(parents=True)
    prediction_path = output / "predictions.json"
    prediction_path.write_text(json.dumps(outcomes, indent=2, sort_keys=True) + "\n")
    manifest = {
        "experiment": "032",
        "followup_to": "031",
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decoding": {"do_sample": False},
        "source_generation_manifest_sha256": hashlib.sha256(
            (PRIVATE_031 / "manifest.json").read_bytes()
        ).hexdigest(),
        "source_answer_file_sha256": source_manifest["private_generation_sha256"],
        "qwen_model": MODEL_SPECS["qwen-3b"],
        "phi_model": {"model": PHI_ID, "revision": PHI_REVISION},
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "n_decisions": len(outcomes),
        "n_unparseable": sum(row["judge_label"] is None for row in outcomes),
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote label-only local cross-judge results to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
