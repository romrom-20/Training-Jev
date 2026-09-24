"""Dose-response test of generated answers under learned/random Granite residuals."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
import tripr_layer_confirmation as tripr
from granite_tripr_residual_transfer import (
    CONTROL_SEEDS,
    LAYER,
    MODEL,
    training_directions,
)
from prompt_effect_forecast import sha

from latent_decisions.experiment import provenance, write_json
from latent_decisions.steering import decompose_shared_residual
from latent_decisions.target import block_tensor, load_target, replace_block_tensor

PROTOCOL = Path("docs/experiments/022-granite-generated-dose-response.md")
ROOT_021 = Path("runs/granite-tripr-residual-transfer-v1")
ROOT = Path("runs/granite-tripr-generation-dose-response-v1")
DOSES = (0.10, 0.20, 0.40)
BOOTSTRAP_SEED = 20261023


def collect(offline=False):
    if (ROOT / "evaluation" / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite {ROOT}; remove or rename it first")
    baseline_path = ROOT_021 / "evaluation" / "generation.json"
    baseline = json.loads(baseline_path.read_text())
    stimuli, labels, _ = tripr.load_tripr()
    conflict_ids = {sid for sid, aspects in labels.items() if len(set(aspects.values())) > 1}
    conflict_stimuli = [row for row in stimuli if row["sentence_id"] in conflict_ids]
    if len(conflict_stimuli) != 63 or len(conflict_ids) != 29:
        raise ValueError("Frozen TripR conflict subset changed")
    if len(baseline) != 189:
        raise ValueError("021 baseline generation set is incomplete")

    native, base_dose = training_directions(ROOT_021)
    trained = decompose_shared_residual(native)["residual"]
    random_by_seed = {
        seed: decompose_shared_residual(native, seed=seed)["random_residual"]
        for seed in CONTROL_SEEDS
    }
    cases = []
    for dose_fraction in DOSES:
        scale = dose_fraction / 0.05
        for stimulus in conflict_stimuli:
            source = stimulus["target"]
            cases.append(
                (stimulus, dose_fraction, "trained", None, trained[source] * scale)
            )
            for seed, directions in random_by_seed.items():
                cases.append(
                    (
                        stimulus,
                        dose_fraction,
                        "random",
                        seed,
                        directions[source] * scale,
                    )
                )

    model, tokenizer, device = load_target(MODEL, "auto", offline)
    if len(model.model.layers) != 40:
        raise ValueError("Pinned Granite model must have 40 layers")
    block = model.model.layers[LAYER - 1]
    output = []
    for offset in range(0, len(cases), 4):
        batch = cases[offset : offset + 4]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": case[0]["user"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for case in batch
        ]
        tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        delta = torch.stack([case[4] for case in batch]).to(device) * base_dose

        def patch(_module, _inputs, result):
            value = block_tensor(result).clone()
            value[:, -1, :] += delta
            return replace_block_tensor(result, value)

        handle = block.register_forward_hook(patch)
        try:
            with torch.inference_mode():
                generated = model.generate(
                    **tokens,
                    max_new_tokens=4,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        finally:
            handle.remove()
        generated_text = tokenizer.batch_decode(
            generated[:, tokens.input_ids.shape[1] :], skip_special_tokens=True
        )
        for case, text in zip(batch, generated_text):
            stimulus, dose_fraction, condition, seed, _vector = case
            answer = text.strip().strip(".,!?;:").lower()
            gold = "positive" if stimulus["label"] else "negative"
            output.append(
                {
                    "id": stimulus["id"],
                    "sentence_id": stimulus["sentence_id"],
                    "target": stimulus["target"],
                    "label": stimulus["label"],
                    "dose_fraction": dose_fraction,
                    "condition": condition,
                    "seed": seed,
                    "valid_exact_one_word": int(answer in {"positive", "negative"}),
                    "strict_correct": int(answer == gold),
                }
            )
        print(f"Granite generation dose outcomes: {offset + len(batch)}/{len(cases)}", flush=True)

    destination = ROOT / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "outcomes.json", output)
    write_json(
        destination / "baseline.json",
        [row for row in baseline if row["condition"] == "baseline"],
    )
    write_json(
        destination / "manifest.json",
        {
            "model": MODEL,
            "device": str(device),
            "dtype": "float32",
            "layer": LAYER,
            "dose_fractions": DOSES,
            "n_conflict_sentences": len(conflict_ids),
            "n_conflict_queries": len(conflict_stimuli),
            "n_random_seeds": len(CONTROL_SEEDS),
            "control_seeds": CONTROL_SEEDS,
            "training_activation_sha256": sha(ROOT_021 / "training" / "activations.npz"),
            "baseline_generation_sha256": sha(baseline_path),
            "dataset_xml_sha256": sha(tripr.DATA),
            "protocol_sha256": sha(PROTOCOL),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "provenance": provenance(),
        },
    )
    del model
    gc.collect()
    if str(device).startswith("mps"):
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    collect(args.offline)


if __name__ == "__main__":
    main()
