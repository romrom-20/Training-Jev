"""Run an aspect-only prior baseline paired to natural prefixes in Experiment 037."""

import argparse
import json
from pathlib import Path

import torch
from run_natural_opinion_span_abstention_037 import (
    BATCH_SIZE,
    MODEL_SPECS,
    PHI_ID,
    PHI_REVISION,
    PUBLIC_STIMULUS_KEYS,
    SOURCE_REVISION,
    load_source_items,
    run_binary_judge,
    run_laya,
    sha256,
)
from run_natural_opinion_span_abstention_037 import (
    PROTOCOL as PROTOCOL_037,
)

PROTOCOL = Path("docs/experiments/038-aspect-only-prior-control.md")
PARENT_RESULT = Path("results/natural-opinion-span-abstention-v1")
OUT = Path(".context/aspect-only-prior-control-038")
ASPECT_ONLY_TEXT = "[no review text provided]"
EXPECTED_PROTOCOL_SHA256 = "ddbf0c681f3a8444ea0ca607df4519a2399e0684a330235b8b7392b2bddba91c"
EXPECTED_PARENT_PROTOCOL_SHA256 = "820fa15fb757d75d2191e0dfea022364e371fe6a4eeed6c52317e3d33b6c346f"


def make_aspect_only_jobs(selected):
    jobs = []
    for item in selected:
        jobs.append(
            {
                **{key: item[key] for key in PUBLIC_STIMULUS_KEYS},
                "condition": "aspect_only",
                "aspect": item["_aspect"],
                "visible_text": ASPECT_ONLY_TEXT,
                "opinion_span_visible": False,
                "expected_decision": "insufficient",
            }
        )
    return jobs


def run(output=OUT):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Experiment 038 requires local MPS")
    if sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("Experiment 038 protocol changed after preregistration")
    if sha256(PROTOCOL_037) != EXPECTED_PARENT_PROTOCOL_SHA256:
        raise ValueError("Parent experiment 037 protocol changed")

    selected = load_source_items()
    public_stimuli = [{key: item[key] for key in PUBLIC_STIMULUS_KEYS} for item in selected]
    parent_stimuli_path = PARENT_RESULT / "stimuli.json"
    parent_predictions_path = PARENT_RESULT / "predictions.json"
    parent_manifest_path = PARENT_RESULT / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text())
    if sha256(parent_predictions_path) != parent_manifest["predictions_sha256"]:
        raise ValueError("Experiment 037 prediction hash mismatch")
    if sha256(parent_stimuli_path) != parent_manifest["stimuli_sha256"]:
        raise ValueError("Experiment 037 stimulus hash mismatch")
    if json.loads(parent_stimuli_path.read_text()) != public_stimuli:
        raise ValueError("Experiment 037 stimuli do not match current pinned source")

    jobs = make_aspect_only_jobs(selected)
    if len(jobs) != 234:
        raise ValueError(f"Expected 234 aspect-only prompts, got {len(jobs)}")
    laya_rows, laya_hash, laya_warnings, laya_seconds = run_laya(jobs)
    qwen_rows, qwen_seconds = run_binary_judge("qwen2.5-3b", jobs)
    phi_rows, phi_seconds = run_binary_judge("phi3-mini", jobs)
    outcomes = laya_rows + qwen_rows + phi_rows
    if len(outcomes) != 1170:
        raise ValueError(f"Expected 1,170 aspect-only outputs; got {len(outcomes)}")

    output.mkdir(parents=True)
    (output / "stimuli.json").write_text(json.dumps(public_stimuli, indent=2) + "\n")
    (output / "predictions.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    manifest = {
        "experiment": "038",
        "exploratory_followup_to": "037",
        "protocol_sha256": sha256(PROTOCOL),
        "runner_sha256": sha256(Path(__file__)),
        "parent_protocol_sha256": sha256(PROTOCOL_037),
        "parent_prediction_sha256": sha256(parent_predictions_path),
        "parent_stimuli_sha256": sha256(parent_stimuli_path),
        "source_revision": SOURCE_REVISION,
        "stimuli_sha256": sha256(output / "stimuli.json"),
        "predictions_sha256": sha256(output / "predictions.json"),
        "stimuli": len(selected),
        "clusters": len({row["cluster_id"] for row in selected}),
        "outcomes": len(outcomes),
        "device": "mps",
        "batch_size": BATCH_SIZE,
        "seed": 20260938,
        "models": {
            "laya": {
                "id": "convaiinnovations/laya",
                "revision": "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
                "package": "laya==0.3.20",
                "weights_sha256": laya_hash,
                "seconds": laya_seconds,
            },
            "qwen2.5-3b": {
                "revision": MODEL_SPECS["qwen-3b"]["revision"],
                "seconds": qwen_seconds,
            },
            "phi3-mini": {"id": PHI_ID, "revision": PHI_REVISION, "seconds": phi_seconds},
        },
        "laya_warnings": laya_warnings,
        "label_only_outputs": True,
        "raw_review_text_published": False,
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
