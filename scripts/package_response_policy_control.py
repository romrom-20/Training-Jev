"""Validate and package compact artifacts from experiment 009."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ("qwen-1.5b", "smollm2-1.7b")
TASKS = ("isolated_clause", "neutral_distractors", "mixed_review", "keyed_record")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_predictions(rows, name):
    if len(rows) != 11520 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError(f"Wrong row count or duplicate prompt IDs: {name}")
    counts = {fmt: sum(r["format"] == fmt for r in rows) for fmt in range(3)}
    if counts != {0: 3840, 1: 3840, 2: 3840}:
        raise ValueError(f"Unbalanced prompt formats: {name}: {counts}")
    paired = {}
    for row in rows:
        key = row["id"].rsplit("-f", 1)[0]
        paired.setdefault(key, {})[row["format"]] = row
    if len(paired) != 3840 or any(set(group) != {0, 1, 2} for group in paired.values()):
        raise ValueError(f"Prompt conditions are not fully paired: {name}")
    for group in paired.values():
        if (
            len({(r["context"], r["aspect"], r["label"], tuple(r["bits"])) for r in group.values()})
            != 1
        ):
            raise ValueError(f"Context/target/label changed across conditions: {name}")
        if any(
            not isinstance(r["strict_correct"], bool)
            or not isinstance(r["conditional_correct"], bool)
            for r in group.values()
        ):
            raise ValueError(f"Non-boolean correctness field: {name}")
    for model_split in ("selector", "final_test"):
        n_groups = 12 if model_split == "selector" else 24
        for fmt in range(3):
            n = sum(r["split"] == model_split and r["format"] == fmt for r in rows)
            if n != 60 * n_groups:
                raise ValueError(f"Unexpected {model_split} row count for {name}: {n}")
            for task in TASKS:
                for aspect in ("food", "service", "value"):
                    cell = [
                        r
                        for r in rows
                        if r["split"] == model_split
                        and r["format"] == fmt
                        and r["task"] == task
                        and r["aspect"] == aspect
                    ]
                    labels = [r["label"] for r in cell]
                    if len(cell) == 0 or labels.count(0) != labels.count(1):
                        raise ValueError(
                            f"Unbalanced label cell for {name}: {model_split}/{task}/{fmt}/{aspect}"
                        )


def make_figure(rows_by_model, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, layout="constrained")
    x = np.arange(len(TASKS))
    width = 0.24
    for ax, model in zip(axes, MODELS):
        rows = rows_by_model[model]
        values = {fmt: [] for fmt in (1, 2)}
        conditional = []
        for task in TASKS:
            for fmt in (1, 2):
                cell = [
                    r
                    for r in rows
                    if r["split"] == "final_test" and r["task"] == task and r["format"] == fmt
                ]
                values[fmt].append(np.mean([r["strict_correct"] for r in cell]))
            c = [
                r
                for r in rows
                if r["split"] == "final_test" and r["task"] == task and r["format"] == 2
            ]
            conditional.append(np.mean([r["conditional_correct"] for r in c]))
        ax.bar(x - width, values[1], width, color="#c77551", label="B: no exact-word constraint")
        ax.bar(x, values[2], width, color="#3e806f", label="C: exact-word constraint")
        ax.bar(
            x + width, conditional, width, color="#5b77a6", label="C: conditional label accuracy"
        )
        ax.set_title(model)
        ax.set_xticks(x, ["Isolated", "Neutral\ncontext", "Mixed\nreview", "Keyed\nrecord"])
        ax.set_ylim(0, 1.04)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Final-test accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Explicit format constraint restores label compliance, not task accuracy")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analysis_path = args.run / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    rows_by_model = {}
    audit = {
        "protocol": "009",
        "models": {},
        "smollm_simple_task_rule": analysis["smollm_simple_task_rule"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    runner = repo / "scripts/response_policy_control.py"
    protocol = repo / "docs/experiments/009-output-constraint-control.md"
    for name in MODELS:
        source = args.run / name
        predictions_path = source / "predictions.json"
        manifest_path = source / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if sha(predictions_path) != manifest["predictions_sha256"]:
            raise ValueError(f"Prediction checksum mismatch: {name}")
        if sha(protocol) != manifest["protocol_sha256"]:
            raise ValueError(f"Frozen protocol checksum mismatch: {name}")
        rows = json.loads(predictions_path.read_text())
        verify_predictions(rows, name)
        rows_by_model[name] = rows
        destination = args.output / name
        destination.mkdir(exist_ok=True)
        zipped = destination / "predictions.json.gz"
        with zipped.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(json.dumps(rows, separators=(",", ":")).encode())
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        audit["models"][name] = {
            "n": len(rows),
            "packed_predictions_sha256": sha(zipped),
            "source_predictions_sha256": manifest["predictions_sha256"],
            "source_dataset_sha256": manifest["dataset_sha256"],
            "protocol_sha256": manifest["protocol_sha256"],
            "runner_sha256_at_packaging": sha(runner),
            "git_commit": manifest["provenance"]["git_commit"],
            "device": manifest["device"],
            "capture_seconds": manifest["capture_seconds"],
            "final_test_cells": [
                cell for cell in analysis["cells"][name] if cell["split"] == "final_test"
            ],
            "paired_format_comparisons": analysis["paired_format_comparisons"][name],
        }
    analysis_destination = args.output / "analysis.json"
    analysis_destination.write_text(json.dumps(analysis, indent=2) + "\n")
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    make_figure(rows_by_model, args.output / "response-format-control.png")
    entries = []
    for path in sorted(p for p in args.output.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        entries.append(f"{sha(path)}  {path.relative_to(args.output)}")
    (args.output / "SHA256SUMS").write_text("\n".join(entries) + "\n")


if __name__ == "__main__":
    main()
