"""Validate and package compact artifacts from an 008 local capture."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analysis_path = args.run / "task-ladder-analysis.json"
    analysis = json.loads(analysis_path.read_text())
    if analysis["selected_task"] is not None or analysis["probes"]:
        raise ValueError("This compact capability bundle expects no task to clear the gate")
    audit = {"protocol": "008", "models": {}, "selected_task": analysis["selected_task"]}
    args.output.mkdir(parents=True, exist_ok=True)
    for name in analysis["completed_models"]:
        source = args.run / name
        rows = json.loads((source / "dataset.json").read_text())
        if len(rows) != 7680 or len({row["id"] for row in rows}) != 7680:
            raise ValueError(f"Unexpected or duplicate rows for {name}")
        if any(
            sum(
                row["label"] == bit
                for row in rows
                if row["task"] == "mixed_review"
                and row["split"] == split
                and row["format"] == fmt
                and row["active"] == active
            )
            != expected
            for split, expected in (("selector", 48), ("final_test", 96))
            for fmt in (0, 1)
            for active in range(3)
            for bit in (0, 1)
        ):
            raise ValueError(f"Unbalanced mixed review rows for {name}")
        dest = args.output / name
        dest.mkdir(exist_ok=True)
        compressed = dest / "dataset.json.gz"
        with compressed.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as out:
            out.write(json.dumps(rows, separators=(",", ":")).encode())
        manifest = json.loads((source / "manifest.json").read_text())
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        audit["models"][name] = {
            "n": len(rows),
            "dataset_sha256": sha(compressed),
            "source_dataset_sha256": manifest["dataset_sha256"],
            "device": manifest["device"],
            "capture_seconds": manifest["capture_seconds"],
            "selector_gates": analysis["selector_gates"][name],
            "final_test_gates": analysis["final_test_gates"][name],
            "final_test_cells": [c for c in analysis["cells"][name] if c["split"] == "final_test"],
            "mixed_review_specificity": [
                x
                for x in analysis["paired_specificity"][name]
                if x["split"] == "final_test"
                and x["task"] in ("mixed_review", "mixed_review_global_majority")
            ],
        }
    audit["incomplete_models"] = analysis["incomplete_models"]
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    shutil_manifest = args.output / "task-ladder-analysis.json"
    shutil_manifest.write_text(json.dumps(analysis, indent=2) + "\n")
    entries = []
    for path in sorted(p for p in args.output.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        entries.append(f"{sha(path)}  {path.relative_to(args.output)}")
    (args.output / "SHA256SUMS").write_text("\n".join(entries) + "\n")


if __name__ == "__main__":
    main()
