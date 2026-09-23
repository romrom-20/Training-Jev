"""Audit and package target-only artifacts from the aspect sentiment run."""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from aspect_sentiment_study import ASPECTS, TEXT_VARIANTS


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination
    if destination.exists() and any(path.name != "README.md" for path in destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {destination}")
    rows = json.loads((args.source / "dataset.json").read_text())
    manifest = json.loads((args.source / "manifest.json").read_text())
    assert len(rows) == 4032
    assert len({row["id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["label"] == row["bits"][row["active"]]
        for j, aspect in enumerate(ASPECTS):
            expected = TEXT_VARIANTS[aspect][1 - row["bits"][j]][row["group_no"] % 2]
            assert expected in row["review"], (row["id"], expected)
    group_counts = Counter((row["split"], row["group"], row["format"]) for row in rows)
    assert len(group_counts) == 168
    assert set(group_counts.values()) == {24}
    split_counts = Counter(row["split"] for row in rows)
    assert split_counts == {
        "train": 768,
        "validation": 192,
        "calibration": 192,
        "source_test": 576,
        "shift_test": 2304,
    }
    capability = []
    for aspect_id, aspect in enumerate(ASPECTS):
        selected = [
            row for row in rows if row["split"] == "source_test" and row["active"] == aspect_id
        ]
        capability.append(
            {
                "aspect": aspect,
                "n": len(selected),
                "strict_accuracy": sum(row["strict_correct"] for row in selected) / len(selected),
                "conditional_accuracy": sum(row["conditional_correct"] for row in selected)
                / len(selected),
            }
        )
    assert capability == manifest["capability"]
    gate = all(item["strict_accuracy"] >= 0.9 for item in capability)
    assert not gate and gate == manifest["capability_gate_passed"]
    shifted = []
    for fmt in (0, 1, 2, 3):
        for aspect_id, aspect in enumerate(ASPECTS):
            selected = [
                row
                for row in rows
                if row["split"] == "shift_test"
                and row["format"] == fmt
                and row["active"] == aspect_id
            ]
            shifted.append(
                {
                    "format": fmt,
                    "aspect": aspect,
                    "n": len(selected),
                    "strict_accuracy": sum(row["strict_correct"] for row in selected)
                    / len(selected),
                    "conditional_accuracy": sum(row["conditional_correct"] for row in selected)
                    / len(selected),
                }
            )

    destination.mkdir(parents=True, exist_ok=True)
    output_data = destination / "qwen-1.5b"
    output_data.mkdir()
    (output_data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
    (output_data / "dataset.json.gz").write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    audit = {
        "rows": len(rows),
        "unique_ids": len({row["id"] for row in rows}),
        "split_prompt_counts": dict(split_counts),
        "groups_times_formats": len(group_counts),
        "labels_match_review_cues": True,
        "capability_gate_passed": gate,
        "source_test": capability,
        "shift_test": shifted,
        "protocol_sha256": manifest["protocol_sha256"],
    }
    (output_data / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(10, 5.2), layout="constrained")
    colors = {"food": "#197c83", "service": "#c16a3a", "value": "#536a9b"}
    for aspect in ASPECTS:
        vals = [
            next(
                x["strict_accuracy"]
                for x in shifted
                if x["format"] == fmt and x["aspect"] == aspect
            )
            for fmt in (0, 1, 2, 3)
        ]
        ax.plot(
            ("Source\ntemplate", "Format 1", "Format 2", "Format 3"),
            vals,
            marker="o",
            linewidth=2,
            label=aspect.title(),
            color=colors[aspect],
        )
    ax.axhline(0.9, color="#6a6a6a", linestyle="--", linewidth=1.4, label="Capability gate (90%)")
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Greedy label accuracy (fraction correct)")
    ax.set_xlabel("Prompt format on 24 held-out scenario groups")
    ax.set_title("Qwen2.5-1.5B aspect-sentiment task accuracy")
    ax.legend(frameon=False, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(destination / "capability-accuracy.png", dpi=180)
    plt.close(fig)
    hashes = []
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            hashes.append(f"{sha256(path)}  {path.relative_to(destination)}")
    (destination / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
    print(
        json.dumps(
            {
                "output": str(destination),
                "audit": audit,
                "figure": str(destination / "capability-accuracy.png"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
