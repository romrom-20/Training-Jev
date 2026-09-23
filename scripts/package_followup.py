"""Package measured follow-up artifacts without weights or activation caches."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def copy_artifact(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.name in ("dataset.json", "intervention-rows.jsonl"):
        dest = dest.with_name(dest.name + ".gz")
        with dest.open("wb") as output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
                compressed.write(source.read_bytes())
    else:
        shutil.copy2(source, dest)


def package(remapping, fresh, dest):
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("Refusing to replace an existing bundle")
    dest.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model in ("qwen-0.5b", "qwen-1.5b"):
        for name in (
            "dataset.json",
            "manifest.json",
            "metrics.json",
            "predictions.npz",
            "intervention-rows.jsonl",
            "intervention-manifest.json",
            "analysis.json",
            "score-transport.json",
        ):
            copy_artifact(remapping / model / name, dest / "remapping" / model / name)
        for name in (
            "dataset.json",
            "analysis.json",
            "audit.json",
            "probe-scores.npz",
            "target-output.json",
        ):
            copy_artifact(fresh / model / name, dest / "fresh" / model / name)
        summaries.append(json.loads((fresh / model / "analysis.json").read_text()))
    shutil.copy2(remapping / "overview.png", dest / "remapping.png")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    plt.rcParams.update({"font.size": 10})
    for row, s in zip(axes, summaries):
        x = np.arange(4)
        labels = ["Labeled prose", "JSON", "Sentences", "Options"]
        for metric, axis in [("accuracy", row[0]), ("brier", row[1])]:
            axis.bar(
                x - 0.18,
                [r["baseline"][metric] for r in s["primary"]],
                width=0.35,
                color="#9aa9ad",
                label="Frozen source probe",
            )
            axis.bar(
                x + 0.18,
                [r["corrected"][metric] for r in s["primary"]],
                width=0.35,
                color="#17768a",
                label="Unlabeled query centering",
            )
            axis.axhline(
                0.5 if metric == "accuracy" else 0.25,
                color="#9f6c48",
                linestyle="--",
                label="Uninformative 50/50 predictor",
            )
            axis.set_xticks(x, labels, rotation=12)
            axis.set(
                title=s["model"]["name"]
                + " · "
                + ("readout accuracy" if metric == "accuracy" else "probability quality"),
                ylabel="Accuracy (fraction correct)"
                if metric == "accuracy"
                else "Binary Brier score (lower is better)",
                xlabel="Fresh held-out prompt format",
            )
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(fontsize=7)
            axis.set_ylim(0, 1.05 if metric == "accuracy" else 0.55)
    fig.suptitle(
        "Fresh-format replication: improvements can coexist with unusable readouts", fontsize=14
    )
    fig.savefig(dest / "fresh-formats.png", dpi=170)
    plt.close(fig)
    (dest / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(dest)}\n"
            for p in sorted(dest.rglob("*"))
            if p.is_file()
        )
    )
    print("Packaged", dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remapping", type=Path, default=Path("runs/answer-remapping-v1"))
    parser.add_argument("--fresh", type=Path, default=Path("runs/fresh-template-v1"))
    parser.add_argument("--output", type=Path, default=Path("results/followup"))
    args = parser.parse_args()
    package(args.remapping, args.fresh, args.output)
