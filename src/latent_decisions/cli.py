import argparse
import hashlib
import json
import resource
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import make_dataset, validate_splits
from .experiment import provenance, train, write_json


def main():
    parser = argparse.ArgumentParser(
        description="Latent Decisions: laptop-scale probing experiments"
    )
    parser.add_argument("command", choices=("collect", "train", "intervene", "report", "all"))
    parser.add_argument("--config", type=Path, default=Path("configs/macbook.toml"))
    parser.add_argument("--run", type=Path, default=Path("runs/macbook"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--offline", action="store_true", help="Use cached target weights only")
    parser.add_argument("--intervention-groups", type=int, default=4)
    parser.add_argument("--dose", type=float, default=0.05)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    run = args.run
    run.mkdir(parents=True, exist_ok=True)
    commands = (
        ("collect", "train", "intervene", "report") if args.command == "all" else (args.command,)
    )
    for command in commands:
        if command == "collect":
            if (run / "manifest.json").exists() or (run / "activations.npz").exists():
                parser.error("Run already has captures; use a new --run directory")
            from .target import collect

            counts = {
                s: config[f"{s}_groups"]
                for s in ("train", "validation", "calibration", "test", "ood")
            }
            rows = make_dataset(counts)
            validate_splits(rows)
            write_json(run / "dataset.json", rows)
            h, q, probs, metadata = collect(rows, config, args.device, args.offline)
            np.savez_compressed(run / "activations.npz", h=h, q=q, color_probs=probs)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            metadata["peak_process_rss_gib"] = rss / (
                1024**3 if sys.platform == "darwin" else 1024**2
            )
            metadata["memory_note"] = (
                "process RSS high-water mark; excludes some Metal/driver allocations"
            )
            metadata["target_color_choice_accuracy"] = float(
                np.mean(probs.argmax(-1) == np.array([r["labels"][0] for r in rows]))
            )
            metadata["target_greedy_color_accuracy"] = float(
                np.mean(
                    [
                        token == ("red" if row["labels"][0] else "blue")
                        for token, row in zip(metadata["top_tokens"], rows)
                    ]
                )
            )
            metadata["mean_color_token_mass"] = float(probs.sum(-1).mean())
            metadata["split_counts"] = {s: n * 8 for s, n in counts.items()}
            metadata["activation_cache_mib"] = (run / "activations.npz").stat().st_size / 1024**2
            write_json(
                run / "manifest.json",
                dict(
                    config=config,
                    provenance=provenance(),
                    collection=metadata,
                    evidence="controlled real-model pilot",
                    collected_at=datetime.now(timezone.utc).isoformat(),
                    dataset_sha256=hashlib.sha256((run / "dataset.json").read_bytes()).hexdigest(),
                    activations_sha256=hashlib.sha256(
                        (run / "activations.npz").read_bytes()
                    ).hexdigest(),
                ),
            )
        else:
            manifest = json.loads((run / "manifest.json").read_text())
            if manifest["config"] != config:
                parser.error(
                    "Config differs from captured manifest; use the original config or a new run"
                )
            for filename, key in (
                ("dataset.json", "dataset_sha256"),
                ("activations.npz", "activations_sha256"),
            ):
                if (
                    command != "report"
                    and hashlib.sha256((run / filename).read_bytes()).hexdigest() != manifest[key]
                ):
                    parser.error(f"{filename} differs from recorded capture")
            if command == "train":
                if (run / "metrics.json").exists():
                    parser.error("This run is already trained; use a new run to preserve results")
                train(run, config)
            elif command == "intervene":
                if (run / "interventions.json").exists():
                    parser.error("Interventions already exist; use a new run to preserve results")
                from .intervene import intervene

                intervene(
                    run, config, args.device, args.offline, args.intervention_groups, args.dose
                )
            else:
                from .report import render

                render(run)
        print(f"Completed {command}: {run}", flush=True)


if __name__ == "__main__":
    main()
