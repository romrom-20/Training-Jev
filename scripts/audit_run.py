"""Recompute published scores from per-example predictions; fail loudly on mismatch."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from latent_decisions.data import validate_splits
from latent_decisions.experiment import write_json
from latent_decisions.metrics import metrics


def audit(run):
    rows = json.loads((run / "dataset.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    report = json.loads((run / "metrics.json").read_text())
    predictions = json.loads((run / "predictions.json").read_text())
    validate_splits(rows)
    assert (
        hashlib.sha256((run / "dataset.json").read_bytes()).hexdigest()
        == manifest["dataset_sha256"]
    )
    by_id = {r["id"]: r for r in rows}
    assert len(by_id) == len(rows)
    expected = {
        (seed, layer, method, row["id"])
        for seed in report["config"]["seeds"]
        for layer in report["config"]["layers"]
        for method in ("bilinear", "independent_linear", "query_only", "shuffled_labels")
        for row in rows
        if row["split"] in ("test", "ood")
    }
    actual = {(p["seed"], p["layer"], p["method"], p["id"]) for p in predictions}
    assert actual == expected and len(actual) == len(predictions), (
        "Missing or duplicate predictions"
    )
    for p in predictions:
        row = by_id[p["id"]]
        assert (
            p["group"] == row["group"]
            and p["labels"] == row["labels"]
            and p["split"] == row["split"]
        )
    checks = 0
    for record in report["records"]:
        for split in ("test", "ood"):
            selected = [
                p
                for p in predictions
                if p["split"] == split
                and all(p[k] == record[k] for k in ("seed", "layer", "method"))
            ]
            recomputed = metrics(
                [p["labels"] for p in selected], [p["probabilities"] for p in selected]
            )
            for key in ("accuracy", "brier", "nll", "auroc", "ece"):
                # Original label means are float32; JSON labels load as integers.
                # Allow rounding in ECE frequencies, not a material metric mismatch.
                np.testing.assert_allclose(
                    recomputed[key], record["evaluations"][split]["calibrated"][key], atol=1e-7
                )
                checks += 1
    layer_scores = {
        layer: np.mean(
            [
                r["validation_nll"]
                for r in report["records"]
                if r["method"] == "bilinear" and r["layer"] == layer
            ]
        )
        for layer in report["config"]["layers"]
    }
    assert report["selected_layer"] == min(layer_scores, key=layer_scores.get)
    causal = json.loads((run / "interventions.json").read_text())
    for name, summary in causal["summary"].items():
        records = [r for r in causal["records"] if r["direction"] == name]
        assert len(records) == causal["prompts"]
        for r in records:
            assert by_id[r["id"]]["split"] == "test"
            np.testing.assert_allclose(
                r["delta_log_odds"], r["plus"]["log_odds"] - r["minus"]["log_odds"]
            )
        np.testing.assert_allclose(summary["mean"], np.mean([r["delta_log_odds"] for r in records]))
    assert abs(causal["summary"]["zero"]["mean"]) < 1e-6
    result = dict(
        status="passed",
        prediction_rows=len(predictions),
        recomputed_metric_checks=checks,
        checked=[
            "dataset hash and scenario isolation",
            "prediction completeness and label identity",
            "saved metrics vs per-example predictions",
            "validation-only layer selection",
            "intervention arithmetic and test-only prompts",
            "zero-vector sham",
        ],
        scope="Artifact consistency audit; does not establish scientific generalization",
    )
    write_json(run / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    audit(parser.parse_args().run)
