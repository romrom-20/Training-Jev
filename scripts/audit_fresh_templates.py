"""Independently reconstruct fresh-template corrections from saved score artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
from score_transport import compact_metrics, estimate_offsets


def audit(source, fresh):
    records = json.loads((fresh / "dataset.json").read_text())
    report = json.loads((fresh / "analysis.json").read_text())
    source_records = json.loads((source / "dataset.json").read_text())
    source_report = json.loads((source / "metrics.json").read_text())
    with np.load(fresh / "probe-scores.npz") as cache:
        target_scores = cache["scores"].copy()
    with np.load(source / "predictions.npz") as cache:
        source_logits = cache["logits"].copy()
        source_rows = [source_records[i] for i in cache["indices"]]
    keys = [
        k
        for k in source_report["prediction_keys"]
        if k["layer"] == report["model"]["intervention_layer"] and k["seed"] == 0
    ]
    assert len(keys) == len(target_scores) == 4
    q = np.array([r["active"] for r in records])
    y = np.array([r["labels"][r["active"]] for r in records])
    src_mask = np.array([r["split"] == "test" and r["group_number"] < 6 for r in source_rows])
    src_q = np.array([r["active"] for r in source_rows])[src_mask]
    checks = 0
    for bank, key in enumerate(keys):
        key_index = source_report["prediction_keys"].index(key)
        src = source_logits[key_index][src_mask, src_q].astype(float)
        raw = target_scores[bank]
        for r in report["results"]:
            if r["regime"] != key["regime"] or r["probe"] != key["method"]:
                continue
            anchors = np.array(
                [x["style"] == r["style"] and x["split"] == "adaptation" for x in records]
            )
            evaluation = np.array(
                [x["style"] == r["style"] and x["split"] == "evaluation" for x in records]
            )
            assert not (anchors & evaluation).any()
            weights = np.where(
                y[anchors] == 1,
                2 * r["adaptation_prevalence"],
                2 * (1 - r["adaptation_prevalence"]),
            )
            offsets = estimate_offsets(src, src_q, raw[anchors], q[anchors], weights)
            # Independently evaluated float32 GEMMs can vary by a few ulps with batch shape.
            np.testing.assert_allclose(offsets[r["correction"]], r["offset"], atol=1e-4)
            score = raw - np.asarray(r["offset"])[q]
            measured = compact_metrics(y[evaluation], score[evaluation], key["temperature"])
            for field in ("accuracy", "brier", "nll", "ece"):
                np.testing.assert_allclose(measured[field], r["metrics"][field], atol=1e-7)
                checks += 1
    result = dict(
        status="passed",
        metric_checks=checks,
        source="stored source predictions and independently saved fresh scores",
        checked=[
            "frozen bank identity",
            "source-anchor reconstruction",
            "disjoint fresh anchors/evaluation",
            "offsets and reported metrics",
        ],
    )
    (fresh / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(fresh.name, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("runs/answer-remapping-v1"))
    parser.add_argument("--fresh", type=Path, default=Path("runs/fresh-template-v1"))
    args = parser.parse_args()
    for fresh in sorted(args.fresh.glob("qwen-*")):
        if (fresh / "analysis.json").exists():
            audit(args.source / fresh.name, fresh)
