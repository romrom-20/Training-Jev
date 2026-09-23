"""Frozen-logit adaptation with disjoint unlabeled anchors and explicit prior-shift stress."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from latent_decisions.experiment import write_json
from latent_decisions.metrics import cluster_mean_ci, metrics, sigmoid


def estimate_offsets(source_scores, source_queries, target_scores, target_queries, weights=None):
    """No label argument: fit one shared or three property-specific additive shifts."""
    weights = np.ones(len(target_scores)) if weights is None else np.asarray(weights)
    shared = np.average(target_scores, weights=weights) - np.mean(source_scores)
    per_query = np.array(
        [
            np.average(target_scores[target_queries == q], weights=weights[target_queries == q])
            - np.mean(source_scores[source_queries == q])
            for q in range(3)
        ]
    )
    return dict(none=np.zeros(3), global_offset=np.repeat(shared, 3), query_offset=per_query)


def compact_metrics(y, logits, temperature):
    result = metrics(y, sigmoid(logits / temperature))
    return {k: result[k] for k in ("n", "accuracy", "brier", "nll", "ece")}


def analyze(run):
    manifest = json.loads((run / "metrics.json").read_text())
    all_rows = json.loads((run / "dataset.json").read_text())
    with np.load(run / "predictions.npz") as cache:
        predictions = cache["logits"].copy()
        rows = [all_rows[i] for i in cache["indices"]]
    q = np.array([r["active"] for r in rows])
    y = np.array([r["labels"][r["active"]] for r in rows])
    source = np.array([r["split"] == "test" and r["group_number"] < 6 for r in rows])
    target = np.array([r["split"] == "ood" and r["group_number"] < 6 for r in rows])
    evaluation = np.array([r["split"] == "ood" and r["group_number"] >= 6 for r in rows])
    assert not (target & evaluation).any()
    assert not (
        {r["group"] for r, m in zip(rows, target) if m}
        & {r["group"] for r, m in zip(rows, evaluation) if m}
    )
    results = []
    loss_records = []
    checks = 0
    for key, logits in zip(manifest["prediction_keys"], predictions):
        active_logits = logits[np.arange(len(rows)), q].astype(float)
        for prevalence in (0.1, 0.5, 0.9):
            # Labels construct an artificial population; the estimator sees scores and weights.
            weights = np.where(y[target] == 1, 2 * prevalence, 2 * (1 - prevalence))
            offsets = estimate_offsets(
                active_logits[source], q[source], active_logits[target], q[target], weights
            )
            for method, offset in offsets.items():
                adjusted = active_logits - offset[q]
                for code in (1, -1):
                    mask = evaluation & np.array([r["code"] == code for r in rows])
                    by_property = []
                    for prop in range(3):
                        cell = mask & (q == prop)
                        before = roc_auc_score(y[cell], active_logits[cell])
                        after = roc_auc_score(y[cell], adjusted[cell])
                        assert abs(before - after) < 1e-10
                        checks += 1
                        by_property.append(
                            dict(
                                property=prop,
                                rank_auroc=after,
                                **compact_metrics(y[cell], adjusted[cell], key["temperature"]),
                            )
                        )
                    results.append(
                        dict(
                            model=manifest["model"]["name"],
                            layer=key["layer"],
                            regime=key["regime"],
                            probe=key["method"],
                            seed=key["seed"],
                            code=code,
                            adaptation_prevalence=prevalence,
                            correction=method,
                            offset=offset.tolist(),
                            metrics=compact_metrics(y[mask], adjusted[mask], key["temperature"]),
                            by_property=by_property,
                        )
                    )
                    if method == "query_offset" and prevalence == 0.5:
                        p = sigmoid(adjusted[mask] / key["temperature"])
                        original = sigmoid(active_logits[mask] / key["temperature"])
                        for row, a, b, label in zip(
                            np.array(rows, dtype=object)[mask], p, original, y[mask]
                        ):
                            loss_records.append(
                                dict(
                                    layer=key["layer"],
                                    regime=key["regime"],
                                    probe=key["method"],
                                    seed=key["seed"],
                                    code=code,
                                    id=row["id"],
                                    group=row["group"],
                                    delta_brier=float((a - label) ** 2 - (b - label) ** 2),
                                )
                            )
    comparisons = []
    for layer in manifest["model"]["layers"]:
        for regime in ("fixed_code", "balanced_code"):
            for probe in ("bilinear", "independent"):
                for code in (1, -1):
                    records = [
                        r
                        for r in loss_records
                        if r["layer"] == layer
                        and r["regime"] == regime
                        and r["probe"] == probe
                        and r["code"] == code
                    ]
                    ids = sorted({r["id"] for r in records})
                    delta = [
                        np.mean([r["delta_brier"] for r in records if r["id"] == id_])
                        for id_ in ids
                    ]
                    groups = [next(r["group"] for r in records if r["id"] == id_) for id_ in ids]
                    comparisons.append(
                        dict(
                            layer=layer,
                            regime=regime,
                            probe=probe,
                            code=code,
                            delta_brier=cluster_mean_ci(delta, groups),
                        )
                    )
    output = dict(
        model=manifest["model"],
        results=results,
        paired_comparisons=comparisons,
        audit=dict(
            status="passed",
            rank_invariance_checks=checks,
            source_anchor_groups=6,
            target_anchor_groups=6,
            disjoint_evaluation_groups=6,
        ),
        scope="Post-failure prospective diagnostic; balanced evaluation and artificial prior-shift stress; not a new calibration method",
    )
    write_json(run / "score-transport.json", output)
    print(manifest["model"]["name"], output["audit"])
    middle = manifest["model"]["intervention_layer"]
    for regime in ("fixed_code", "balanced_code"):
        for method in ("none", "global_offset", "query_offset"):
            rs = [
                r
                for r in results
                if r["layer"] == middle
                and r["regime"] == regime
                and r["probe"] == "bilinear"
                and r["correction"] == method
                and r["adaptation_prevalence"] == 0.5
            ]
            print(
                regime,
                method,
                "accuracy",
                round(np.mean([r["metrics"]["accuracy"] for r in rs]), 4),
                "Brier",
                round(np.mean([r["metrics"]["brier"] for r in rs]), 4),
            )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    for run in sorted(args.run.glob("qwen-*")):
        analyze(run)
