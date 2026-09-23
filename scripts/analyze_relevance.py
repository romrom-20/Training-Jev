"""Audit and summarize the frozen answer-remapping study, including failed gates."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latent_decisions.data import validate_splits
from latent_decisions.experiment import write_json
from latent_decisions.metrics import cluster_mean_ci, metrics, sigmoid
from latent_decisions.relevance import regime_mask


def analyze(run):
    manifest = json.loads((run / "manifest.json").read_text())
    report = json.loads((run / "metrics.json").read_text())
    rows = json.loads((run / "dataset.json").read_text())
    validate_splits(rows)
    assert (
        hashlib.sha256((run / "dataset.json").read_bytes()).hexdigest()
        == manifest["dataset_sha256"]
    )
    assert manifest["config"] == report["config"]
    with np.load(run / "predictions.npz") as cache:
        logits = cache["logits"].copy()
        indices = cache["indices"].copy()
    eval_rows = [rows[i] for i in indices]
    labels = np.array([r["labels"] for r in eval_rows])
    assert len(logits) == len(report["records"]) == 36
    assert [r for r in rows if r["split"] in ("test", "ood")] == eval_rows
    audits = 0
    records = []
    for ri, r in enumerate(report["records"]):
        for name, result in r["evaluations"].items():
            split, codepart, relevance = name.split("/")
            code = int(codepart.removeprefix("code"))
            mask = np.array(
                [
                    [
                        row["split"] == split
                        and row["code"] == code
                        and ((j == row["active"]) == (relevance == "relevant"))
                        for j in range(3)
                    ]
                    for row in eval_rows
                ]
            )
            for readout, temp in (("raw", 1), ("calibrated", r["temperature"])):
                recomputed = metrics(labels[mask], sigmoid(logits[ri][mask] / temp))
                for key in ("accuracy", "brier", "nll", "auroc", "ece"):
                    np.testing.assert_allclose(recomputed[key], result[readout][key], atol=1e-7)
                    audits += 1
        train = regime_mask(rows, report["config"], "train", r["regime"])
        assert int(train.sum()) == r["train_examples"] == 384
    for layer in manifest["model"]["layers"]:
        for regime in ("fixed_code", "balanced_code"):
            for method in ("bilinear", "independent"):
                key_indices = [
                    i
                    for i, k in enumerate(report["prediction_keys"])
                    if k["layer"] == layer and k["regime"] == regime and k["method"] == method
                ]
                for split in ("test", "ood"):
                    for code in (1, -1):
                        mask = np.array(
                            [r["split"] == split and r["code"] == code for r in eval_rows]
                        )
                        active = np.array([r["active"] for r in eval_rows])[mask]
                        truth = labels[mask, active]
                        losses = []
                        correct = []
                        for k in key_indices:
                            p = sigmoid(
                                logits[k][mask, active]
                                / report["prediction_keys"][k]["temperature"]
                            )
                            losses.append((p - truth) ** 2)
                            correct.append((p >= 0.5) == truth)
                        groups = [r["group"] for r, m in zip(eval_rows, mask) if m]
                        records.append(
                            dict(
                                layer=layer,
                                regime=regime,
                                method=method,
                                split=split,
                                code=code,
                                accuracy=cluster_mean_ci(np.mean(correct, axis=0), groups),
                                brier=cluster_mean_ci(np.mean(losses, axis=0), groups),
                            )
                        )
    # Immutable derived summary; all raw prediction slices remain available.
    summary = dict(
        model=manifest["model"],
        capability=manifest["capability"],
        capability_gate_passed=manifest["capability_gate_passed"],
        prediction=records,
        audit=dict(metric_values_recomputed=audits, status="passed"),
        capture_seconds=manifest["capture_seconds"],
        fit_seconds=report["seconds"],
    )
    path = run / "intervention-manifest.json"
    if not path.exists():
        write_json(run / "analysis-partial.json", summary)
        return summary
    intervention_manifest = json.loads(path.read_text())
    data = (run / "intervention-rows.jsonl").read_bytes()
    assert hashlib.sha256(data).hexdigest() == intervention_manifest["records_sha256"]
    interventions = [json.loads(line) for line in data.splitlines()]
    by_id = {r["id"]: r for r in rows}
    groups = set(intervention_manifest["groups"])
    assert len(interventions) == intervention_manifest["records"]
    seen = set()
    for row in interventions:
        assert (row["name"], row["id"]) not in seen
        seen.add((row["name"], row["id"]))
        source = by_id[row["id"]]
        assert source["split"] == "test" and row["group"] in groups
        assert source["code"] == row["code"] and source["active"] == row["active"]
        if row["kind"] == "add":
            delta = row["outcomes"]["1"]["log_odds"] - row["outcomes"]["-1"]["log_odds"]
            np.testing.assert_allclose(row["semantic_effect"], source["code"] * delta)
        else:
            donor = by_id[row["donor_id"]]
            assert all(source[k] == donor[k] for k in ("group", "active", "code", "template"))
            assert sum(a != b for a, b in zip(source["labels"], donor["labels"])) == 1
            assert source["labels"][row["property"]] != donor["labels"][row["property"]]
            delta = row["outcomes"]["1"]["log_odds"] - row["baseline_log_odds"]
            np.testing.assert_allclose(
                row["semantic_effect"],
                source["code"] * (1 - 2 * source["labels"][row["property"]]) * delta,
            )
        np.testing.assert_allclose(row["delta_log_odds"], delta)
        if row["name"] == "zero":
            assert abs(delta) < 1e-6
            assert abs(row["outcomes"]["1"]["log_odds"] - row["baseline_log_odds"]) < 1e-4
        audits += 1
    grouped = defaultdict(list)
    for r in interventions:
        family = "/".join(r["name"].split("/")[:-1]) if r["property"] >= 0 else r["name"]
        relation = "relevant" if r["property"] == r["active"] else "irrelevant"
        if r["property"] == -1:
            relation = "output_control"
        grouped[(family, r["code"], relation)].append(r)
    steering = []
    for (family, code, relation), rs in grouped.items():
        steering.append(
            dict(
                family=family,
                code=code,
                relation=relation,
                signed_effect=cluster_mean_ci(
                    [r["semantic_effect"] for r in rs], [r["group"] for r in rs]
                ),
                absolute_effect=cluster_mean_ci(
                    [abs(r["semantic_effect"]) for r in rs], [r["group"] for r in rs]
                ),
                mean_option_mass=float(
                    np.mean([o["mass"] for r in rs for o in r["outcomes"].values()])
                ),
                positive_fraction=float(np.mean([r["semantic_effect"] > 0 for r in rs])),
            )
        )
    summary.update(
        steering=steering,
        intervention_seconds=intervention_manifest["seconds"],
        intervention_timing_scope=intervention_manifest.get(
            "timing_scope", "entire intervention stage"
        ),
        output_A_control_meaning="Task-correct A label, not the target's observed A/B output",
        intervention_layer=intervention_manifest["layer"],
        intervention_records=len(interventions),
    )
    summary["audit"].update(
        intervention_rows_checked=len(interventions),
        checks_total=audits,
        scope="Metric reconstruction, split/budget checks, donor matching, hook-zero equivalence and effect arithmetic",
    )
    write_json(run / "analysis.json", summary)
    return summary


def plot(summaries, root):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(
        len(summaries), 3, figsize=(15, 5 * len(summaries)), squeeze=False, layout="constrained"
    )
    for row, s in zip(axes, summaries):
        name = s["model"]["name"]
        for regime, color in (("fixed_code", "#176e89"), ("balanced_code", "#a75726")):
            for code, style in ((1, "-"), (-1, "--")):
                records = [
                    r
                    for r in s["prediction"]
                    if r["regime"] == regime
                    and r["method"] == "bilinear"
                    and r["split"] == "test"
                    and r["code"] == code
                ]
                row[0].plot(
                    [r["layer"] for r in records],
                    [r["accuracy"]["mean"] for r in records],
                    style,
                    marker="o",
                    color=color,
                    label=f"{regime}, code {code:+d}",
                )
        row[0].set(
            title=f"{name}: active-property readout",
            xlabel="Transformer block (1-indexed)",
            ylabel="Test accuracy (mean of 3 seeds)",
            ylim=(0, 1.05),
        )
        row[0].legend(fontsize=7)
        caps = [c for c in s["capability"] if c["split"] == "test"]
        row[1].bar(range(len(caps)), [c["greedy_accuracy"] for c in caps], color="#364954")
        row[1].axhline(0.9, color="#b26336", linestyle="--", label="Prespecified competence gate")
        row[1].set_xticks(range(len(caps)), [f"{c['active']}\n{c['code']:+d}" for c in caps])
        row[1].set(
            title=f"Target task competence: {'PASS' if s['capability_gate_passed'] else 'FAIL'}",
            ylabel="Greedy A/B answer accuracy",
            xlabel="Requested property and answer mapping",
            ylim=(0, 1.05),
        )
        row[1].legend(fontsize=7)
        if "steering" in s:
            families = [
                "fixed_code/bilinear",
                "balanced_code/bilinear",
                "fixed_code/independent",
                "balanced_code/independent",
                "donor",
            ]
            for code, color, offset in ((1, "#176e89", -0.16), (-1, "#a75726", 0.16)):
                selected = [
                    next(
                        r
                        for r in s["steering"]
                        if r["family"] == family
                        and r["code"] == code
                        and r["relation"] == "relevant"
                    )
                    for family in families
                ]
                means = [r["signed_effect"]["mean"] for r in selected]
                errors = np.array(
                    [
                        [r["signed_effect"]["mean"] - r["signed_effect"]["low"] for r in selected],
                        [r["signed_effect"]["high"] - r["signed_effect"]["mean"] for r in selected],
                    ]
                )
                row[2].barh(
                    np.arange(len(families)) + offset,
                    means,
                    height=0.3,
                    xerr=errors,
                    color=color,
                    label=f"code {code:+d}",
                )
            row[2].set_yticks(
                range(len(families)),
                [
                    "Fixed / shared",
                    "Balanced / shared",
                    "Fixed / linear",
                    "Balanced / linear",
                    "Matched donor",
                ],
            )
            row[2].axvline(0, color="gray", linewidth=0.5)
            row[2].set(
                title=f"Block {s['intervention_layer']}: semantic-oriented effect",
                xlabel="Mean Δ log odds (orientation corrected for mapping)",
                ylabel="Intervention method",
            )
            row[2].legend(fontsize=7)
    fig.suptitle(
        "Answer remapping study — task competence is a prerequisite for semantic causal claims",
        fontsize=14,
    )
    fig.savefig(root / "overview.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    summaries = []
    for run in sorted(args.run.glob("qwen-*")):
        if (run / "metrics.json").exists():
            s = analyze(run)
            summaries.append(s)
            print(
                json.dumps(
                    {
                        "model": s["model"]["name"],
                        "gate": s["capability_gate_passed"],
                        "audit": s["audit"],
                    }
                )
            )
    if summaries:
        plot(summaries, args.run)
