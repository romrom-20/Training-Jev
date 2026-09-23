"""Capture four fresh formats and test frozen score-transport corrections."""

import argparse
import gc
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from score_transport import compact_metrics, estimate_offsets
from sklearn.metrics import roc_auc_score

from latent_decisions.data import FACTORS, VALUES, validate_splits
from latent_decisions.experiment import provenance, write_json
from latent_decisions.metrics import cluster_mean_ci, sigmoid
from latent_decisions.probes import BilinearProbe, IndependentProbe
from latent_decisions.target import block_tensor, encode_rows, forward_last, load_target

FORMATS = ("labeled_prose", "json", "sentences", "options")


def fresh_rows():
    rows = []
    for style, group, bits, active, code in itertools.product(
        FORMATS, range(12), itertools.product((0, 1), repeat=3), range(3), (1, -1)
    ):
        record = hashlib.sha256(f"20260923/{style}/{group}".encode()).hexdigest()[:10]
        values = {FACTORS[j]: VALUES[j][bits[j]] for j in range(3)}
        field = FACTORS[active]
        positive = VALUES[active][1]
        yes, no = ("A", "B") if code == 1 else ("B", "A")
        rule = f"For the {field} field, use {yes} for {positive} and {no} for the other value. Return only A or B."
        if style == "labeled_prose":
            body = (
                f"Entry {record}. "
                + " | ".join(f"{k}: {v}" for k, v in values.items())
                + ". "
                + rule
            )
        elif style == "json":
            body = (
                f"Use this JSON record, ID {record}: "
                + json.dumps(values, sort_keys=True)
                + ". "
                + rule
            )
        elif style == "sentences":
            body = (
                f"Item {record} has a {values['shape']} shape. Its animal is {values['animal']}, and its color is {values['color']}. "
                + rule
            )
        else:
            a = VALUES[active][1 if code == 1 else 0]
            b = VALUES[active][0 if code == 1 else 1]
            body = (
                f"Record {record}: "
                + "; ".join(f"{k}={v}" for k, v in values.items())
                + f". What is the {field}? Options: A: {a}. B: {b}. Answer with only the correct letter."
            )
        rows.append(
            dict(
                id=f"{style}-{group}-" + "".join(map(str, bits)) + f"-{active}-{code}",
                group=f"{style}-{group}",
                group_number=group,
                style=style,
                split="adaptation" if group < 6 else "evaluation",
                active=active,
                code=code,
                labels=list(bits),
                answer_a=int((2 * bits[active] - 1) * code > 0),
                system=body,
                user="Apply the rule to this record.",
            )
        )
    validate_splits(rows)
    return rows


def capture(rows, source, dest):
    manifest = json.loads((source / "manifest.json").read_text())
    model_config = manifest["model"]
    config = manifest["config"]
    model, tokenizer, device = load_target(model_config, "auto", True)
    captured = []

    def hook(module, inputs, output):
        captured.append(block_tensor(output)[:, -1].detach().float().cpu().numpy())

    handle = model.model.layers[model_config["intervention_layer"] - 1].register_forward_hook(hook)
    ids = [tokenizer.encode(v, add_special_tokens=False) for v in ["B", "A"]]
    if any(len(x) != 1 for x in ids):
        raise ValueError("Expected single A/B tokens")
    ids = [x[0] for x in ids]
    tops = []
    mass = []
    log_odds = []
    try:
        with torch.inference_mode():
            for offset in range(0, len(rows), config["batch_size"]):
                tokens = encode_rows(
                    tokenizer,
                    rows[offset : offset + config["batch_size"]],
                    device,
                    config["max_length"],
                )
                logits = forward_last(model, tokens)
                tops.extend(tokenizer.batch_decode(logits.argmax(-1)[:, None]))
                mass.extend(logits.softmax(-1)[:, ids].sum(-1).cpu().tolist())
                log_odds.extend((logits[:, ids[1]] - logits[:, ids[0]]).cpu().tolist())
                if offset % 384 == 0:
                    print(f"{model_config['name']} fresh capture {offset}/{len(rows)}", flush=True)
    finally:
        handle.remove()
    np.savez_compressed(
        dest / "activations.npz", h=np.concatenate(captured), mass=mass, log_odds=log_odds
    )
    write_json(dest / "target-output.json", dict(top=tops, mass=mass, log_odds=log_odds))
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return np.concatenate(captured)


def analyze(source, dest):
    start = time.perf_counter()
    if (dest / "analysis.json").exists():
        raise ValueError("Completed replication already exists")
    dest.mkdir(parents=True, exist_ok=True)
    rows = fresh_rows()
    write_json(dest / "dataset.json", rows)
    h = capture(rows, source, dest)
    source_manifest = json.loads((source / "manifest.json").read_text())
    model_config = source_manifest["model"]
    source_rows = json.loads((source / "dataset.json").read_text())
    with np.load(source / "activations.npz") as cache:
        source_h = cache["h"][
            :, model_config["layers"].index(model_config["intervention_layer"])
        ].copy()
    source_ix = np.array(
        [i for i, r in enumerate(source_rows) if r["split"] == "test" and r["group_number"] < 6]
    )
    source_queries = np.array([source_rows[i]["active"] for i in source_ix])
    states = torch.load(source / "directions.pt", map_location="cpu", weights_only=True)["banks"]
    qid = np.array([r["active"] for r in rows])
    y = np.array([r["labels"][r["active"]] for r in rows])
    saved = []
    results = []
    checks = 0
    for state in states:
        dim = len(state["mean"])
        q = state["q"]
        probe = (
            BilinearProbe(dim, q.shape[-1], source_manifest["config"]["rank"])
            if state["method"] == "bilinear"
            else IndependentProbe(dim)
        )
        probe.load_state_dict(state["state_dict"])
        probe.eval()

        def logits(x, queries):
            with torch.no_grad():
                hn = (torch.from_numpy(x) - state["mean"]) / state["scale"]
                tasks = torch.tensor(queries)
                return probe(hn, q[tasks], tasks).numpy().astype(float)

        src = logits(source_h[source_ix], source_queries)
        target_scores = logits(h, qid)
        saved.append(target_scores)
        for style in FORMATS:
            anchors = np.array([r["style"] == style and r["split"] == "adaptation" for r in rows])
            evaluation = np.array(
                [r["style"] == style and r["split"] == "evaluation" for r in rows]
            )
            assert not (anchors & evaluation).any()
            for prevalence in (0.1, 0.5, 0.9):
                weights = np.where(y[anchors] == 1, 2 * prevalence, 2 * (1 - prevalence))
                offsets = estimate_offsets(
                    src, source_queries, target_scores[anchors], qid[anchors], weights
                )
                for method, offset in offsets.items():
                    adjusted = target_scores - offset[qid]
                    per_property = []
                    for prop, code in itertools.product(range(3), (1, -1)):
                        cell = (
                            evaluation & (qid == prop) & np.array([r["code"] == code for r in rows])
                        )
                        rank = roc_auc_score(y[cell], adjusted[cell])
                        assert abs(rank - roc_auc_score(y[cell], target_scores[cell])) < 1e-10
                        checks += 1
                        per_property.append(
                            dict(
                                property=prop,
                                code=code,
                                rank_auroc=rank,
                                **compact_metrics(y[cell], adjusted[cell], state["temperature"]),
                            )
                        )
                    p = sigmoid(adjusted[evaluation] / state["temperature"])
                    p0 = sigmoid(target_scores[evaluation] / state["temperature"])
                    loss_delta = (p - y[evaluation]) ** 2 - (p0 - y[evaluation]) ** 2
                    groups = [r["group"] for r, m in zip(rows, evaluation) if m]
                    results.append(
                        dict(
                            style=style,
                            regime=state["regime"],
                            probe=state["method"],
                            seed=state["seed"],
                            correction=method,
                            adaptation_prevalence=prevalence,
                            offset=offset.tolist(),
                            temperature=state["temperature"],
                            metrics=compact_metrics(
                                y[evaluation], adjusted[evaluation], state["temperature"]
                            ),
                            by_property=per_property,
                            delta_brier=cluster_mean_ci(loss_delta, groups),
                        )
                    )
    np.savez_compressed(dest / "probe-scores.npz", scores=np.stack(saved))
    primary = []
    for style in FORMATS:
        rs = [
            r
            for r in results
            if r["style"] == style
            and r["regime"] == "balanced_code"
            and r["probe"] == "bilinear"
            and r["adaptation_prevalence"] == 0.5
        ]
        baseline = next(r for r in rs if r["correction"] == "none")
        corrected = next(r for r in rs if r["correction"] == "query_offset")
        primary.append(
            dict(
                style=style,
                baseline=baseline["metrics"],
                corrected=corrected["metrics"],
                delta_brier=corrected["delta_brier"],
            )
        )
    passed = (
        sum(r["delta_brier"]["mean"] < 0 for r in primary) >= 3
        and max(r["delta_brier"]["mean"] for r in primary) <= 0.05
    )
    write_json(
        dest / "analysis.json",
        dict(
            model=model_config,
            primary=primary,
            continuation_gate_passed=passed,
            results=results,
            seconds=time.perf_counter() - start,
            provenance=provenance(),
            source_run=str(source),
            source_directions_sha256=hashlib.sha256(
                (source / "directions.pt").read_bytes()
            ).hexdigest(),
            dataset_sha256=hashlib.sha256((dest / "dataset.json").read_bytes()).hexdigest(),
            protocol_sha256=hashlib.sha256(
                Path("docs/experiments/004-fresh-template-replication.md").read_bytes()
            ).hexdigest(),
            audit=dict(
                status="passed", rank_invariance_checks=checks, disjoint_anchor_and_eval_groups=True
            ),
        ),
    )
    print(model_config["name"], "CONTINUATION GATE", passed, flush=True)
    for r in primary:
        print(
            r["style"],
            "accuracy",
            r["baseline"]["accuracy"],
            "->",
            r["corrected"]["accuracy"],
            "Brier delta",
            r["delta_brier"],
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("runs/answer-remapping-v1"))
    parser.add_argument("--run", type=Path, default=Path("runs/fresh-template-v1"))
    args = parser.parse_args()
    for source in sorted(args.source.glob("qwen-*")):
        analyze(source, args.run / source.name)
