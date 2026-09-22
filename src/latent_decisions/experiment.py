import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import FACTORS
from .metrics import cluster_mean_ci, fit_temperature, metrics, sigmoid
from .probes import (
    BilinearProbe,
    IndependentProbe,
    QueryOnlyProbe,
    Standardizer,
    expand,
    fit_probe,
    predict,
    seed_all,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def provenance():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = None, None
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(path.name.encode() + path.read_bytes())
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        git_commit=commit,
        dirty_worktree=dirty,
        source_sha256=source_hash.hexdigest(),
        packages={
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "numpy", "scikit-learn")
        },
    )


def evaluate(y, logits, temperature):
    p = sigmoid(logits / temperature)
    return dict(
        raw=metrics(y, sigmoid(logits)),
        calibrated=metrics(y, p),
        by_property={name: metrics(y[:, j], p[:, j]) for j, name in enumerate(FACTORS)},
    )


def train(run, config):
    start = time.perf_counter()
    torch.set_num_threads(4)
    rows = json.loads((run / "dataset.json").read_text())
    with np.load(run / "activations.npz", allow_pickle=False) as cache:
        h = torch.from_numpy(cache["h"].copy()).float()
        q = torch.from_numpy(cache["q"].copy()).float()
    labels = np.array([r["labels"] for r in rows], dtype=np.float32)
    masks = {
        s: np.array([r["split"] == s for r in rows])
        for s in ("train", "validation", "calibration", "test", "ood")
    }
    # RMS-normalize each query, preserving relative semantic geometry. No fit on test queries.
    q = q / q.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    records, predictions, candidates = [], [], []
    for seed in config["seeds"]:
        for li, layer in enumerate(config["layers"]):
            norm = Standardizer().fit(h[masks["train"], li])
            hn = norm(h[:, li])
            data = {s: expand(hn[m], q[0], labels[m]) for s, m in masks.items()}
            for name in ("bilinear", "independent_linear", "query_only", "shuffled_labels"):
                seed_all(seed)
                if name in ("bilinear", "shuffled_labels"):
                    model = BilinearProbe(h.shape[-1], q.shape[-1], config["rank"])
                elif name == "independent_linear":
                    model = IndependentProbe(h.shape[-1])
                else:
                    model = QueryOnlyProbe(q.shape[-1])
                train_data = data["train"]
                if name == "shuffled_labels":
                    rng = np.random.default_rng(seed)
                    shuffled = labels[masks["train"]].copy()
                    for j in range(3):
                        rng.shuffle(shuffled[:, j])
                    train_data = expand(hn[masks["train"]], q[0], shuffled)
                fit = fit_probe(
                    model,
                    train_data,
                    data["validation"],
                    steps=config["steps"],
                    lr=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                    brier_weight=config["brier_weight"],
                )
                temperature = fit_temperature(
                    predict(model, data["calibration"]), labels[masks["calibration"]]
                )
                record = dict(
                    method=name,
                    seed=seed,
                    layer=layer,
                    temperature=temperature,
                    **fit,
                    evaluations={},
                )
                for split in ("test", "ood"):
                    logits = predict(model, data[split])
                    record["evaluations"][split] = evaluate(
                        labels[masks[split]], logits, temperature
                    )
                    for row, probs in zip(
                        np.array(rows, dtype=object)[masks[split]], sigmoid(logits / temperature)
                    ):
                        predictions.append(
                            dict(
                                id=row["id"],
                                group=row["group"],
                                split=split,
                                method=name,
                                seed=seed,
                                layer=layer,
                                labels=row["labels"],
                                probabilities=probs.tolist(),
                            )
                        )
                if name == "bilinear":
                    mask = masks["test"]
                    paraphrase = expand(hn[mask], q[1], labels[mask])
                    rng = np.random.default_rng(seed)
                    permuted = expand(hn[mask][rng.permutation(mask.sum())], q[0], labels[mask])
                    swapped = expand(hn[mask], q[0].roll(1, 0), labels[mask])
                    zero = expand(torch.zeros_like(hn[mask]), q[0], labels[mask])
                    for control, dat in (
                        ("paraphrase", paraphrase),
                        ("shuffled_activations", permuted),
                        ("swapped_queries", swapped),
                        ("mean_activation", zero),
                    ):
                        record["evaluations"][control] = evaluate(
                            labels[mask], predict(model, dat), temperature
                        )
                    state = dict(
                        state_dict=model.state_dict(),
                        hidden_dim=h.shape[-1],
                        query_dim=q.shape[-1],
                        rank=config["rank"],
                        mean=norm.mean,
                        scale=norm.scale,
                        q=q[0],
                        temperature=temperature,
                        layer=layer,
                        seed=seed,
                        validation_nll=fit["validation_nll"],
                    )
                    candidates.append(state)
                records.append(record)
            print(f"Fitted seed {seed}, layer {layer}", flush=True)
    # Select intervention layer by mean validation NLL across seeds, never by test results.
    layer_scores = {
        layer: float(np.mean([c["validation_nll"] for c in candidates if c["layer"] == layer]))
        for layer in config["layers"]
    }
    selected_layer = min(layer_scores, key=layer_scores.get)
    selected = next(
        c for c in candidates if c["layer"] == selected_layer and c["seed"] == config["seeds"][0]
    )
    torch.save(selected, run / "probe.pt")
    baselines = text_baselines(rows, labels, masks)
    # Paired model comparison: average loss over seeds first, bootstrap complete test groups.
    chosen = [r for r in predictions if r["layer"] == selected_layer and r["split"] == "test"]
    ids = [r["id"] for r in rows if r["split"] == "test"]
    loss = {}
    for method in ("bilinear", "independent_linear"):
        loss[method] = np.array(
            [
                np.mean(
                    [
                        (np.array(p["probabilities"]) - p["labels"]) ** 2
                        for p in chosen
                        if p["method"] == method and p["id"] == id_
                    ]
                )
                for id_ in ids
            ]
        )
    groups = [r["group"] for r in rows if r["split"] == "test"]
    report = dict(
        schema_version=1,
        evidence="controlled real-model pilot",
        config=config,
        provenance=provenance(),
        selected_layer=selected_layer,
        selection_rule="minimum mean validation NLL across seeds; first listed seed for interventions",
        validation_layer_scores=layer_scores,
        records=records,
        text_baselines=baselines,
        paired_brier_difference=cluster_mean_ci(
            loss["bilinear"] - loss["independent_linear"], groups
        ),
        training_seconds=time.perf_counter() - start,
    )
    write_json(run / "metrics.json", report)
    write_json(run / "predictions.json", predictions)
    return report


def text_baselines(rows, labels, masks):
    results = {}
    for name in ("full_prompt_text", "user_only_text"):
        texts = [
            r["system"] + " " + r["user"] if name == "full_prompt_text" else r["user"] for r in rows
        ]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        xtrain = vectorizer.fit_transform(np.array(texts)[masks["train"]])
        models = [
            LogisticRegression(C=10, max_iter=1000).fit(xtrain, labels[masks["train"], j])
            for j in range(3)
        ]
        cal = vectorizer.transform(np.array(texts)[masks["calibration"]])
        cal_logits = np.stack([m.decision_function(cal) for m in models], axis=1)
        temperature = fit_temperature(cal_logits, labels[masks["calibration"]])
        results[name] = dict(temperature=temperature)
        for split in ("test", "ood"):
            x = vectorizer.transform(np.array(texts)[masks[split]])
            logits = np.stack([m.decision_function(x) for m in models], axis=1)
            results[name][split] = evaluate(labels[masks[split]], logits, temperature)
    return results
