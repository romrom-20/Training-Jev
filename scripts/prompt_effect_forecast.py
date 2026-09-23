"""Measure and forecast prompt-level effects of aspect-specific activation steering."""

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from task_ladder import ASPECTS, MODEL_SPECS

from latent_decisions.experiment import provenance, write_json
from latent_decisions.target import block_tensor, forward_last, load_target, replace_block_tensor

LAYER = 24
DOSE_FRACTION = 0.05
SPLITS = ("train", "validation", "calibration", "final_test")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def paired_rows(rows):
    out = []
    for index, row in enumerate(rows):
        if row["task"] != "mixed_review" or row["format"] != 0 or row["split"] not in SPLITS:
            continue
        for source in range(3):
            out.append(
                {
                    "effect_id": f"{row['id']}|source={source}",
                    "base_id": row["id"],
                    "index": index,
                    "group": row["group"],
                    "group_no": row["group_no"],
                    "split": row["split"],
                    "source": source,
                    "target": row["active"],
                    "context": row["context"],
                    "label": row["label"],
                    "conditional_p_positive": row["conditional_p_positive"],
                }
            )
    return out


def source_directions(rows, h, layer_index, train_indices):
    directions = []
    norms = []
    for source in range(3):
        chosen = [
            i
            for i in train_indices
            if rows[i]["task"] == "mixed_review"
            and rows[i]["format"] == 0
            and rows[i]["active"] == source
        ]
        positive = h[[i for i in chosen if rows[i]["label"] == 1], layer_index]
        negative = h[[i for i in chosen if rows[i]["label"] == 0], layer_index]
        if not len(positive) or not len(negative):
            raise ValueError(f"No balanced training examples for source {ASPECTS[source]}")
        direction = torch.from_numpy(positive.mean(0) - negative.mean(0)).float()
        direction = direction / direction.norm().clamp_min(1e-12)
        directions.append(direction)
        norms.extend(np.linalg.norm(h[chosen, layer_index], axis=-1).tolist())
    dose = DOSE_FRACTION * float(np.median(norms))
    return torch.stack(directions), dose


def capture(model_name, root, source_root, offline):
    source = source_root / model_name
    rows = json.loads((source / "dataset.json").read_text())
    with np.load(source / "activations.npz", allow_pickle=False) as cache:
        h = cache["h"].copy()
    spec = dict(MODEL_SPECS[model_name], name=model_name)
    layer_index = spec["layers"].index(LAYER)
    train_indices = [i for i, r in enumerate(rows) if r["split"] == "train"]
    directions, dose = source_directions(rows, h, layer_index, train_indices)
    effects = paired_rows(rows)
    positions = {row["id"]: i for i, row in enumerate(rows)}
    partial = root / "effect-outcomes.partial.jsonl"
    completed = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                completed[item["effect_id"]] = item
    expected = {item["effect_id"] for item in effects}
    if not set(completed).issubset(expected):
        raise ValueError("Checkpoint contains effects outside the frozen dataset")
    pending = [item for item in effects if item["effect_id"] not in completed]
    model, tokenizer, device = load_target(spec, "auto", offline)
    positive_ids = [tokenizer.encode(w, add_special_tokens=False) for w in ("positive", "negative")]
    if any(len(ids) != 1 for ids in positive_ids):
        raise ValueError("Positive/negative must be single-token labels")
    positive_id, negative_id = (ids[0] for ids in positive_ids)
    block = model.model.layers[LAYER - 1]
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": rows[positions[item["base_id"]]]["user"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for item in pending
    ]
    start = time.perf_counter()
    with partial.open("a") as stream:
        for offset in range(0, len(pending), 8):
            batch = pending[offset : offset + 8]
            tokens = tokenizer(texts[offset : offset + 8], padding=True, return_tensors="pt").to(
                device
            )
            delta = torch.stack([directions[item["source"]] for item in batch]).to(device) * dose

            def patch(module, inputs, output):
                value = block_tensor(output).clone()
                value[:, -1, :] += delta
                return replace_block_tensor(output, value)

            hook = block.register_forward_hook(patch)
            try:
                with torch.inference_mode():
                    logits = forward_last(model, tokens)
            finally:
                hook.remove()
            log_odds = (logits[:, positive_id] - logits[:, negative_id]).float().cpu().tolist()
            for j, item in enumerate(batch):
                baseline_p = min(max(float(item["conditional_p_positive"]), 1e-8), 1 - 1e-8)
                baseline = math.log(baseline_p / (1 - baseline_p))
                record = {
                    **item,
                    "unmodified_log_odds": baseline,
                    "steered_log_odds": log_odds[j],
                    "observed_delta": log_odds[j] - baseline,
                }
                completed[item["effect_id"]] = record
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
            if (offset + len(batch)) % 256 == 0 or offset + len(batch) == len(pending):
                print(f"{model_name}: patched {len(completed)}/{len(effects)}", flush=True)
    if set(completed) != expected:
        raise ValueError("Effect capture is incomplete")
    outcomes = [completed[item["effect_id"]] for item in effects]
    write_json(root / "effects.json", outcomes)

    # First-order directional derivative on untouched final-test prompts.
    directions_device = directions.to(device)
    final_rows = [
        r
        for r in rows
        if r["split"] == "final_test" and r["task"] == "mixed_review" and r["format"] == 0
    ]
    gradient_partial = root / "gradient-final-test.partial.jsonl"
    gradient_records = {}
    if gradient_partial.exists():
        for line in gradient_partial.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                gradient_records[record["base_id"]] = record
    expected_gradient_ids = {row["id"] for row in final_rows}
    if not set(gradient_records).issubset(expected_gradient_ids):
        raise ValueError("Gradient checkpoint contains prompts outside final test")
    pending_final = [row for row in final_rows if row["id"] not in gradient_records]
    pending_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["user"]}], tokenize=False, add_generation_prompt=True
        )
        for row in pending_final
    ]
    with gradient_partial.open("a") as stream:
        for offset in range(0, len(pending_final), 4):
            batch = pending_final[offset : offset + 4]
            tokens = tokenizer(
                pending_texts[offset : offset + 4], padding=True, return_tensors="pt"
            ).to(device)
            captured = {}

            def tap(module, inputs, output):
                state = block_tensor(output).detach().requires_grad_(True)
                captured["state"] = state
                return replace_block_tensor(output, state)

            hook = block.register_forward_hook(tap)
            try:
                with torch.enable_grad():
                    logits = forward_last(model, tokens)
                    contrast = logits[:, positive_id] - logits[:, negative_id]
                    grad = torch.autograd.grad(contrast.sum(), captured["state"])[0][:, -1, :]
            finally:
                hook.remove()
            deriv = dose * (grad @ directions_device.T)
            for j, row in enumerate(batch):
                record = {
                    "base_id": row["id"],
                    "group": row["group"],
                    "target": row["active"],
                    "gradient_delta_by_source": [
                        float(x) for x in deriv[j].detach().cpu().tolist()
                    ],
                }
                gradient_records[row["id"]] = record
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
            if offset % 64 == 0 or offset + len(batch) == len(pending_final):
                print(
                    f"{model_name}: gradient reference {len(gradient_records)}/{len(final_rows)}",
                    flush=True,
                )
    gradients = [gradient_records[row["id"]] for row in final_rows]
    write_json(root / "gradient-final-test.json", gradients)
    manifest = {
        "model": spec,
        "device": str(device),
        "layer": LAYER,
        "dose_fraction": DOSE_FRACTION,
        "dose_l2": dose,
        "n_effects": len(outcomes),
        "n_final_prompts_gradient": len(gradients),
        "capture_seconds": time.perf_counter() - start,
        "source_dataset_sha256": sha(source / "dataset.json"),
        "source_activation_sha256": json.loads((source / "manifest.json").read_text())[
            "activation_sha256"
        ],
        "protocol_sha256": sha(
            Path(__file__).resolve().parents[1]
            / "docs/experiments/010-prompt-level-effect-forecast.md"
        ),
        "effects_sha256": sha(root / "effects.json"),
        "gradient_sha256": sha(root / "gradient-final-test.json"),
        "provenance": provenance(),
    }
    write_json(root / "manifest.json", manifest)
    del model
    gc.collect()
    if str(device).startswith("mps"):
        torch.mps.empty_cache()


class BilinearEffect(torch.nn.Module):
    def __init__(self, hidden_dim, query_dim, rank=4):
        super().__init__()
        self.h = torch.nn.Linear(hidden_dim, rank, bias=False)
        self.q = torch.nn.Linear(query_dim, rank, bias=False)
        self.bias = torch.nn.Linear(query_dim, 1)
        self.rank = rank

    def forward(self, h, q):
        return (self.h(h) * self.q(q)).sum(-1) / math.sqrt(self.rank) + self.bias(q).squeeze(-1)


def metric(y, pred):
    mse = float(np.mean((y - pred) ** 2))
    return {
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(y - pred))),
        "r2": float(1 - mse / np.var(y)) if np.var(y) else None,
        "sign_accuracy": float(np.mean((y > 0) == (pred > 0))),
        "mse": mse,
    }


def group_bootstrap_rmse_delta(y, pred, reference, groups, reps=5000, seed=20260924):
    unique = np.unique(groups)
    by_group = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(reps):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([by_group[g] for g in sampled])
        values.append(
            math.sqrt(np.mean((y[ix] - pred[ix]) ** 2))
            - math.sqrt(np.mean((y[ix] - reference[ix]) ** 2))
        )
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def analyze_model(root, source_root, name):
    source = source_root / name
    rows = json.loads((source / "dataset.json").read_text())
    with np.load(source / "activations.npz", allow_pickle=False) as cache:
        h_all = torch.from_numpy(cache["h"].copy()).float()[
            :, MODEL_SPECS[name]["layers"].index(LAYER)
        ]
        q_all = torch.from_numpy(cache["q"].copy()).float()
    qnorm = q_all / q_all.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    pair_queries = {}
    for source_ix in range(3):
        for target_ix in range(3):
            qs, qt = qnorm[source_ix], qnorm[target_ix]
            pair_queries[(source_ix, target_ix)] = torch.cat((qs, qt, qs * qt))
    qpair_dim = len(next(iter(pair_queries.values())))
    outcomes = json.loads((root / name / "effects.json").read_text())
    indices = {row["id"]: i for i, row in enumerate(rows)}
    train = [x for x in outcomes if x["split"] == "train"]
    val = [x for x in outcomes if x["split"] == "validation"]
    test = [x for x in outcomes if x["split"] == "final_test"]
    if (len(train), len(val), len(test)) != (1152, 432, 1728):
        raise ValueError(
            f"Unexpected split counts for {name}: {len(train)}, {len(val)}, {len(test)}"
        )

    def tensors(records):
        ix = torch.tensor([indices[x["base_id"]] for x in records])
        hidden = h_all[ix]
        queries = torch.stack([pair_queries[(x["source"], x["target"])] for x in records])
        targets = torch.tensor([x["observed_delta"] for x in records], dtype=torch.float32)
        return hidden, queries, targets

    xtr, qtr, ytr = tensors(train)
    xva, qva, yva = tensors(val)
    xte, qte, yte = tensors(test)
    hmean, hscale = xtr.mean(0), xtr.std(0).clamp_min(0.05)
    ymean, yscale = ytr.mean(), ytr.std().clamp_min(1e-4)
    xtr, xva, xte = tuple((x - hmean) / hscale for x in (xtr, xva, xte))
    ztr, zva = (ytr - ymean) / yscale, (yva - ymean) / yscale
    torch.set_num_threads(4)
    seed_predictions, records = [], []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        model = BilinearEffect(xtr.shape[-1], qpair_dim, rank=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
        best, state, best_step = float("inf"), None, 0
        for step in range(400):
            model.train()
            loss = torch.nn.functional.mse_loss(model(xtr, qtr), ztr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.eval()
                score = torch.nn.functional.mse_loss(model(xva, qva), zva).item()
            if score < best:
                best, best_step = score, step + 1
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state)
        with torch.no_grad():
            pred = model(xte, qte).numpy() * float(yscale) + float(ymean)
        seed_predictions.append(pred)
        records.append(
            {
                "seed": seed,
                "best_validation_mse_standardized": best,
                "selected_step": best_step,
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        )
    head_pred = np.mean(seed_predictions, axis=0)
    y = np.asarray([x["observed_delta"] for x in test])
    groups = np.asarray([x["group_no"] for x in test])
    pair_means = {}
    for x in train:
        key = (x["source"], x["target"])
        pair_means.setdefault(key, []).append(x["observed_delta"])
    mean_pred = np.asarray([np.mean(pair_means[(x["source"], x["target"])]) for x in test])
    texts = [
        f"source aspect {ASPECTS[x['source']]}; queried aspect {ASPECTS[x['target']]}; review: {x['context']}"
        for x in train + val + test
    ]
    vectorizer = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            (
                "char",
                TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, sublinear_tf=True),
            ),
        ]
    )
    text_train = vectorizer.fit_transform(texts[: len(train)])
    text_val = vectorizer.transform(texts[len(train) : len(train) + len(val)])
    text_test = vectorizer.transform(texts[len(train) + len(val) :])
    alpha_scores = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        ridge = Ridge(alpha=alpha).fit(text_train, np.asarray([x["observed_delta"] for x in train]))
        alpha_scores.append(
            (
                float(
                    np.mean(
                        (ridge.predict(text_val) - np.asarray([x["observed_delta"] for x in val]))
                        ** 2
                    )
                ),
                alpha,
            )
        )
    best_alpha = min(alpha_scores)[1]
    text_model = Ridge(alpha=best_alpha).fit(
        text_train, np.asarray([x["observed_delta"] for x in train])
    )
    text_pred = text_model.predict(text_test)

    gradients = json.loads((root / name / "gradient-final-test.json").read_text())
    gradient_map = {
        (x["base_id"], source): x["gradient_delta_by_source"][source]
        for x in gradients
        for source in range(3)
    }
    gradient_pred = np.asarray([gradient_map[(x["base_id"], x["source"])] for x in test])
    ci = group_bootstrap_rmse_delta(y, head_pred, mean_pred, groups)
    pair_metrics = []
    for s in range(3):
        for t in range(3):
            mask = np.asarray([x["source"] == s and x["target"] == t for x in test])
            pair_metrics.append(
                {
                    "source": ASPECTS[s],
                    "target": ASPECTS[t],
                    "n": int(mask.sum()),
                    "mean_effect": float(y[mask].mean()),
                    "within_pair_sd": float(y[mask].std()),
                    "head": metric(y[mask], head_pred[mask]),
                    "pair_mean": metric(y[mask], mean_pred[mask]),
                }
            )
    observed_total_var = float(np.var(y))
    effect_pair_means = np.asarray([np.mean(pair_means[(x["source"], x["target"])]) for x in test])
    within_var = float(np.mean((y - effect_pair_means) ** 2))
    return {
        "model": name,
        "n_train": len(train),
        "n_validation": len(val),
        "n_test": len(test),
        "n_test_groups": len(np.unique(groups)),
        "training_records": records,
        "selected_text_ridge_alpha": best_alpha,
        "within_pair_effect_variance_fraction": within_var / observed_total_var
        if observed_total_var
        else None,
        "pooled": {
            "shared_bilinear_head": metric(y, head_pred),
            "pair_mean": metric(y, mean_pred),
            "text_ridge": metric(y, text_pred),
            "first_order_gradient_reference": metric(y, gradient_pred),
            "head_minus_pair_mean_rmse_group_bootstrap_95_ci": ci,
            "relative_rmse_reduction_vs_pair_mean": float(
                1 - metric(y, head_pred)["rmse"] / metric(y, mean_pred)["rmse"]
            ),
        },
        "by_source_target": pair_metrics,
        "test_predictions": [
            {
                "base_id": x["base_id"],
                "group": x["group"],
                "source": ASPECTS[x["source"]],
                "target": ASPECTS[x["target"]],
                "observed_delta": float(yi),
                "head": float(pi),
                "pair_mean": float(mi),
                "text_ridge": float(ti),
                "gradient": float(gi),
            }
            for x, yi, pi, mi, ti, gi in zip(
                test, y, head_pred, mean_pred, text_pred, gradient_pred
            )
        ],
    }


def analyze(root, source_root):
    results = {}
    for name in ("qwen-1.5b", "smollm2-1.7b"):
        if not (root / name / "manifest.json").exists():
            raise ValueError(f"Missing completed capture for {name}")
        results[name] = analyze_model(root, source_root, name)
    write_json(root / "analysis.json", results)
    print(
        json.dumps({name: value["pooled"] for name, value in results.items()}, indent=2), flush=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("collect", "analyze", "all"))
    parser.add_argument("--root", type=Path, default=Path("runs/prompt-effect-forecast-v1"))
    parser.add_argument("--source-root", type=Path, default=Path("runs/task-ladder-v1"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("qwen-1.5b", "smollm2-1.7b"),
        default=("qwen-1.5b", "smollm2-1.7b"),
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage in ("collect", "all"):
        if args.root.exists() and not args.resume:
            raise FileExistsError(f"Refusing to overwrite {args.root}; pass --resume to continue")
        args.root.mkdir(parents=True, exist_ok=True)
        for name in args.models:
            out = args.root / name
            out.mkdir(exist_ok=args.resume)
            if (out / "manifest.json").exists():
                continue
            capture(name, out, args.source_root, args.offline)
    if args.stage in ("analyze", "all"):
        analyze(args.root, args.source_root)


if __name__ == "__main__":
    main()
