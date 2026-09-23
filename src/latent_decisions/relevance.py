"""Prospective answer-remapping study. Separate from the original pilot protocol."""

import argparse
import gc
import hashlib
import itertools
import json
import random
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .data import FACTORS, QUESTIONS, VALUES, validate_splits
from .experiment import provenance, write_json
from .metrics import fit_temperature, metrics, sigmoid
from .probes import BilinearProbe, IndependentProbe, Standardizer, expand, fit_probe, seed_all
from .target import block_tensor, encode_rows, forward_last, load_target, replace_block_tensor

TEMPLATES = (
    "Record {record}. Settings: {settings}. Read only the {field} setting. "
    "If {field} is {positive}, answer {yes}; otherwise answer {no}. "
    "Output exactly one uppercase letter, A or B, and nothing else.",
    "Configuration {record}: {settings}. Your task concerns {field}. "
    "Return {yes} when {field} equals {positive}; return {no} otherwise. "
    "Give just the single letter A or B.",
    "Registry entry [{record}]: {settings}. Inspect the value of {field}. "
    "The answer code for {positive} is {yes}; the code for the other value is {no}. "
    "Respond using only the appropriate uppercase letter.",
)


def dataset(config):
    rng = random.Random(config["data_seed"])
    rows = []
    for split in ("train", "validation", "calibration", "test", "ood"):
        for group in range(config[f"{split}_groups"]):
            record = hashlib.sha256(f"{config['data_seed']}/{split}/{group}".encode()).hexdigest()[
                :10
            ]
            order = list(range(3))
            rng.shuffle(order)
            template = 2 if split == "ood" else group % 2
            for bits, active, code in itertools.product(
                itertools.product((0, 1), repeat=3), range(3), (1, -1)
            ):
                settings = "; ".join(f"{FACTORS[j]}={VALUES[j][bits[j]]}" for j in order)
                prompt = TEMPLATES[template].format(
                    record=record,
                    settings=settings,
                    field=FACTORS[active],
                    positive=VALUES[active][1],
                    yes="A" if code == 1 else "B",
                    no="B" if code == 1 else "A",
                )
                rows.append(
                    dict(
                        id=f"{split}-{group}-" + "".join(map(str, bits)) + f"-{active}-{code}",
                        group=f"{split}-{group}",
                        group_number=group,
                        split=split,
                        template=template,
                        system=prompt,
                        user="Apply the rule to this record.",
                        labels=list(bits),
                        active=active,
                        code=code,
                        answer_a=int((2 * bits[active] - 1) * code > 0),
                    )
                )
    validate_splits(rows)
    return rows


def regime_mask(rows, config, split, regime):
    return np.array(
        [
            r["split"] == split
            and (
                r["code"] == 1
                if regime == "fixed_code"
                else r["group_number"] < config[f"{split}_groups"] // 2
            )
            for r in rows
        ]
    )


def paired_donors(rows):
    lookup = {
        (r["group"], r["active"], r["code"], tuple(r["labels"])): i for i, r in enumerate(rows)
    }
    donors = []
    for r in rows:
        indices = []
        for j in range(3):
            flipped = r["labels"].copy()
            flipped[j] = 1 - flipped[j]
            indices.append(lookup[(r["group"], r["active"], r["code"], tuple(flipped))])
        donors.append(indices)
    return np.asarray(donors)


def capture(run, config, model_config, offline):
    start = time.perf_counter()
    rows = dataset(config)
    write_json(run / "dataset.json", rows)
    model, tokenizer, device = load_target(model_config, "auto", offline)
    option_ids = [tokenizer.encode(x, add_special_tokens=False) for x in ("B", "A")]
    if any(len(x) != 1 for x in option_ids):
        raise ValueError("A/B must be single tokens")
    option_ids = [x[0] for x in option_ids]
    captures, handles = {}, []
    for layer in model_config["layers"]:

        def hook(module, inputs, output, layer=layer):
            captures[layer] = block_tensor(output)[:, -1].detach().float().cpu().numpy()

        handles.append(model.model.layers[layer - 1].register_forward_hook(hook))
    hs, lodds, masses, tops = [], [], [], []
    max_driver = 0
    try:
        with torch.inference_mode():
            for start_i in range(0, len(rows), config["batch_size"]):
                tokens = encode_rows(
                    tokenizer,
                    rows[start_i : start_i + config["batch_size"]],
                    device,
                    config["max_length"],
                )
                logits = forward_last(model, tokens)
                probs = logits.softmax(-1)
                hs.append(np.stack([captures[layer] for layer in model_config["layers"]], axis=1))
                lodds.extend((logits[:, option_ids[1]] - logits[:, option_ids[0]]).cpu().tolist())
                masses.extend(probs[:, option_ids].sum(-1).cpu().tolist())
                tops.extend(tokenizer.batch_decode(logits.argmax(-1)[:, None]))
                if device == "mps":
                    max_driver = max(max_driver, torch.mps.driver_allocated_memory())
                if start_i % 384 == 0:
                    print(f"{model_config['name']} capture {start_i}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    with torch.inference_mode():
        tokens = tokenizer(list(QUESTIONS[0]), padding=True, return_tensors="pt").to(device)
        hidden = model.model(**tokens, use_cache=False).last_hidden_state
        mask = tokens.attention_mask[..., None]
        q = ((hidden * mask).sum(1) / mask.sum(1)).float().cpu().numpy()
    np.savez_compressed(
        run / "activations.npz",
        h=np.concatenate(hs),
        q=q,
        log_odds=np.asarray(lodds),
        mass=np.asarray(masses),
        donors=paired_donors(rows),
    )
    capability = []
    for split, active, code in itertools.product(("test", "ood"), range(3), (1, -1)):
        ix = [
            i
            for i, r in enumerate(rows)
            if r["split"] == split and r["active"] == active and r["code"] == code
        ]
        capability.append(
            dict(
                split=split,
                active=FACTORS[active],
                code=code,
                n=len(ix),
                greedy_accuracy=float(
                    np.mean([tops[i] == ("A" if rows[i]["answer_a"] else "B") for i in ix])
                ),
                choice_accuracy=float(
                    np.mean([(lodds[i] >= 0) == rows[i]["answer_a"] for i in ix])
                ),
                mean_option_mass=float(np.mean([masses[i] for i in ix])),
            )
        )
    gate = all(
        x["greedy_accuracy"] >= config["capability_gate"]
        for x in capability
        if x["split"] == "test"
    )
    write_json(
        run / "manifest.json",
        dict(
            config=config,
            model=model_config,
            provenance=provenance(),
            created_at=datetime.now(timezone.utc).isoformat(),
            capture_seconds=time.perf_counter() - start,
            capability=capability,
            capability_gate_passed=gate,
            device=device,
            peak_observed_mps_driver_gib=max_driver / 1024**3,
            memory_note="Sampled Metal driver allocation during capture; not total system memory",
            protocol_sha256=hashlib.sha256(
                Path("docs/experiments/002-answer-remapping.md").read_bytes()
            ).hexdigest(),
            dataset_sha256=hashlib.sha256((run / "dataset.json").read_bytes()).hexdigest(),
            activations_sha256=hashlib.sha256((run / "activations.npz").read_bytes()).hexdigest(),
        ),
    )
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def active_examples(h, q, rows, indices):
    task = torch.tensor([rows[i]["active"] for i in indices])
    y = torch.tensor([rows[i]["labels"][rows[i]["active"]] for i in indices], dtype=torch.float32)
    return h[indices], q[task], task, y


def fit(run, config, model_config):
    start = time.perf_counter()
    torch.set_num_threads(4)
    rows = json.loads((run / "dataset.json").read_text())
    with np.load(run / "activations.npz") as cache:
        h = torch.from_numpy(cache["h"].copy())
        q = torch.from_numpy(cache["q"].copy())
    q = q / q.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    test_indices = np.array([i for i, r in enumerate(rows) if r["split"] in ("test", "ood")])
    eval_rows = [rows[i] for i in test_indices]
    eval_y = np.asarray([r["labels"] for r in eval_rows])
    summaries, saved_logits, keys, states = [], [], [], []
    for li, layer in enumerate(model_config["layers"]):
        for regime in ("fixed_code", "balanced_code"):
            indices = {
                s: np.where(regime_mask(rows, config, s, regime))[0]
                for s in ("train", "validation", "calibration")
            }
            norm = Standardizer().fit(h[indices["train"], li])
            hn = norm(h[:, li])
            data = {s: active_examples(hn, q, rows, ix) for s, ix in indices.items()}
            eval_data = expand(hn[test_indices], q, eval_y)
            for seed, method in itertools.product(config["seeds"], ("bilinear", "independent")):
                seed_all(seed)
                probe = (
                    BilinearProbe(h.shape[-1], q.shape[-1], config["rank"])
                    if method == "bilinear"
                    else IndependentProbe(h.shape[-1])
                )
                fitted = fit_probe(
                    probe,
                    data["train"],
                    data["validation"],
                    steps=config["steps"],
                    lr=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                    brier_weight=config["brier_weight"],
                )
                with torch.no_grad():
                    cal_logits = probe(*data["calibration"][:3]).numpy()
                    logits = probe(*eval_data[:3]).numpy().reshape(-1, 3)
                temperature = fit_temperature(cal_logits, data["calibration"][3].numpy())
                key = dict(
                    layer=layer,
                    regime=regime,
                    seed=seed,
                    method=method,
                    temperature=temperature,
                    train_examples=len(indices["train"]),
                    **fitted,
                )
                keys.append(key)
                saved_logits.append(logits)
                evaluations = {}
                for split, code, relevance in itertools.product(
                    ("test", "ood"), (1, -1), ("relevant", "irrelevant")
                ):
                    mask = np.array(
                        [
                            [
                                r["split"] == split
                                and r["code"] == code
                                and ((j == r["active"]) == (relevance == "relevant"))
                                for j in range(3)
                            ]
                            for r in eval_rows
                        ]
                    )
                    name = f"{split}/code{code}/{relevance}"
                    evaluations[name] = dict(
                        raw=metrics(eval_y[mask], sigmoid(logits[mask])),
                        calibrated=metrics(eval_y[mask], sigmoid(logits[mask] / temperature)),
                    )
                summaries.append(dict(**key, evaluations=evaluations))
                if (
                    seed == config["intervention_seed"]
                    and layer == model_config["intervention_layer"]
                ):
                    with torch.no_grad():
                        directions = (
                            probe.raw_direction(q, norm.scale)
                            if method == "bilinear"
                            else probe.linear.weight / norm.scale
                        )
                    states.append(
                        dict(
                            **key,
                            directions=directions.detach(),
                            state_dict=probe.state_dict(),
                            mean=norm.mean,
                            scale=norm.scale,
                            q=q,
                        )
                    )
            print(f"{model_config['name']} fit layer {layer}, {regime}", flush=True)
    # Intended-output control: predicts task-correct A, not the model's observed choice.
    # If target competence fails, this is not an answer-writing-direction control.
    li = model_config["layers"].index(model_config["intervention_layer"])
    train_ix = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    val_ix = np.array([i for i, r in enumerate(rows) if r["split"] == "validation"])
    norm = Standardizer().fit(h[train_ix, li])
    hn = norm(h[:, li])

    def output_data(ix):
        return (
            hn[ix],
            q[0].repeat(len(ix), 1),
            torch.zeros(len(ix), dtype=torch.long),
            torch.tensor([rows[i]["answer_a"] for i in ix], dtype=torch.float32),
        )

    seed_all(0)
    output_probe = IndependentProbe(h.shape[-1], tasks=1)
    fit_probe(output_probe, output_data(train_ix), output_data(val_ix), steps=config["steps"])
    torch.save(
        dict(banks=states, output_direction=(output_probe.linear.weight[0] / norm.scale).detach()),
        run / "directions.pt",
    )
    np.savez_compressed(
        run / "predictions.npz", logits=np.stack(saved_logits), indices=test_indices
    )
    write_json(
        run / "metrics.json",
        dict(
            config=config,
            model=model_config,
            records=summaries,
            prediction_keys=keys,
            seconds=time.perf_counter() - start,
            provenance=provenance(),
        ),
    )


def read_completed_interventions(path, selected_ids, allowed_names):
    """Resume only intact, complete conditions; never silently discard partial measurements."""
    if not path.exists():
        return [], set()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    completed = {}
    for row in records:
        if row["name"] not in allowed_names:
            raise ValueError("Unrecognized saved intervention")
        ids = completed.setdefault(row["name"], set())
        if row["id"] in ids:
            raise ValueError("Duplicate saved intervention row")
        ids.add(row["id"])
    if any(ids != set(selected_ids) for ids in completed.values()):
        raise ValueError("Partial condition found; preserve and inspect it before resuming")
    return records, set(completed)


def interventions(run, config, model_config, offline, resume=False):
    start = time.perf_counter()
    rows = json.loads((run / "dataset.json").read_text())
    with np.load(run / "activations.npz") as cache:
        li = model_config["layers"].index(model_config["intervention_layer"])
        all_h = cache["h"][:, li].copy()
        baseline_logits = cache["log_odds"].copy()
        donors = cache["donors"].copy()
    groups = sorted(
        {r["group"] for r in rows if r["split"] == "test"}, key=lambda x: int(x.split("-")[1])
    )[: config["intervention_groups"]]
    selected = np.array([i for i, r in enumerate(rows) if r["group"] in groups])
    selected_rows = [rows[i] for i in selected]
    magnitude = config["dose"] * float(
        np.median(np.linalg.norm(all_h[[r["split"] == "train" for r in rows]], axis=-1))
    )
    states = torch.load(run / "directions.pt", map_location="cpu", weights_only=True)
    specs = [dict(name="zero", property=-1, kind="add", vector=torch.zeros(all_h.shape[-1]))]
    for bank in states["banks"]:
        for j, v in enumerate(bank["directions"]):
            specs.append(
                dict(
                    name=f"{bank['regime']}/{bank['method']}/{FACTORS[j]}",
                    property=j,
                    kind="add",
                    vector=v / v.norm(),
                )
            )
    generator = torch.Generator().manual_seed(20260922)
    for j in range(config["random_directions"]):
        v = torch.randn(all_h.shape[-1], generator=generator)
        specs.append(dict(name=f"random_{j}", property=-1, kind="add", vector=v / v.norm()))
    v = states["output_direction"]
    specs.append(dict(name="output_A", property=-1, kind="add", vector=v / v.norm()))
    for j in range(3):
        specs.append(dict(name=f"donor/{FACTORS[j]}", property=j, kind="patch", vector=None))
    model, tokenizer, device = load_target(model_config, "auto", offline)
    ids = [tokenizer.encode(t, add_special_tokens=False) for t in ("B", "A")]
    if any(len(x) != 1 for x in ids):
        raise ValueError("A/B must be single tokens")
    ids = [x[0] for x in ids]
    block = model.model.layers[model_config["intervention_layer"] - 1]
    destination = run / "intervention-rows.jsonl"
    if destination.exists() and not resume:
        raise ValueError(
            "Partial intervention file exists; inspect then use --resume-interventions"
        )
    records, completed = read_completed_interventions(
        destination, [r["id"] for r in selected_rows], {s["name"] for s in specs}
    )
    resumed_rows = len(records)
    with destination.open("a") as output_file:
        for spec in specs:
            if spec["name"] in completed:
                print(f"Retained completed condition {spec['name']}", flush=True)
                continue
            outcomes = {}
            for sign in (-1, 1) if spec["kind"] == "add" else (1,):
                lodds, masses, top = [], [], []
                for offset in range(0, len(selected), config["batch_size"]):
                    batch_ix = selected[offset : offset + config["batch_size"]]
                    tokens = encode_rows(
                        tokenizer, [rows[i] for i in batch_ix], device, config["max_length"]
                    )
                    if spec["kind"] == "patch":
                        replacement = torch.from_numpy(
                            all_h[donors[batch_ix, spec["property"]]]
                        ).to(device)
                    else:
                        delta = (sign * magnitude * spec["vector"]).to(device)

                    def hook(module, inputs, result):
                        value = block_tensor(result).clone()
                        if spec["kind"] == "patch":
                            value[:, -1] = replacement
                        else:
                            value[:, -1] += delta
                        return replace_block_tensor(result, value)

                    handle = block.register_forward_hook(hook)
                    try:
                        with torch.inference_mode():
                            logits = forward_last(model, tokens)
                    finally:
                        handle.remove()
                    lodds.extend((logits[:, ids[1]] - logits[:, ids[0]]).cpu().tolist())
                    masses.extend(logits.softmax(-1)[:, ids].sum(-1).cpu().tolist())
                    top.extend(tokenizer.batch_decode(logits.argmax(-1)[:, None]))
                outcomes[sign] = dict(log_odds=lodds, mass=masses, top=top)
            for local, (index, row) in enumerate(zip(selected, selected_rows)):
                if spec["kind"] == "add":
                    delta = outcomes[1]["log_odds"][local] - outcomes[-1]["log_odds"][local]
                    effect = row["code"] * delta
                else:
                    delta = outcomes[1]["log_odds"][local] - baseline_logits[index]
                    effect = row["code"] * (1 - 2 * row["labels"][spec["property"]]) * delta
                record = dict(
                    id=row["id"],
                    group=row["group"],
                    active=row["active"],
                    code=row["code"],
                    name=spec["name"],
                    property=spec["property"],
                    kind=spec["kind"],
                    baseline_log_odds=float(baseline_logits[index]),
                    delta_log_odds=float(delta),
                    semantic_effect=float(effect),
                    outcomes={
                        str(s): {k: v[local] for k, v in o.items()} for s, o in outcomes.items()
                    },
                )
                if spec["kind"] == "patch":
                    donor_index = donors[index, spec["property"]]
                    record["donor_id"] = rows[donor_index]["id"]
                    record["reference_effect"] = float(
                        row["code"]
                        * (1 - 2 * row["labels"][spec["property"]])
                        * (baseline_logits[donor_index] - baseline_logits[index])
                    )
                output_file.write(json.dumps(record) + "\n")
                records.append(record)
            output_file.flush()
            print(
                f"{model_config['name']} intervention {spec['name']} ({time.perf_counter() - start:.0f}s)",
                flush=True,
            )
    write_json(
        run / "intervention-manifest.json",
        dict(
            model=model_config,
            config=config,
            provenance=provenance(),
            seconds=time.perf_counter() - start,
            timing_scope="resumed portion only" if resumed_rows else "entire intervention stage",
            retained_rows=resumed_rows,
            prompts=len(selected),
            records=len(records),
            dose_l2=magnitude,
            layer=model_config["intervention_layer"],
            groups=groups,
            directions_sha256=hashlib.sha256((run / "directions.pt").read_bytes()).hexdigest(),
            records_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        ),
    )
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "train", "intervene", "all"))
    parser.add_argument("--config", type=Path, default=Path("configs/relevance.toml"))
    parser.add_argument("--run", type=Path, default=Path("runs/answer-remapping-v1"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume-interventions", action="store_true")
    parser.add_argument("--model", choices=("qwen-0.5b", "qwen-1.5b"))
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    for m in config["models"]:
        if args.model and args.model != m["name"]:
            continue
        run = args.run / m["name"]
        run.mkdir(parents=True, exist_ok=True)
        for stage in ("collect", "train", "intervene") if args.stage == "all" else (args.stage,):
            artifact = {
                "collect": "manifest.json",
                "train": "metrics.json",
                "intervene": "intervention-manifest.json",
            }[stage]
            if (run / artifact).exists():
                parser.error(f"{run / artifact} exists; stages are immutable, choose a new run")
            if stage != "collect":
                manifest = json.loads((run / "manifest.json").read_text())
                if manifest["config"] != config or manifest["model"] != m:
                    parser.error("Config mismatch with captured run")
                for name in ("dataset", "activations"):
                    path = run / (name + (".json" if name == "dataset" else ".npz"))
                    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[name + "_sha256"]:
                        parser.error("Capture hash mismatch")
            if stage == "collect":
                capture(run, config, m, args.offline)
            elif stage == "train":
                fit(run, config, m)
            else:
                interventions(run, config, m, args.offline, args.resume_interventions)
            print(f"Completed {m['name']} {stage}", flush=True)


if __name__ == "__main__":
    main()
