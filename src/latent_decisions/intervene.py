"""Finite-dose interventions on a frozen target, plus matched random controls."""

import hashlib
import json
import time

import numpy as np
import torch

from .data import FACTORS
from .experiment import provenance, write_json
from .metrics import cluster_mean_ci
from .probes import BilinearProbe
from .target import block_tensor, encode_rows, forward_last, load_target, replace_block_tensor


def intervene(run, config, device="auto", offline=False, groups=4, dose=0.05):
    if groups < 2 or not 0 < dose <= 0.2:
        raise ValueError("Use at least two groups and a dose in (0, .2]")
    start = time.perf_counter()
    state = torch.load(run / "probe.pt", map_location="cpu", weights_only=True)
    probe = BilinearProbe(state["hidden_dim"], state["query_dim"], state["rank"])
    probe.load_state_dict(state["state_dict"])
    probe.eval()
    rows_all = json.loads((run / "dataset.json").read_text())
    group_ids = sorted({r["group"] for r in rows_all if r["split"] == "test"})[:groups]
    if len(group_ids) < groups:
        raise ValueError("Requested more intervention groups than exist in the test split")
    rows = [r for r in rows_all if r["group"] in group_ids]
    with np.load(run / "activations.npz") as cache:
        li = config["layers"].index(state["layer"])
        train_h = cache["h"][[r["split"] == "train" for r in rows_all], li]
        magnitude = dose * float(np.median(np.linalg.norm(train_h, axis=-1)))
    with torch.no_grad():
        directions = probe.raw_direction(state["q"], state["scale"])
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    generator = torch.Generator().manual_seed(2026)
    random = torch.randn((8, state["hidden_dim"]), generator=generator)
    random = random / random.norm(dim=-1, keepdim=True)
    named = [(name, directions[j]) for j, name in enumerate(FACTORS)] + [
        (f"random_{j}", v) for j, v in enumerate(random)
    ]
    model, tokenizer, device = load_target(config, device, offline)
    color_ids = [tokenizer.encode(w, add_special_tokens=False) for w in ("blue", "red")]
    if any(len(x) != 1 for x in color_ids):
        raise ValueError("Expected one-token color labels")
    color_ids = [x[0] for x in color_ids]
    block = model.model.layers[state["layer"] - 1]
    records = []
    baseline = {}
    # Fixed sample set and fixed intervention magnitude; no test-driven strength search.
    for name, direction in [("zero", torch.zeros(state["hidden_dim"]))] + named:
        outputs = {}
        for sign in (-1, 1):
            values, masses, kls, probe_probs = [], [], [], []
            for offset in range(0, len(rows), config["batch_size"]):
                batch = rows[offset : offset + config["batch_size"]]
                tokens = encode_rows(tokenizer, batch, device, config["max_length"])
                captured = []
                delta = (sign * magnitude * direction).to(device)

                def hook(module, inputs, output):
                    value = block_tensor(output).clone()
                    value[:, -1, :] += delta
                    captured.append(value[:, -1, :].detach().float().cpu())
                    return replace_block_tensor(output, value)

                handle = block.register_forward_hook(hook)
                try:
                    with torch.inference_mode():
                        logits = forward_last(model, tokens)
                finally:
                    handle.remove()
                logp = logits.log_softmax(-1).cpu()
                if name == "zero" and sign == -1:
                    baseline[offset] = logp
                p = logp.exp()
                values.extend((logits[:, color_ids[1]] - logits[:, color_ids[0]]).cpu().tolist())
                masses.extend(p[:, color_ids].sum(-1).tolist())
                kls.extend((baseline[offset].exp() * (baseline[offset] - logp)).sum(-1).tolist())
                with torch.no_grad():
                    h = (captured[0] - state["mean"]) / state["scale"]
                    count = len(batch)
                    q = state["q"][0].repeat(count, 1)
                    probe_probs.extend(
                        (probe(h, q, torch.zeros(count, dtype=torch.long)) / state["temperature"])
                        .sigmoid()
                        .tolist()
                    )
            outputs[sign] = dict(
                log_odds=values,
                color_mass=masses,
                kl_from_unmodified=kls,
                probe_red_probability=probe_probs,
            )
        for i, row in enumerate(rows):
            records.append(
                dict(
                    id=row["id"],
                    group=row["group"],
                    direction=name,
                    color_label=row["labels"][0],
                    minus={k: v[i] for k, v in outputs[-1].items()},
                    plus={k: v[i] for k, v in outputs[1].items()},
                    delta_log_odds=outputs[1]["log_odds"][i] - outputs[-1]["log_odds"][i],
                )
            )
        print(f"Intervened: {name}", flush=True)
    summary = {
        name: cluster_mean_ci(
            [r["delta_log_odds"] for r in records if r["direction"] == name],
            [r["group"] for r in records if r["direction"] == name],
        )
        for name in ["zero"] + [n for n, _ in named]
    }
    result = dict(
        provenance=provenance(),
        probe_sha256=hashlib.sha256((run / "probe.pt").read_bytes()).hexdigest(),
        evidence="exploratory steering diagnostic, not mechanism identification",
        layer=state["layer"],
        seed=state["seed"],
        dose_fraction=dose,
        delta_l2=magnitude,
        prompts=len(rows),
        random_directions=8,
        seconds=time.perf_counter() - start,
        estimand="mean log(P(red)/P(blue)) at +dose minus -dose over fixed held-out prompts",
        uncertainty="percentile bootstrap over scenario groups; conditional on trained probe and directions",
        summary=summary,
        records=records,
    )
    write_json(run / "interventions.json", result)
    return result
