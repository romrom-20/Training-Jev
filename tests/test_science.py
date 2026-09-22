import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from latent_decisions.data import make_dataset, validate_splits
from latent_decisions.metrics import cluster_mean_ci, fit_temperature, metrics, sigmoid
from latent_decisions.probes import BilinearProbe, Standardizer, expand, fit_probe, predict
from latent_decisions.target import block_tensor, forward_last, replace_block_tensor


def test_factorial_balance_and_split_isolation():
    rows = make_dataset({s: 2 for s in ("train", "validation", "calibration", "test", "ood")})
    validate_splits(rows)
    for group in {r["group"] for r in rows}:
        labels = np.array([r["labels"] for r in rows if r["group"] == group])
        assert len(np.unique(labels, axis=0)) == 8
        np.testing.assert_array_equal(labels.mean(0), [0.5, 0.5, 0.5])
    assert {r["template"] for r in rows if r["split"] == "ood"} == {2}
    assert 2 not in {r["template"] for r in rows if r["split"] != "ood"}


def test_split_guard_rejects_leakage():
    rows = make_dataset({"train": 1, "test": 1})
    rows[-1]["group"] = "train-0"
    with pytest.raises(ValueError, match="leaked"):
        validate_splits(rows)


def test_scoring_rules_and_ece_edges():
    perfect = metrics([0, 1], [0, 1])
    assert perfect["brier"] == 0
    assert perfect["ece"] == 0
    assert perfect["auroc"] == 1
    assert metrics([0, 1], [0.5, 0.5])["brier"] == 0.25
    assert metrics([1, 1], [0.8, 0.9])["auroc"] is None
    assert sum(r["n"] for r in perfect["reliability"]) == 2
    with pytest.raises(ValueError):
        metrics([0], [np.nan])


def test_temperature_corrects_overconfidence():
    logits = np.array([-8.0] * 10 + [8.0] * 10)
    y = np.array([0] * 8 + [1] * 2 + [1] * 8 + [0] * 2)
    temp = fit_temperature(logits, y)
    assert temp > 1
    assert metrics(y, sigmoid(logits / temp))["nll"] < metrics(y, sigmoid(logits))["nll"]


def test_cluster_interval_and_pairing():
    result = cluster_mean_ci([1, 1, 3, 3], ["a", "a", "b", "b"])
    assert result["mean"] == 2
    assert result["groups"] == 2
    assert result["low"] <= 2 <= result["high"]


def test_normalization_only_fits_training_data():
    train = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    norm = Standardizer().fit(train)
    mean = norm.mean.clone()
    norm(torch.tensor([[1000.0, 2000.0]]))
    assert torch.equal(norm.mean, mean)


def test_raw_direction_matches_autograd_through_scaling():
    torch.manual_seed(0)
    model = BilinearProbe(7, 5, 2)
    scale = torch.rand(7) + 0.1
    h = torch.randn(1, 7, requires_grad=True)
    q = torch.randn(1, 5)
    score = model((h - 2) / scale, q, torch.tensor([0]))
    (gradient,) = torch.autograd.grad(score.sum(), h)
    torch.testing.assert_close(model.raw_direction(q, scale), gradient)


def test_planted_query_specific_signal():
    torch.set_num_threads(2)
    torch.manual_seed(4)
    h = torch.randn(200, 6)
    q = torch.eye(3)
    y = (h[:, :3] > 0).float().numpy()
    model = BilinearProbe(6, 3, 3)
    train = expand(h[:120], q, y[:120])
    val = expand(h[120:160], q, y[120:160])
    fit_probe(model, train, val, steps=160, lr=0.03)
    logits = predict(model, expand(h[160:], q, y[160:]))
    assert metrics(y[160:], sigmoid(logits))["accuracy"] > 0.9
    shuffled = predict(model, expand(h[160:].flip(0), q, y[160:]))
    assert metrics(y[160:], sigmoid(shuffled))["accuracy"] < 0.7


def tiny_target():
    torch.manual_seed(1)
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=50,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            attn_implementation="eager",
        )
    ).eval()


def test_last_only_head_matches_full_forward():
    model = tiny_target()
    tokens = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    with torch.no_grad():
        expected = model(**tokens, use_cache=False).logits[:, -1]
        actual = forward_last(model, tokens)
    torch.testing.assert_close(actual, expected)


def test_left_padding_and_intervention_at_block_output():
    model = tiny_target()
    original = torch.tensor([[1, 2, 3]])
    padded = torch.tensor([[0, 0, 1, 2, 3]])
    mask = torch.tensor([[0, 0, 1, 1, 1]])
    with torch.no_grad():
        plain = forward_last(
            model, {"input_ids": original, "attention_mask": torch.ones_like(original)}
        )
        padded_out = forward_last(model, {"input_ids": padded, "attention_mask": mask})
    torch.testing.assert_close(plain, padded_out, atol=1e-5, rtol=1e-5)
    observed = []

    def hook(module, inputs, output):
        before = block_tensor(output)
        after = before.clone()
        after[:, -1, 0] += 1
        torch.testing.assert_close(before[:, :-1], after[:, :-1])
        observed.append(True)
        return replace_block_tensor(output, after)

    handle = model.model.layers[0].register_forward_hook(hook)
    try:
        with torch.no_grad():
            edited = forward_last(
                model, {"input_ids": original, "attention_mask": torch.ones_like(original)}
            )
    finally:
        handle.remove()
    assert observed
    assert not torch.allclose(edited, plain)
    assert len(model.model.layers[0]._forward_hooks) == 0
