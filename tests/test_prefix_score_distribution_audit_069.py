from __future__ import annotations

import numpy as np
import pytest
from analyze_prefix_score_distribution_audit_069 import bootstrap_mean, validate_distribution
from run_prefix_score_distribution_audit_069 import candidate_token_ids, normalize_score_logprobs


class TinyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]


class BrokenTokenizer(TinyTokenizer):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = super().encode(text, add_special_tokens)
        if text == "1.0":
            return tokens[:2]
        return tokens


def test_candidate_grid_is_exactly_ordered_and_complete() -> None:
    values, encodings = candidate_token_ids(TinyTokenizer())
    assert len(values) == len(encodings) == 81
    assert values[0] == 1.0 and values[-1] == 9.0
    assert values[10] == 2.0
    assert encodings[0] == [ord("1"), ord("."), ord("0")]
    assert encodings[-1] == [ord("9"), ord("."), ord("0")]


def test_candidate_grid_rejects_unregistered_tokenization() -> None:
    with pytest.raises(ValueError, match="three tokens"):
        candidate_token_ids(BrokenTokenizer())


def test_probability_normalization_requires_exact_registered_support() -> None:
    normalized = normalize_score_logprobs(np.zeros(81))
    assert normalized.shape == (81,)
    assert np.isclose(normalized.sum(), 1.0)
    with pytest.raises(ValueError, match="81 finite"):
        normalize_score_logprobs(np.zeros(90))


def test_distribution_validation_recomputes_normalized_scores() -> None:
    logprobs = np.linspace(-5.0, -1.0, 81)
    probs = np.exp(logprobs - logprobs.max())
    probs /= probs.sum()
    assert np.allclose(validate_distribution({
        "score_logprobs": logprobs.tolist(),
        "restricted_probabilities": probs.tolist(),
    }), probs)
    with pytest.raises(ValueError, match="do not match"):
        validate_distribution({
            "score_logprobs": logprobs.tolist(),
            "restricted_probabilities": np.full(81, 1 / 81).tolist(),
        })


def test_recipient_bootstrap_is_seeded_and_paired_vector_mean() -> None:
    values = np.array([-1.0, 0.0, 2.0, 3.0])
    first = bootstrap_mean(values, seed=17, replicates=500)
    second = bootstrap_mean(values, seed=17, replicates=500)
    assert first == second
    assert first["estimate"] == 1.0
    assert len(first["ci95"]) == 2
    assert first["ci95"][0] <= first["ci95"][1]
