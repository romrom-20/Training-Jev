from __future__ import annotations

import numpy as np
from analyze_cross_family_prefix_coupling_071 import paired_effect


def row_at(value_index: int, greedy: float) -> dict:
    logprobs = np.full(81, -100.0)
    logprobs[value_index] = 0.0
    probabilities = np.zeros(81)
    probabilities[value_index] = 1.0
    return {
        "score_logprobs": logprobs.tolist(),
        "restricted_probabilities": probabilities.tolist(),
        "greedy_second_score": greedy,
    }


def test_paired_effect_uses_shared_recipients_and_score_support() -> None:
    values = np.arange(10, 91, dtype=np.float64) / 10
    result = paired_effect(row_at(10, 2.0), row_at(20, 3.0), values)
    assert result["expected_shift"] == 1.0
    assert result["greedy_shift"] == 1.0
    assert result["total_variation"] == 1.0
    assert np.isclose(result["wasserstein1"], 1.0)
    assert np.isclose(result["grid_mass_low"], 1.0)
    assert np.isclose(result["grid_mass_high"], 1.0)
