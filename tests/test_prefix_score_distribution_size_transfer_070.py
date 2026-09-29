from __future__ import annotations

import numpy as np
from analyze_prefix_score_distribution_size_transfer_070 import paired_order_metrics


def make_row(index: int, greedy: float) -> dict:
    logprobs = np.full(81, -100.0)
    logprobs[index] = 0.0
    probs = np.zeros(81)
    probs[index] = 1.0
    return {
        "score_logprobs": logprobs.tolist(),
        "restricted_probabilities": probs.tolist(),
        "greedy_second_score": greedy,
    }


def test_paired_metrics_measure_score_shift_and_distribution_movement() -> None:
    values = np.arange(10, 91, dtype=np.float64) / 10.0
    result = paired_order_metrics(make_row(10, 2.0), make_row(20, 3.0), values)
    assert result["low_expectation"] == 2.0
    assert result["high_expectation"] == 3.0
    assert result["greedy_high"] - result["greedy_low"] == 1.0
    assert result["total_variation"] == 1.0
    assert np.isclose(result["wasserstein1"], 1.0)
    assert np.isclose(result["low_grid_mass"], 1.0)
    assert np.isclose(result["high_grid_mass"], 1.0)
