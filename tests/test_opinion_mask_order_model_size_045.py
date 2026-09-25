import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_opinion_mask_order_model_size_045 as analysis


def _synthetic_rows(invalid_count=0):
    parent, current = {}, {}
    invalid = 0
    for cluster in range(217):
        case_id = f"case-{cluster}"
        for lang in analysis.LANGS:
            for condition, prediction in (
                ("aspect_only", [3.0, 3.0]),
                ("opinion_masked", [5.0, 5.0]),
            ):
                parent[(case_id, lang, condition)] = {
                    "case_id": case_id,
                    "lang": lang,
                    "condition": condition,
                    "gold": [5.0, 5.0],
                    "prediction": prediction,
                }
            for condition, prediction in (
                ("opinion_masked", [5.0, 5.0]),
                ("opinion_shuffled", [4.0, 4.0]),
            ):
                output = None if invalid < invalid_count else prediction
                invalid += output is None
                current[(case_id, lang, condition)] = {
                    "case_id": case_id,
                    "lang": lang,
                    "condition": condition,
                    "prediction": output,
                }
    return parent, current


def test_invalid_output_gate_withholds_model_size_score_comparison():
    parent, current = _synthetic_rows(invalid_count=27)
    result = analysis.analyze(parent, current)
    assert result["status"] == "protocol_execution_failure"
    assert result["n_invalid_outputs"] == 27
    assert result["score_analysis_performed"] is False


def test_paired_analysis_uses_natural_minus_shuffled_rmse_contrast():
    parent, current = _synthetic_rows()
    result = analysis.analyze(parent, current)
    assert result["status"] == "scored"
    assert result["primary_exploratory"]["contrast"] == (
        "opinion_shuffled_rmse_minus_opinion_masked_rmse"
    )
    assert result["primary_exploratory"]["estimate"] == 1.0
    assert result["primary_exploratory"]["ci95"] == [1.0, 1.0]
    assert result["primary_exploratory"]["registered_order_gate_passed"] is True


def test_incomplete_pair_grid_is_rejected():
    parent, current = _synthetic_rows()
    current.pop(next(iter(current)))
    with pytest.raises(ValueError, match="frozen paired grid"):
        analysis.analyze(parent, current)
