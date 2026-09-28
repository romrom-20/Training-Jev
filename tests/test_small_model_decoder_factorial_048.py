import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_small_model_decoder_factorial_048 as analysis
from run_small_model_decoder_factorial_048 import parse_free_va


def _synthetic_rows(invalid_free=0):
    rows = {}
    invalid = 0
    for cluster in range(217):
        case_id = f"case-{cluster}"
        for lang in analysis.LANGS:
            for decoder, aspect_prediction, context_prediction in (
                ("finite_grid", [3.0, 3.0], [5.0, 5.0]),
                ("free_greedy", [4.0, 4.0], [5.0, 5.0]),
            ):
                for condition, prediction in (
                    ("aspect_only", aspect_prediction),
                    ("opinion_masked", context_prediction),
                ):
                    value = None if decoder == "free_greedy" and invalid < invalid_free else prediction
                    invalid += value is None
                    rows[(case_id, lang, condition, decoder)] = {
                        "case_id": case_id,
                        "lang": lang,
                        "condition": condition,
                        "decoder": decoder,
                        "gold": [5.0, 5.0],
                        "prediction": value,
                    }
    return rows


def test_free_output_parser_accepts_finite_numeric_va_and_rejects_invalid_json():
    assert parse_free_va('{"valence":4.25,"arousal":8}') == [4.25, 8.0]
    assert parse_free_va('```json\n{"arousal":8,"valence":4.25}\n```') == [4.25, 8.0]
    for raw in (
        "answer: {\"valence\":4,\"arousal\":5}",
        '{"valence":true,"arousal":5}',
        '{"valence":NaN,"arousal":5}',
        '{"valence":4,"arousal":5,"confidence":0.9}',
        '{"valence":10,"arousal":5}',
    ):
        assert parse_free_va(raw) is None


def test_matched_decoder_analysis_computes_context_gains_and_interaction():
    result = analysis.analyze(_synthetic_rows())
    assert result["status"] == "scored"
    assert result["context_gain_by_decoder"]["finite_grid"]["estimate"] == 2.0
    assert result["context_gain_by_decoder"]["free_greedy"]["estimate"] == 1.0
    primary = result["primary_decoder_interaction"]
    assert primary["estimate"] == 1.0
    assert primary["ci95"] == [1.0, 1.0]
    assert primary["registered_interaction_gate_passed"] is True


def test_free_invalid_rate_over_two_percent_withholds_all_scores():
    result = analysis.analyze(_synthetic_rows(invalid_free=27))
    assert result["status"] == "protocol_execution_failure"
    assert result["invalid_free_outputs"] == 27
    assert result["score_analysis_performed"] is False
    assert "primary_decoder_interaction" not in result


def test_free_invalid_rate_at_two_percent_keeps_complete_clusters_only():
    result = analysis.analyze(_synthetic_rows(invalid_free=26))
    assert result["status"] == "scored"
    assert result["n_complete_clusters"] == 212
    assert result["primary_decoder_interaction"]["estimate"] == 1.0


def test_incomplete_factorial_is_rejected():
    rows = _synthetic_rows()
    rows.pop(next(iter(rows)))
    with pytest.raises(ValueError, match="complete paired grid"):
        analysis.analyze(rows)
