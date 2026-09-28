import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_free_decode_opinion_mask_sensitivity_047 as analysis
from run_free_decode_opinion_mask_sensitivity_047 import parse_va


def _synthetic_rows(invalid_count=0):
    rows = {}
    invalid = 0
    for cluster in range(217):
        case_id = f"case-{cluster}"
        for lang in analysis.LANGS:
            for condition, prediction in (
                ("aspect_only", [3.0, 3.0]),
                ("opinion_masked", [5.0, 5.0]),
            ):
                value = None if invalid < invalid_count else prediction
                invalid += value is None
                rows[(case_id, lang, condition)] = {
                    "case_id": case_id,
                    "lang": lang,
                    "condition": condition,
                    "gold": [5.0, 5.0],
                    "prediction": value,
                }
    return rows


def test_free_decoder_parser_accepts_finite_in_range_json_only():
    assert parse_va('{"valence":4.25,"arousal":8}') == [4.25, 8.0]
    assert parse_va('```json\n{"valence":4.25,"arousal":8}\n```') == [4.25, 8.0]
    for raw in (
        "score: {\"valence\":4,\"arousal\":5}",
        '{"valence":true,"arousal":5}',
        '{"valence":NaN,"arousal":5}',
        '{"valence":4,"arousal":5,"confidence":0.8}',
        '{"valence":0,"arousal":5}',
    ):
        assert parse_va(raw) is None


def test_invalid_rate_above_two_percent_withholds_all_score_analysis():
    result = analysis.analyze(_synthetic_rows(invalid_count=27))
    assert result["status"] == "protocol_execution_failure"
    assert result["score_analysis_performed"] is False
    assert "primary" not in result


def test_invalid_rate_at_two_percent_or_less_allows_complete_cluster_analysis():
    result = analysis.analyze(_synthetic_rows(invalid_count=26))
    assert result["status"] == "scored"
    assert result["n_complete_clusters"] == 212
    assert result["primary"]["estimate"] == 2.0
    assert result["primary"]["ci95"] == [2.0, 2.0]
    assert result["primary"]["registered_free_decode_gate_passed"] is True


def test_complete_paired_sample_analysis():
    result = analysis.analyze(_synthetic_rows())
    assert result["n_invalid_outputs"] == 0
    assert result["primary"]["estimate"] == 2.0
    assert result["primary"]["registered_free_decode_gate_passed"] is True


def test_incomplete_pair_grid_is_rejected():
    rows = _synthetic_rows()
    rows.pop(next(iter(rows)))
    with pytest.raises(ValueError, match="complete paired grid"):
        analysis.analyze(rows)
