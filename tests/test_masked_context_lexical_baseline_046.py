import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_masked_context_lexical_baseline_046 as analysis
from run_masked_context_lexical_baseline_046 import (
    ALPHAS,
    group_id,
    mask_annotated_opinions,
)


def test_masking_replaces_all_occurrences_and_merges_overlapping_spans():
    masked = mask_annotated_opinions(
        "good soup, not good service",
        [{"Opinion": "good"}, {"Opinion": "not good"}],
    )
    assert "good" not in masked
    assert masked.count("[MASKED]") == 2


def test_review_group_uses_prefix_before_sentence_separator():
    assert group_id("1052:7_25") == "1052"


def test_analysis_reports_paired_lexical_context_gain():
    rows = {}
    for cluster in range(217):
        case_id = f"case-{cluster}"
        for lang in analysis.LANGS:
            for condition, prediction in (
                ("aspect_only", [3.0, 3.0]),
                ("masked_context", [5.0, 5.0]),
            ):
                rows[(case_id, lang, condition)] = {
                    "case_id": case_id,
                    "lang": lang,
                    "condition": condition,
                    "gold": [5.0, 5.0],
                    "prediction": prediction,
                }
    result = analysis.analyze(rows)
    assert result["primary_diagnostic"]["estimate"] == 2.0
    assert result["primary_diagnostic"]["ci95"] == [2.0, 2.0]
    assert result["primary_diagnostic"]["registered_lexical_gain_gate_passed"] is True


def test_analysis_rejects_incomplete_or_nonfinite_paired_predictions():
    rows = {}
    for cluster in range(217):
        for lang in analysis.LANGS:
            for condition in analysis.CONDITIONS:
                rows[(f"case-{cluster}", lang, condition)] = {
                    "case_id": f"case-{cluster}",
                    "lang": lang,
                    "condition": condition,
                    "gold": [5.0, 5.0],
                    "prediction": [5.0, 5.0],
                }
    rows.pop(next(iter(rows)))
    with pytest.raises(ValueError, match="complete frozen paired grid"):
        analysis.analyze(rows)
    rows[("case-0", "rus", "aspect_only")] = {
        "case_id": "case-0",
        "lang": "rus",
        "condition": "aspect_only",
        "gold": [5.0, 5.0],
        "prediction": [float("nan"), 5.0],
    }
    with pytest.raises(ValueError, match="Invalid VA"):
        analysis.analyze(rows)


def test_registered_alpha_grid_is_frozen():
    assert ALPHAS == (0.01, 0.1, 1.0, 10.0, 100.0)
