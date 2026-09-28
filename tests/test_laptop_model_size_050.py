import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_laptop_model_size_050 as analyzer


def _rows_for_model(size):
    rows = {}
    for case_id, gold in (("a", [5.0, 5.0]), ("b", [6.0, 4.0])):
        for decoder in analyzer.DECODERS:
            for condition in analyzer.CONDITIONS:
                if size == "3b":
                    offset = (1.0 if decoder == "finite_grid" else 0.5) if condition == "aspect_only" else 0.0
                else:
                    offset = (0.5 if decoder == "finite_grid" else 0.25) if condition == "aspect_only" else 0.0
                rows[(case_id, condition, decoder)] = {
                    "case_id": case_id,
                    "condition": condition,
                    "decoder": decoder,
                    "gold": gold,
                    "prediction": [gold[0] + offset, gold[1] + offset],
                }
    return rows


def test_analyzer_bootstraps_primary_and_paired_size_contrast(monkeypatch):
    monkeypatch.setattr(analyzer, "EXPECTED_IDS", 2)
    monkeypatch.setattr(analyzer, "BOOTSTRAPS", 100)
    summary = analyzer.analyze(_rows_for_model("1.5b"), _rows_for_model("3b"))

    assert summary["status"] == "scored"
    assert summary["primary_3b_interaction"]["estimate"] == pytest.approx(0.5)
    assert summary["primary_3b_interaction"]["ci95"] == pytest.approx([0.5, 0.5])
    assert summary["paired_size_moderation_secondary"]["estimate"] == pytest.approx(0.25)
    assert summary["paired_size_moderation_secondary"]["confirmatory"] is False


def test_analyzer_withholds_3b_scores_if_free_invalid_gate_fails(monkeypatch):
    monkeypatch.setattr(analyzer, "EXPECTED_IDS", 2)
    model_3b = _rows_for_model("3b")
    model_3b[("a", "aspect_only", "free_greedy")]["prediction"] = None
    summary = analyzer.analyze(_rows_for_model("1.5b"), model_3b)

    assert summary["status"] == "protocol_execution_failure"
    assert summary["score_analysis_performed"] is False
    assert "primary_3b_interaction" not in summary


def test_analyzer_requires_identical_paired_gold_labels(monkeypatch):
    monkeypatch.setattr(analyzer, "EXPECTED_IDS", 2)
    parent = _rows_for_model("1.5b")
    model_3b = _rows_for_model("3b")
    model_3b[("a", "aspect_only", "finite_grid")]["gold"] = [4.0, 5.0]

    with pytest.raises(ValueError, match="gold VA"):
        analyzer.analyze(parent, model_3b)
