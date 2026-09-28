import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_restaurant_domain_decoder_051 as analyzer
import run_restaurant_domain_decoder_051 as runner


def _case():
    row = {
        "ID": "rest-1",
        "Text": "The noodles are good; the wait was awful.",
        "Triplet": [
            {"Aspect": "noodles", "Opinion": "good", "VA": "7.0#6.0"},
            {"Aspect": "wait", "Opinion": "awful", "VA": "2.0#3.0"},
        ],
    }
    return {"case_id": "rest-1", "target_index": 0, "gold": [7.0, 6.0], "row": row}


def test_jobs_use_same_prompt_in_both_decoders():
    jobs = runner.build_jobs([_case()])
    assert len(jobs) == 4
    by_cell = {}
    for job in jobs:
        by_cell.setdefault(job["condition"], {})[job["decoder"]] = job
    assert all(arms["finite_grid"]["prompt"] == arms["free_greedy"]["prompt"] for arms in by_cell.values())
    assert "1.0 to 9.0 in increments of 0.1" in by_cell["aspect_only"]["finite_grid"]["prompt"]
    assert "[MASKED]" in by_cell["opinion_masked"]["finite_grid"]["prompt"]


def _rows(prefix, finite_offset, free_offset):
    rows = {}
    for case_id, gold in ((f"{prefix}-a", [5.0, 5.0]), (f"{prefix}-b", [6.0, 4.0])):
        for decoder, offset in (("finite_grid", finite_offset), ("free_greedy", free_offset)):
            for condition in analyzer.CONDITIONS:
                shift = offset if condition == "aspect_only" else 0.0
                rows[(case_id, condition, decoder)] = {
                    "case_id": case_id,
                    "condition": condition,
                    "decoder": decoder,
                    "gold": gold,
                    "prediction": [gold[0] + shift, gold[1] + shift],
                }
    return rows


def test_analyzer_reports_primary_and_independent_domain_difference(monkeypatch):
    monkeypatch.setattr(analyzer, "EXPECTED_RESTAURANT_IDS", 2)
    monkeypatch.setattr(analyzer, "EXPECTED_LAPTOP_IDS", 2)
    monkeypatch.setattr(analyzer, "BOOTSTRAPS", 100)
    restaurant = _rows("r", finite_offset=1.0, free_offset=0.5)
    laptop = _rows("l", finite_offset=0.75, free_offset=0.5)

    summary = analyzer.analyze(restaurant, laptop)
    assert summary["status"] == "scored"
    assert summary["primary_decoder_interaction"]["estimate"] == pytest.approx(0.5)
    assert summary["primary_decoder_interaction"]["ci95"] == pytest.approx([0.5, 0.5])
    assert summary["domain_moderation_secondary"]["estimate"] == pytest.approx(-0.25)
    assert summary["domain_moderation_secondary"]["ci95"] == pytest.approx([-0.25, -0.25])


def test_analyzer_withholds_all_contrasts_if_free_invalid_gate_fails(monkeypatch):
    monkeypatch.setattr(analyzer, "EXPECTED_RESTAURANT_IDS", 2)
    monkeypatch.setattr(analyzer, "EXPECTED_LAPTOP_IDS", 2)
    restaurant = _rows("r", finite_offset=1.0, free_offset=0.5)
    laptop = _rows("l", finite_offset=0.75, free_offset=0.5)
    restaurant[("r-a", "aspect_only", "free_greedy")]["prediction"] = None

    summary = analyzer.analyze(restaurant, laptop)
    assert summary["status"] == "protocol_execution_failure"
    assert summary["score_analysis_performed"] is False
    assert "primary_decoder_interaction" not in summary
