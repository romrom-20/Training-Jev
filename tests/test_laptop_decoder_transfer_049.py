import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_laptop_decoder_transfer_049 as analyzer
import run_laptop_decoder_transfer_049 as runner


def make_row(case_id="row-1"):
    return {
        "ID": case_id,
        "Text": "Battery is great; screen is dim.",
        "Triplet": [
            {"Aspect": "Battery", "Opinion": "great", "VA": "7.0#6.0"},
            {"Aspect": "screen", "Opinion": "dim", "VA": "3.0#4.0"},
        ],
    }


def test_jobs_pair_identical_prompt_across_decoder_arms_and_keep_targets():
    row = make_row()
    case = {"case_id": row["ID"], "target_index": 0, "gold": [7.0, 6.0], "row": row}
    jobs = runner.build_jobs([case])

    assert len(jobs) == 4
    by_cell = {}
    for job in jobs:
        by_cell.setdefault(job["condition"], {})[job["decoder"]] = job
    for arms in by_cell.values():
        assert arms["finite_grid"]["prompt"] == arms["free_greedy"]["prompt"]
    assert '1.0 to 9.0 in increments of 0.1' in by_cell["aspect_only"]["finite_grid"]["prompt"]
    assert "[MASKED]" in by_cell["opinion_masked"]["finite_grid"]["prompt"]
    assert "Battery" in by_cell["opinion_masked"]["finite_grid"]["prompt"]


def test_protocol_grid_prompt_is_stable_and_declares_its_format():
    plain = runner.prompt_with_registered_grid(None, "screen")
    assert '"valence"' in plain and '"arousal"' in plain
    assert "exactly one decimal place" in plain
    assert runner.parse_free_va('{"valence":4.2,"arousal":6}') == [4.2, 6.0]
    assert runner.parse_free_va('{"valence":4.2,"arousal":9.1}') is None


def _synthetic_rows():
    rows = {}
    for case_id, gold in (("a", [5.0, 5.0]), ("b", [6.0, 4.0])):
        rows[case_id] = {}
        for decoder in analyzer.DECODERS:
            for condition in analyzer.CONDITIONS:
                if decoder == "finite_grid":
                    offset = 1 if condition == "aspect_only" else 0
                    pred = [gold[0] + offset, gold[1] + offset]
                else:
                    offset = 0.5 if condition == "aspect_only" else 0
                    pred = [gold[0] + offset, gold[1] + offset]
                rows[case_id][decoder, condition] = {
                    "case_id": case_id,
                    "decoder": decoder,
                    "condition": condition,
                    "gold": gold,
                    "prediction": pred,
                }
    return {
        (case_id, condition, decoder): values[decoder, condition]
        for case_id, values in rows.items()
        for decoder in analyzer.DECODERS
        for condition in analyzer.CONDITIONS
    }


def test_analyzer_computes_paired_decoder_interaction(monkeypatch):
    monkeypatch.setattr(analyzer, "EXPECTED_IDS", 2)
    monkeypatch.setattr(analyzer, "BOOTSTRAPS", 100)
    summary = analyzer.analyze(_synthetic_rows())
    assert summary["status"] == "scored"
    assert summary["primary_decoder_interaction"]["estimate"] == pytest.approx(0.5)
    assert summary["primary_decoder_interaction"]["ci95"] == pytest.approx([0.5, 0.5])


def test_analyzer_withholds_scores_when_free_invalid_rate_exceeds_gate(monkeypatch):
    monkeypatch.setattr(analyzer, "EXPECTED_IDS", 2)
    rows = _synthetic_rows()
    rows[("a", "aspect_only", "free_greedy")]["prediction"] = None
    summary = analyzer.analyze(rows)
    assert summary["status"] == "protocol_execution_failure"
    assert summary["score_analysis_performed"] is False
    assert "primary_decoder_interaction" not in summary


def test_analyzer_output_is_private_aggregate_only(tmp_path):
    summary = {
        "status": "scored",
        "n_clusters": 943,
        "n_outputs": 3772,
        "invalid_free_outputs": 0,
        "invalid_free_rate": 0.0,
        "context_gain_by_decoder": {
            "finite_grid": {"estimate": 0.4, "ci95": [0.2, 0.6]},
            "free_greedy": {"estimate": 0.1, "ci95": [-0.1, 0.3]},
        },
        "primary_decoder_interaction": {
            "estimate": 0.3,
            "ci95": [0.1, 0.5],
            "registered_interaction_gate_passed": True,
        },
    }
    analyzer.write_report(summary, tmp_path)
    report = (tmp_path / "README.md").read_text()
    assert "No review text" in report
    assert "0.300" in report
