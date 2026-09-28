import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_sighan_chinese_decoder_transfer_052 as analyzer
import run_sighan_chinese_decoder_transfer_052 as runner


def _row(case_id="cn-1"):
    return {
        "ID": case_id,
        "Text": "食物美味，服务很慢。",
        "Triplet": [
            {"Aspect": "食物", "Category": "食物#品质", "Opinion": "美味", "VA": "7.0#6.0"},
            {"Aspect": "服务", "Category": "服务#概括", "Opinion": "很慢", "VA": "3.0#4.0"},
        ],
    }


def test_source_parser_pairs_gold_and_input_and_reads_continuous_va():
    input_raw = "ID, Sentence\ncn-1, 食物美味，服务很慢。\n".encode()
    gold_raw = (
        "ID Quadruples\n"
        "cn-1 (食物,食物#品质,美味,7.0#6.0)(服务,服务#概括,很慢,3.0#4.0)\n"
    ).encode()
    rows = runner.parse_source(input_raw, gold_raw)
    assert len(rows) == 1
    assert rows[0]["ID"] == "cn-1"
    assert rows[0]["Triplet"][0]["VA"] == "7.0#6.0"
    assert rows[0]["Triplet"][1]["Opinion"] == "很慢"


def test_source_parser_rejects_input_gold_id_mismatch():
    input_raw = "ID, Sentence\ncn-1, 食物美味。\n".encode()
    gold_raw = "ID Quadruples\ncn-2 (食物,食物#品质,美味,7.0#6.0)\n".encode()
    with pytest.raises(ValueError, match="IDs differ"):
        runner.parse_source(input_raw, gold_raw)


def test_jobs_pair_prompts_and_mask_all_annotated_opinions():
    row = _row()
    case = {"case_id": row["ID"], "target_index": 0, "gold": [7.0, 6.0], "row": row}
    jobs = runner.build_shared_jobs([case])
    by_cell = {}
    for job in jobs:
        by_cell.setdefault(job["condition"], {})[job["decoder"]] = job
    assert len(jobs) == 4
    for arms in by_cell.values():
        assert arms["finite_grid"]["prompt"] == arms["free_greedy"]["prompt"]
    masked = by_cell["opinion_masked"]["finite_grid"]["prompt"]
    assert "[MASKED]" in masked
    assert "美味" not in masked
    assert "很慢" not in masked
    assert "食物" in masked


def _synthetic_rows(prefix, finite_shift, free_shift):
    rows = {}
    for case_id, gold in ((f"{prefix}-a", [5.0, 5.0]), (f"{prefix}-b", [6.0, 4.0])):
        for decoder, shift in (("finite_grid", finite_shift), ("free_greedy", free_shift)):
            for condition in analyzer.CONDITIONS:
                offset = shift if condition == "aspect_only" else 0.0
                rows[(case_id, condition, decoder)] = {
                    "case_id": case_id,
                    "condition": condition,
                    "decoder": decoder,
                    "gold": gold,
                    "prediction": [gold[0] + offset, gold[1] + offset],
                }
    return rows


def test_analyzer_computes_registered_interaction_and_transfer(monkeypatch):
    monkeypatch.setattr(analyzer, "EXPECTED_CHINESE_IDS", 2)
    monkeypatch.setattr(analyzer, "EXPECTED_ENGLISH_IDS", 2)
    monkeypatch.setattr(analyzer, "BOOTSTRAPS", 100)
    chinese = _synthetic_rows("cn", finite_shift=1.0, free_shift=0.5)
    english = _synthetic_rows("en", finite_shift=0.75, free_shift=0.5)
    summary = analyzer.analyze(chinese, english)
    assert summary["status"] == "scored"
    assert summary["primary_decoder_interaction"]["estimate"] == pytest.approx(0.5)
    assert summary["primary_decoder_interaction"]["ci95"] == pytest.approx([0.5, 0.5])
    assert summary["cross_release_language_transfer_secondary"]["estimate"] == pytest.approx(-0.25)


def test_analyzer_withholds_scores_when_free_invalid_rate_exceeds_gate(monkeypatch):
    monkeypatch.setattr(analyzer, "EXPECTED_CHINESE_IDS", 2)
    monkeypatch.setattr(analyzer, "EXPECTED_ENGLISH_IDS", 2)
    chinese = _synthetic_rows("cn", finite_shift=1.0, free_shift=0.5)
    english = _synthetic_rows("en", finite_shift=0.75, free_shift=0.5)
    chinese[("cn-a", "aspect_only", "free_greedy")]["prediction"] = None
    summary = analyzer.analyze(chinese, english)
    assert summary["status"] == "protocol_execution_failure"
    assert summary["score_analysis_performed"] is False
    assert "primary_decoder_interaction" not in summary
