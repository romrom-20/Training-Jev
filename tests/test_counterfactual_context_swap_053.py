import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_counterfactual_context_swap_053 as analyzer
import run_counterfactual_context_swap_053 as runner


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


def _case(case_id, valence, text="review text"):
    return {
        "case_id": case_id,
        "gold": [valence, 5.0],
        "target_index": 0,
        "row": {
            "ID": case_id,
            "Text": text,
            "Triplet": [{"Aspect": "battery", "Opinion": "NULL", "VA": f"{valence}#5"}],
        },
    }


def test_sample_is_reproducible_and_has_frozen_bucket_counts():
    cases = (
        [_case(f"neg-{index}", 3.0) for index in range(100)]
        + [_case(f"neu-{index}", 5.0) for index in range(35)]
        + [_case(f"pos-{index}", 7.0) for index in range(100)]
    )
    first = runner.choose_sample(cases)
    second = runner.choose_sample(list(reversed(cases)))
    assert [case["case_id"] for case in first] == [case["case_id"] for case in second]
    assert {bucket: sum(runner.valence_bucket(case) == bucket for case in first)
            for bucket in runner.BUCKET_COUNTS} == runner.BUCKET_COUNTS


def test_donor_map_is_length_binned_derangement_and_permutation():
    cases = [_case(f"id-{index:03}", 3 + index % 5, "x" * (index + 3)) for index in range(20)]
    donor_map, lengths = runner.build_donor_map(cases, _Tokenizer())
    assert set(donor_map) == set(donor_map.values()) == set(lengths)
    assert all(recipient != donor for recipient, donor in donor_map.items())


def _synthetic_rows(prefix, swapped_offset):
    rows = {}
    for case_id in (f"{prefix}-1", f"{prefix}-2"):
        for decoder, offset in (("finite_grid", swapped_offset), ("free_greedy", swapped_offset / 2)):
            rows[(case_id, "swapped_context", decoder)] = {
                "case_id": case_id, "condition": "swapped_context", "decoder": decoder,
                "gold": [5.0, 5.0], "prediction": [5.0 + offset, 5.0 + offset],
            }
    return rows


def test_analyzer_reports_decoder_difference_in_matched_review_advantage(monkeypatch):
    monkeypatch.setattr(analyzer, "EXPECTED_IDS", 2)
    monkeypatch.setattr(analyzer, "BOOTSTRAPS", 50)
    swapped = _synthetic_rows("s", 2.0)
    matched = {
        (case_id, "opinion_masked", decoder): {
            "case_id": case_id, "condition": "opinion_masked", "decoder": decoder,
            "gold": [5.0, 5.0], "prediction": [5.0, 5.0],
        }
        for case_id in ("s-1", "s-2")
        for decoder in ("finite_grid", "free_greedy")
    }
    summary = analyzer.analyze(swapped, matched)
    assert summary["status"] == "scored"
    assert summary["matched_review_advantage"]["finite_grid"]["matched_review_advantage_rmse"] == pytest.approx(2.0)
    assert summary["matched_review_advantage"]["free_greedy"]["matched_review_advantage_rmse"] == pytest.approx(1.0)
    assert summary["primary_decoder_interaction"]["estimate"] == pytest.approx(1.0)


def test_analyzer_withholds_scores_above_invalid_free_gate(monkeypatch):
    monkeypatch.setattr(analyzer, "EXPECTED_IDS", 2)
    swapped = _synthetic_rows("s", 1.0)
    swapped[("s-1", "swapped_context", "free_greedy")]["prediction"] = None
    matched = {
        (case_id, "opinion_masked", decoder): {
            "case_id": case_id, "condition": "opinion_masked", "decoder": decoder,
            "gold": [5.0, 5.0], "prediction": [5.0, 5.0],
        }
        for case_id in ("s-1", "s-2")
        for decoder in ("finite_grid", "free_greedy")
    }
    summary = analyzer.analyze(swapped, matched)
    assert summary["status"] == "protocol_execution_failure"
    assert summary["score_analysis_performed"] is False
