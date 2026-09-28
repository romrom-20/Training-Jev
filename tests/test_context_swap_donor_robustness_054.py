import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_context_swap_donor_robustness_054 as analyzer
import run_context_swap_donor_robustness_054 as runner


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


def _case(case_id, valence, text):
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


def test_each_permutation_is_deterministic_and_deranged():
    cases = [_case(f"id-{index:03}", 3 + index % 5, "x" * (index + 5)) for index in range(30)]
    first, _ = runner.make_donor_map(cases, _Tokenizer(), 1)
    again, _ = runner.make_donor_map(list(reversed(cases)), _Tokenizer(), 1)
    second, _ = runner.make_donor_map(cases, _Tokenizer(), 2)
    assert first == again
    assert set(first) == set(first.values())
    assert all(key != value for key, value in first.items())
    assert first != second


def _synthetic(permutation_offsets):
    swapped = {}
    matched = {}
    for case_id in ("id-1", "id-2"):
        for decoder, matched_offset in (("finite_grid", 0.0), ("free_greedy", 0.0)):
            matched[(case_id, "opinion_masked", decoder)] = {
                "case_id": case_id, "condition": "opinion_masked", "decoder": decoder,
                "gold": [5.0, 5.0], "prediction": [5.0 + matched_offset, 5.0 + matched_offset],
            }
        for permutation, offsets in permutation_offsets.items():
            for decoder, offset in (("finite_grid", offsets[0]), ("free_greedy", offsets[1])):
                swapped[(case_id, permutation, "swapped_context", decoder)] = {
                    "case_id": case_id, "permutation": permutation,
                    "condition": "swapped_context", "decoder": decoder,
                    "gold": [5.0, 5.0], "prediction": [5.0 + offset, 5.0 + offset],
                }
    return swapped, matched


def test_analyzer_averages_fixed_permutation_interactions(monkeypatch):
    monkeypatch.setattr(analyzer, "EXPECTED_IDS", 2)
    monkeypatch.setattr(analyzer, "BOOTSTRAPS", 50)
    swapped, matched = _synthetic({1: (2.0, 1.0), 2: (1.0, 0.5), 3: (3.0, 1.0)})
    result = analyzer.analyze(swapped, matched)
    assert result["status"] == "scored"
    assert result["permutation_sensitivity"]["range"] == pytest.approx([0.5, 2.0])
    assert result["mean_primary_interaction"]["estimate"] == pytest.approx(7 / 6)


def test_analyzer_withholds_contrasts_if_free_invalid_rate_exceeds_gate(monkeypatch):
    monkeypatch.setattr(analyzer, "EXPECTED_IDS", 2)
    swapped, matched = _synthetic({1: (1.0, 0.5), 2: (1.0, 0.5), 3: (1.0, 0.5)})
    swapped[("id-1", 1, "swapped_context", "free_greedy")]["prediction"] = None
    result = analyzer.analyze(swapped, matched)
    assert result["status"] == "protocol_execution_failure"
    assert result["score_analysis_performed"] is False
