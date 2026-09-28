import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_context_swap_donor_robustness_054 as base_analyzer
import analyze_fresh_context_swap_replication_055 as analyzer
import run_fresh_context_swap_replication_055 as runner


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


def _case(case_id, valence, text="sample review"):
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


def test_fresh_sample_is_disjoint_and_uses_expected_quotas():
    cases = (
        [_case(f"neg-{index}", 3.0) for index in range(224)]
        + [_case(f"neu-{index}", 5.0) for index in range(30)]
        + [_case(f"pos-{index}", 7.0) for index in range(689)]
    )
    old = {case["case_id"] for case in runner.exp053.choose_sample(cases)}
    fresh = runner.fresh_sample(cases)
    assert len(fresh) == 217
    assert not old.intersection(case["case_id"] for case in fresh)
    assert {polarity: sum(runner.bucket(case) == polarity for case in fresh)
            for polarity in runner.BUCKET_COUNTS} == runner.BUCKET_COUNTS


def test_three_donor_maps_are_bijective_derangements():
    cases = [_case(f"fresh-{index:03}", 3 + index % 5, "x" * (index + 5))
             for index in range(30)]
    maps, lengths = runner.donor_maps(cases, _Tokenizer())
    assert set(maps) == {1, 2, 3}
    ids = {case["case_id"] for case in cases}
    for mapping in maps.values():
        assert set(mapping) == set(mapping.values()) == ids
        assert all(recipient != donor for recipient, donor in mapping.items())
    assert set(lengths) == ids


def test_fresh_analyzer_wraps_frozen_054_scoring(monkeypatch):
    monkeypatch.setattr(base_analyzer, "EXPECTED_IDS", 2)
    monkeypatch.setattr(base_analyzer, "BOOTSTRAPS", 30)
    swapped = {}
    matched = {}
    ids = ("fresh-1", "fresh-2")
    for case_id in ids:
        for decoder in base_analyzer.DECODERS:
            matched[(case_id, "opinion_masked", decoder)] = {
                "case_id": case_id, "condition": "opinion_masked", "decoder": decoder,
                "gold": [5.0, 5.0], "prediction": [5.0, 5.0],
            }
        for permutation in base_analyzer.PERMUTATIONS:
            for decoder, offset in (("finite_grid", 2.0), ("free_greedy", 1.0)):
                swapped[(case_id, permutation, "swapped_context", decoder)] = {
                    "case_id": case_id, "permutation": permutation,
                    "condition": "swapped_context", "decoder": decoder,
                    "gold": [5.0, 5.0], "prediction": [5.0 + offset, 5.0 + offset],
                }
    summary = analyzer.analyze(swapped, matched)
    assert summary["experiment"] == "055-fresh-context-swap-replication"
    assert summary["analysis_seed"] == 20260955
    assert summary["sample_design"]["valence"] == "108 negative, 109 positive; no neutral cases"
    assert summary["mean_primary_interaction"]["estimate"] == pytest.approx(1.0)
