import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_context_swap_donor_robustness_054 as shared
import analyze_cross_category_donor_control_060 as analyzer
import run_cross_category_donor_control_060 as runner


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


def _case(case_id, polarity, category, length):
    valence = 3.0 if polarity == "neg" else 7.0
    return {
        "case_id": case_id,
        "gold": [valence, 5.0],
        "target_index": 0,
        "category": category,
        "row": {
            "ID": case_id,
            "Text": "x" * length,
            "Triplet": [{"Aspect": "battery", "Opinion": "NULL", "VA": f"{valence}#5"}],
        },
    }


def test_cross_category_maps_are_bijective_polarity_matched_and_distinct():
    cases = []
    for polarity in ("neg", "pos"):
        for category in ("A#QUALITY", "B#QUALITY", "C#QUALITY"):
            for index in range(2):
                cases.append(_case(f"{polarity}-{category}-{index}", polarity, category, 8 + index))
    maps, meta = runner.cross_category_maps(cases, _Tokenizer())
    by_id = {case["case_id"]: case for case in cases}
    assert meta["n_unique_assignment_maps"] == 3
    for mapping in maps.values():
        assert set(mapping) == set(mapping.values()) == set(by_id)
        for recipient_id, donor_id in mapping.items():
            recipient, donor = by_id[recipient_id], by_id[donor_id]
            assert recipient_id != donor_id
            assert runner.exp057.exp055.bucket(recipient) == runner.exp057.exp055.bucket(donor)
            assert recipient["category"] != donor["category"]


def test_registered_direct_contrast_is_cross_category_minus_same_category(monkeypatch):
    monkeypatch.setattr(shared, "EXPECTED_IDS", 2)
    monkeypatch.setattr(shared, "BOOTSTRAPS", 30)
    ids = ("a", "b")
    own, same, cross = {}, {}, {}
    for case_id in ids:
        for decoder in shared.DECODERS:
            own[(case_id, "opinion_masked", decoder)] = {
                "case_id": case_id, "condition": "opinion_masked", "decoder": decoder,
                "gold": [5.0, 5.0], "prediction": [5.0, 5.0],
            }
            for permutation in shared.PERMUTATIONS:
                for rows, condition, finite_offset in (
                    (same, "category_polarity_matched_context", 2.0),
                    (cross, "cross_category_polarity_matched_context", 3.0),
                ):
                    offset = finite_offset if decoder == "finite_grid" else 1.0
                    rows[(case_id, permutation, "swapped_context", decoder)] = {
                        "case_id": case_id, "permutation": permutation, "condition": condition,
                        "decoder": decoder, "gold": [5.0, 5.0],
                        "prediction": [5.0 + offset, 5.0 + offset],
                    }
    summary = analyzer.analyze(cross, same, own, None)
    assert summary["category_mismatched_interaction"]["estimate"] == pytest.approx(2.0)
    assert summary["category_matched_interaction"]["estimate"] == pytest.approx(1.0)
    assert summary["mean_primary_interaction"]["estimate"] == pytest.approx(1.0)
    assert summary["mean_primary_interaction"]["n_recipient_ids"] == 2
