import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_sighan_output_key_order_065 as runner


def fake_case(case_id, cat, valence=2.0):
    return {
        "case_id": case_id,
        "gold": [valence, 5.0],
        "target_index": 0,
        "row": {"Triplet": [{"Category": cat}]},
    }


def test_extract_maps_reuses_complete_same_and_cross_category_derangements(tmp_path):
    cases = [fake_case("a0", "A"), fake_case("a1", "A"),
             fake_case("b0", "B"), fake_case("b1", "B")]
    rows = []
    for condition in (runner.SAME, runner.CROSS):
        for permutation in runner.PERMUTATIONS:
            donor = ({"a0": "a1", "a1": "a0", "b0": "b1", "b1": "b0"}
                     if condition == runner.SAME
                     else {"a0": "b0", "b0": "a0", "a1": "b1", "b1": "a1"})
            for order in runner.ORDERS:
                for case_id, donor_id in donor.items():
                    rows.append({
                        "case_id": case_id, "donor_id": donor_id,
                        "condition": condition, "permutation": permutation,
                        "order": order, "decoder": "finite_grid",
                    })
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    same, cross = runner.extract_maps(path, cases)
    assert same == {permutation: {"a0": "a1", "a1": "a0", "b0": "b1", "b1": "b0"}
                    for permutation in runner.PERMUTATIONS}
    assert cross[1]["a0"] == "b0"


def test_extract_maps_rejects_order_specific_donor_changes(tmp_path):
    cases = [fake_case("a0", "A"), fake_case("a1", "A"),
             fake_case("b0", "B"), fake_case("b1", "B")]
    rows = []
    for order, donor in (("valence_first", "a1"), ("arousal_first", "a0")):
        for permutation in runner.PERMUTATIONS:
            for case_id in ("a0", "a1", "b0", "b1"):
                rows.append({
                    "case_id": case_id,
                    "donor_id": donor if case_id == "a0" else ({"a1": "a0", "b0": "b1", "b1": "b0"}[case_id]),
                    "condition": runner.SAME, "permutation": permutation,
                    "order": order, "decoder": "finite_grid",
                })
    path = tmp_path / "predictions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="assignment changed"):
        runner.extract_maps(path, cases)
