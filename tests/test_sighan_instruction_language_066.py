import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_sighan_instruction_language_066 as runner


def case(case_id, cat, valence):
    return {
        "case_id": case_id,
        "gold": [valence, 5.0],
        "target_index": 0,
        "row": {"Text": "meal was tasty", "Triplet": [{
            "Category": cat, "Aspect": "meal", "Opinion": "tasty",
        }]},
    }


def test_frozen_subset_has_exact_88_id_strata():
    rows = []
    index = 0
    for polarity, quotas in runner.SAMPLE_QUOTAS.items():
        valence = 2.0 if polarity == "neg" else 7.0
        for cat, quota in quotas.items():
            for _ in range(quota + 2):
                rows.append(case(f"id-{index:03}", cat, valence))
                index += 1

    selected, stats = runner.select_cases(rows)

    assert len(selected) == 88
    assert len({row["case_id"] for row in selected}) == 88
    assert stats["polarity_counts"] == {"neg": 44, "pos": 44}
    assert stats["n_category_polarity_groups"] == 6
    assert sum(stats["selection_quotas"].values()) == 88


def test_frozen_subset_fails_closed_for_a_short_stratum():
    with pytest.raises(ValueError, match="Insufficient"):
        runner.select_cases([case("one", "食物#品质", 2.0)])


def test_chinese_prompt_changes_only_serialized_json_key_order():
    va = runner.chinese_prompt("masked review", "meal", "valence_first")
    av = runner.chinese_prompt("masked review", "meal", "arousal_first")

    assert '"valence"和"arousal"' in va
    assert '"arousal"和"valence"' in av
    assert va.replace('"valence"和"arousal"', "KEYS") == av.replace(
        '"arousal"和"valence"', "KEYS"
    )
    assert "1.0 至 9.0" in va
    assert "masked review" in va and "meal" in va
    with pytest.raises(ValueError, match="Unexpected output order"):
        runner.chinese_prompt("review", "aspect", "other")


def test_extract_maps_allows_frozen_donors_outside_recipient_subset(tmp_path):
    # The four donor-only IDs are part of Exp064's fixed 180-case cohort.
    base = [case("a0", "A", 2.0), case("a1", "A", 2.0),
            case("a2", "A", 2.0), case("a3", "A", 2.0),
            case("b0", "B", 2.0), case("b1", "B", 2.0),
            case("b2", "B", 2.0), case("b3", "B", 2.0)]
    recipients = [base[0], base[1], base[4], base[5]]
    same = {1: {"a0": "a1", "a1": "a0", "b0": "b1", "b1": "b0"},
            2: {"a0": "a2", "a1": "a3", "b0": "b2", "b1": "b3"},
            3: {"a0": "a3", "a1": "a2", "b0": "b3", "b1": "b2"}}
    cross = {1: {"a0": "b0", "a1": "b1", "b0": "a0", "b1": "a1"},
             2: {"a0": "b2", "a1": "b3", "b0": "a2", "b1": "a3"},
             3: {"a0": "b3", "a1": "b2", "b0": "a3", "b1": "a2"}}
    base_by_id = {item["case_id"]: item for item in base}
    rows = []
    for condition, mappings in ((runner.SAME, same), (runner.CROSS, cross)):
        for permutation, mapping in mappings.items():
            for order in runner.ORDERS:
                for source, donor_id in mapping.items():
                    donor = base_by_id[donor_id]
                    rows.append({
                        "case_id": source, "donor_id": donor_id,
                        "donor_category": runner.category(donor),
                        "donor_polarity": runner.polarity(donor),
                        "condition": condition, "permutation": permutation,
                            "order": order, "decoder": "finite_grid",
                        })
    rows.extend({
        "case_id": donor_only["case_id"], "condition": runner.OWN,
        "permutation": 0, "order": runner.ORDERS[0], "decoder": "finite_grid",
    } for donor_only in (base[2], base[3], base[6], base[7]))
    path = tmp_path / "parent.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    extracted_same, extracted_cross = runner.extract_maps(path, recipients)

    assert extracted_same == same
    assert extracted_cross == cross


def test_extract_maps_rejects_wrong_polarity_metadata(tmp_path):
    cases = [case("a0", "A", 2.0), case("a1", "A", 2.0),
             case("b0", "B", 2.0), case("b1", "B", 2.0)]
    rows = []
    for condition in (runner.SAME, runner.CROSS):
        maps = ({"a0": "a1", "a1": "a0", "b0": "b1", "b1": "b0"}
                if condition == runner.SAME else
                {"a0": "b0", "a1": "b1", "b0": "a0", "b1": "a1"})
        for permutation in runner.PERMUTATIONS:
            for order in runner.ORDERS:
                for source, donor_id in maps.items():
                    donor = next(item for item in cases if item["case_id"] == donor_id)
                    rows.append({
                        "case_id": source, "donor_id": donor_id,
                        "donor_category": runner.category(donor),
                        "donor_polarity": "pos" if source == "a0" else runner.polarity(donor),
                        "condition": condition, "permutation": permutation,
                        "order": order, "decoder": "finite_grid",
                    })
    path = tmp_path / "parent.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="preserve polarity"):
        runner.extract_maps(path, cases)
