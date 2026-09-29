import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_fresh_sample_prefix_replication_068 as runner


def case(case_id, polarity):
    return {
        "case_id": case_id,
        "gold": [2.0 if polarity == "neg" else 7.0, 5.0],
        "target_index": 0,
        "row": {"Text": "meal tasted wonderful", "Triplet": [{
            "Category": "食物#品质", "Aspect": "meal", "Opinion": "wonderful",
        }]},
    }


def test_fresh_selection_is_64_food_quality_ids_balanced_and_disjoint():
    source = ([case(f"n-{i:03}", "neg") for i in range(40)]
              + [case(f"p-{i:03}", "pos") for i in range(40)])
    excluded = {"n-000", "p-000"}
    selected, stats = runner.select_cases(source, excluded)

    assert len(selected) == 64
    assert stats["polarity_counts"] == {"neg": 32, "pos": 32}
    assert stats["category_polarity_counts"] == {"neg|食物#品质": 32, "pos|食物#品质": 32}
    assert stats["disjoint_from_exp064"] is True
    assert not ({case["case_id"] for case in selected} & excluded)


def test_fresh_selection_fails_closed_if_negative_food_quality_pool_is_short():
    source = [case(f"n-{i:03}", "neg") for i in range(31)]
    source += [case(f"p-{i:03}", "pos") for i in range(40)]
    with pytest.raises(ValueError, match="Insufficient fresh neg"):
        runner.select_cases(source, set())


def test_exp068_reuses_exp067_four_arm_jobs_exactly():
    cases = [case("fresh-neg", "neg"), case("fresh-pos", "pos")]
    jobs = runner.jobs_for(cases)

    assert len(jobs) == 8
    assert {(job["order"], job["forced_first_value"]) for job in jobs} == {
        (order, forced) for order in runner.ORDERS for forced in runner.FORCED_VALUES
    }
    assert all(job["assistant_prefix"] for job in jobs)


def test_protocol_binding_matches_literal_prefix_implementation():
    assert runner.exp067.assistant_prefix("valence_first", 2.0) == '{"valence": 2.0, "arousal": '
    assert runner.exp067.assistant_prefix("arousal_first", 8.0) == '{"arousal": 8.0, "valence": '
