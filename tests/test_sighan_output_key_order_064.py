import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_sighan_output_key_order_064 as runner


def case(case_id, pol, cat):
    value = 2.0 if pol == "neg" else 7.0
    return {
        "case_id": case_id,
        "gold": [value, 5.0],
        "target_index": 0,
        "row": {"Triplet": [{"Category": cat}]},
    }


def test_frozen_sample_uses_exact_balanced_polarity_category_quotas():
    rows = []
    i = 0
    for pol, quotas in runner.SAMPLE_QUOTAS.items():
        for cat, quota in quotas.items():
            for _ in range(quota):
                rows.append(case(f"id-{i:03}", pol, cat))
                i += 1
    selected, stats = runner.select_cases(rows)
    assert len(selected) == 180
    assert stats["polarity_counts"] == {"neg": 90, "pos": 90}
    assert stats["n_category_polarity_groups"] == 7


def test_sample_selection_fails_closed_if_a_registered_stratum_is_short():
    with pytest.raises(ValueError, match="Insufficient"):
        runner.select_cases([case("one", "neg", "食物#品质")])


def test_donor_assignments_preserve_a_full_derangement_and_cross_categories():
    rows = [case(f"a{i}", "neg", "A") for i in range(3)]
    rows += [case(f"b{i}", "neg", "B") for i in range(3)]
    lengths = {row["case_id"]: 10 + i for i, row in enumerate(rows)}
    mapping = runner.assign(rows, lengths, 101, require_different_category=True)
    by_id = {row["case_id"]: row for row in rows}
    assert len(mapping) == len(rows)
    assert set(mapping.values()) == set(mapping)
    assert all(source != donor for source, donor in mapping.items())
    assert all(runner.category(by_id[source]) != runner.category(by_id[donor])
               for source, donor in mapping.items())


def test_cross_category_assignment_rejects_impossible_category_mix():
    rows = [case("a", "neg", "A"), case("b", "neg", "A"), case("c", "neg", "B")]
    with pytest.raises(ValueError, match="No complete donor assignment"):
        runner.assign(rows, {"a": 1, "b": 2, "c": 3}, 101,
                      require_different_category=True)
