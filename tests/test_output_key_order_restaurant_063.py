import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_output_key_order_restaurant_063 as runner
import run_output_key_order_topic_match_062 as order_runner


def fake_case(case_id, category, valence=2.0):
    return {"case_id": case_id, "category": category, "gold": [valence, 5.0]}


def test_same_category_assignment_is_polarity_stratum_derangement():
    rows = [fake_case("a", "FOOD"), fake_case("b", "FOOD"), fake_case("c", "FOOD")]
    lengths = {"a": 10, "b": 11, "c": 12}
    mapping = runner.assign(rows, lengths, 1, require_different_category=False)
    assert set(mapping) == {"a", "b", "c"}
    assert set(mapping.values()) == set(mapping)
    assert all(source != donor for source, donor in mapping.items())


def test_cross_category_assignment_rejects_category_collisions():
    rows = [fake_case("a", "FOOD"), fake_case("b", "FOOD"),
            fake_case("c", "SERVICE"), fake_case("d", "SERVICE")]
    lengths = {"a": 10, "b": 12, "c": 11, "d": 13}
    mapping = runner.assign(rows, lengths, 101, require_different_category=True)
    by_id = {row["case_id"]: row for row in rows}
    assert set(mapping.values()) == set(mapping)
    assert all(source != donor for source, donor in mapping.items())
    assert all(by_id[source]["category"] != by_id[donor]["category"]
               for source, donor in mapping.items())


def test_cross_category_assignment_fails_when_no_derangement_exists():
    rows = [fake_case("a", "FOOD"), fake_case("b", "FOOD"), fake_case("c", "SERVICE")]
    with pytest.raises(ValueError, match="No complete donor assignment"):
        runner.assign(rows, {"a": 1, "b": 2, "c": 3}, 101,
                      require_different_category=True)


def test_arousal_first_finite_parser_returns_canonical_va_order():
    assert order_runner.parse_finite_arousal_first(
        '{"arousal":4.2,"valence":7.1}'
    ) == [7.1, 4.2]
