import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from analyze_laya_decision_audit_030 import bootstrap_ratio
from run_laya_decision_audit_030 import CHOICES, build_question, choice_key_maps


def test_choice_key_rotation_balances_each_semantic_class_across_four_items():
    mappings = choice_key_maps([f"item-{index}" for index in range(4)], seed=1)
    assert mappings == choice_key_maps([f"item-{index}" for index in range(4)], seed=1)
    for key in "ABCD":
        assert {mapping[key] for mapping in mappings.values()} == set(CHOICES)


def test_typed_choice_schema_has_fixed_descriptions_and_neutral_keys():
    mapping = {"A": "negative", "B": "unclear", "C": "positive", "D": "mixed"}
    question = build_question(mapping)["polarity"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == set(mapping)
    assert question["criteria"]["A"] == CHOICES["negative"]
    assert "true" not in question["criteria"]
    assert "false" not in question["criteria"]


def test_bootstrap_resamples_source_sentences_and_is_deterministic():
    rows = [
        {"sentence_id": "s1", "correct": 1},
        {"sentence_id": "s1", "correct": 1},
        {"sentence_id": "s2", "correct": 0},
    ]

    def numerator(row):
        return row["correct"]

    first = bootstrap_ratio(rows, numerator, lambda row: 1, reps=500, seed=8)
    second = bootstrap_ratio(rows, numerator, lambda row: 1, reps=500, seed=8)
    assert first == second
    assert first["estimate"] == 2 / 3
