import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from analyze_cross_judge_length_robustness_032 import is_correct, paired_accuracy_bootstrap
from run_cross_judge_length_robustness_032 import parse_label


def test_parse_label_accepts_only_exact_single_polarity():
    assert parse_label("Positive.") == 1
    assert parse_label("negative\n") == 0
    assert parse_label("positive, but also negative") is None
    assert parse_label("") is None


def test_unparseable_is_not_scored_as_correct():
    assert is_correct(None, 1) is False
    assert is_correct("positive", 1) is True
    assert is_correct("negative", 1) is False


def test_paired_cluster_bootstrap_is_one_when_all_pairs_improve():
    rows = [
        {"sentence_id": "a", "gold": 1, "label_8": "negative", "label_32": "positive"},
        {"sentence_id": "a", "gold": 0, "label_8": "positive", "label_32": "negative"},
        {"sentence_id": "b", "gold": 1, "label_8": None, "label_32": "positive"},
    ]
    result = paired_accuracy_bootstrap(rows, reps=100, seed=7)
    assert result["difference_strict_accuracy_32_minus_8"] == 1
    assert result["sentence_cluster_bootstrap_95_ci"] == [1, 1]
