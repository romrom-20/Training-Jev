import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from analyze_answer_length_censoring_031 import bootstrap_difference, validate_generation_pair
from run_answer_length_censoring_031 import eos_content


def test_eos_is_excluded_from_generated_content():
    assert eos_content([11, 12, 99, 0], eos_id=99) == ([11, 12], True)
    assert eos_content([11, 12, 13], eos_id=99) == ([11, 12, 13], False)


def test_pair_validator_requires_true_prefix_for_cap_hits():
    short = {"capped": True, "token_ids": list(range(8))}
    validate_generation_pair(short, {"token_ids": list(range(12))})
    with pytest.raises(ValueError, match="Non-prefix"):
        validate_generation_pair(short, {"token_ids": [0, 1, 2, 3, 4, 5, 6, 99]})


def test_pair_validator_requires_natural_early_stop_to_be_stable():
    short = {"capped": False, "token_ids": [5, 6]}
    validate_generation_pair(short, {"token_ids": [5, 6]})
    with pytest.raises(ValueError, match="Naturally ended"):
        validate_generation_pair(short, {"token_ids": [5, 6, 7]})


def test_primary_bootstrap_uses_only_capped_pairs_and_is_reproducible():
    rows = [
        {"sentence_id": "a", "capped_at_8": True, "ambiguous_8": 1, "ambiguous_32": 0},
        {"sentence_id": "b", "capped_at_8": True, "ambiguous_8": 0, "ambiguous_32": 1},
        {"sentence_id": "c", "capped_at_8": False, "ambiguous_8": 1, "ambiguous_32": 0},
    ]
    result = bootstrap_difference(rows, reps=100, seed=11)
    assert result["n_pairs"] == 2
    assert result["difference_ambiguous_32_minus_8"] == 0
    assert result["sentence_cluster_bootstrap_95_ci"] == [-1, 1]
