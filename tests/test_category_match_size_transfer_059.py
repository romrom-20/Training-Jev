import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_category_match_size_transfer_059 as analyzer
import analyze_context_swap_donor_robustness_054 as shared


def _design(interaction):
    donors, own = {}, {}
    for case_id in ("a", "b"):
        for decoder, offset in (("finite_grid", interaction + 1.0), ("free_greedy", 1.0)):
            own[(case_id, "opinion_masked", decoder)] = {
                "case_id": case_id, "condition": "opinion_masked", "decoder": decoder,
                "gold": [5.0, 5.0], "prediction": [5.0, 5.0],
            }
            for permutation in shared.PERMUTATIONS:
                donors[(case_id, permutation, "swapped_context", decoder)] = {
                    "case_id": case_id, "condition": "swapped_context", "decoder": decoder,
                    "permutation": permutation, "gold": [5.0, 5.0],
                    "prediction": [5.0 + offset, 5.0 + offset],
                }
    return donors, own


def test_interaction_and_size_contrast_use_same_recipient_ids():
    donor_15, own_15 = _design(0.3)
    donor_3, own_3 = _design(0.1)
    estimate = analyzer.interaction_by_size(donor_15, own_15, ["a", "b"])
    contrast = analyzer.paired_size_contrast(
        donor_15, own_15, donor_3, own_3, bootstrap_replicates=40
    )
    assert estimate == pytest.approx(0.3)
    assert contrast["estimate_1_5b"] == pytest.approx(0.3)
    assert contrast["estimate_3b"] == pytest.approx(0.1)
    assert contrast["difference"] == pytest.approx(0.2)
    assert contrast["n_recipient_ids"] == 2
