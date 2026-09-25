import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_opinion_mask_word_order_control_044 as analysis
from run_opinion_mask_word_order_control_044 import shuffle_tokens


def test_shuffle_is_deterministic_changes_order_and_preserves_token_multiset():
    text = "The meal was [MASKED] but service felt [MASKED] ."
    first = shuffle_tokens(text, "case-1", "rus")
    second = shuffle_tokens(text, "case-1", "rus")
    assert first == second
    assert first != text
    assert Counter(first.split()) == Counter(text.split())
    assert first.count("[MASKED]") == 2


def _synthetic_rows(invalid_count=0):
    parent, shuffled = {}, {}
    invalid = 0
    for cluster in range(217):
        case_id = f"case-{cluster}"
        for lang in analysis.LANGS:
            for condition, prediction in (
                ("aspect_only", [3.0, 3.0]),
                ("opinion_masked", [5.0, 5.0]),
            ):
                parent[(case_id, lang, condition)] = {
                    "case_id": case_id,
                    "lang": lang,
                    "condition": condition,
                    "gold": [5.0, 5.0],
                    "prediction": prediction,
                }
            shuffled_prediction = None if invalid < invalid_count else [4.0, 4.0]
            invalid += shuffled_prediction is None
            shuffled[(case_id, lang, "opinion_shuffled")] = {
                "case_id": case_id,
                "lang": lang,
                "condition": "opinion_shuffled",
                "prediction": shuffled_prediction,
            }
    return parent, shuffled


def test_analysis_gate_withholds_scores_when_invalid_rate_exceeds_two_percent():
    parent, shuffled = _synthetic_rows(invalid_count=14)
    result = analysis.analyze(parent, shuffled)
    assert result["status"] == "protocol_execution_failure"
    assert result["n_invalid_outputs"] == 14
    assert result["score_analysis_performed"] is False


def test_analysis_computes_the_order_contrast_with_sentence_level_pairs():
    parent, shuffled = _synthetic_rows()
    result = analysis.analyze(parent, shuffled)
    assert result["status"] == "scored"
    assert result["primary_exploratory"]["estimate"] == 1.0
    assert result["primary_exploratory"]["ci95"] == [1.0, 1.0]
    assert result["primary_exploratory"]["practical_order_gate_passed"] is True
    assert result["descriptive_shuffled_vs_aspect_prior"]["estimate"] == 1.0
