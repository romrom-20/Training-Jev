import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_output_key_order_topic_match_062 as analyzer
import run_output_key_order_topic_match_062 as runner


def test_arousal_first_candidate_and_parser_round_trip():
    values = runner.arousal_first_candidates()
    assert len(values) == 6561
    assert values[0] == ('{"arousal":1.0,"valence":1.0}', {"valence": 1.0, "arousal": 1.0})
    assert values[-1] == ('{"arousal":9.0,"valence":9.0}', {"valence": 9.0, "arousal": 9.0})
    assert runner.parse_finite_arousal_first('{"arousal":4.2,"valence":7.1}') == [7.1, 4.2]
    assert runner.parse_finite_arousal_first('{"valence":7.1,"arousal":4.2}') is None


def test_arousal_first_prompt_reverses_only_numeric_instruction():
    prompt = runner.arousal_first_prompt("review [MASKED]", "battery")
    assert 'numeric keys "arousal" and "valence"' in prompt
    assert 'numeric keys "valence" and "arousal"' not in prompt
    assert "increments of 0.1" in prompt


def test_interaction_by_condition_computes_finite_minus_free_advantage():
    own, same = {}, {}
    for case_id in ("a", "b"):
        for decoder in ("finite_grid", "free_greedy"):
            own[(case_id, "opinion_masked", decoder)] = {
                "case_id": case_id, "condition": "opinion_masked", "decoder": decoder,
                "gold": [5.0, 5.0], "prediction": [5.0, 5.0],
            }
            for permutation in (1, 2, 3):
                same[(case_id, permutation, "swapped_context", decoder)] = {
                    "case_id": case_id, "permutation": permutation,
                    "condition": "swapped_context", "decoder": decoder,
                    "gold": [5.0, 5.0],
                    "prediction": [7.0, 7.0] if decoder == "finite_grid" else [6.0, 6.0],
                }
    assert analyzer.interaction_by_condition(same, own, ["a", "b"]) == {
        1: pytest.approx(1.0), 2: pytest.approx(1.0), 3: pytest.approx(1.0)
    }
