import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_structured_output_feasibility_042 as analysis
import pytest
from run_structured_output_feasibility_042 import (
    allowed_next_tokens,
    output_candidates,
    parse_response,
    revised_prompt,
)


def test_candidate_set_contains_81_numeric_outputs_and_one_abstention():
    candidates = output_candidates()
    assert len(candidates) == 82
    assert len({text for text, _ in candidates}) == 82
    assert sum(value["status"] == "estimate" for _, value in candidates) == 81
    assert sum(value["status"] == "insufficient" for _, value in candidates) == 1
    assert all(1 <= value["valence"] <= 9 and 1 <= value["arousal"] <= 9
               for _, value in candidates if value["status"] == "estimate")


def test_finite_trie_returns_only_next_tokens_for_valid_candidate_prefixes():
    sequences = [[1, 2, 3], [1, 2, 4], [1, 5]]
    assert allowed_next_tokens([], sequences) == [1]
    assert allowed_next_tokens([1], sequences) == [2, 5]
    assert allowed_next_tokens([1, 2], sequences) == [3, 4]
    with pytest.raises(ValueError, match="not a prefix"):
        allowed_next_tokens([9], sequences)


def test_parser_accepts_only_the_two_registered_response_shapes():
    assert parse_response('{"status":"estimate","valence":5,"arousal":3}') == {
        "status": "estimate",
        "valence": 5.0,
        "arousal": 3.0,
    }
    assert parse_response('```json\n{"status":"insufficient"}\n```') == {
        "status": "insufficient"
    }
    assert parse_response('{"status":"estimate","valence":10,"arousal":3}') is None
    assert parse_response('{"valence":5,"arousal":3}') is None
    assert parse_response('I cannot estimate from this phrase.') is None


def test_prompt_changes_only_output_contract():
    prompt = (
        'Return exactly one JSON object with numeric keys "valence" and "arousal", '
        "both from 1 to 9.\nReview text: example\nTarget aspect: room"
    )
    revised = revised_prompt(prompt)
    assert "Review text: example" in revised
    assert "Target aspect: room" in revised
    assert '"status":"insufficient"' in revised
    assert '"status":"estimate"' in revised


def test_aggregate_reports_prior_validity_groups_without_score_analysis():
    rows = []
    groups = [
        ("prior_invalid", lang, None)
        for index in range(48)
        for lang in (("rus", "ukr", "tat")[index % 3],)
    ] + [
        ("prior_valid_control", lang, {"status": "estimate", "valence": 7.0, "arousal": 6.0})
        for lang in ("rus", "ukr", "tat")
        for _ in range(16)
    ]
    for index, (group, lang, response) in enumerate(groups):
        case_id = f"case-{group}-{lang}-{index:03}"
        for decoder, parsed in (("free", response), ("constrained", {"status": "insufficient"})):
            rows.append(
                {
                    "case_id": case_id,
                    "lang": lang,
                    "source_condition": "aspect_opinion",
                    "group": group,
                    "decoder": decoder,
                    "response": parsed,
                }
            )
    result = analysis.analyze(rows)
    assert result["interpretation"].startswith("adaptive technical audit")
    assert result["n_generations"] == 192
    assert "gold_accuracy" not in result
