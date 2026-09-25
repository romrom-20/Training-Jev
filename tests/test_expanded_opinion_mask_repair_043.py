import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_expanded_opinion_mask_repair_043 as analysis
import run_expanded_opinion_mask_repair_043 as experiment


def test_candidate_grid_has_6561_unique_one_decimal_pairs():
    values = experiment.candidates()
    assert len(values) == 6561
    assert len({text for text, _ in values}) == 6561
    assert values[0] == ('{"valence":1.0,"arousal":1.0}', {"valence": 1.0, "arousal": 1.0})
    assert values[-1] == ('{"valence":9.0,"arousal":9.0}', {"valence": 9.0, "arousal": 9.0})


def test_parser_accepts_only_in_range_tenth_resolution_va():
    assert experiment.parse_va('{"valence":4.3,"arousal":8.0}') == [4.3, 8.0]
    assert experiment.parse_va('{"valence":4.35,"arousal":8.0}') is None
    assert experiment.parse_va('{"valence":0.9,"arousal":8.0}') is None
    assert experiment.parse_va('{"valence":4,"arousal":8}') is None
    assert experiment.parse_va('{"valence":4,"arousal":8,"confidence":0.5}') is None


def test_finite_trie_allows_only_registered_candidate_sequences():
    trie = experiment.build_trie([[1, 2, 9], [1, 3, 9]])
    assert trie[()] == (1,)
    assert trie[(1,)] == (2, 3)
    assert trie[(1, 2)] == (9,)
    assert trie[(1, 2, 9)] == (9,)
    assert trie[(1, 2, 9, 9)] == (9,)


def test_predeclared_sample_uses_largest_near_balanced_disjoint_pool():
    assert experiment.QUOTAS == {"neg": 103, "neu": 11, "pos": 103}
    assert sum(experiment.QUOTAS.values()) == 217
    assert len(experiment.CONDITIONS) == 2


def test_analysis_stops_without_scoring_when_invalid_threshold_fails():
    rows = []
    for cluster in range(217):
        for lang in analysis.LANGS:
            for condition in analysis.CONDITIONS:
                invalid = cluster < 27 and lang == "rus" and condition == "aspect_only"
                rows.append(
                    {
                        "case_id": f"case-{cluster}",
                        "lang": lang,
                        "condition": condition,
                        "gold": [4.0, 5.0],
                        "prediction": None if invalid else [4.0, 5.0],
                    }
                )
    result = analysis.analyze(rows)
    assert result["status"] == "protocol_execution_failure"
    assert result["n_invalid_outputs"] == 27
    assert result["score_analysis_performed"] is False
