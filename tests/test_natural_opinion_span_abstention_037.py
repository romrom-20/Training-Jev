import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_natural_opinion_span_abstention_037 as analysis
from run_natural_opinion_span_abstention_037 import (
    PUBLIC_STIMULUS_KEYS,
    decode_laya_choice,
    laya_map_key,
    load_source_items,
    make_jobs,
    parse_label,
)


def test_frozen_selection_balances_domains_and_polarities_without_public_text():
    selected = load_source_items()
    assert len(selected) == 234
    assert Counter((row["dataset"], row["polarity"]) for row in selected) == {
        ("14res", "positive"): 27,
        ("14res", "negative"): 27,
        ("14lap", "positive"): 29,
        ("14lap", "negative"): 29,
        ("15res", "positive"): 39,
        ("15res", "negative"): 39,
        ("16res", "positive"): 22,
        ("16res", "negative"): 22,
    }
    public_rows = [{key: row[key] for key in PUBLIC_STIMULUS_KEYS} for row in selected]
    assert all("_tokens" not in row and "_aspect" not in row for row in public_rows)
    assert len({row["cluster_id"] for row in public_rows}) == 234


def test_prefix_pair_hides_then_reveals_complete_annotated_opinion_span():
    selected = load_source_items()
    jobs = make_jobs(selected)
    assert len(jobs) == 468
    for item in selected:
        pair = [row for row in jobs if row["id"] == item["id"]]
        assert {row["condition"] for row in pair} == {"before_opinion", "opinion_visible"}
        before = next(row for row in pair if row["condition"] == "before_opinion")
        visible = next(row for row in pair if row["condition"] == "opinion_visible")
        opinion_tokens = [item["_tokens"][i] for i in item["opinion_token_indices"]]
        assert before["visible_text"].split()[-1] != opinion_tokens[0]
        assert all(token in visible["visible_text"].split() for token in opinion_tokens)
        assert before["expected_decision"] == "insufficient"
        assert visible["expected_decision"] == item["polarity"]


def test_laya_mapping_key_reuses_same_aspect_and_visible_prefix():
    rows = make_jobs(load_source_items())
    keyed = {}
    for row in rows:
        key = laya_map_key(row)
        keyed.setdefault(key, set()).add((row["aspect"], row["visible_text"]))
    assert all(len(values) == 1 for values in keyed.values())
    assert decode_laya_choice("B", {"A": "negative", "B": "unclear"}) == "unclear"
    with pytest.raises(ValueError, match="unknown choice"):
        decode_laya_choice("E", {"A": "positive"})


def test_label_parsers_are_strict_about_allowed_output_sets():
    assert parse_label("positive", "forced_binary") == "positive"
    assert parse_label("Negative.", "forced_binary") == "negative"
    assert parse_label("insufficient", "abstention") == "insufficient"
    assert parse_label("insufficient", "forced_binary") is None
    assert parse_label("The answer is positive", "forced_binary") is None


def test_analyzer_scores_abstention_policy_and_detects_incomplete_join(monkeypatch):
    monkeypatch.setattr(analysis, "REPS", 20)
    stimuli = [
        {
            "id": f"d-{index}",
            "cluster_id": f"d-{index}",
            "dataset": "14res",
            "polarity": "positive" if index % 2 else "negative",
        }
        for index in range(6)
    ]
    outcomes = []
    for stimulus in stimuli:
        for condition in ("before_opinion", "opinion_visible"):
            expected = "insufficient" if condition == "before_opinion" else stimulus["polarity"]
            for judge in ("qwen2.5-3b", "phi3-mini"):
                for wrapper in ("forced_binary", "abstention"):
                    label = expected if wrapper == "abstention" else stimulus["polarity"]
                    outcomes.append(
                        {
                            "id": stimulus["id"],
                            "cluster_id": stimulus["cluster_id"],
                            "dataset": stimulus["dataset"],
                            "condition": condition,
                            "expected_decision": expected,
                            "polarity": stimulus["polarity"],
                            "judge": judge,
                            "wrapper": wrapper,
                            "label": label,
                        }
                    )
            laya_label = "unclear" if condition == "before_opinion" else stimulus["polarity"]
            outcomes.append(
                {
                    "id": stimulus["id"],
                    "cluster_id": stimulus["cluster_id"],
                    "dataset": stimulus["dataset"],
                    "condition": condition,
                    "expected_decision": expected,
                    "polarity": stimulus["polarity"],
                    "judge": "laya",
                    "wrapper": "four_way",
                    "label": laya_label,
                }
            )

    result = analysis.analyze(stimuli, outcomes)
    assert result["primary_pooled_change"]["estimate"] == 0.5
    assert result["preregistered_success_rule_passed"]
    assert result["pre_opinion_predictive_polarity_diagnostics"]["qwen2.5-3b"][
        "forced_binary_polarity_accuracy_before_opinion"
    ]["estimate"] == 1.0
    assert result["laya"]["before_opinion"]["abstention_recall"]["estimate"] == 1.0
    with pytest.raises(ValueError, match="Expected"):
        analysis.analyze(stimuli, outcomes[:-1])
