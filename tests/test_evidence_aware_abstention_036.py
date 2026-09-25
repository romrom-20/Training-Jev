import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_evidence_aware_abstention_036 as analysis
from run_evidence_aware_abstention_036 import load_jobs, map_key, parse_decision


def test_clear_subset_is_balanced_and_gold_tracks_visible_evidence():
    stimuli, jobs = load_jobs()
    assert len(stimuli) == 192
    assert Counter(row["polarity"] for row in stimuli) == {"positive": 96, "negative": 96}
    assert len({row["cluster_id"] for row in stimuli}) == 48
    no_cue = [row for row in jobs if row["position"] == "late" and row["budget"] == 8]
    visible = [row for row in jobs if row["cue_visible"]]
    assert len(no_cue) == 96
    assert all(row["expected_decision"] == "insufficient" for row in no_cue)
    assert len(visible) == 288
    assert all(row["expected_decision"] == row["polarity"] for row in visible)


def test_identical_prefixes_share_laya_mapping_key():
    _stimuli, jobs = load_jobs()
    no_cue = [row for row in jobs if row["position"] == "late" and row["budget"] == 8]
    for cluster in {row["cluster_id"] for row in no_cue}:
        keys = {map_key(row) for row in no_cue if row["cluster_id"] == cluster}
        assert len(keys) == 1


def test_abstention_label_parser_is_strict():
    assert parse_decision("insufficient") == "insufficient"
    assert parse_decision("Positive.") == "positive"
    assert parse_decision("negative") == "negative"
    assert parse_decision("insufficient evidence") is None


def test_paired_analyzer_rewards_appropriate_abstention(monkeypatch):
    monkeypatch.setattr(analysis, "REPS", 20)
    stimuli, jobs = load_jobs()
    new_outcomes = []
    base_outcomes = []
    for judge in analysis.JUDGES:
        for row in jobs:
            if judge == "laya":
                label = "unclear" if row["expected_decision"] == "insufficient" else row["polarity"]
            else:
                label = row["expected_decision"]
            new_outcomes.append(
                {
                    "judge": judge,
                    "id": row["id"],
                    "cluster_id": row["cluster_id"],
                    "gold": row["gold"],
                    "expected_decision": row["expected_decision"],
                    "label": label,
                    "position": row["position"],
                    "budget": row["budget"],
                    "cue_visible": row["cue_visible"],
                }
            )
            if judge in analysis.BINARIES:
                forced_label = row["gold"] if row["cue_visible"] else 1 - row["gold"]
                base_outcomes.append(
                    {
                        "judge": judge,
                        "id": row["id"],
                        "budget": row["budget"],
                        "label": forced_label,
                    }
                )

    result = analysis.analyze(stimuli, new_outcomes, base_outcomes)
    assert result["primary_pooled_change"]["estimate"] == 0.25
    assert result["preregistered_success_rule_passed"]
