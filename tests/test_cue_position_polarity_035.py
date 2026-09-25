import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_cue_position_polarity_035 as analysis
from run_cue_position_polarity_035 import (
    BUDGETS,
    STIMULI_FILE,
    STIMULI_SHA256,
    make_jobs,
    make_stimuli,
)


def test_cue_position_factorial_is_balanced_and_word_locked():
    stimuli = make_stimuli()
    assert hashlib.sha256(STIMULI_FILE.read_bytes()).hexdigest() == STIMULI_SHA256
    assert json.loads(STIMULI_FILE.read_text()) == stimuli
    jobs = make_jobs(stimuli)
    assert len(stimuli) == 384
    assert len({row["cluster_id"] for row in stimuli}) == 48
    assert Counter(row["polarity"] for row in stimuli) == {"negative": 192, "positive": 192}
    assert len(jobs) == len(stimuli) * len(BUDGETS)
    assert all(len(row["frame"].split()) == 7 for row in stimuli)
    assert all(len(row["answer"].split()) <= 12 for row in stimuli)


def test_early_and_late_conditions_move_the_same_clause_across_prefix_boundary():
    stimuli = make_stimuli()
    jobs = make_jobs(stimuli)
    by_id_budget = {(row["id"], row["budget"]): row for row in jobs}
    paired = defaultdict(dict)
    for row in stimuli:
        pair_key = (row["cluster_id"], row["polarity"], row["form"])
        paired[pair_key][row["position"]] = row
        prefix8 = by_id_budget[(row["id"], 8)]["visible_text"]
        prefix12 = by_id_budget[(row["id"], 12)]["visible_text"]
        if row["position"] == "early":
            assert row["clause"] in prefix8
        else:
            assert "good" not in prefix8 and "bad" not in prefix8
            assert prefix12 == row["answer"]
    for positions in paired.values():
        assert positions["early"]["frame"] == positions["late"]["frame"]
        assert positions["early"]["clause"] == positions["late"]["clause"]


def test_analyzer_detects_late_cue_rescue_with_scaffold_pairing(monkeypatch):
    monkeypatch.setattr(analysis, "REPS", 20)
    stimuli = make_stimuli()
    outcomes = []
    for judge in analysis.JUDGES:
        for item in stimuli:
            for budget in BUDGETS:
                correct = item["position"] == "early" or budget == 12
                label = item["gold"] if correct else 1 - item["gold"]
                if judge == "laya":
                    label = ("positive" if label else "negative") if correct else "unclear"
                outcomes.append(
                    {
                        "judge": judge,
                        "id": item["id"],
                        "cluster_id": item["cluster_id"],
                        "gold": item["gold"],
                        "polarity": item["polarity"],
                        "form": item["form"],
                        "position": item["position"],
                        "budget": budget,
                        "label": label,
                    }
                )

    report = analysis.analyze(stimuli, outcomes)
    assert report["primary_late_minus_early_difference_in_differences"]["estimate"] == 1
    assert all(row["estimate"] == 1 for row in report["primary_by_binary_judge"].values())
    assert report["capability_check_passed"]
