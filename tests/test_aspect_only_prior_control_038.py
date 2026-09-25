import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_aspect_only_prior_control_038 as analysis
from run_aspect_only_prior_control_038 import ASPECT_ONLY_TEXT, make_aspect_only_jobs


def test_aspect_only_jobs_keep_frozen_ids_and_remove_review_words():
    selected = [
        {
            "id": "14res-1",
            "cluster_id": "14res-1",
            "dataset": "14res",
            "line_index": 1,
            "aspect_token_indices": [2],
            "opinion_token_indices": [4],
            "polarity": "positive",
            "_aspect": "coffee",
            "_tokens": ["the", "great", "coffee", "was", "amazing"],
        }
    ]
    jobs = make_aspect_only_jobs(selected)
    assert len(jobs) == 1
    assert jobs[0]["id"] == "14res-1"
    assert jobs[0]["aspect"] == "coffee"
    assert jobs[0]["visible_text"] == ASPECT_ONLY_TEXT
    assert jobs[0]["expected_decision"] == "insufficient"
    assert "amazing" not in jobs[0]["visible_text"]


def test_aspect_only_analysis_pairs_with_parent_and_scores_context_gain(monkeypatch):
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
    natural, control = [], []
    for stimulus in stimuli:
        gold = stimulus["polarity"]
        opposite = "negative" if gold == "positive" else "positive"
        for judge in ("qwen2.5-3b", "phi3-mini"):
            for wrapper, label in (("forced_binary", gold), ("abstention", "insufficient")):
                natural.append(
                    {
                        "judge": judge,
                        "wrapper": wrapper,
                        "id": stimulus["id"],
                        "condition": "before_opinion",
                        "cluster_id": stimulus["cluster_id"],
                        "polarity": gold,
                        "label": label,
                    }
                )
                control.append(
                    {
                        "judge": judge,
                        "wrapper": wrapper,
                        "id": stimulus["id"],
                        "condition": "aspect_only",
                        "cluster_id": stimulus["cluster_id"],
                        "polarity": gold,
                        "label": opposite if wrapper == "forced_binary" else "insufficient",
                    }
                )
        control.append(
            {
                "judge": "laya",
                "wrapper": "four_way",
                "id": stimulus["id"],
                "condition": "aspect_only",
                "cluster_id": stimulus["cluster_id"],
                "polarity": gold,
                "label": "unclear",
            }
        )

    result = analysis.analyze(stimuli, natural, control)
    assert result["primary_pooled_context_gain"]["estimate"] == 1.0
    assert result["preregistered_diagnostic_gate_passed"]
    assert result["laya_aspect_only"]["unclear_rate"]["estimate"] == 1.0
