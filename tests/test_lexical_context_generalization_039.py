import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_lexical_context_generalization_039 as analysis
from run_lexical_context_generalization_039 import build_features


def test_feature_variants_mask_target_without_writing_source_text():
    item = {
        "_tokens": ["the", "service", "was", "not", "very", "good"],
        "_aspect": "service",
        "aspect_token_indices": [1],
        "opinion_token_indices": [5],
    }
    features = build_features(item)
    assert features["aspect_only"] == "aspect service"
    assert features["masked_context"] == "review prefix the ASPECT was not very"
    assert features["context_plus_aspect"] == (
        "aspect service review prefix the service was not very"
    )
    assert "good" not in features["masked_context"]


def test_analysis_scores_paired_context_advantage(monkeypatch):
    monkeypatch.setattr(analysis, "REPS", 20)
    stimuli = [
        {
            "id": f"d-{index}",
            "cluster_id": f"d-{index}",
            "dataset": "14res" if index < 4 else "14lap",
            "polarity": "positive" if index % 2 else "negative",
        }
        for index in range(8)
    ]
    predictions = []
    for stimulus in stimuli:
        gold = stimulus["polarity"]
        opposite = "negative" if gold == "positive" else "positive"
        predictions.append(
            {
                "id": stimulus["id"],
                "dataset": stimulus["dataset"],
                "gold": gold,
                "aspect_only_label": opposite,
                "masked_context_label": gold,
                "context_plus_aspect_label": gold,
            }
        )
    result = analysis.analyze(stimuli, predictions)
    assert result["paired_differences"]["masked_context_minus_aspect_only"]["estimate"] == 1.0
    assert result["preregistered_diagnostic_gate_passed"]
    with pytest.raises(ValueError, match="Duplicate prediction ID"):
        analysis.analyze(stimuli, predictions + [predictions[0]])
