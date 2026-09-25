import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_tripr_prefix_threshold_034 as analysis
from analyze_tripr_prefix_threshold_034 import correct, paired_bootstrap
from run_tripr_prefix_threshold_034 import generated_content


def test_eos_is_excluded_from_tripr_completion_tokens():
    assert generated_content([4, 5, 99, 0], eos_id=99) == ([4, 5], True)
    assert generated_content([4, 5, 6], eos_id=99) == ([4, 5, 6], False)


def test_tripr_accuracy_accepts_binary_and_named_labels():
    assert correct(1, 1)
    assert correct(0, 0)
    assert correct("positive", 1)
    assert correct("negative", 0)
    assert not correct(None, 1)


def test_sentence_bootstrap_preserves_clustered_paired_difference():
    rows = [
        {"sentence_id": "s1", "delta": 1},
        {"sentence_id": "s1", "delta": 1},
        {"sentence_id": "s2", "delta": 1},
    ]
    result = paired_bootstrap(rows, lambda row: row["delta"], reps=100, seed=3)
    assert result["estimate"] == 1
    assert result["sentence_cluster_bootstrap_95_ci"] == [1, 1]
    assert result["n_source_sentence_clusters"] == 2


def test_target_agreement_breakdown_uses_target_rows_not_pooled_rows(monkeypatch):
    monkeypatch.setattr(analysis, "REPS", 20)
    rows = []
    for model in analysis.MODELS:
        for judge in analysis.JUDGES:
            for budget in analysis.BUDGETS:
                if judge == "laya":
                    label = "clear"
                else:
                    label = int(judge == "qwen2.5-3b")
                rows.append(
                    {
                        "model": model,
                        "judge": judge,
                        "id": "item-1",
                        "sentence_id": f"sentence-{model}",
                        "budget": budget,
                        "label": label,
                        "gold": 1,
                        "actual_tokens": budget,
                    }
                )

    metrics = analysis.series_metrics(rows)
    for model in analysis.MODELS:
        target = metrics["by_target"][model]["agreement_by_budget"]["8"]
        assert target["n"] == 1
        assert target["estimate"] == 0
