import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_trabl_prior_opinion_interference_040 as analysis
from trabl_prior_opinion_interference_040 import make_jobs, public_stimuli, select_pairs


def _rows(text, labels_1, labels_2=None, source_id="private-review"):
    return [
        {
            "line_index": 10,
            "source_id": source_id,
            "annotator": 1,
            "text": text,
            "labels": labels_1,
        },
        {
            "line_index": 11,
            "source_id": source_id,
            "annotator": 2,
            "text": text,
            "labels": labels_2 if labels_2 is not None else labels_1,
        },
    ]


def _tuple(aspect, category, opinion, sentiment, snippet):
    return {
        "aspect term": aspect,
        "aspect category": category,
        "opinion span": opinion,
        "sentiment": sentiment,
        "snippet": snippet,
    }


def test_selects_opposite_polarity_targets_in_opinion_order_and_builds_intervention():
    text = "Room was awful but staff were excellent."
    labels = [
        _tuple("Room", "room", "awful", "negative", "Room was awful"),
        _tuple("staff", "service", "excellent", "positive", "staff were excellent"),
    ]
    items = select_pairs(_rows(text, labels))
    assert len(items) == 2
    assert [item["role"] for item in items] == ["earlier", "later"]
    assert [item["gold"] for item in items] == ["negative", "positive"]
    assert items[1]["prior_gold"] == "negative"
    jobs = make_jobs(items)
    later = {job["condition"]: job for job in jobs if job["role"] == "later"}
    assert later["aspect_only"]["visible_text"] == ""
    assert later["natural_prefix"]["visible_text"] == "Room was awful but staff were"
    assert later["prior_opinion_deleted"]["visible_text"] == "Room was  but staff were"
    assert later["opinion_visible"]["visible_text"] == "Room was awful but staff were excellent"


def test_requires_exact_annotator_agreement_and_distinct_aspects():
    text = "Room was awful but staff were excellent."
    positive = _tuple("staff", "service", "excellent", "positive", "staff were excellent")
    negative = _tuple("Room", "room", "awful", "negative", "Room was awful")
    disagreeing = [negative, positive | {"sentiment": "negative"}]
    assert select_pairs(_rows(text, [negative, positive], disagreeing)) == []
    same_aspect = [
        negative,
        _tuple("Room", "room", "excellent", "positive", "Room was awful but staff were excellent"),
    ]
    assert select_pairs(_rows(text, same_aspect)) == []


def test_rejects_ambiguous_target_spans_and_nonvisible_opinion():
    text = "Room was awful; the room felt awful."
    labels = [
        _tuple("Room", "room", "awful", "negative", "Room was awful"),
        _tuple("staff", "service", "excellent", "positive", "staff were excellent"),
    ]
    assert select_pairs(_rows(text, labels)) == []


def test_public_stimuli_never_include_review_text_aspects_or_source_ids():
    text = "Room was awful but staff were excellent."
    labels = [
        _tuple("Room", "room", "awful", "negative", "Room was awful"),
        _tuple("staff", "service", "excellent", "positive", "staff were excellent"),
    ]
    public = public_stimuli(select_pairs(_rows(text, labels)))[0]
    assert "_text" not in public
    assert "source_id" not in public
    assert "aspect" not in public
    assert "awful" not in repr(public)
    assert "excellent" not in repr(public)


def test_primary_analysis_is_natural_copy_rate_minus_deleted_copy_rate():
    analysis.REPS = 100
    stimuli = []
    predictions = []
    for cluster_index in range(2):
        cluster = f"review-{cluster_index}"
        earlier_id, later_id = f"{cluster}-neg", f"{cluster}-pos"
        earlier = {
            "id": earlier_id,
            "cluster_id": cluster,
            "role": "earlier",
            "gold": "negative",
            "prior_gold": None,
            "prior_opinion_span": None,
        }
        later = {
            "id": later_id,
            "cluster_id": cluster,
            "role": "later",
            "gold": "positive",
            "prior_gold": "negative",
            "prior_opinion_span": [3, 8],
        }
        stimuli.extend((earlier, later))
        for item in (earlier, later):
            for condition in ("aspect_only", "natural_prefix", "opinion_visible"):
                for judge, wrapper in (
                    ("laya", "four_way"),
                    ("qwen2.5-3b", "forced_binary"),
                    ("qwen2.5-3b", "abstention"),
                    ("phi3-mini", "forced_binary"),
                    ("phi3-mini", "abstention"),
                ):
                    label = item["gold"]
                    predictions.append(
                        {
                            "id": item["id"],
                            "cluster_id": cluster,
                            "role": item["role"],
                            "gold": item["gold"],
                            "prior_gold": item["prior_gold"],
                            "judge": judge,
                            "wrapper": wrapper,
                            "condition": condition,
                            "label": label,
                        }
                    )
        for judge, wrapper in (
            ("laya", "four_way"),
            ("qwen2.5-3b", "forced_binary"),
            ("qwen2.5-3b", "abstention"),
            ("phi3-mini", "forced_binary"),
            ("phi3-mini", "abstention"),
        ):
            predictions.append(
                {
                    "id": later["id"],
                    "cluster_id": cluster,
                    "role": "later",
                    "gold": "positive",
                    "prior_gold": "negative",
                    "judge": judge,
                    "wrapper": wrapper,
                    "condition": "prior_opinion_deleted",
                    "label": "positive",
                }
            )
        # A natural-prefix error follows the preceding target; deletion fixes it.
        for row in predictions:
            if row["id"] == later_id and row["judge"] == "qwen2.5-3b" and row["wrapper"] == "forced_binary":
                if row["condition"] == "natural_prefix":
                    row["label"] = "negative"

    report = analysis.analyze(stimuli, predictions)
    primary = report["later_target_copy_rate_contrasts"]["qwen2.5-3b/forced_binary"]
    assert primary["natural_copy_rate"] == 1.0
    assert primary["deleted_copy_rate"] == 0.0
    assert primary["estimate"] == 1.0
