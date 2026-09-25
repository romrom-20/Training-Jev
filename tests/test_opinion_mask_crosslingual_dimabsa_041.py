import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import analyze_opinion_mask_crosslingual_dimabsa_041 as analysis
from run_opinion_mask_crosslingual_dimabsa_041 import (
    CONDITIONS,
    LANGS,
    make_jobs,
    mask_opinions,
    parse_va,
    select_cases,
)


def _parallel_rows(case_id, va="7.00#6.00"):
    return {
        lang: {
            "ID": case_id,
            "Text": f"{lang} battery is weak but screen feels bright.",
            "Triplet": [
                {"Aspect": f"{lang} battery", "Opinion": "weak", "VA": va},
                {"Aspect": f"{lang} screen", "Opinion": "bright", "VA": "7.00#6.00"},
            ],
        }
        for lang in LANGS
    }


def test_masking_removes_each_unique_annotated_opinion_and_preserves_aspect():
    row = {
        "ID": "x",
        "Text": "Battery life is poor, but the screen is bright.",
        "Triplet": [
            {"Aspect": "Battery life", "Opinion": "poor", "VA": "2#4"},
            {"Aspect": "screen", "Opinion": "bright", "VA": "8#6"},
            {"Aspect": "battery", "Opinion": "poor", "VA": "3#4"},
        ],
    }
    assert mask_opinions(row) == "Battery life is [MASKED], but the screen is [MASKED]."


def test_selects_stratified_aligned_cases_and_emits_all_four_conditions():
    data = {lang: [] for lang in LANGS}
    for i in range(120):
        valence = 3.0 if i < 55 else 5.0 if i < 65 else 7.0
        rowset = _parallel_rows(f"id-{i:03d}", f"{valence:.2f}#6.00")
        for lang in LANGS:
            data[lang].append(rowset[lang])
    cases = select_cases(data)
    assert len(cases) == 120
    assert {bucket: sum(case["bucket"] == bucket for case in cases) for bucket in ("neg", "neu", "pos")} == {
        "neg": 55,
        "neu": 10,
        "pos": 55,
    }
    jobs = make_jobs(cases[:1])
    assert len(jobs) == len(LANGS) * len(CONDITIONS)
    assert {job["condition"] for job in jobs} == set(CONDITIONS)
    assert all("Review text:" in job["prompt"] for job in jobs)


def test_rejects_repeated_opinion_string_without_span_offsets():
    row = {
        "ID": "ambiguous",
        "Text": "The food was good; the service was good.",
        "Triplet": [
            {"Aspect": "food", "Opinion": "good", "VA": "7#6"},
            {"Aspect": "service", "Opinion": "good", "VA": "7#6"},
        ],
    }
    with pytest.raises(ValueError, match="not unique"):
        mask_opinions(row)


def test_rejects_overlapping_opinion_annotations():
    row = {
        "ID": "overlap",
        "Text": "The battery is very poor.",
        "Triplet": [
            {"Aspect": "battery", "Opinion": "very poor", "VA": "2#4"},
            {"Aspect": "battery", "Opinion": "poor", "VA": "2#4"},
        ],
    }
    with pytest.raises(ValueError, match="Overlapping"):
        mask_opinions(row)


def test_parses_only_valid_json_va_pairs():
    assert parse_va('{"valence": 6.25, "arousal": 4}') == [6.25, 4.0]
    assert parse_va('```json\n{"valence": 6.25, "arousal": 4}\n```') == [6.25, 4.0]
    assert parse_va("valence is 6 and arousal is 4") is None
    assert parse_va('Here is the answer: {"valence": 6, "arousal": 4}') is None
    assert parse_va('```json\n{"valence": 6, "arousal": 4}\n``` extra') is None
    assert parse_va('{"valence": 10, "arousal": 4}') is None
    assert parse_va('{"valence": null, "arousal": 4}') is None


def test_cluster_bootstrap_keeps_language_versions_together():
    predictions = {}
    for case_id, gold in (("one", [7.0, 6.0]), ("two", [3.0, 4.0])):
        for lang in LANGS:
            for condition in CONDITIONS:
                predicted = gold if condition == "opinion_masked" else [gold[0] + 1, gold[1] + 1]
                predictions[(case_id, lang, condition)] = {
                    "case_id": case_id,
                    "lang": lang,
                    "condition": condition,
                    "gold": gold,
                    "prediction": predicted,
                }
    result = analysis.paired_bootstrap(
        predictions, ["one", "two"], "aspect_only", "opinion_masked", n_boot=50
    )
    assert result["estimate"] == 1.0
    assert result["n_clusters"] == 2


def test_invalid_output_gate_returns_coverage_only_without_scoring():
    predictions = {}
    for i in range(120):
        case_id = f"case-{i:03}"
        for lang in LANGS:
            for condition in CONDITIONS:
                invalid = condition == "aspect_opinion" and i < 29
                predictions[(case_id, lang, condition)] = {
                    "case_id": case_id,
                    "lang": lang,
                    "condition": condition,
                    "gold": [7.0, 6.0],
                    "prediction": None if invalid else [7.0, 6.0],
                }
    summary = analysis.analyze(predictions)
    assert summary["status"] == "protocol_execution_failure"
    assert summary["invalid_by_condition"]["aspect_opinion"] == 29 * len(LANGS)
    assert summary["score_analysis_performed"] is False
    assert "primary" not in summary
