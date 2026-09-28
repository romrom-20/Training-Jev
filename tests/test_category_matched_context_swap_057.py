import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_category_matched_context_swap_057 as analyzer
import analyze_context_swap_donor_robustness_054 as shared
import run_category_matched_context_swap_057 as runner


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


def _case(case_id, valence, category, text):
    return {
        "case_id": case_id,
        "gold": [valence, 5.0],
        "target_index": 0,
        "category": category,
        "row": {
            "ID": case_id,
            "Text": text,
            "Triplet": [{"Aspect": "battery", "Opinion": "NULL", "VA": f"{valence}#5"}],
        },
    }


def test_offsets_are_unique_nonzero_cyclic_shifts():
    assert runner.offsets_for_group(2) == [1]
    assert runner.offsets_for_group(3) == [1, 2]
    assert runner.offsets_for_group(4) == [1, 3, 2]


def test_category_maps_preserve_polarity_and_category():
    cases = []
    for category in ("BATTERY#GENERAL", "DISPLAY#QUALITY"):
        cases.extend(_case(f"neg-{category}-{i}", 3.0, category, "n" * (i + 5))
                     for i in range(3))
        cases.extend(_case(f"pos-{category}-{i}", 7.0, category, "p" * (i + 5))
                     for i in range(3))
    mappings, meta = runner.donor_maps(cases, _Tokenizer())
    by_id = {case["case_id"]: case for case in cases}
    assert meta["n_unique_assignment_maps"] >= 2
    for mapping in mappings.values():
        assert set(mapping) == set(mapping.values()) == set(by_id)
        for recipient_id, donor_id in mapping.items():
            assert recipient_id != donor_id
            recipient, donor = by_id[recipient_id], by_id[donor_id]
            assert runner.exp055.bucket(recipient) == runner.exp055.bucket(donor)
            assert recipient["category"] == donor["category"]


def test_attach_categories_requires_exactly_one_official_category():
    case = _case("r1", 3.0, "unused", "review")
    case["row"]["Triplet"][0].update({"Aspect": "screen", "Opinion": "dark", "VA": "3#5"})
    task3 = [{"Text": "review", "Quadruplet": [
        {"Aspect": "screen", "Opinion": "dark", "VA": "3#5", "Category": "DISPLAY#QUALITY"}
    ]}]
    result = runner.attach_categories([case], task3)
    assert result[0]["category"] == "DISPLAY#QUALITY"


def test_analyzer_applies_shared_estimator_and_adds_control_metadata(monkeypatch):
    monkeypatch.setattr(shared, "EXPECTED_IDS", 2)
    monkeypatch.setattr(shared, "BOOTSTRAPS", 30)
    swapped = {}
    matched = {}
    for case_id in ("a", "b"):
        for decoder in shared.DECODERS:
            matched[(case_id, "opinion_masked", decoder)] = {
                "case_id": case_id, "condition": "opinion_masked", "decoder": decoder,
                "gold": [5.0, 5.0], "prediction": [5.0, 5.0],
            }
        for permutation in shared.PERMUTATIONS:
            for decoder, offset in (("finite_grid", 2.0), ("free_greedy", 1.0)):
                swapped[(case_id, permutation, "swapped_context", decoder)] = {
                    "case_id": case_id, "permutation": permutation,
                    "condition": "category_polarity_matched_context", "decoder": decoder,
                    "gold": [5.0, 5.0], "prediction": [5.0 + offset, 5.0 + offset],
                }
    summary = analyzer.analyze(swapped, matched, {
        "n_unique_assignment_maps": 2,
        "protocol_sha256": "protocol",
        "source_revision": "revision",
        "source_sha256": "source",
        "parent_output_sha256": "parent",
        "model": "model",
        "model_revision": "model-revision",
        "device": "mps",
        "donor_assignment_sha256": {1: "a", 2: "a", 3: "b"},
        "output_sha256": "output",
        "generation_seconds_this_process_only": 1.0,
    })
    assert summary["experiment"] == "057-category-matched-context-swap"
    assert summary["analysis_seed"] == 20260957
    assert summary["control"]["donors_match_recipient_gold_valence_polarity"] is True
    assert summary["permutation_sensitivity"]["n_permutations"] == 2
    assert summary["mean_primary_interaction"]["estimate"] == pytest.approx(1.0)


def test_loader_checks_condition_and_unique_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, "EXPECTED_IDS", 2)
    rows = []
    for case_id in ("a", "b"):
        for permutation in shared.PERMUTATIONS:
            for decoder in shared.DECODERS:
                rows.append({
                    "case_id": case_id, "permutation": permutation,
                    "condition": "category_polarity_matched_context", "decoder": decoder,
                    "prediction": [5.0, 5.0],
                })
    path = tmp_path / "outputs.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert len(analyzer.load_rows(path)) == len(rows)


def test_post_hoc_same_recipient_contrast_pairs_the_two_donor_controls(monkeypatch):
    monkeypatch.setattr(shared, "EXPECTED_IDS", 2)
    category = {}
    polarity = {}
    matched = {}
    for case_id in ("a", "b"):
        for decoder in shared.DECODERS:
            matched[(case_id, "opinion_masked", decoder)] = {
                "case_id": case_id, "condition": "opinion_masked", "decoder": decoder,
                "gold": [5.0, 5.0], "prediction": [5.0, 5.0],
            }
        for permutation in shared.PERMUTATIONS:
            for decoder in shared.DECODERS:
                category_offset = 2.0 if decoder == "finite_grid" else 1.0
                polarity_offset = 3.0 if decoder == "finite_grid" else 1.0
                for target, offset, condition in (
                    (category, category_offset, "category_polarity_matched_context"),
                    (polarity, polarity_offset, "polarity_matched_context"),
                ):
                    target[(case_id, permutation, "swapped_context", decoder)] = {
                        "case_id": case_id, "condition": condition, "permutation": permutation,
                        "decoder": decoder, "gold": [5.0, 5.0],
                        "prediction": [5.0 + offset, 5.0 + offset],
                    }
    result = analyzer.exploratory_same_recipient_contrast(
        category, polarity, matched, bootstrap_replicates=40
    )
    assert result["status"] == "post_hoc_exploratory"
    assert result["category_matched_estimate"] == pytest.approx(1.0)
    assert result["polarity_only_estimate_same_recipients"] == pytest.approx(2.0)
    assert result["difference"] == pytest.approx(-1.0)
    assert result["n_recipient_ids"] == 2
