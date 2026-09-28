import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_context_swap_donor_robustness_054 as shared
import analyze_polarity_matched_context_swap_056 as analyzer
import run_polarity_matched_context_swap_056 as runner


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


def _case(case_id, valence, text="sample review"):
    return {
        "case_id": case_id,
        "gold": [valence, 5.0],
        "target_index": 0,
        "row": {
            "ID": case_id,
            "Text": text,
            "Triplet": [{"Aspect": "battery", "Opinion": "NULL", "VA": f"{valence}#5"}],
        },
    }


def test_donor_permutations_preserve_gold_polarity_and_are_derangements():
    cases = ([_case(f"neg-{i:02}", 3.0, "n" * (i + 5)) for i in range(30)]
             + [_case(f"pos-{i:02}", 7.0, "p" * (i + 5)) for i in range(30)])
    mappings, lengths = runner.donor_maps(cases, _Tokenizer())
    by_id = {case["case_id"]: case for case in cases}
    assert set(mappings) == {1, 2, 3}
    for mapping in mappings.values():
        assert set(mapping) == set(mapping.values()) == set(by_id)
        assert all(recipient != donor for recipient, donor in mapping.items())
        assert all(runner.exp055.bucket(by_id[r]) == runner.exp055.bucket(by_id[d])
                   for r, d in mapping.items())
    assert set(lengths) == set(by_id)


def _synthetic():
    swapped = {}
    matched = {}
    for case_id in ("fresh-1", "fresh-2"):
        for decoder in shared.DECODERS:
            matched[(case_id, "opinion_masked", decoder)] = {
                "case_id": case_id, "condition": "opinion_masked", "decoder": decoder,
                "gold": [5.0, 5.0], "prediction": [5.0, 5.0],
            }
        for permutation in shared.PERMUTATIONS:
            for decoder, offset in (("finite_grid", 2.0), ("free_greedy", 1.0)):
                swapped[(case_id, permutation, "swapped_context", decoder)] = {
                    "case_id": case_id, "permutation": permutation,
                    "condition": "polarity_matched_context", "decoder": decoder,
                    "gold": [5.0, 5.0], "prediction": [5.0 + offset, 5.0 + offset],
                }
    return swapped, matched


def test_analyzer_labels_the_registered_polarity_control(monkeypatch):
    monkeypatch.setattr(shared, "EXPECTED_IDS", 2)
    monkeypatch.setattr(shared, "BOOTSTRAPS", 30)
    swapped, matched = _synthetic()
    normalized = {
        (case_id, permutation, "swapped_context", decoder): row
        for (case_id, permutation, _condition, decoder), row in swapped.items()
    }
    summary = analyzer.analyze(normalized, matched)
    assert summary["experiment"] == "056-polarity-matched-context-swap"
    assert summary["analysis_seed"] == 20260956
    assert summary["control"]["donors_match_recipient_gold_valence_polarity"] is True
    assert summary["mean_primary_interaction"]["estimate"] == pytest.approx(1.0)


def test_loader_requires_only_polarity_matched_condition(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, "EXPECTED_IDS", 2)
    rows = []
    for case_id in ("a", "b"):
        for permutation in shared.PERMUTATIONS:
            for decoder in shared.DECODERS:
                rows.append({
                    "case_id": case_id, "permutation": permutation,
                    "condition": "polarity_matched_context", "decoder": decoder,
                    "prediction": [5.0, 5.0],
                })
    path = tmp_path / "outputs.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    loaded = analyzer.load_rows(path)
    assert len(loaded) == 2 * len(shared.PERMUTATIONS) * len(shared.DECODERS)
