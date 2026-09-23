"""Tests for the score-only correction and the fresh replication dataset."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


transport = load_script("score_transport")
fresh = load_script("fresh_template_replication")


def test_offsets_recover_known_nuisance_shift_without_labels():
    queries = np.repeat(np.arange(3), 4)
    source = np.tile([-2.0, -1.0, 1.0, 2.0], 3)
    shift = np.array([10.0, -4.0, 7.0])
    target = source + shift[queries]
    correction = transport.estimate_offsets(source, queries, target, queries)
    np.testing.assert_allclose(correction["query_offset"], shift)
    np.testing.assert_allclose(target - correction["query_offset"][queries], source)
    np.testing.assert_allclose(correction["none"], 0)


def test_weighted_population_changes_fitted_offset():
    queries = np.repeat(np.arange(3), 2)
    scores = np.tile([-1.0, 1.0], 3)
    weights = np.tile([0.2, 1.8], 3)
    correction = transport.estimate_offsets(scores, queries, scores, queries, weights)
    np.testing.assert_allclose(correction["query_offset"], 0.8)


def test_fresh_formats_are_balanced_and_disjoint():
    rows = fresh.fresh_rows()
    assert len(rows) == 2304
    assert len({r["id"] for r in rows}) == 2304
    for style in fresh.FORMATS:
        for split in ("adaptation", "evaluation"):
            subset = [r for r in rows if r["style"] == style and r["split"] == split]
            assert len(subset) == 288
            assert len({r["group"] for r in subset}) == 6
            assert sum(r["labels"][r["active"]] for r in subset) == 144
        a = {r["group"] for r in rows if r["style"] == style and r["split"] == "adaptation"}
        b = {r["group"] for r in rows if r["style"] == style and r["split"] == "evaluation"}
        assert not a & b
