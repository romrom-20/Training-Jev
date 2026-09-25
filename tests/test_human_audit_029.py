import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from natural_aspect_selectivity import DATA, load_stimuli
from prepare_human_audit_029 import QUOTAS, select_sample


def test_frozen_human_audit_sample_is_reproducible_and_sentence_unique():
    stimuli = load_stimuli(DATA)
    first = select_sample(stimuli)
    second = select_sample(stimuli)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len(first) == 20
    assert len({row["sentence_id"] for row in first}) == 20
    assert {
        (category, label): sum(
            row["category"] == category and row["label"] == label for row in first
        )
        for category, label in QUOTAS
    } == QUOTAS
    assert sum(row["label"] for row in first) == 10
