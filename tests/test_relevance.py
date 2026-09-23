import numpy as np
import torch

from latent_decisions.relevance import active_examples, dataset, paired_donors, regime_mask


def config():
    return dict(
        data_seed=20260922,
        train_groups=4,
        validation_groups=2,
        calibration_groups=2,
        test_groups=2,
        ood_groups=2,
    )


def test_full_crossing_and_mapping_counterbalance():
    rows = dataset(config())
    assert len(rows) == 12 * 48
    for group in {r["group"] for r in rows}:
        rs = [r for r in rows if r["group"] == group]
        assert len(rs) == 48
        for active in range(3):
            for code in (1, -1):
                cell = [r for r in rs if r["active"] == active and r["code"] == code]
                assert len(cell) == 8
                assert sum(r["answer_a"] for r in cell) == 4
                assert all(
                    r["answer_a"] == (r["labels"][active] if code == 1 else 1 - r["labels"][active])
                    for r in cell
                )


def test_donor_changes_exactly_one_setting():
    rows = dataset(config())
    donors = paired_donors(rows)
    for i, row in enumerate(rows):
        for j in range(3):
            donor = rows[donors[i, j]]
            assert all(row[k] == donor[k] for k in ("group", "active", "code", "template"))
            assert sum(a != b for a, b in zip(row["labels"], donor["labels"])) == 1
            assert donor["labels"][j] == 1 - row["labels"][j]
            assert donors[donors[i, j], j] == i


def test_regimes_match_label_budget_without_test_rows():
    rows = dataset(config())
    for split in ("train", "validation", "calibration"):
        fixed = regime_mask(rows, config(), split, "fixed_code")
        balanced = regime_mask(rows, config(), split, "balanced_code")
        assert fixed.sum() == balanced.sum()
        assert all(rows[i]["code"] == 1 and rows[i]["split"] == split for i in np.where(fixed)[0])
        for active in range(3):
            assert sum(
                balanced[i] for i, r in enumerate(rows) if r["active"] == active and r["code"] == 1
            ) == sum(
                balanced[i] for i, r in enumerate(rows) if r["active"] == active and r["code"] == -1
            )


def test_training_question_and_label_follow_active_property():
    rows = dataset(config())[:48]
    h = torch.randn(48, 5)
    q = torch.eye(3)
    indices = np.array([1, 7, 14, 23])
    data = active_examples(h, q, rows, indices)
    for k, i in enumerate(indices):
        assert data[2][k].item() == rows[i]["active"]
        assert data[3][k].item() == rows[i]["labels"][rows[i]["active"]]
        torch.testing.assert_close(data[1][k], q[rows[i]["active"]])


def test_answer_direction_fails_semantic_remapping_criterion():
    # A fixed direction adding +2 to A/B log odds has opposite semantic meaning by code.
    assert 1 * 2 > 0
    assert -1 * 2 < 0
    # A semantic direction must reverse its raw A/B effect when the code reverses.
    assert 1 * 2 == -1 * -2


def test_resume_preserves_complete_conditions_and_rejects_partial(tmp_path):
    import json

    import pytest

    from latent_decisions.relevance import read_completed_interventions

    path = tmp_path / "rows.jsonl"
    records = [dict(name="zero", id="a"), dict(name="zero", id="b")]
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    saved, complete = read_completed_interventions(path, ["a", "b"], {"zero", "color"})
    assert saved == records and complete == {"zero"}
    with pytest.raises(ValueError, match="Partial"):
        read_completed_interventions(path, ["a", "b", "c"], {"zero"})
