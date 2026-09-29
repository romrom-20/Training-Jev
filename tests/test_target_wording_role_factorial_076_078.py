"""Tests for the post-hoc 076–078 factorial aggregation."""

import analyze_target_wording_role_factorial_076_078 as analysis


def test_three_runs_form_the_full_posthoc_factorial_without_filling_bad_cells(tmp_path):
    rows, hashes = analysis.load_rows()
    assert len(rows) == 768
    assert set(hashes) == {"generic", "explicit_0.5b", "explicit_1.5b"}

    summary = analysis.analyze(output_dir=tmp_path)
    assert summary["preregistered"] is False
    assert summary["n_recipients"] == 24
    qwen05 = summary["results"]["qwen-0.5b"]["arousal_first"]
    assert qwen05["assistant_minus_user_shift"]["generic"] is None
    assert qwen05["wording_by_role_difference_in_differences"] is None
    assert qwen05["assistant_minus_user_shift"]["explicit"] is not None
