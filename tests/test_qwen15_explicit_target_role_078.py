"""Protocol tests for Experiment 078."""

import analyze_qwen15_explicit_target_role_078 as analysis
import numpy as np
import run_qwen15_explicit_target_role_078 as runner


def test_exp078_mirrors_exp077_prompt_design_on_pinned_qwen15():
    jobs = runner.build_jobs()
    assert len(jobs) == 192
    assert runner.MODEL_CONFIG["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert {job["study"] for job in jobs} == {"explicit_named_role"}
    assert {job["anchor_location"] for job in jobs} == {"prior_assistant", "prior_user"}
    assert all(job["messages"][-1]["content"].find(
        f'key "{job["target_axis"]}"') >= 0 for job in jobs)


def test_exp078_expected_score_normalizes_the_registered_support():
    row = {"canonical_logprobs": np.log(np.full(81, 0.25 / 81)).tolist()}
    mean, mass = analysis._score(row)
    assert abs(mean - 5.0) < 1e-12
    assert abs(mass - 0.25) < 1e-12
