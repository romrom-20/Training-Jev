"""Protocol checks for the explicit target-name follow-up."""

import analyze_explicit_target_role_control_077 as analysis
import numpy as np
import run_explicit_target_role_control_077 as runner


def test_explicit_target_prompt_names_key_and_keeps_role_final_request_equal():
    jobs = runner.build_jobs()
    assert len(jobs) == 192
    by = {(j["case_id"], j["order"], j["forced_value"], j["anchor_location"]): j
          for j in jobs}
    for case_id in {job["case_id"] for job in jobs}:
        for order in runner.exp076.ORDERS:
            for value in (2.0, 8.0):
                assistant = by[(case_id, order, value, "prior_assistant")]
                user = by[(case_id, order, value, "prior_user")]
                assert assistant["messages"][-1]["content"] == user["messages"][-1]["content"]
                assert f'key "{assistant["target_axis"]}"' in assistant["messages"][-1]["content"]
                assert assistant["target_axis"] == user["target_axis"]


def test_probability_support_and_bootstrap_are_well_formed():
    row = {"canonical_logprobs": np.log(np.full(81, 0.4 / 81)).tolist()}
    mean, mass = analysis._score(row)
    assert abs(mean - 5.0) < 1e-12
    assert abs(mass - 0.4) < 1e-12
