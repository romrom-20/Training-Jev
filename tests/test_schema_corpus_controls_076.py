"""Protocol and parser checks for Experiment 076."""

from pathlib import Path

import run_schema_corpus_controls_076 as exp


def _cohorts():
    role, _ = exp.select_role_cases(Path(".context/dimabsa"))
    aste, _ = exp.select_aste_cases(Path(".context/aste14res/14lap_test_triplets.txt"))
    return role, aste


def test_frozen_cohorts_are_balanced_and_distinct():
    role, aste = _cohorts()
    assert len(role) == len(aste) == 24
    assert sum(case["gold"][0] < 4.5 for case in role) == 12
    assert sum(case["gold"][0] > 5.5 for case in role) == 12
    assert {case["case_id"] for case in role}.isdisjoint(
        {case["case_id"] for case in aste}
    )


def test_job_counts_and_role_control_final_request_are_matched():
    role, aste = _cohorts()
    jobs = exp.build_jobs(role, aste)
    assert len(jobs) == 288
    role_jobs = [job for job in jobs if job["study"] == "role_control"]
    assert len(role_jobs) == 192
    by = {(j["case_id"], j["order"], j["forced_value"], j["anchor_location"]): j
          for j in role_jobs}
    case_id = role[0]["case_id"]
    for order in exp.ORDERS:
        for value in exp.FORCED_VALUES:
            assistant = by[(case_id, order, value, "prior_assistant")]
            user = by[(case_id, order, value, "prior_user")]
            assert assistant["messages"][-1] == user["messages"][-1]
            assert assistant["assistant_prefix"] == user["assistant_prefix"]
            assert assistant["messages"][-2]["content"] == (
                f'{{"{assistant["anchor_axis"]}": {value:.1f}}}'
            )
            assert assistant["messages"][-2]["role"] == "assistant"
            assert f'{{"{user["anchor_axis"]}": {value:.1f}}}' in user["messages"][1]["content"]


def test_aste_masks_all_opinion_spans_and_parses_full_pair():
    role, aste = _cohorts()
    masked = exp.mask_aste_opinions(aste[0])
    assert "[MASKED]" in masked
    assert len([token for token in masked.split() if token == "[MASKED]"]) >= 1
    job = next(j for j in exp.build_jobs(role, aste)
               if j["order"] == "valence_first" and j["forced_value"] == 2.0)
    assert exp.parse_target(job, '7.2}') == 7.2
    assert exp.parse_target(job, '6.0, "arousal": 7.2}') is None


def test_role_control_parser_requires_exact_target_key():
    job = {
        "study": "role_control", "target_axis": "arousal",
        "assistant_prefix": '{"arousal": ',
    }
    assert exp.parse_target(job, "5.4}") == 5.4
    assert exp.parse_target(job, '5.4, "valence": 8.0}') is None
