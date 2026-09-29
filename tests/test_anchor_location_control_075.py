from __future__ import annotations

from pathlib import Path

import run_anchor_location_control_075 as runner


def test_exp075_design_matches_sample_and_two_location_factorial() -> None:
    cases, metadata = runner.select_cases(Path(".context/dimabsa"))
    jobs = runner.build_jobs(cases)
    assert len(cases) == 64
    assert len(jobs) == 512
    assert metadata["recipient_ids_sha256"] == (
        "a14b6300c88940b3ccb6ac6b1b78c78b1cb5e202c4ce945ea3fe8220d36c7ddf"
    )
    assert len({job["user_prompt_sha256"] for job in jobs
                if job["anchor_location"] == "assistant_prefix"}) == 64
    assert len({job["user_prompt_sha256"] for job in jobs
                if job["anchor_location"] == "user_message"}) == 256
    assert sum(job["anchor_location"] == "assistant_prefix" for job in jobs) == 256
    assert sum(job["anchor_location"] == "user_message" for job in jobs) == 256


def test_target_parser_reads_both_registered_output_schemas() -> None:
    assistant = {
        "anchor_location": "assistant_prefix", "anchor_axis": "valence",
        "target_axis": "arousal", "forced_value": 8.0,
        "assistant_prefix": '{"valence": 8.0, "arousal": ',
    }
    score, pair = runner.parse_target(assistant, "4.3}")
    assert score == 4.3
    assert pair == [8.0, 4.3]

    user = {
        "anchor_location": "user_message", "anchor_axis": "valence",
        "target_axis": "arousal", "forced_value": 8.0,
        "assistant_prefix": '{"arousal": ',
    }
    score, pair = runner.parse_target(user, "4.3}")
    assert score == 4.3
    assert pair is None
    assert runner.parse_target(user, '"4.3"}')[0] is None
