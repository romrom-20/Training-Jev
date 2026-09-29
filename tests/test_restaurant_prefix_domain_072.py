from __future__ import annotations

from pathlib import Path

import run_restaurant_prefix_domain_072 as runner


def test_frozen_restaurant_sample_and_complete_factorial() -> None:
    cases, metadata = runner.select_cases(Path(".context/dimabsa"))
    jobs = runner.build_jobs(cases)
    assert metadata["recipient_ids_sha256"] == (
        "f15f8acb5cf3cc80bf9eea082033fd652b336fcfe3732d8294c831ddd8ae4343"
    )
    assert metadata["prior_exp051_full_split_exposure"] is True
    assert len(cases) == 64
    assert len(jobs) == 256
    assert metadata["polarity_counts"] == {"negative": 32, "positive": 32}
    assert len({case["case_id"] for case in cases}) == 64
    assert len({job["user_prompt_sha256"] for job in jobs}) == 64


def test_models_and_context_count_are_frozen() -> None:
    assert set(runner.MODEL_CONFIGS) == {"qwen-1.5b", "qwen-3b", "smollm2-1.7b"}
    assert len(runner.MODEL_CONFIGS) * 256 == 768
