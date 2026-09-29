from __future__ import annotations

from pathlib import Path

import run_qwen05_prefix_size_073 as runner


def test_exp073_reuses_exact_laptop_sample_and_all_prefix_cells() -> None:
    cases, metadata = runner.select_cases(Path(".context/dimabsa"))
    jobs = runner.build_jobs(cases)
    assert metadata["recipient_ids_sha256"] == (
        "a14b6300c88940b3ccb6ac6b1b78c78b1cb5e202c4ce945ea3fe8220d36c7ddf"
    )
    assert len(cases) == 64
    assert len(jobs) == 256
    assert len({job["user_prompt_sha256"] for job in jobs}) == 64


def test_only_pinned_half_billion_checkpoint_is_run() -> None:
    assert runner.MODEL_CONFIGS == {
        "qwen-0.5b": {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        }
    }
