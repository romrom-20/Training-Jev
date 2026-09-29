from __future__ import annotations

from pathlib import Path

import numpy as np
import run_qwen05_numeric_support_074 as runner
from analyze_qwen05_numeric_support_074 import _mass_expected


def test_exp074_sample_and_surface_family_sizes() -> None:
    cases, metadata = runner.select_cases(Path(".context/dimabsa"))
    jobs = runner.build_jobs(cases)
    assert len(cases) == 24
    assert metadata["polarity_counts"] == {"negative": 12, "positive": 12}
    assert metadata["recipient_ids_sha256"] == (
        "4f03b671acd8aedfce6cef7eefae4c738a80d29fbcd7f30f042d4243fd9e001d"
    )
    assert len(jobs) == 96
    assert len(runner.canonical_strings()) == 81
    assert len(runner.extended_strings()) == 81
    assert len(runner.integer_strings()) == 9
    assert runner.canonical_strings()[0] == "1.0}"
    assert runner.extended_strings()[32] == "4.20}"
    assert runner.integer_strings()[-1] == "9}"


def test_parser_accepted_surface_forms_aggregate_by_numeric_value() -> None:
    canonical = np.full(81, -100.0)
    extended = np.full(81, -100.0)
    integers = np.full(9, -100.0)
    canonical[0] = 0.0
    extended[0] = 0.0
    integers[0] = 0.0
    row = {
        "canonical_logprobs": canonical.tolist(),
        "extended_logprobs": extended.tolist(),
        "integer_logprobs": integers.tolist(),
    }
    mass, expected = _mass_expected(row, "all_parser_forms")
    assert np.isclose(mass, 3.0)
    assert np.isclose(expected, 1.0)
