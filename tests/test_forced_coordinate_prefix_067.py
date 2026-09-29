import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_forced_coordinate_prefix_067 as runner
from run_small_model_decoder_factorial_048 import parse_free_va


def case(case_id, category, valence):
    return {
        "case_id": case_id,
        "gold": [valence, 5.0],
        "target_index": 0,
        "row": {"Text": "meal tasted wonderful", "Triplet": [{
            "Category": category, "Aspect": "meal", "Opinion": "wonderful",
        }]},
    }


def test_fixed_sample_has_64_balanced_recipients_in_six_strata():
    source = []
    n = 0
    for polarity, quotas in runner.exp066.SAMPLE_QUOTAS.items():
        gold = 2.0 if polarity == "neg" else 7.0
        for category, quota in quotas.items():
            for _ in range(quota):
                source.append(case(f"id-{n:03}", category, gold))
                n += 1
    recipients, stats = runner.select_cases(source)

    assert len(recipients) == 64
    assert stats["polarity_counts"] == {"neg": 32, "pos": 32}
    assert len(stats["category_polarity_counts"]) == 6
    assert stats["recipient_ids_sha256"] == runner.sha256(
        "\n".join(sorted(row["case_id"] for row in recipients)).encode()
    )


def test_sample_selection_fails_closed_if_any_quota_is_short():
    with pytest.raises(ValueError, match="Insufficient"):
        runner.select_cases([case("one", "食物#品质", 2.0)])


@pytest.mark.parametrize("order,key,other", [
    ("valence_first", "valence", "arousal"),
    ("arousal_first", "arousal", "valence"),
])
@pytest.mark.parametrize("forced", [2.0, 8.0])
def test_prefix_forces_first_field_and_leaves_second_as_continuation(order, key, other, forced):
    prefix = runner.assistant_prefix(order, forced)
    assert prefix == f'{{"{key}": {forced:.1f}, "{other}": '
    full = prefix + ("7.0}" if other == "arousal" else "3.0}")
    prediction = parse_free_va(full)
    assert prediction is not None
    assert prediction[0 if key == "valence" else 1] == forced


def test_jobs_form_complete_four_arm_grid_with_prompt_fixed_within_order():
    cases = [case("r1", "食物#品质", 2.0), case("r2", "食物#份量与款式", 7.0)]
    jobs = runner.jobs_for(cases)

    assert len(jobs) == 8
    grouped = {}
    for job in jobs:
        grouped.setdefault((job["case_id"], job["order"]), set()).add(job["prompt"])
    assert all(len(prompts) == 1 for prompts in grouped.values())
    assert {job["forced_first_value"] for job in jobs} == {2.0, 8.0}
    counts = Counter((job["order"], job["forced_first_value"]) for job in jobs)
    assert len(counts) == 4 and set(counts.values()) == {2}


def test_prefix_rejects_unregistered_order_and_value():
    with pytest.raises(ValueError, match="Unexpected prefix arm"):
        runner.assistant_prefix("other", 2.0)
    with pytest.raises(ValueError, match="Unexpected prefix arm"):
        runner.assistant_prefix("valence_first", 5.0)
