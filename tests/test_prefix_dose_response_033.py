import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from run_prefix_dose_response_033 import BUDGETS, prefix_records


class TokenizerStub:
    def decode(self, values, skip_special_tokens=True):
        assert skip_special_tokens
        return " ".join(map(str, values))


def test_prefixes_are_nested_views_of_the_same_completion():
    source = [
        {
            "model": "target",
            "id": "item-1",
            "sentence_id": "sentence-1",
            "category": "food",
            "gold": 1,
            "token_ids": list(range(1, 33)),
        }
    ]
    prefixes = prefix_records(source, TokenizerStub())
    assert tuple(row["budget"] for row in prefixes) == BUDGETS
    assert [row["actual_tokens"] for row in prefixes] == [4, 12, 16, 24]
    assert prefixes[0]["answer"] == "1 2 3 4"
    assert prefixes[-1]["answer"] == "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"


def test_prefix_budget_beyond_natural_end_reuses_completed_text():
    source = [
        {
            "model": "target",
            "id": "item-2",
            "sentence_id": "sentence-2",
            "category": "service",
            "gold": 0,
            "token_ids": [8, 9, 10],
        }
    ]
    prefixes = prefix_records(source, TokenizerStub())
    assert [row["actual_tokens"] for row in prefixes] == [3, 3, 3, 3]
    assert len({row["answer"] for row in prefixes}) == 1
