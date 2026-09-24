import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from analyze_tripr_residual_control_seeds import sentence_contrast


def test_diagonal_off_diagonal_ignores_sources_without_a_matching_target():
    rows = [
        {"sentence_id": "s1", "source": 0, "target": 0, "margin_delta": 2.0},
        {"sentence_id": "s1", "source": 0, "target": 1, "margin_delta": 0.0},
        {"sentence_id": "s1", "source": 1, "target": 0, "margin_delta": 0.0},
        {"sentence_id": "s1", "source": 1, "target": 1, "margin_delta": 2.0},
        # No target-2 query exists in this sentence; source 2 has no diagonal cell.
        {"sentence_id": "s1", "source": 2, "target": 0, "margin_delta": 100.0},
        {"sentence_id": "s1", "source": 2, "target": 1, "margin_delta": -100.0},
    ]

    assert sentence_contrast(rows) == {"s1": 2.0}
