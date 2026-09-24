import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from freeform_evaluator_feasibility import aspect_input


def test_aspect_input_contains_aspect_and_sentence_but_not_label():
    value = aspect_input("service", "The waiter was kind.")
    assert value == "Aspect: service. Review: The waiter was kind."
    assert "positive" not in value
    assert "negative" not in value
