import pytest
from pydantic import ValidationError
from rav.core.types import BoundingBox, ObjectObservation


def test_observation_is_typed() -> None:
    observation = ObjectObservation(
        id="cup-1",
        class_name="cup",
        confidence=0.93,
        bbox=BoundingBox(x=10, y=20, width=30, height=40),
        source="detector",
    )

    assert observation.bbox is not None
    assert observation.bbox.width == 30


def test_observation_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        ObjectObservation(id="cup-1", class_name="cup", confidence=1.1, source="detector")
