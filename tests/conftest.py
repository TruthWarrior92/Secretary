"""Shared fixtures for Secretary tests."""
import pytest
from secretary.models import Segment


@pytest.fixture
def sample_segments() -> list[Segment]:
    return [
        Segment(start=0.0, end=2.5, text="Hello world.", speaker_id="SPEAKER_00"),
        Segment(start=2.5, end=5.0, text="How are you?", speaker_id="SPEAKER_01"),
        Segment(start=5.0, end=7.5, text="I'm fine thanks.", speaker_id="SPEAKER_00"),
        Segment(start=7.5, end=10.0, text="Great.", speaker_id="SPEAKER_01"),
    ]


@pytest.fixture
def label_map() -> dict[str, str]:
    return {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"}
