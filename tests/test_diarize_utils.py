"""Tests for secretary.diarize_utils."""
import pytest
from secretary.models import Segment
from secretary.diarize_utils import get_timestamp_texts, merge_sentence, merge_short_segments


def test_get_timestamp_texts():
    result = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello."},
            {"start": 1.0, "end": 2.0, "text": "World."},
        ]
    }
    ts = get_timestamp_texts(result)
    assert len(ts) == 2
    assert ts[0][1] == "Hello."
    assert ts[1][0].start == 1.0


def test_merge_sentence_same_speaker():
    from pyannote.core import Segment as PS
    data = [
        (PS(0, 1), "SPK_A", "Hello"),
        (PS(1, 2), "SPK_A", " world."),
    ]
    merged = merge_sentence(data)
    assert len(merged) == 1
    assert merged[0][2] == "Hello world."


def test_merge_sentence_different_speakers():
    from pyannote.core import Segment as PS
    data = [
        (PS(0, 1), "SPK_A", "Hi."),
        (PS(1, 2), "SPK_B", "Bye."),
    ]
    merged = merge_sentence(data)
    assert len(merged) == 2
    assert merged[0][1] == "SPK_A"
    assert merged[1][1] == "SPK_B"


def test_merge_short_segments():
    segs = [
        Segment(start=0, end=2, text="A", speaker_id="S0"),
        Segment(start=2, end=2.3, text="B", speaker_id="S0"),  # short
        Segment(start=2.3, end=5, text="C", speaker_id="S1"),
    ]
    result = merge_short_segments(segs, min_dur=0.5)
    assert len(result) == 2
    assert result[0].text == "A B"
    assert result[0].end == 2.3


def test_merge_short_segments_no_op():
    segs = [Segment(start=0, end=3, text="X", speaker_id="S")]
    result = merge_short_segments(segs, min_dur=0.0)
    assert len(result) == 1
