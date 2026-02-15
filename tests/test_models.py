"""Tests for secretary.models."""
import pytest
from secretary.models import (
    SPEAKER_COLORS,
    Segment,
    apply_speaker_labels,
    get_display_speaker,
    snapshot_segments,
    split_segment,
)


def test_segment_duration():
    s = Segment(start=1.0, end=3.5, text="hi", speaker_id="SPK")
    assert s.duration() == pytest.approx(2.5)


def test_get_display_speaker_with_map():
    assert get_display_speaker("SPEAKER_00", {"SPEAKER_00": "Alice"}) == "Alice"


def test_get_display_speaker_without_map():
    assert get_display_speaker("SPEAKER_00", {}) == "SPEAKER_00"


def test_apply_speaker_labels(sample_segments, label_map):
    apply_speaker_labels(sample_segments, label_map)
    assert sample_segments[0].speaker_id == "Alice"
    assert sample_segments[1].speaker_id == "Bob"


def test_split_segment():
    seg = Segment(start=0.0, end=4.0, text="Hello world.", speaker_id="SPK")
    a, b = split_segment(seg, 2.0)
    assert a.start == 0.0
    assert a.end == 2.0
    assert a.text == "Hello world."
    assert b.start == 2.0
    assert b.end == 4.0
    assert b.text == ""
    assert b.speaker_id == "SPK"


def test_split_segment_clamps():
    seg = Segment(start=1.0, end=3.0, text="x", speaker_id="S")
    a, b = split_segment(seg, 0.5)  # before start
    assert a.start == 1.0 and a.end == 1.0
    a2, b2 = split_segment(seg, 9.0)  # after end
    assert a2.end == 3.0 and b2.start == 3.0


def test_snapshot_segments(sample_segments):
    snap = snapshot_segments(sample_segments)
    assert len(snap) == len(sample_segments)
    snap[0].text = "CHANGED"
    assert sample_segments[0].text != "CHANGED"


def test_speaker_colors_not_empty():
    assert len(SPEAKER_COLORS) >= 10
