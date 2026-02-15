"""Tests for secretary.export."""
import json
import pytest
from pathlib import Path
from secretary.models import Segment
from secretary.export import to_txt, to_srt, to_vtt, to_json, load_json


def test_to_txt(sample_segments, label_map, tmp_path):
    out = tmp_path / "out.txt"
    to_txt(sample_segments, out, label_map=label_map)
    text = out.read_text(encoding="utf-8")
    lines = text.strip().split("\n")
    assert len(lines) == 4
    assert "[Alice]" in lines[0]
    assert "[Bob]" in lines[1]
    assert "0.00 - " in lines[0]


def test_to_srt(sample_segments, label_map, tmp_path):
    out = tmp_path / "out.srt"
    to_srt(sample_segments, out, label_map=label_map)
    text = out.read_text(encoding="utf-8")
    assert "1\n" in text
    assert "-->" in text
    assert "[Alice]" in text


def test_to_vtt(sample_segments, label_map, tmp_path):
    out = tmp_path / "out.vtt"
    to_vtt(sample_segments, out, label_map=label_map)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("WEBVTT")
    assert "-->" in text
    assert "[Bob]" in text


def test_to_json_roundtrip(sample_segments, label_map, tmp_path):
    out = tmp_path / "project.json"
    to_json(sample_segments, out, label_map=label_map, audio_path="C:\\audio.wav")
    segs, lm, ap = load_json(out)
    assert len(segs) == 4
    assert lm == label_map
    assert ap == "C:\\audio.wav"
    assert segs[0].text == "Hello world."


def test_to_json_no_audio(sample_segments, tmp_path):
    out = tmp_path / "project.json"
    to_json(sample_segments, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "audio_path" not in data
