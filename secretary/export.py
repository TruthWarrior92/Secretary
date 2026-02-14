"""Export segments to TXT, SRT, VTT, JSON. Uses label_map for display names."""
from pathlib import Path
from typing import Any

from secretary.models import Segment, get_display_speaker


def to_txt(
    segments: list[Segment],
    path: str | Path,
    *,
    label_map: dict[str, str] | None = None,
) -> None:
    """Pyannote-whisper style: 'start end SPEAKER_00 text' per line."""
    label_map = label_map or {}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for seg in segments:
        spk = get_display_speaker(seg.speaker_id, label_map)
        line = f"{seg.start:.2f} {seg.end:.2f} {spk} {seg.text}"
        lines.append(line)
    path.write_text("\n".join(lines), encoding="utf-8")


def to_srt(
    segments: list[Segment],
    path: str | Path,
    *,
    label_map: dict[str, str] | None = None,
) -> None:
    """SRT subtitle format with speaker prefix in text."""
    label_map = label_map or {}

    def _ts(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h:02d}:{m:02d}:{int(s):02d},{int(s % 1 * 1000):03d}"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for i, seg in enumerate(segments, 1):
        spk = get_display_speaker(seg.speaker_id, label_map)
        blocks.append(f"{i}\n{_ts(seg.start)} --> {_ts(seg.end)}\n[{spk}] {seg.text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def to_vtt(
    segments: list[Segment],
    path: str | Path,
    *,
    label_map: dict[str, str] | None = None,
) -> None:
    """WebVTT format with speaker in text."""
    label_map = label_map or {}

    def _ts(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h:02d}:{m:02d}:{int(s):02d}.{int(s % 1 * 1000):03d}"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["WEBVTT", ""]
    for seg in segments:
        spk = get_display_speaker(seg.speaker_id, label_map)
        lines.append(f"{_ts(seg.start)} --> {_ts(seg.end)}")
        lines.append(f"[{spk}] {seg.text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def to_json(
    segments: list[Segment],
    path: str | Path,
    *,
    label_map: dict[str, str] | None = None,
    audio_path: str | None = None,
) -> None:
    """Full project: segments + label_map + optional audio_path for save/load project."""
    label_map = label_map or {}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "speaker_id": seg.speaker_id,
            }
            for seg in segments
        ],
        "speaker_labels": label_map,
    }
    if audio_path is not None:
        data["audio_path"] = audio_path
    import json

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> tuple[list[Segment], dict[str, str], str | None]:
    """Load project JSON; return (segments, label_map, audio_path)."""
    import json

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = [
        Segment(
            start=o["start"],
            end=o["end"],
            text=o["text"],
            speaker_id=o["speaker_id"],
        )
        for o in data.get("segments", [])
    ]
    label_map = data.get("speaker_labels") or {}
    audio_path = data.get("audio_path")
    return segments, label_map, audio_path
