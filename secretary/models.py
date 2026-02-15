"""Segment dataclass, speaker label map, colors, and helpers."""
import copy
from dataclasses import dataclass, field


# Distinct colors for up to 12 speakers; wraps around for more.
SPEAKER_COLORS = [
    "#2563EB",  # blue
    "#DC2626",  # red
    "#16A34A",  # green
    "#D97706",  # amber
    "#7C3AED",  # violet
    "#0891B2",  # cyan
    "#DB2777",  # pink
    "#65A30D",  # lime
    "#EA580C",  # orange
    "#4F46E5",  # indigo
    "#0D9488",  # teal
    "#9333EA",  # purple
]


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker_id: str  # e.g. SPEAKER_00 or user label like "Alice"

    def duration(self) -> float:
        return self.end - self.start


def get_display_speaker(speaker_id: str, label_map: dict[str, str]) -> str:
    """Resolve display name: label_map[speaker_id] or speaker_id."""
    return label_map.get(speaker_id, speaker_id)


def apply_speaker_labels(segments: list[Segment], label_map: dict[str, str]) -> None:
    """In-place: replace speaker_id with label where present."""
    for seg in segments:
        seg.speaker_id = label_map.get(seg.speaker_id, seg.speaker_id)


def split_segment(seg: Segment, at: float) -> tuple[Segment, Segment]:
    """Split *seg* at *at* seconds into two segments. Text goes to the first."""
    at = max(seg.start, min(at, seg.end))
    return (
        Segment(start=seg.start, end=at, text=seg.text, speaker_id=seg.speaker_id),
        Segment(start=at, end=seg.end, text="", speaker_id=seg.speaker_id),
    )


def snapshot_segments(segments: list[Segment]) -> list[Segment]:
    """Deep-copy a segment list for undo/redo."""
    return copy.deepcopy(segments)
