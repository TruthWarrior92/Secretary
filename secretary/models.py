"""Segment dataclass and speaker label map."""
from dataclasses import dataclass, field


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
    """In-place: replace speaker_id with label where present. Prefer keeping canonical ID in model and only map on display/export; this is for when user wants to persist renames."""
    for seg in segments:
        seg.speaker_id = label_map.get(seg.speaker_id, seg.speaker_id)
