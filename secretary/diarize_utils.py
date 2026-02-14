"""Assign speakers to Whisper segments using pyannote Annotation; merge by speaker and sentence."""
from pyannote.core import Segment as PyannoteSegment

from secretary.models import Segment

# Sentence-ending punctuation for merging
PUNC_SENT_END = (".", "?", "!")


def get_timestamp_texts(transcribe_result: dict) -> list[tuple[PyannoteSegment, str]]:
    """From Whisper transcribe result -> list of (Segment, text)."""
    timestamp_texts = []
    for item in transcribe_result.get("segments", []):
        start = item["start"]
        end = item["end"]
        text = (item.get("text") or "").strip()
        timestamp_texts.append((PyannoteSegment(start, end), text))
    return timestamp_texts


def add_speaker_to_text(
    timestamp_texts: list[tuple[PyannoteSegment, str]], annotation
) -> list[tuple[PyannoteSegment, str, str]]:
    """For each (seg, text), assign dominant speaker from annotation. Returns (seg, speaker_id, text)."""
    result = []
    for seg, text in timestamp_texts:
        try:
            spk = annotation.crop(seg).argmax()
        except Exception:
            spk = "SPEAKER_00"
        result.append((seg, spk, text))
    return result


def _merge_cache(
    text_cache: list[tuple[PyannoteSegment, str, str]],
) -> tuple[PyannoteSegment, str, str]:
    """Merge a list of (seg, spk, text) into one (seg, spk, sentence)."""
    sentence = "".join(item[2] for item in text_cache).strip()
    spk = text_cache[0][1]
    start = text_cache[0][0].start
    end = text_cache[-1][0].end
    return (PyannoteSegment(start, end), spk, sentence)


def merge_sentence(
    spk_text: list[tuple[PyannoteSegment, str, str]],
) -> list[tuple[PyannoteSegment, str, str]]:
    """Merge consecutive (seg, speaker, text) by same speaker and sentence boundaries."""
    merged = []
    pre_spk = None
    text_cache: list[tuple[PyannoteSegment, str, str]] = []
    for seg, spk, text in spk_text:
        if spk != pre_spk and pre_spk is not None and len(text_cache) > 0:
            merged.append(_merge_cache(text_cache))
            text_cache = [(seg, spk, text)]
            pre_spk = spk
        elif text and len(text) > 0 and text[-1] in PUNC_SENT_END:
            text_cache.append((seg, spk, text))
            merged.append(_merge_cache(text_cache))
            text_cache = []
            pre_spk = spk
        else:
            text_cache.append((seg, spk, text))
            pre_spk = spk
    if text_cache:
        merged.append(_merge_cache(text_cache))
    return merged


def diarize_text(transcribe_result: dict, diarization_annotation) -> list[Segment]:
    """Whisper result + pyannote Annotation -> list of Segment(start, end, text, speaker_id)."""
    timestamp_texts = get_timestamp_texts(transcribe_result)
    spk_text = add_speaker_to_text(timestamp_texts, diarization_annotation)
    merged = merge_sentence(spk_text)
    return [
        Segment(start=seg.start, end=seg.end, text=text, speaker_id=spk)
        for seg, spk, text in merged
    ]
