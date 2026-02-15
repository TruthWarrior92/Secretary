"""Assign speakers to Whisper segments using pyannote Annotation; merge by speaker and sentence."""
import json
import time
from pathlib import Path

from pyannote.core import Segment as PyannoteSegment

from secretary.models import Segment

# Sentence-ending punctuation for merging
PUNC_SENT_END = (".", "?", "!")


# #region agent log
def _debug_log(message: str, data: dict):
    try:
        path = Path(__file__).resolve().parent.parent / ".cursor" / "debug.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp": int(time.time() * 1000), "location": "diarize_utils.py", "message": message, "data": data}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
# #endregion


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
    labels = list(annotation.labels()) if hasattr(annotation, "labels") else []
    fallback = labels[0] if labels else "SPEAKER_00"
    # #region agent log
    _debug_log("add_speaker_to_text annotation labels", {"labels": labels, "fallback": fallback, "num_whisper_segments": len(timestamp_texts)})
    sample_debug = []
    # #endregion
    for i, (seg, text) in enumerate(timestamp_texts):
        try:
            cropped = annotation.crop(seg)
            spk = cropped.argmax()
            cropped_labels = list(cropped.labels()) if hasattr(cropped, "labels") else []
            used_fallback = spk is None
            if spk is None:
                spk = fallback
        except Exception as e:
            cropped_labels = []
            used_fallback = True
            spk = fallback
            # #region agent log
            _debug_log("add_speaker_to_text exception", {"seg_idx": i, "start": seg.start, "end": seg.end, "error": str(e)})
            # #endregion
        result.append((seg, spk, text))
        # #region agent log
        if i < 5 or i >= len(timestamp_texts) - 2 or (len(timestamp_texts) > 10 and i % max(1, len(timestamp_texts) // 5) == 0):
            sample_debug.append({"idx": i, "start": round(seg.start, 4), "end": round(seg.end, 4), "cropped_labels": cropped_labels, "argmax": str(spk), "used_fallback": used_fallback})
        # #endregion
    # #region agent log
    _debug_log("add_speaker_to_text sample", {"sample": sample_debug})
    # #endregion
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
    # #region agent log
    try:
        diar_labels = list(diarization_annotation.labels()) if hasattr(diarization_annotation, "labels") else []
        diar_segments = []
        for j, (seg, _, label) in enumerate(diarization_annotation.itertracks(yield_label=True)):
            if j >= 25:
                diar_segments.append({"note": "... (truncated)"})
                break
            diar_segments.append({"start": round(seg.start, 4), "end": round(seg.end, 4), "label": label})
        _debug_log("diarize_text diarization annotation", {"labels": diar_labels, "first_25_segments": diar_segments})
    except Exception as e:
        _debug_log("diarize_text annotation inspect failed", {"error": str(e)})
    # #endregion
    timestamp_texts = get_timestamp_texts(transcribe_result)
    spk_text = add_speaker_to_text(timestamp_texts, diarization_annotation)
    merged = merge_sentence(spk_text)
    return [
        Segment(start=seg.start, end=seg.end, text=text, speaker_id=spk)
        for seg, spk, text in merged
    ]
