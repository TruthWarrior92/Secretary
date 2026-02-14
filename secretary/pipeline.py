"""Load Whisper and pyannote diarization; run pipeline on one audio file -> list of Segment."""
import os
from pathlib import Path
from typing import Callable

from secretary.diarize_utils import diarize_text
from secretary.models import Segment

# Lazy imports so GUI can start without loading heavy libs
_whisper_model = None
_diarization_pipeline = None


def _ensure_audio_path(audio_path: str | Path) -> str:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return str(path.resolve())


def load_whisper(model_name: str = "base", device: str | None = None):
    import torch
    import whisper

    global _whisper_model
    if _whisper_model is None or getattr(_whisper_model, "_model_name", None) != model_name:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _whisper_model = whisper.load_model(model_name, device=dev)
        _whisper_model._model_name = model_name
    return _whisper_model


def load_diarization_pipeline(token: str | None = None):
    from pyannote.audio import Pipeline

    from config import get_hf_token, DIARIZATION_MODEL

    global _diarization_pipeline
    if _diarization_pipeline is None:
        hf_token = token or get_hf_token()
        if not hf_token:
            raise ValueError(
                "Hugging Face token required. Set HUGGINGFACE_TOKEN in .env or ~/.secretary/config. "
                "Create token at https://hf.co/settings/tokens and accept terms at https://huggingface.co/pyannote/speaker-diarization"
            )
        _diarization_pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL,
            use_auth_token=hf_token,
        )
    return _diarization_pipeline


def run_pipeline(
    audio_path: str | Path,
    *,
    model_name: str = "base",
    language: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> list[Segment]:
    """Run Whisper then diarization; return list of Segment. progress_callback(message) optional."""
    audio_path = _ensure_audio_path(audio_path)

    if progress_callback:
        progress_callback("Loading Whisper...")
    model = load_whisper(model_name)
    if progress_callback:
        progress_callback("Transcribing...")
    transcribe_result = model.transcribe(audio_path, language=language or None)

    if progress_callback:
        progress_callback("Running diarization...")
    pipeline = load_diarization_pipeline()
    diarization_result = pipeline(audio_path)

    if progress_callback:
        progress_callback("Assigning speakers...")
    segments = diarize_text(transcribe_result, diarization_result)
    return segments
