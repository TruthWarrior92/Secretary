"""Load Whisper and pyannote diarization; run pipeline on one audio file -> list of Segment."""
import json
import os
import time
import warnings
from pathlib import Path
from typing import Callable

# Suppress HF "use_auth_token is deprecated, use token=" so users aren't confused; we pass token=
warnings.filterwarnings("ignore", message=".*use_auth_token.*", category=FutureWarning)
# pyannote pooling std() degrees-of-freedom warning (benign)
warnings.filterwarnings("ignore", message=".*degrees of freedom.*")
# Whisper Triton/CUDA toolkit fallback warnings (benign — uses slower CPU kernels)
warnings.filterwarnings("ignore", message=".*Failed to launch Triton kernels.*")

from secretary.diarize_utils import diarize_text
from secretary.models import Segment

# #region agent log
def _debug_log(message: str, data: dict, hypothesis_id: str = ""):
    try:
        path = Path(__file__).resolve().parent.parent / ".cursor" / "debug.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp": int(time.time() * 1000), "location": "pipeline.py", "message": message, "data": data, "hypothesisId": hypothesis_id}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def log_pipeline_error(message: str, traceback_text: str | None = None) -> None:
    """Append pipeline error to .cursor/debug.log so it can be read without copying the error dialog."""
    try:
        path = Path(__file__).resolve().parent.parent / ".cursor" / "debug.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": int(time.time() * 1000),
            "location": "pipeline_error",
            "message": message,
            "data": {} if not traceback_text else {"traceback": traceback_text},
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
# #endregion

# Lazy imports so GUI can start without loading heavy libs
_whisper_model = None
_diarization_pipeline = None


def _ensure_audio_path(audio_path: str | Path) -> str:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    return str(path.resolve())


def _load_audio_as_numpy(audio_path: str, target_sr: int = 16000):
    """Load audio as 1-D float32 numpy array at target_sr (mono). For Whisper."""
    import numpy as np

    path = _ensure_audio_path(audio_path)
    try:
        import soundfile as sf
        waveform_np, sr = sf.read(path, dtype="float32")
        if waveform_np.ndim == 2:
            waveform_np = waveform_np.mean(axis=1)
    except Exception:
        import librosa
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", UserWarning)
            waveform_np, sr = librosa.load(path, sr=None, mono=True, dtype="float32")
    if waveform_np.ndim > 1:
        waveform_np = waveform_np.mean(axis=-1) if waveform_np.shape[-1] < waveform_np.shape[0] else waveform_np.mean(axis=0)
    if sr != target_sr:
        import librosa
        waveform_np = librosa.resample(waveform_np, orig_sr=sr, target_sr=target_sr)
    return waveform_np.astype(np.float32)


def _load_audio_for_diarization(audio_path: str, target_sr: int = 16000) -> dict:
    """Load audio as (channels, samples) torch tensor at target_sr. Avoids pyannote's torchcodec/FFmpeg dependency."""
    import torch

    path = _ensure_audio_path(audio_path)
    try:
        import soundfile as sf
        waveform_np, sr = sf.read(path, dtype="float32")
        # soundfile returns (samples,) or (samples, channels) -> need (channels, samples)
        if waveform_np.ndim == 2:
            waveform_np = waveform_np.T
    except Exception:
        import librosa
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", UserWarning)
            waveform_np, sr = librosa.load(path, sr=None, mono=False, dtype="float32")
        if waveform_np.ndim == 1:
            waveform_np = waveform_np.reshape(1, -1)

    if waveform_np.ndim == 1:
        waveform_np = waveform_np.reshape(1, -1)
    if sr != target_sr:
        import librosa
        waveform_np = librosa.resample(waveform_np, orig_sr=sr, target_sr=target_sr)
        if waveform_np.ndim == 1:
            waveform_np = waveform_np.reshape(1, -1)
        sr = target_sr
    # (samples,) or (channels, samples) -> (channels, samples)
    tensor = torch.from_numpy(waveform_np).float()
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return {"waveform": tensor, "sample_rate": sr}


def load_whisper(model_name: str = "base", device: str | None = None):
    import torch
    import whisper

    global _whisper_model
    if _whisper_model is None or getattr(_whisper_model, "_model_name", None) != model_name:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _whisper_model = whisper.load_model(model_name, device=dev)
        _whisper_model._model_name = model_name
    return _whisper_model


def _patch_pyannote_revision_in_checkpoint():
    """If pyannote config uses model@revision, from_pretrained raises. Patch to split into checkpoint + revision."""
    def _wrap(orig, is_pipeline: bool = False):
        @classmethod
        def _from_pretrained(cls, checkpoint, *args, revision=None, **kwargs):
            if isinstance(checkpoint, str) and "@" in checkpoint:
                checkpoint, rev = checkpoint.split("@", 1)
                revision = revision or rev
            kwargs.pop("revision", None)
            if is_pipeline and args:
                revision = revision or args[0]
                args = args[1:]
            # Pass revision only in kwargs to avoid "multiple values for argument 'revision'"
            if revision is not None:
                kwargs["revision"] = revision
            return orig(cls, checkpoint, *args, **kwargs)
        _from_pretrained.__name__ = "from_pretrained"
        return _from_pretrained

    try:
        from pyannote.audio.core import model as _model_mod
        from pyannote.audio.core import pipeline as _pipeline_mod
        from pyannote.audio.core import plda as _plda_mod
        from pyannote.audio.core import calibration as _cal_mod
    except ImportError:
        return
    # Only patch Model, PLDA, Calibration (for model@revision in config). Do NOT patch Pipeline:
    # patching Pipeline causes "multiple values for argument 'revision'" when we call from_pretrained.
    for Klass in (
        _model_mod.Model,
        getattr(_plda_mod, "PLDA", None),
        getattr(_cal_mod, "Calibration", None),
    ):
        if Klass is None or getattr(Klass, "_secretary_revision_patched", False):
            continue
        Klass.from_pretrained = _wrap(Klass.from_pretrained, is_pipeline=False)
        Klass._secretary_revision_patched = True


def load_diarization_pipeline(token: str | None = None):
    # Suppress torchcodec/FFmpeg warning: we pass preloaded waveform dict, so pyannote never decodes from file
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
        from pyannote.audio import Pipeline

    from config import get_hf_token, DIARIZATION_MODEL

    # Disabled: patch caused "multiple values for revision" and "path should be ... not type"
    # _patch_pyannote_revision_in_checkpoint()

    global _diarization_pipeline
    if _diarization_pipeline is None:
        hf_token = token or get_hf_token()
        if not hf_token:
            raise ValueError(
                "Hugging Face token required. Use config.json (copy config.json.example), .env, or ~/.secretary/config.json. "
                "Create a read token at https://hf.co/settings/tokens and accept conditions at https://huggingface.co/pyannote/speaker-diarization-community-1"
            )
        # #region agent log
        _debug_log("About to Pipeline.from_pretrained", {"model": DIARIZATION_MODEL}, "H5")
        # #endregion
        # Set env so huggingface_hub and any internal code that only reads env get the token
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_TOKEN"] = hf_token
        _diarization_pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL,
            token=hf_token,
        )
        # #region agent log
        _debug_log("Pipeline.from_pretrained returned", {}, "H5")
        # #endregion
    return _diarization_pipeline


def run_pipeline(
    audio_path: str | Path,
    *,
    model_name: str = "base",
    language: str | None = None,
    use_vad: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> list[Segment]:
    """Run Whisper then diarization; return list of Segment. progress_callback(message) optional."""
    import numpy as np
    audio_path = _ensure_audio_path(audio_path)

    if progress_callback:
        progress_callback("Loading Whisper...")
    model = load_whisper(model_name)

    # Pre-load audio as numpy so Whisper doesn't use ffmpeg on M4A (avoids codec padding/offset)
    if progress_callback:
        progress_callback("Loading audio...")
    whisper_audio = _load_audio_as_numpy(audio_path, target_sr=16000)

    # Optional VAD pre-filter: zero-out non-speech regions so Whisper skips silence
    if use_vad:
        if progress_callback:
            progress_callback("Running VAD pre-filter...")
        vad_regions = _run_vad_filter(audio_path)
        if vad_regions:
            mask = np.zeros_like(whisper_audio)
            sr = 16000
            for vs, ve in vad_regions:
                s0, s1 = int(vs * sr), min(int(ve * sr), len(mask))
                mask[s0:s1] = whisper_audio[s0:s1]
            whisper_audio = mask
    # #region agent log
    _debug_log("whisper_audio_loaded", {"shape": list(whisper_audio.shape), "duration_sec": round(len(whisper_audio) / 16000, 4), "dtype": str(whisper_audio.dtype)})
    # #endregion

    if progress_callback:
        progress_callback("Transcribing...")
    transcribe_result = model.transcribe(
        whisper_audio,
        language=language or None,
        word_timestamps=True,
    )
    # #region agent log
    whisper_segs = transcribe_result.get("segments", [])
    _debug_log("whisper_raw_segments", {"count": len(whisper_segs), "audio_duration_sec": round(len(whisper_audio) / 16000, 4), "segments": [{"start": s["start"], "end": s["end"], "text_preview": (s.get("text") or "")[:40]} for s in whisper_segs]})
    # #endregion

    if progress_callback:
        progress_callback("Running diarization...")
    # #region agent log
    _debug_log("About to load diarization pipeline", {"audio_path": audio_path, "file_size_mb": round(os.path.getsize(audio_path) / (1024 * 1024), 2)}, "H5")
    # #endregion
    t_load_start = time.perf_counter()
    pipeline = load_diarization_pipeline()
    t_load_end = time.perf_counter()
    # #region agent log
    _debug_log("Diarization pipeline loaded", {"load_sec": round(t_load_end - t_load_start, 2)}, "H5")
    try:
        import torch
        _debug_log("Device check", {"cuda_available": torch.cuda.is_available(), "pipeline_device": str(getattr(pipeline, "device", "unknown"))}, "H1")
    except Exception as e:
        _debug_log("Device check failed", {"error": str(e)}, "H1")
    # #endregion
    # #region agent log
    _debug_log("About to load audio for diarization (preload avoids torchcodec)", {"audio_path": audio_path}, "H2")
    # #endregion
    audio_input = _load_audio_for_diarization(audio_path)
    duration_sec = float(audio_input["waveform"].shape[-1]) / audio_input["sample_rate"]
    # #region agent log
    _debug_log("About to call pipeline(audio_input)", {"waveform_shape": list(audio_input["waveform"].shape), "sample_rate": audio_input["sample_rate"], "audio_duration_sec": round(duration_sec, 4)}, "H2")
    t_infer_start = time.perf_counter()
    # #endregion
    try:
        diarization_result = pipeline(
            audio_input,
            min_speakers=2,
            max_speakers=25,
        )
    except TypeError:
        diarization_result = pipeline(audio_input)
    # #region agent log
    t_infer_end = time.perf_counter()
    _debug_log("pipeline(audio_input) returned", {"infer_sec": round(t_infer_end - t_infer_start, 2), "result_type": type(diarization_result).__name__}, "H2")
    # #endregion

    # Extract actual Annotation from DiarizeOutput wrapper if needed
    annotation = diarization_result
    if hasattr(diarization_result, "speaker_diarization"):
        annotation = diarization_result.speaker_diarization
    # #region agent log
    _debug_log("annotation_extracted", {"annotation_type": type(annotation).__name__, "has_crop": hasattr(annotation, "crop"), "has_labels": hasattr(annotation, "labels"), "labels": list(annotation.labels()) if hasattr(annotation, "labels") else []})
    # #endregion

    if progress_callback:
        progress_callback("Assigning speakers...")
    segments = diarize_text(transcribe_result, annotation)
    # #region agent log
    _debug_log("segments_before_clamp", {"count": len(segments), "duration_sec": round(duration_sec, 4), "segments": [{"start": round(s.start, 4), "end": round(s.end, 4), "speaker_id": s.speaker_id, "text_preview": (s.text or "")[:30]} for s in segments]})
    # #endregion
    _post_process_segments(segments, duration_sec)
    # #region agent log
    _debug_log("segments_after_clamp", {"count": len(segments), "duration_sec": round(duration_sec, 4), "segments": [{"start": round(s.start, 4), "end": round(s.end, 4), "speaker_id": s.speaker_id} for s in segments]})
    # #endregion
    return segments


def run_diarization_only(
    audio_path: str | Path,
    transcribe_result: dict,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> list[Segment]:
    """Re-run diarization using a cached Whisper transcribe_result (skips Whisper)."""
    audio_path = _ensure_audio_path(audio_path)

    if progress_callback:
        progress_callback("Loading diarization pipeline...")
    pipeline = load_diarization_pipeline()

    if progress_callback:
        progress_callback("Loading audio...")
    audio_input = _load_audio_for_diarization(audio_path)
    duration_sec = float(audio_input["waveform"].shape[-1]) / audio_input["sample_rate"]

    if progress_callback:
        progress_callback("Running diarization...")
    try:
        diarization_result = pipeline(audio_input, min_speakers=2, max_speakers=25)
    except TypeError:
        diarization_result = pipeline(audio_input)

    annotation = diarization_result
    if hasattr(diarization_result, "speaker_diarization"):
        annotation = diarization_result.speaker_diarization

    if progress_callback:
        progress_callback("Assigning speakers...")
    segments = diarize_text(transcribe_result, annotation)

    _post_process_segments(segments, duration_sec)
    return segments


def run_pipeline_accurate(
    audio_path: str | Path,
    *,
    model_name: str = "base",
    language: str | None = None,
    use_vad: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> list[Segment]:
    """Accurate mode: diarize first, then run Whisper per speaker segment for cleaner per-turn text."""
    import numpy as np
    audio_path = _ensure_audio_path(audio_path)

    if progress_callback:
        progress_callback("Loading audio…")
    full_audio = _load_audio_as_numpy(audio_path, target_sr=16000)
    sr = 16000
    audio_input = _load_audio_for_diarization(audio_path)
    duration_sec = float(audio_input["waveform"].shape[-1]) / audio_input["sample_rate"]

    if progress_callback:
        progress_callback("Loading diarization pipeline…")
    dia_pipeline = load_diarization_pipeline()

    if progress_callback:
        progress_callback("Running diarization…")
    try:
        diarization_result = dia_pipeline(audio_input, min_speakers=2, max_speakers=25)
    except TypeError:
        diarization_result = dia_pipeline(audio_input)
    annotation = diarization_result
    if hasattr(diarization_result, "speaker_diarization"):
        annotation = diarization_result.speaker_diarization

    if progress_callback:
        progress_callback("Loading Whisper…")
    model = load_whisper(model_name)

    if progress_callback:
        progress_callback("Transcribing per-speaker segments…")
    segments: list[Segment] = []
    tracks = list(annotation.itertracks(yield_label=True))
    for i, (seg, _, label) in enumerate(tracks):
        start_sample = int(seg.start * sr)
        end_sample = min(int(seg.end * sr), len(full_audio))
        if end_sample <= start_sample:
            continue
        chunk = full_audio[start_sample:end_sample].astype(np.float32)
        result = model.transcribe(chunk, language=language or None, word_timestamps=True)
        text = result.get("text", "").strip()
        if text:
            segments.append(Segment(start=seg.start, end=seg.end, text=text, speaker_id=label))
        if progress_callback and (i % 5 == 0 or i == len(tracks) - 1):
            progress_callback(f"Transcribing segment {i + 1}/{len(tracks)}…")

    _post_process_segments(segments, duration_sec)
    return segments


def _run_vad_filter(audio_path: str | Path) -> list[tuple[float, float]]:
    """Run VAD and return list of (start, end) speech regions in seconds."""
    audio_input = _load_audio_for_diarization(audio_path)
    try:
        from pyannote.audio.pipelines import VoiceActivityDetection
        from pyannote.audio import Model
        from config import get_hf_token
        token = get_hf_token()
        vad_model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=token)
        vad_pipeline = VoiceActivityDetection(segmentation=vad_model)
        HYPER_PARAMS = {"onset": 0.5, "offset": 0.5, "min_duration_on": 0.2, "min_duration_off": 0.2}
        vad_pipeline.instantiate(HYPER_PARAMS)
        vad_result = vad_pipeline(audio_input)
        regions = [(seg.start, seg.end) for seg in vad_result.get_timeline()]
        return regions
    except Exception:
        return []


def _post_process_segments(segments: list[Segment], duration_sec: float) -> None:
    """End-pad, clamp, and remove zero-length segments in-place."""
    END_PAD_SEC = 0.25
    for i, seg in enumerate(segments):
        next_start = segments[i + 1].start if i + 1 < len(segments) else duration_sec
        seg.end = min(seg.end + END_PAD_SEC, next_start, duration_sec)
    for seg in segments:
        seg.start = max(0.0, min(seg.start, duration_sec))
        seg.end = max(0.0, min(seg.end, duration_sec))
        if seg.end < seg.start:
            seg.end = seg.start
    segments[:] = [s for s in segments if s.end > s.start]
