"""Paths and config: HF token, model defaults. Load from .env or user config."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Hugging Face token (required for pyannote diarization)
def get_hf_token() -> str | None:
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    config_dir = Path.home() / ".secretary"
    config_file = config_dir / "config"
    if config_file.exists():
        for line in config_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("HUGGINGFACE_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def get_config_dir() -> Path:
    d = Path.home() / ".secretary"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Defaults (can be overridden by GUI and persisted later)
DEFAULT_WHISPER_MODEL = "base"
DEFAULT_LANGUAGE = None  # auto-detect
# Use pyannote/speaker-diarization (accept terms on HF). Alternatives: speaker-diarization-3.1, speaker-diarization-community-1
DIARIZATION_MODEL = "pyannote/speaker-diarization"
