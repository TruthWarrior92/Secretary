"""Paths and config: HF token, model defaults. Load from JSON, .env, or legacy config file."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (directory containing config.py)
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)
load_dotenv()  # also allow cwd .env

# Avoid huggingface_hub symlinks warning on Windows (cache still works)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_PROJECT_ROOT = Path(__file__).resolve().parent

# Hugging Face token (required for pyannote diarization). Prefer JSON so terminal env issues don't affect it.
def get_hf_token() -> str | None:
    # 1) JSON config (most reliable; no terminal env involved)
    for candidate in (_PROJECT_ROOT / "config.json", Path.home() / ".secretary" / "config.json"):
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                token = data.get("HUGGINGFACE_TOKEN") or data.get("huggingface_token")
                if token and isinstance(token, str):
                    return token.strip()
            except (json.JSONDecodeError, OSError):
                pass
    # 2) Environment variables
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    # 3) Legacy ~/.secretary/config (key=value lines)
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
# Community-1: better benchmarks, simpler setup. Accept conditions: https://huggingface.co/pyannote/speaker-diarization-community-1
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
