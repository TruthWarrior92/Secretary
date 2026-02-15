"""Configure user-facing logging to ~/.secretary/logs/."""
import logging
import os
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path.home() / ".secretary" / "logs"
_configured = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Return the 'secretary' logger, writing to a daily rotating file."""
    global _configured
    logger = logging.getLogger("secretary")
    if _configured:
        return logger

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = _LOG_DIR / f"secretary_{today}.log"

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(level)
    _configured = True

    # Cleanup: keep last 7 days
    try:
        for f in sorted(_LOG_DIR.glob("secretary_*.log"))[:-7]:
            f.unlink(missing_ok=True)
    except Exception:
        pass

    return logger


def get_log_dir() -> Path:
    """Return the log directory path."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR
