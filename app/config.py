from __future__ import annotations

from pathlib import Path

from app.schemas import ProcessingConfig

BASE_DIR = Path(__file__).resolve().parent.parent
RUNTIME_DIR = BASE_DIR / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
OUTPUT_DIR = RUNTIME_DIR / "outputs"
TMP_DIR = RUNTIME_DIR / "tmp"

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}

DEFAULT_PROCESSING_CONFIG = ProcessingConfig()
MAX_HISTORY_POINTS = 180


def ensure_runtime_dirs() -> None:
    for directory in (RUNTIME_DIR, UPLOAD_DIR, OUTPUT_DIR, TMP_DIR):
        directory.mkdir(parents=True, exist_ok=True)
