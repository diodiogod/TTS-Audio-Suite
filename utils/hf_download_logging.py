"""Shared Hugging Face download logging policy for TTS Audio Suite."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Dict, Iterator


HF_DOWNLOAD_LOGGERS = ("httpx", "httpcore", "huggingface_hub")


def configure_hf_download_logging() -> None:
    """Suppress per-request Hub chatter while preserving actual errors."""
    for name in HF_DOWNLOAD_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)


@contextmanager
def quiet_hf_download_logs() -> Iterator[None]:
    """Apply the shared policy temporarily for standalone downloader use."""
    original_levels: Dict[str, int] = {
        name: logging.getLogger(name).level for name in HF_DOWNLOAD_LOGGERS
    }
    try:
        configure_hf_download_logging()
        yield
    finally:
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)
