"""
Lazy Russian stress support for ChatterBox Official 23-Lang.

This keeps the heavy Russian dictionary assets out of the global installer and
downloads them on demand into the organized TTS model cache.
"""

import importlib
import logging
import os
import threading
import zipfile
from pathlib import Path

from utils.downloads.unified_downloader import unified_downloader

logger = logging.getLogger(__name__)

RUSSIAN_STRESS_DATA_URL = "https://github.com/Vuizur/add-stress-to-epub/releases/download/v1.0.1/russian_dict.zip"
RUSSIAN_STRESS_DATA_ENV = "RUSSIAN_TEXT_STRESSER_DATA_DIR"

_russian_stresser = None
_setup_lock = threading.Lock()
_setup_attempted = False
_setup_failed = False

def _has_russian_stress_runtime() -> bool:
    try:
        module = importlib.import_module("russian_text_stresser.russian_dictionary")
        return hasattr(module, "DATA_DIR_ENV_VAR")
    except Exception:
        return False


def _get_russian_stress_data_dir() -> Path:
    base_dir = Path(unified_downloader.get_organized_path("chatterbox_official_23lang"))
    data_dir = base_dir / "russian_text_stresser"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _ensure_russian_stress_data() -> bool:
    data_dir = _get_russian_stress_data_dir()
    required_files = [
        data_dir / "russian_dict.db",
        data_dir / "simple_cases.pkl",
    ]

    if all(path.exists() for path in required_files):
        print("📁 Using cached Russian stress dictionary")
        os.environ[RUSSIAN_STRESS_DATA_ENV] = str(data_dir)
        return True

    zip_path = data_dir / "russian_dict.zip"
    if not unified_downloader.download_file(
        RUSSIAN_STRESS_DATA_URL,
        str(zip_path),
        "Official 23-Lang Russian stress dictionary",
    ):
        return False

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
    except Exception as e:
        logger.warning("Failed to extract Russian stress dictionary: %s", e)
        return False
    finally:
        try:
            if zip_path.exists():
                zip_path.unlink()
        except Exception:
            pass

    if not all(path.exists() for path in required_files):
        logger.warning("Russian stress dictionary extracted but expected files are missing")
        return False

    os.environ[RUSSIAN_STRESS_DATA_ENV] = str(data_dir)
    return True


def get_russian_text_stresser():
    global _russian_stresser, _setup_attempted, _setup_failed

    if _russian_stresser is not None:
        return _russian_stresser

    if _setup_failed and _setup_attempted:
        return None

    with _setup_lock:
        if _russian_stresser is not None:
            return _russian_stresser

        if _setup_failed and _setup_attempted:
            return None

        _setup_attempted = True
        print("🔤 Preparing Russian stress support for ChatterBox Official 23-Lang")

        if not _has_russian_stress_runtime():
            _setup_failed = True
            logger.warning(
                "Russian stress support package is missing or outdated. "
                "Rerun ComfyUI Manager install or install.py to add the patched dependency."
            )
            return None

        if not _ensure_russian_stress_data():
            _setup_failed = True
            logger.warning("Russian stress support setup failed during dictionary download")
            return None

        try:
            from russian_text_stresser.text_stresser import RussianTextStresser

            _russian_stresser = RussianTextStresser()
            print("✅ Russian stress support ready")
            return _russian_stresser
        except Exception as e:
            _setup_failed = True
            logger.warning("Russian stress support initialization failed: %s", e)
            return None
