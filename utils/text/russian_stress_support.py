"""
Lazy Russian stress support for ChatterBox Official 23-Lang.

This keeps the heavy Russian dictionary assets out of the global installer and
downloads them on demand into the organized TTS model cache.
"""

import importlib
import logging
import os
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

from utils.downloads.unified_downloader import unified_downloader

logger = logging.getLogger(__name__)

RUSSIAN_STRESS_FORK_REF = "git+https://github.com/diodiogod/add-stress-to-epub.git@98f53b9"
RUSSIAN_STRESS_DATA_URL = "https://github.com/Vuizur/add-stress-to-epub/releases/download/v1.0.1/russian_dict.zip"
RUSSIAN_STRESS_DATA_ENV = "RUSSIAN_TEXT_STRESSER_DATA_DIR"

_russian_stresser = None
_setup_lock = threading.Lock()
_setup_attempted = False
_setup_failed = False


def _purge_stresser_modules():
    for module_name in list(sys.modules.keys()):
        if module_name == "russian_text_stresser" or module_name.startswith("russian_text_stresser."):
            sys.modules.pop(module_name, None)


def _check_package_installed(package_spec: str) -> bool:
    try:
        import re
        from importlib.metadata import version

        match = re.match(r"^([a-zA-Z0-9_.\\-]+)([><=!~]+)?(.+)?$", package_spec)
        if not match:
            return False

        package_name = match.group(1)
        operator = match.group(2)
        required_version = match.group(3)

        installed_version = version(package_name)
        if not operator or not required_version:
            return True

        try:
            from packaging.specifiers import SpecifierSet
            from packaging.version import Version

            spec = SpecifierSet(f"{operator}{required_version}")
            return Version(installed_version) in spec
        except Exception:
            return True
    except Exception:
        return False


def _run_pip_install(args, description: str) -> bool:
    cmd = [sys.executable, "-m", "pip"] + args
    logger.info("%s", description)
    print(f"📦 {description}")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=1800,
        )
    except Exception as e:
        logger.warning("%s failed: %s", description, e)
        return False

    if result.returncode == 0:
        return True

    if result.stdout:
        logger.warning("%s stdout: %s", description, result.stdout.strip())
    if result.stderr:
        logger.warning("%s stderr: %s", description, result.stderr.strip())
    logger.warning("%s failed with exit code %s", description, result.returncode)
    return False


def _fork_supports_external_data_dir() -> bool:
    try:
        module = importlib.import_module("russian_text_stresser.russian_dictionary")
        return hasattr(module, "DATA_DIR_ENV_VAR")
    except Exception:
        return False


def _ensure_python_dependencies() -> bool:
    dependency_specs = [
        "poetry-core>=2.0.0",
        "spacy<4",
        "pymorphy2>=0.9.1",
        "stressed-cyrillic-tools>=0.1.10",
    ]

    for spec in dependency_specs:
        if _check_package_installed(spec):
            continue
        if not _run_pip_install(["install", spec], f"Installing Russian stress dependency {spec}"):
            return False

    if not _fork_supports_external_data_dir():
        if not _run_pip_install(
            [
                "install",
                "--force-reinstall",
                "--no-deps",
                "--no-build-isolation",
                "--ignore-requires-python",
                RUSSIAN_STRESS_FORK_REF,
            ],
            "Installing patched russian-text-stresser",
        ):
            return False

    _purge_stresser_modules()
    importlib.invalidate_caches()
    return True


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

        if not _ensure_python_dependencies():
            _setup_failed = True
            logger.warning("Russian stress support setup failed during dependency install")
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
