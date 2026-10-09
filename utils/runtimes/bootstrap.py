from __future__ import annotations

"""
Resolve and validate prepared runtimes without installing dependencies.
"""

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .profiles import RuntimeProfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "runtimes"
BOOTSTRAP_STRATEGY = "profile_packages_v6"


def _venv_python_path(runtime_dir: Path) -> Path:
    if os.name == "nt":
        return runtime_dir / "Scripts" / "python.exe"
    return runtime_dir / "bin" / "python"


def resolve_runtime_dir(profile: RuntimeProfile) -> Path:
    hint = Path(profile.python_path_hint or "")
    if hint.parts and hint.parts[0] == "runtimes" and len(hint.parents) >= 2:
        return PROJECT_ROOT / hint.parent.parent
    return RUNTIME_ROOT / profile.name


def resolve_runtime_python(profile: RuntimeProfile) -> Path:
    return _venv_python_path(resolve_runtime_dir(profile))


def _venv_site_packages_path(runtime_dir: Path) -> Path:
    if os.name == "nt":
        return runtime_dir / "Lib" / "site-packages"
    major = sys.version_info.major
    minor = sys.version_info.minor
    return runtime_dir / "lib" / f"python{major}.{minor}" / "site-packages"


def runtime_is_ready(profile: RuntimeProfile, *, base_python: Optional[str] = None) -> bool:
    runtime_dir = resolve_runtime_dir(profile)
    if not _venv_python_path(runtime_dir).is_file():
        return False
    try:
        metadata = json.loads((runtime_dir / "runtime_metadata.json").read_text(encoding="utf-8"))
        return (
            metadata.get("bootstrap_strategy") == BOOTSTRAP_STRATEGY
            and metadata.get("install_complete") is True
            and metadata.get("profile") == asdict(profile)
            and os.path.normcase(os.path.realpath(metadata.get("source_python", "")))
                == os.path.normcase(os.path.realpath(base_python or sys.executable))
        )
    except (OSError, ValueError, TypeError):
        return False


def ensure_runtime(profile: RuntimeProfile, *, base_python: Optional[str] = None) -> Path:
    """Generation only reuses prepared environments; missing/stale ones need repair."""
    if runtime_is_ready(profile, base_python=base_python):
        return resolve_runtime_python(profile)
    raise RuntimeError(
        "Shared Runtime is missing or needs repair. Enable 'Install shared runtime automatically' "
        "in ComfyUI Settings > TTS Audio Suite > Runtime installation, then repair/reinstall "
        "TTS Audio Suite through Manager. Restart if Manager requests it. "
        "Running a workflow or restarting alone does not install dependencies. "
        f"Runtime folder: {resolve_runtime_dir(profile)}"
    )
