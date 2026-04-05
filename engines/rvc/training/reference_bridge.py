"""
Bridge loader for the bundled RVC training runtime.

This intentionally bypasses unrelated custom-node package initialization. We only
expose the small bundled training runtime needed by the RVC trainer.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


_REFERENCE_PACKAGE_NAME = "engines.rvc.training.runtime_shim"
_REFERENCE_ROOT = (
    Path(__file__).resolve().parents[3] / "engines" / "rvc" / "training" / "bundled_runtime"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _ensure_reference_package() -> str:
    if not _REFERENCE_ROOT.exists():
        raise FileNotFoundError(f"Missing bundled RVC runtime package: {_REFERENCE_ROOT}")

    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.append(str(_PROJECT_ROOT))

    importlib.import_module(_REFERENCE_PACKAGE_NAME)

    return _REFERENCE_PACKAGE_NAME


def import_reference_module(module_name: str):
    package_name = _ensure_reference_package()
    return importlib.import_module(f"{package_name}.{module_name}")


def reference_root() -> Path:
    _ensure_reference_package()
    return _REFERENCE_ROOT
