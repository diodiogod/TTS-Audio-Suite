"""
Load maintained RVC infer-pack modules by file path.

This avoids importing ``engines.rvc`` as a package during Windows spawn,
which would otherwise drag in unrelated engine imports and path conflicts.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MAINTAINED_INFER_PACK_ROOT = (
    Path(__file__).resolve().parents[4] / "impl" / "lib" / "infer_pack"
)


def load_maintained_module(source_filename: str, caller_name: str, caller_package: str):
    source_path = MAINTAINED_INFER_PACK_ROOT / source_filename
    if not source_path.exists():
        raise FileNotFoundError(f"Missing maintained infer-pack module: {source_path}")

    module_name = f"{caller_name}__maintained"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {source_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = caller_package
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def export_public_symbols(source_filename: str, caller_globals: dict):
    module = load_maintained_module(
        source_filename=source_filename,
        caller_name=caller_globals["__name__"],
        caller_package=caller_globals["__package__"],
    )
    exported = getattr(module, "__all__", None)
    if exported is None:
        exported = [name for name in module.__dict__ if not name.startswith("_")]

    for name in exported:
        caller_globals[name] = getattr(module, name)

    caller_globals["__all__"] = exported
    caller_globals["__doc__"] = getattr(module, "__doc__", caller_globals.get("__doc__"))
    return module
