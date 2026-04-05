"""
Import bridge package for the bundled RVC training runtime.

This package exists so Windows multiprocessing ``spawn`` can import
``engines.rvc.training.runtime_shim.training_cli`` in child processes.
We intentionally expose the bundled runtime tree as this package's
search path without executing unrelated custom-node code.
"""

from __future__ import annotations

from pathlib import Path


REFERENCE_ROOT = (
    Path(__file__).resolve().parents[1] / "bundled_runtime"
)
BRIDGE_ROOT = Path(__file__).resolve().parent

if not REFERENCE_ROOT.exists():
    raise FileNotFoundError(f"Missing bundled RVC runtime package: {REFERENCE_ROOT}")

# Search the local bridge package first, then fall back to the bundled
# runtime tree. This lets us selectively override broken upstream
# modules such as ``lib.infer_pack.models`` without copying the whole repo.
__path__ = [str(BRIDGE_ROOT), str(REFERENCE_ROOT)]
