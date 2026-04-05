"""
Bridge infer-pack package for the bundled RVC training runtime.

Local wrapper modules are searched first, then the maintained engine infer-pack
directory is available as fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path


BRIDGE_INFER_PACK_ROOT = Path(__file__).resolve().parent
REFERENCE_INFER_PACK_ROOT = (
    Path(__file__).resolve().parents[4] / "impl" / "lib" / "infer_pack"
)

if not REFERENCE_INFER_PACK_ROOT.exists():
    raise FileNotFoundError(f"Missing maintained RVC infer-pack package: {REFERENCE_INFER_PACK_ROOT}")

__path__ = [str(BRIDGE_INFER_PACK_ROOT), str(REFERENCE_INFER_PACK_ROOT)]

# Some vendored modules import ``infer_pack`` as a top-level package.
sys.modules["infer_pack"] = sys.modules[__name__]
