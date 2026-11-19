"""
ZONOS3-Mini engine integration (vendored official `zonos3.reference` implementation).
"""

import os
import sys

VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def ensure_vendor_path() -> str:
    """Put the vendored `zonos3` package first on sys.path; refuse a foreign `zonos3` that is already imported."""
    loaded = sys.modules.get("zonos3")
    loaded_file = getattr(loaded, "__file__", None)
    if loaded_file and not os.path.abspath(loaded_file).startswith(VENDOR_DIR + os.sep):
        raise ImportError(f"ZONOS3-Mini: a different 'zonos3' package is already imported from {loaded_file}")
    if VENDOR_DIR not in sys.path:
        sys.path.insert(0, VENDOR_DIR)
    return VENDOR_DIR
