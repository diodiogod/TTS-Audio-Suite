"""
Bridge package for ``engines.rvc.training.runtime_shim.lib``.

The bundled runtime ships a real ``lib/__init__.py``, so Python would
otherwise lock onto that package and ignore local shadow modules under this
bridge package. Extend ``__path__`` so local overrides win first and the rest
of the bundled runtime tree remains available as fallback.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


BRIDGE_LIB_ROOT = Path(__file__).resolve().parent
REFERENCE_LIB_ROOT = (
    Path(__file__).resolve().parents[2] / "bundled_runtime" / "lib"
)

if not REFERENCE_LIB_ROOT.exists():
    raise FileNotFoundError(f"Missing bundled RVC runtime lib package: {REFERENCE_LIB_ROOT}")

__path__ = [str(BRIDGE_LIB_ROOT), str(REFERENCE_LIB_ROOT)]

# Mirror the useful public symbols from the bundled ``lib/__init__.py`` so
# broken absolute imports like ``from lib import ObjectNamespace`` keep working.
REFERENCE_LIB_INIT = REFERENCE_LIB_ROOT / "__init__.py"


def _load_reference_lib_init():
    module_name = f"{__name__}__reference_init"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(module_name, REFERENCE_LIB_INIT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {REFERENCE_LIB_INIT}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_reference_lib = _load_reference_lib_init()
for _name in (
    "get_cwd",
    "ObjectNamespace",
    "PersistedDict",
    "BASE_DIR",
    "BASE_MODELS_DIR",
    "SONG_DIR",
    "BASE_CACHE_DIR",
    "DATASETS_DIR",
    "LOG_DIR",
    "OUTPUT_DIR",
):
    globals()[_name] = getattr(_reference_lib, _name)

# The vendored reference code uses broken absolute imports like ``from lib.utils``.
# Publish this bridge package under the legacy top-level name so those imports
# resolve back into the bridge instead of relying on fragile sys.path state.
sys.modules["lib"] = sys.modules[__name__]
