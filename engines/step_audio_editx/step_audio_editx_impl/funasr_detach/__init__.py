"""Initialize funasr package."""

import os
import pkgutil
import importlib

dirname = os.path.dirname(__file__)
version_file = os.path.join(dirname, "version.txt")
with open(version_file, "r") as f:
    __version__ = f.read().strip()


import importlib
import pkgutil


def import_submodules(package, recursive=True):
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        try:
            results[name] = importlib.import_module(name)
        except Exception as e:
            # Silently ignore import errors for unused FunASR modules
            # (many bundled modules have missing dependencies we don't need)
            pass
        if recursive and is_pkg:
            results.update(import_submodules(name))
    return results


# TTS Audio Suite Patch: Explicit ParaformerStreaming import
# CRITICAL: Import ParaformerStreaming explicitly BEFORE import_submodules
# to ensure it's registered even if import_submodules fails to import it
# Root cause: paraformer_streaming/model.py imports distutils (removed in Python 3.12+)
# causing silent import failure in import_submodules try/except block
# This explicit import ensures ParaformerStreaming gets registered in tables.model_classes
# Fixes issue #226: 'NoneType' object is not callable at auto_model.py:209
try:
    from funasr_detach.models.paraformer_streaming.model import ParaformerStreaming
except Exception:
    pass  # If it fails, import_submodules might still catch it

import_submodules(__name__)

from funasr_detach.auto.auto_model import AutoModel
from funasr_detach.auto.auto_frontend import AutoFrontend
