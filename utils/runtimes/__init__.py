"""
Isolated engine runtime scaffolding.

This package defines the stable contract for running fragile engines in
separate Python environments without changing the existing ComfyUI-side UI.
"""

from .bootstrap import ensure_runtime, resolve_runtime_dir, resolve_runtime_python
from .launcher import IsolatedRuntimeLauncher
from .profiles import RuntimeProfile, get_runtime_profile, list_runtime_profiles
from .protocol import RuntimeJobRequest, RuntimeJobResponse
from .vibevoice_proxy import VibeVoiceIsolatedProxy, build_vibevoice_isolated_proxy

__all__ = [
    "ensure_runtime",
    "resolve_runtime_dir",
    "resolve_runtime_python",
    "IsolatedRuntimeLauncher",
    "RuntimeProfile",
    "RuntimeJobRequest",
    "RuntimeJobResponse",
    "VibeVoiceIsolatedProxy",
    "build_vibevoice_isolated_proxy",
    "get_runtime_profile",
    "list_runtime_profiles",
]
