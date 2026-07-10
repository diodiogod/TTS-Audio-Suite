"""
Isolated engine runtime scaffolding.

This package defines the stable contract for running fragile engines in
separate Python environments without changing the existing ComfyUI-side UI.
"""

from .bootstrap import ensure_runtime, resolve_runtime_dir, resolve_runtime_python
from .higgs_audio_proxy import HiggsAudioIsolatedProxy, build_higgs_audio_isolated_proxy
from .fish_audio_s2_proxy import FishAudioS2Proxy, build_fish_audio_s2_proxy
from .launcher import IsolatedRuntimeLauncher
from .profiles import RuntimeProfile, get_runtime_profile, list_runtime_profiles
from .protocol import RuntimeJobRequest, RuntimeJobResponse
from .qwen3_asr_proxy import Qwen3ASRIsolatedProxy, build_qwen3_asr_isolated_proxy
from .qwen3_tts_proxy import Qwen3TTSIsolatedProxy, build_qwen3_tts_isolated_proxy
from .vibevoice_proxy import VibeVoiceIsolatedProxy, build_vibevoice_isolated_proxy

__all__ = [
    "ensure_runtime",
    "resolve_runtime_dir",
    "resolve_runtime_python",
    "HiggsAudioIsolatedProxy",
    "IsolatedRuntimeLauncher",
    "RuntimeProfile",
    "RuntimeJobRequest",
    "RuntimeJobResponse",
    "build_higgs_audio_isolated_proxy",
    "FishAudioS2Proxy",
    "build_fish_audio_s2_proxy",
    "Qwen3ASRIsolatedProxy",
    "build_qwen3_asr_isolated_proxy",
    "Qwen3TTSIsolatedProxy",
    "build_qwen3_tts_isolated_proxy",
    "VibeVoiceIsolatedProxy",
    "build_vibevoice_isolated_proxy",
    "get_runtime_profile",
    "list_runtime_profiles",
]
