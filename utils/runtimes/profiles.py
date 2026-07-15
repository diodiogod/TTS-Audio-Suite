from __future__ import annotations

"""
Named isolated-runtime profiles.

Profiles describe which external Python runtime should be used for a fragile
engine family. Paths stay user-configurable; these names are the stable keys.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    engine_names: List[str]
    python_path_hint: Optional[str] = None
    description: str = ""
    runtime_mode: str = "isolated"
    env_vars: Dict[str, str] = field(default_factory=dict)
    inherit_base_site_packages: bool = False
    pip_packages: List[str] = field(default_factory=list)
    pip_packages_no_deps: List[str] = field(default_factory=list)


_VIBEVOICE_T4_PACKAGES = [
    "numpy>=1.26.4,<2.3.0",
    "soundfile>=0.12.0",
    "omegaconf>=2.3.0",
    "transformers>=4.51.3,<=4.57.3",
    "accelerate",
    "requests",
    "av",
    "bitsandbytes>=0.47.0",
    "safetensors>=0.6.2",
    "sentencepiece>=0.2.1",
    "tqdm",
    "scipy",
    "librosa",
    "llvmlite>=0.40.0",
    "numba>=0.57.0",
    "diffusers",
    "ml-collections",
    "absl-py",
    "conformer>=0.3.2",
    "x-transformers",
]

_QWEN3_T4_PACKAGES = [
    "numpy>=1.26.4,<2.3.0",
    "soundfile>=0.12.0",
    "librosa",
    "transformers>=4.51.3,<=4.57.3",
    "accelerate",
    "huggingface-hub<1.0",
    "safetensors>=0.6.2",
]


RUNTIME_PROFILES: Dict[str, RuntimeProfile] = {
    "vibevoice_transformers4_shared": RuntimeProfile(
        name="vibevoice_transformers4_shared",
        engine_names=["vibevoice"],
        python_path_hint="runtimes/shared_legacy_t4/Scripts/python.exe",
        description="Shared legacy Transformers 4 runtime for VibeVoice/Kugel and similar engines.",
        inherit_base_site_packages=True,
        pip_packages=list(_VIBEVOICE_T4_PACKAGES),
        pip_packages_no_deps=[
            "git+https://github.com/FushionHub/VibeVoice.git",
        ],
    ),
    "vibevoice_transformers4_dedicated": RuntimeProfile(
        name="vibevoice_transformers4_dedicated",
        engine_names=["vibevoice"],
        python_path_hint="runtimes/vibevoice_t4_dedicated/Scripts/python.exe",
        description="Dedicated legacy Transformers 4 runtime reserved for VibeVoice/Kugel only.",
        inherit_base_site_packages=True,
        pip_packages=list(_VIBEVOICE_T4_PACKAGES),
        pip_packages_no_deps=[
            "git+https://github.com/FushionHub/VibeVoice.git",
        ],
    ),
    "qwen3_tts_transformers4_dedicated": RuntimeProfile(
        name="qwen3_tts_transformers4_dedicated",
        engine_names=["qwen3_tts"],
        python_path_hint="runtimes/qwen3_tts_t4_dedicated/Scripts/python.exe",
        description="Dedicated legacy Transformers 4 runtime reserved for Qwen3-TTS.",
        inherit_base_site_packages=True,
        pip_packages=list(_QWEN3_T4_PACKAGES),
    ),
    "step_audio_editx_transformers5": RuntimeProfile(
        name="step_audio_editx_transformers5",
        engine_names=["step_audio_editx"],
        python_path_hint="runtimes/step_audio_editx_t5/Scripts/python.exe",
        description="Dedicated Step Audio EditX runtime with pinned remote-code stack.",
    ),
    "moss_tts_transformers5": RuntimeProfile(
        name="moss_tts_transformers5",
        engine_names=["moss_tts"],
        python_path_hint="runtimes/moss_tts_t5/Scripts/python.exe",
        description="Dedicated MOSS-TTS runtime isolated from VibeVoice/Qwen.",
    ),
    "higgs_audio_embedded": RuntimeProfile(
        name="higgs_audio_embedded",
        engine_names=["higgs_audio"],
        description="Placeholder profile for future Higgs isolation if needed.",
        runtime_mode="embedded",
    ),
}


def get_runtime_profile(profile_name: Optional[str]) -> Optional[RuntimeProfile]:
    if not profile_name:
        return None
    return RUNTIME_PROFILES.get(profile_name)


def list_runtime_profiles() -> List[RuntimeProfile]:
    return list(RUNTIME_PROFILES.values())
