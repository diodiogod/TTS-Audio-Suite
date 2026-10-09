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
    "kernels>=0.6.1,<=0.9",
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

RUNTIME_PROFILES: Dict[str, RuntimeProfile] = {
    "vibevoice_transformers4_shared": RuntimeProfile(
        name="vibevoice_transformers4_shared",
        engine_names=["vibevoice", "step_audio_editx"],
        python_path_hint="runtimes/shared_legacy_t4/Scripts/python.exe",
        description="Shared legacy Transformers 4 runtime for engines that still require the older dependency stack.",
        inherit_base_site_packages=True,
        pip_packages=list(_VIBEVOICE_T4_PACKAGES),
        pip_packages_no_deps=[
            "git+https://github.com/FushionHub/VibeVoice.git",
        ],
    ),

}


_LEGACY_DEDICATED_PROFILES = {
    "vibevoice_transformers4_dedicated",
    "qwen3_tts_transformers4_dedicated",
    "step_audio_editx_transformers4",
}


def get_runtime_profile(profile_name: Optional[str]) -> Optional[RuntimeProfile]:
    if not profile_name:
        return None
    if profile_name in _LEGACY_DEDICATED_PROFILES:
        profile_name = "vibevoice_transformers4_shared"
    return RUNTIME_PROFILES.get(profile_name)


def list_runtime_profiles() -> List[RuntimeProfile]:
    return list(RUNTIME_PROFILES.values())
