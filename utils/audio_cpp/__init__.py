"""Pinned audio.cpp integration data, discovery, and installation helpers."""

from .catalog import (
    AUDIO_CPP_RELEASE_COMMIT,
    AUDIO_CPP_RELEASE_TAG,
    AUDIO_CPP_RELEASE_VERSION,
    AudioCppCatalog,
    FamilyRecord,
    PackageRecord,
    family_choices,
    get_family,
    get_model_specs_dir,
    get_package,
    load_catalog,
    package_choices,
    recommended_package,
    resolve_task,
)
from .settings import AudioCppSettings, get_settings_path, load_settings, save_settings
from .resolver import AudioCppResolutionError, resolve_audio_cpp_config

__all__ = [
    "AUDIO_CPP_RELEASE_COMMIT",
    "AUDIO_CPP_RELEASE_TAG",
    "AUDIO_CPP_RELEASE_VERSION",
    "AudioCppCatalog",
    "AudioCppSettings",
    "AudioCppResolutionError",
    "FamilyRecord",
    "PackageRecord",
    "family_choices",
    "get_family",
    "get_model_specs_dir",
    "get_package",
    "get_settings_path",
    "load_catalog",
    "load_settings",
    "package_choices",
    "recommended_package",
    "resolve_audio_cpp_config",
    "resolve_task",
    "save_settings",
]
