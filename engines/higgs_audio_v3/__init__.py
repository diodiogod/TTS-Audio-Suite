"""Higgs Audio v3 engine package."""

__all__ = ["HiggsAudioV3Engine"]


def __getattr__(name):
    if name == "HiggsAudioV3Engine":
        from .higgs_audio_v3 import HiggsAudioV3Engine

        return HiggsAudioV3Engine
    raise AttributeError(name)
