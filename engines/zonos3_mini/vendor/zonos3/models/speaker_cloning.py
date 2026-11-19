import os
from contextlib import nullcontext
from functools import cache
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# TTS Audio Suite Patch: the legacy Qwen3 / W2V-BERT speaker backends (transformers, huggingface_hub downloads) are
# removed; only the speaker encoder registry (ownmect_s2) remains.


def speaker_embedder_device() -> str:
    """The servers' speaker-encoder device: ``ZONOS3_SPEAKER_EMBEDDER_DEVICE``, else the CPU (it does not occupy GPU
    memory next to the TTS model)."""
    return os.environ.get("ZONOS3_SPEAKER_EMBEDDER_DEVICE", "cpu")


def resolve_speaker_embedding_backend(codec: str | None, expected_dim: int) -> str:
    """Resolve a checkpoint codec to an installed reference-audio encoder."""
    normalized = str(codec or "").strip().lower().replace("-", "_")
    # TTS Audio Suite Patch: no built-in legacy aliases.
    if normalized:
        # TTS Audio Suite Patch: (alias lookup removed)
        # codecs beyond the built-in encoders (e.g. ownmect_s2) come from the speaker encoder registry, keyed by the
        # exact training codec name; the backend key is that name
        from zonos3.models.mect_speaker import registry_entry, registry_path

        entry = registry_entry(codec)
        if entry is not None:
            if int(entry.get("embed_dim", expected_dim)) != int(expected_dim):
                raise ValueError(
                    f"Speaker embedding codec {codec!r}: the checkpoint expects {expected_dim} dimensions but the "
                    f"registry entry produces {entry.get('embed_dim')} ({registry_path()})."
                )
            return entry["codec"]
        raise ValueError(
            f"Unsupported speaker embedding codec: {codec!r} (not a built-in encoder and not in the speaker encoder "
            f"registry {registry_path()})."
        )

    # TTS Audio Suite Patch: legacy 2048D / 256D fallbacks removed.
    raise ValueError(f"Unsupported speaker embedding codec: {codec!r} (the checkpoint does not identify one).")


def create_speaker_embedder(
    codec: str | None,
    expected_dim: int,
    device: str = "cpu",
) -> nn.Module:
    backend = resolve_speaker_embedding_backend(codec, expected_dim)
    # TTS Audio Suite Patch: registry branch only.
    from zonos3.models.mect_speaker import create_registry_speaker_embedder, registry_entry

    entry = registry_entry(backend)
    if entry is not None:
        return create_registry_speaker_embedder(entry, device=device)
    raise AssertionError(f"Unhandled speaker embedding backend: {backend}")
