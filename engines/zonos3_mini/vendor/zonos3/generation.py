"""The generation registry (``registries/generation.json``): the request defaults and limits of every server.

``SAMPLING`` holds the v2 sampling defaults (``servers/api/openapi.yaml`` repeats them), the rest the request limits the
request rules (``zonos3.tts.plan``) and the servers' upload paths enforce. The ggml server embeds the same file at
build time; the in-browser server imports it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

REGISTRY = Path(__file__).resolve().parent / "registries" / "generation.json"

_data = json.loads(REGISTRY.read_text(encoding="utf-8"))
SAMPLE_RATE: int = int(_data["sample_rate"])
# temperature, topk, top_p, min_p, repetition_penalty, repetition_window, repetition_codebooks
SAMPLING = MappingProxyType(dict(_data["sampling"]))
SILENCE_PREFILL_DEFAULT_FRAMES: int = int(_data["silence_prefill_frames"]["default"])
SILENCE_PREFILL_MAX_FRAMES: int = int(_data["silence_prefill_frames"]["max"])
DECODE_CONTEXT_FRAMES: int = int(_data["decode_context_frames"])
REFERENCE_MAX_FRAMES: int = int(_data["reference_max_frames"])
MAX_FRAMES: int = int(_data["max_frames"])
REFERENCE_AUDIO_MAX_SECONDS: float = float(_data["audio_max_seconds"]["reference"])
UPLOAD_MAX_SECONDS: float = float(_data["audio_max_seconds"]["upload"])
EDIT_MAX_SECONDS: float = float(_data["audio_max_seconds"]["edit"])
# prefix streams: decode a frame once its first STREAM_CODEBOOKS codebooks arrived, for the "first" chunk or "all"
STREAM_CODEBOOKS: int = int(_data["stream_codebooks"]["default"])
STREAM_CODEBOOKS_SCOPE: str = str(_data["stream_codebooks"]["scope"])
# the soft limiter's knee after the decoder's output gain (zonos3.tts.audio.output_pcm)
OUTPUT_LIMIT_DBFS: float = float(_data["output_limit_dbfs"])
del _data
