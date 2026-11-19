from __future__ import annotations

# TTS Audio Suite Patch: the offline API (TTSLLM, tts/llm.py) and its sequence helpers are not part of this copy;
# engines/zonos3_mini/zonos3_mini_fast.py drives the scheduler instead.
from ..message.tts import TTSSamplingParams

__all__ = ["TTSSamplingParams"]
