from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# Import TTS message classes so they're available in globals() for deserialization
from .tts import (  # noqa: F401
    BatchTTSFrontendMsg,
    TTSAudioReply,
    TTSCodesReply,
)
from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        return deserialize_type(globals(), json)
