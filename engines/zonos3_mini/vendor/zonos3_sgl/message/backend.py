from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# Import TTS message classes so they're available in globals() for deserialization
from .tts import (  # noqa: F401
    BatchTTSBackendMsg,
    TTSSamplingParams,
    TTSUserMsg,
)
from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class ExitMsg(BaseBackendMsg):
    pass
