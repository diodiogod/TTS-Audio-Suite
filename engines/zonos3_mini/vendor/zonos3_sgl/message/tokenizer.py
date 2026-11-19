from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# Import TTS message classes so they're available in globals() for deserialization
from .tts import (  # noqa: F401
    BatchTTSTokenizerMsg,
    TTSDecodeCodesMsg,
    TTSDetokenizeMsg,
    TTSNormalizeMsg,
    TTSSamplingParams,
    TTSTokenizeMsg,
)
from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
