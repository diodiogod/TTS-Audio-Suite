"""Basic data types for multimodal ChatML format."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class AudioContent:
    audio_url: str
    # Base64 encoded audio bytes
    raw_audio: Optional[str] = None
    offset: Optional[float] = None
    duration: Optional[float] = None
    row_id: Optional[int] = None
    type: str = "audio"


@dataclass
class TextContent:
    text: str
    type: str = "text"


@dataclass
class Message:
    role: str
    content: Union[str, AudioContent, TextContent, List[Union[str, AudioContent, TextContent]]]
    recipient: Optional[str] = None


@dataclass
class ChatMLSample:
    """Dataclass to hold multimodal ChatML data."""

    messages: List[Message]
    start_index: Optional[int] = None  # We will mask the messages[:start_index] when finetuning the LLM.
    misc: Optional[Dict] = None
    speaker: Optional[str] = None
