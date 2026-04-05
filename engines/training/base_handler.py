"""
Base abstractions for engine-specific training handlers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from utils.config_sanitizer import ConfigSanitizer


def unpack_tts_engine(tts_engine: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Normalize a TTS_ENGINE value into ``(engine_type, config_dict)``.

    The suite currently uses two shapes:
    - dict-based engine configs for most engines
    - adapter instances for RVC
    """
    if hasattr(tts_engine, "engine_type"):
        engine_type = getattr(tts_engine, "engine_type", None)
        config = getattr(tts_engine, "config", {}) or {}
        if not engine_type:
            raise ValueError("TTS engine adapter is missing engine_type")
        if not isinstance(config, dict):
            raise ValueError(f"{engine_type} engine config must be a dict")
        return engine_type, ConfigSanitizer.sanitize(config)

    if isinstance(tts_engine, dict):
        engine_type = tts_engine.get("engine_type")
        config = tts_engine.get("config", {}) or {}
        if not config:
            config = tts_engine
        if not engine_type:
            raise ValueError("TTS engine is missing engine_type")
        if not isinstance(config, dict):
            raise ValueError(f"{engine_type} engine config must be a dict")
        return engine_type, ConfigSanitizer.sanitize(config)

    raise ValueError("Invalid TTS_engine input")


class BaseTrainingHandler(ABC):
    """
    Base class for engine-specific training backends.
    """

    engine_type: str = ""
    artifact_type: str = "model"

    def ensure_engine_type(self, tts_engine: Any) -> Dict[str, Any]:
        engine_type, config = unpack_tts_engine(tts_engine)
        if engine_type != self.engine_type:
            raise ValueError(
                f"This training handler only supports '{self.engine_type}', got '{engine_type}'"
            )
        return config

    @abstractmethod
    def build_default_training_config(self, tts_engine: Any) -> Dict[str, Any]:
        """Return a default TRAINING_CONFIG payload for this engine."""

    @abstractmethod
    def prepare_dataset(self, tts_engine: Any, **kwargs) -> Dict[str, Any]:
        """Create a TRAINING_DATASET payload for this engine."""

    @abstractmethod
    def train(
        self,
        tts_engine: Any,
        training_dataset: Dict[str, Any],
        training_config: Dict[str, Any],
        output_name: str = "",
        resume: bool = False,
        overwrite: bool = False,
        continue_from: Any = None,
        node_id: str = "",
    ) -> Dict[str, Any]:
        """Run the engine-specific training job and return TRAINING_ARTIFACTS."""
