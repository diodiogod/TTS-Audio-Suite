"""
Engine Capability Registry

Defines what each TTS engine can do and any special handling requirements.
Allows generic code to handle engine-specific needs without hardcoded if/else checks.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any


@dataclass
class EngineCapabilities:
    """
    Describes what an engine can do and how to handle it.

    Note: This is about MODEL LOADING decisions, not generation-time language handling.
    Language parameters used during generation are handled by each engine separately.

    Attributes:
        supports_voice_conversion: Engine has VC capability
        multilingual_model_switching: Whether engine loads different model weights per language:
            - False: Load ONE model for all supported languages (ChatterBox 23-Lang, VibeVoice, F5-TTS)
                     Language handling happens at generation time within the engine
            - True: Load DIFFERENT models based on language (Classic ChatterBox, F5-TTS variants)
                    Each language has its own model weights
        can_corrupt_on_reload: Model can become corrupted if reloaded (needs recovery)
        recovery_handler: Function to call if model corruption detected
        requires_special_init: Engine needs special initialization
        fallback_languages: Languages to try if primary loading fails
    """

    supports_voice_conversion: bool = False
    multilingual_model_switching: bool = False  # True = load different models per language
    can_corrupt_on_reload: bool = False
    recovery_handler: Optional[Callable] = None
    requires_special_init: bool = False
    fallback_languages: list = None

    def __post_init__(self):
        if self.fallback_languages is None:
            self.fallback_languages = ["English"]


# Registry of all TTS engines and their capabilities
ENGINE_REGISTRY: Dict[str, EngineCapabilities] = {
    "chatterbox": EngineCapabilities(
        supports_voice_conversion=True,
        multilingual_model_switching=True,  # Load different models per language
        can_corrupt_on_reload=False,
        fallback_languages=["English", "German", "Italian"],
    ),

    "chatterbox_official_23lang": EngineCapabilities(
        supports_voice_conversion=True,
        multilingual_model_switching=False,  # One model for all 23 languages
        can_corrupt_on_reload=False,
        fallback_languages=["English"],
    ),

    "f5tts": EngineCapabilities(
        supports_voice_conversion=False,
        multilingual_model_switching=True,  # Load language-specific models (F5-FR, F5-PT-BR, etc)
        can_corrupt_on_reload=False,
        fallback_languages=["English"],
    ),

    "higgs_audio": EngineCapabilities(
        supports_voice_conversion=False,
        multilingual_model_switching=False,  # Single model
        can_corrupt_on_reload=True,  # Higgs has CUDA graph corruption issues
        recovery_handler=None,  # Would be set to reset function
        requires_special_init=True,
    ),

    "rvc": EngineCapabilities(
        supports_voice_conversion=True,
        multilingual_model_switching=False,  # Single model
        can_corrupt_on_reload=False,
        fallback_languages=[],
    ),

    "vibevoice": EngineCapabilities(
        supports_voice_conversion=False,
        multilingual_model_switching=False,  # One model handles multiple languages
        can_corrupt_on_reload=False,
        fallback_languages=["English"],
    ),

    "index_tts": EngineCapabilities(
        supports_voice_conversion=False,
        multilingual_model_switching=False,  # Single model
        can_corrupt_on_reload=False,
        fallback_languages=[],
    ),
}


def get_engine_capabilities(engine_name: str) -> Optional[EngineCapabilities]:
    """
    Get capabilities for an engine.

    Args:
        engine_name: Name of the engine

    Returns:
        EngineCapabilities if engine is registered, None otherwise
    """
    return ENGINE_REGISTRY.get(engine_name)


def engine_supports_feature(engine_name: str, feature: str) -> bool:
    """
    Check if engine supports a feature.

    Args:
        engine_name: Name of the engine
        feature: Feature name ("voice_conversion", "multilingual")

    Returns:
        True if engine supports the feature
    """
    caps = get_engine_capabilities(engine_name)
    if not caps:
        return False

    feature_map = {
        "voice_conversion": caps.supports_voice_conversion,
        "multilingual": caps.supports_multilingual,
        "corruption_recovery": caps.can_corrupt_on_reload,
    }

    return feature_map.get(feature, False)


def register_engine_recovery_handler(engine_name: str, handler: Callable) -> None:
    """
    Register a recovery handler for an engine.

    Called at startup to register recovery functions for engines that can corrupt.

    Args:
        engine_name: Name of the engine
        handler: Function to call for recovery
    """
    if engine_name in ENGINE_REGISTRY:
        ENGINE_REGISTRY[engine_name].recovery_handler = handler


def get_recovery_handler(engine_name: str) -> Optional[Callable]:
    """
    Get recovery handler for an engine if it has corruption issues.

    Args:
        engine_name: Name of the engine

    Returns:
        Recovery function if engine can corrupt, None otherwise
    """
    caps = get_engine_capabilities(engine_name)
    if caps and caps.can_corrupt_on_reload:
        return caps.recovery_handler
    return None
