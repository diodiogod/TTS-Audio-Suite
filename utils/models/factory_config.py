"""
Standardized Model Loading Configuration

Unified configuration object for model loading across the entire system.
Used by both the unified interface and all model factory functions.
Ensures all TTS engines use consistent parameters for model loading.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from utils.device import resolve_torch_device


@dataclass
class ModelLoadConfig:
    """
    Standard configuration for model loading and factory functions.

    All TTS engines (ChatterBox, F5-TTS, Higgs, RVC, VibeVoice, etc.)
    use this same config object. Reduces code duplication and makes
    adding new engines straightforward.

    Attributes:
        device: Device to load model on ("auto", "cuda", "mps", "xpu", "cpu")
        model_name: Name/identifier of the model
        engine_name: Optional engine identifier (e.g., "chatterbox", "f5tts")
        model_type: Optional model type (e.g., "tts", "vc", "tokenizer")
        language: Optional language code (for multilingual engines)
        model_path: Optional local path to model directory
        repo_id: Optional HuggingFace repository ID
        additional_params: Engine-specific parameters (quantization, etc.)
    """

    # Required parameters
    device: str
    model_name: str

    # Interface parameters (optional, used by unified interface)
    engine_name: Optional[str] = None
    model_type: Optional[str] = None

    # Optional parameters for model loading
    language: Optional[str] = None
    model_path: Optional[str] = None
    repo_id: Optional[str] = None

    # Engine-specific parameters
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Post-initialization validation and normalization.

        - Resolve device to actual device ("auto" → "cuda"/"mps"/"cpu")
        - Validate required parameters
        """
        # Resolve device immediately for consistency
        self.device = resolve_torch_device(self.device)

        # Normalize model_name (remove "local:" prefix if present)
        if self.model_name and self.model_name.startswith("local:"):
            self.model_name = self.model_name.replace("local:", "")

        # Ensure additional_params is a dict
        if self.additional_params is None:
            self.additional_params = {}

    def get_cache_key_parts(self) -> Dict[str, Any]:
        """
        Get dictionary of parameters suitable for cache key generation.

        Returns all parameters that should affect caching.
        Used by cache_key_generator.py to create consistent keys.
        """
        return {
            "device": self.device,
            "model_name": self.model_name,
            "language": self.language,
            "model_path": self.model_path,
            "repo_id": self.repo_id,
            **self.additional_params
        }

    def for_engine(self, engine_name: str) -> 'ModelLoadConfig':
        """
        Create a copy configured for a specific engine.

        Allows engines to validate that they received expected parameters.
        """
        return ModelLoadConfig(
            device=self.device,
            model_name=self.model_name,
            language=self.language,
            model_path=self.model_path,
            repo_id=self.repo_id,
            additional_params=dict(self.additional_params)
        )

    def __repr__(self) -> str:
        """User-friendly representation."""
        parts = [
            f"device={self.device}",
            f"model={self.model_name}"
        ]
        if self.language:
            parts.append(f"lang={self.language}")
        if self.model_path:
            parts.append(f"path={self.model_path}")
        if self.repo_id:
            parts.append(f"repo={self.repo_id}")
        if self.additional_params:
            parts.append(f"extra={self.additional_params}")

        return f"ModelLoadConfig({', '.join(parts)})"
