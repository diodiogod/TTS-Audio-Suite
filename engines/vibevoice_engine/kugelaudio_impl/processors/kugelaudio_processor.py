"""Main processor for KugelAudio combining text and audio processing."""

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    TruncationStrategy,
)
from transformers.utils import TensorType, cached_file, logging

from .audio_processor import AudioNormalizer, AudioProcessor

logger = logging.get_logger(__name__)


class KugelAudioProcessor:
    """Combined processor for KugelAudio text and audio.

    Wraps a text tokenizer and audio processor into a single interface
    for preparing inputs for KugelAudio models.

    Voice cloning from raw audio is not supported. Instead, pre-encoded
    voices are loaded from .pt files referenced in a voices.json registry.

    Example:
        >>> processor = KugelAudioProcessor.from_pretrained("kugelaudio/kugelaudio-0-open")
        >>> inputs = processor(text="Hello world", voice="default")
    """

    def __init__(
        self,
        tokenizer=None,
        audio_processor: Optional[AudioProcessor] = None,
        speech_compression_ratio: int = 3200,
        db_normalize: bool = True,
        voices_registry: Optional[Dict[str, Any]] = None,
        voices_dir: Optional[str] = None,
        model_name_or_path: Optional[str] = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor or AudioProcessor()
        self.speech_compression_ratio = speech_compression_ratio
        self.db_normalize = db_normalize
        self.audio_normalizer = AudioNormalizer() if db_normalize else None
        self.voices_registry = voices_registry or {}
        self.voices_dir = voices_dir
        self._model_name_or_path = model_name_or_path

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load processor from pretrained model.

        Args:
            pretrained_model_name_or_path: Model ID or local path

        Returns:
            KugelAudioProcessor instance
        """
        from .text_tokenizer import KugelAudioTextTokenizer

        # Try to load config
        config_path = os.path.join(pretrained_model_name_or_path, "preprocessor_config.json")
        config = None

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            try:
                config_file = cached_file(
                    pretrained_model_name_or_path, "preprocessor_config.json", **kwargs
                )
                with open(config_file, "r") as f:
                    config = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load config: {e}. Using defaults.")
                config = {
                    "speech_compression_ratio": 3200,
                    "db_normalize": True,
                }

        # Extract parameters
        speech_compression_ratio = config.get("speech_compression_ratio", 3200)
        db_normalize = config.get("db_normalize", True)

        # Load tokenizer
        lm_name = config.get("language_model_pretrained_name") or kwargs.pop(
            "language_model_pretrained_name", "Qwen/Qwen2.5-1.5B"
        )
        logger.info(f"Loading tokenizer from {lm_name}")
        tokenizer = KugelAudioTextTokenizer.from_pretrained(lm_name, **kwargs)

        # Load audio processor
        if "audio_processor" in config:
            audio_config = config["audio_processor"]
            audio_processor = AudioProcessor(
                sampling_rate=audio_config.get("sampling_rate", 24000),
                normalize_audio=audio_config.get("normalize_audio", True),
                target_dB_FS=audio_config.get("target_dB_FS", -25),
            )
        else:
            audio_processor = AudioProcessor()

        # Load voices registry (voices.json)
        voices_registry = {}
        voices_dir = None
        voices_json_path = os.path.join(pretrained_model_name_or_path, "voices", "voices.json")
        if os.path.exists(voices_json_path):
            with open(voices_json_path, "r") as f:
                voices_registry = json.load(f)
            voices_dir = os.path.join(pretrained_model_name_or_path, "voices")
        else:
            try:
                voices_file = cached_file(
                    pretrained_model_name_or_path, "voices/voices.json", **kwargs
                )
                if voices_file:
                    with open(voices_file, "r") as f:
                        voices_registry = json.load(f)
                    voices_dir = os.path.dirname(voices_file)
            except Exception:
                logger.warning("No voices.json found. Pre-encoded voices will not be available.")

        return cls(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
            speech_compression_ratio=speech_compression_ratio,
            db_normalize=db_normalize,
            voices_registry=voices_registry,
            voices_dir=voices_dir,
            model_name_or_path=pretrained_model_name_or_path,
        )

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """Save processor to directory."""
        os.makedirs(save_directory, exist_ok=True)

        config = {
            "processor_class": "KugelAudioProcessor",
            "speech_compression_ratio": self.speech_compression_ratio,
            "db_normalize": self.db_normalize,
            "audio_processor": {
                "feature_extractor_type": "AudioProcessor",
                "sampling_rate": getattr(self.audio_processor, "sampling_rate", 24000),
                "normalize_audio": getattr(self.audio_processor, "normalize_audio", True),
                "target_dB_FS": getattr(self.audio_processor, "target_dB_FS", -25),
            },
        }

        config_path = os.path.join(save_directory, "preprocessor_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Processor saved to {config_path}")

    def get_available_voices(self) -> List[str]:
        """Return list of available pre-encoded voice names."""
        return list(self.voices_registry.keys())

    def load_voice_cache(self, voice_name: str) -> dict:
        """Load a pre-encoded voice by name from the voices registry.

        Supports loading from local directories and from HuggingFace Hub
        repositories. When the model was loaded from a HuggingFace repo,
        individual .pt voice files are downloaded automatically on demand.

        Args:
            voice_name: Name of the voice (must be in voices.json registry).

        Returns:
            Dict with "acoustic_mean" tensor, suitable for passing as voice_cache.

        Raises:
            ValueError: If voice_name is not found in the registry.
        """
        if voice_name not in self.voices_registry:
            available = ", ".join(self.voices_registry.keys()) or "(none)"
            raise ValueError(f"Voice '{voice_name}' not found. Available voices: {available}")

        voice_info = self.voices_registry[voice_name]
        voice_file = voice_info["file"]
        voice_path = None

        # Strategy 1: Try local voices_dir (works for local model paths)
        if self.voices_dir:
            candidate = os.path.join(self.voices_dir, voice_file)
            if os.path.exists(candidate):
                voice_path = candidate

        # Strategy 2: Download from HuggingFace Hub if not found locally
        if voice_path is None and self._model_name_or_path:
            try:
                voice_path = cached_file(
                    self._model_name_or_path,
                    f"voices/{voice_file}",
                )
            except Exception as e:
                logger.warning(f"Could not download voice file 'voices/{voice_file}' from hub: {e}")

        if voice_path is None:
            raise ValueError(
                f"Could not find voice file '{voice_file}' for voice '{voice_name}'. "
                f"Checked local dir: {self.voices_dir}, "
                f"HuggingFace repo: {self._model_name_or_path}"
            )

        voice_cache = torch.load(voice_path, map_location="cpu", weights_only=True)
        if "acoustic_mean" not in voice_cache:
            raise ValueError(
                f"Voice file '{voice_file}' does not contain 'acoustic_mean'. "
                f"Available keys: {list(voice_cache.keys())}"
            )
        return voice_cache

    def __call__(
        self,
        text: Optional[str] = None,
        voice: Optional[str] = None,
        voice_cache: Optional[dict] = None,
        voice_prompt: Optional[Union[np.ndarray, torch.Tensor, str]] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        truncation: Union[bool, str, TruncationStrategy] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        """Process text and optional voice input.

        Supports three modes of voice cloning:
        1. voice_prompt: Raw audio (numpy, tensor, or file path) - model encodes internally
        2. voice: Name of a pre-encoded voice from voices.json registry
        3. voice_cache: Pre-encoded voice features dict

        Args:
            text: Input text to synthesize
            voice: Name of a pre-encoded voice (from voices.json registry)
            voice_cache: Pre-encoded voice features dict (alternative to voice name)
            voice_prompt: Raw voice audio for cloning (numpy array, torch tensor, or path)
            padding: Padding strategy
            truncation: Truncation strategy
            max_length: Maximum sequence length
            return_tensors: Return format

        Returns:
            BatchEncoding with processed inputs including speech_input_mask
        """
        if text is None:
            raise ValueError("Text input is required")

        # Special token IDs
        speech_start_id = 151652  # <|vision_start|> repurposed for speech
        speech_diffusion_id = 151654  # VAE token used as placeholder

        # Format text with proper template
        # KugelAudio uses 0-based speaker numbering internally (Speaker 0, 1, 2...)
        # Our pipeline uses 1-based (Speaker 1, 2, 3...) so convert
        formatted_text = text.strip()
        
        def _convert_speaker_to_zero_based(match):
            speaker_num = int(match.group(1))
            # Convert 1-based to 0-based (Speaker 1 → Speaker 0)
            internal_num = max(0, speaker_num - 1)
            return f"Speaker {internal_num}:"
        
        if re.search(r'Speaker\s+\d+\s*:', formatted_text):
            # Text has Speaker N: format — convert 1-based to 0-based
            formatted_text = re.sub(r'Speaker\s+(\d+)\s*:', _convert_speaker_to_zero_based, formatted_text)
        else:
            # Plain text — wrap as Speaker 0
            formatted_text = f"Speaker 0: {formatted_text}"

        # Build the full prompt template matching the training format
        system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"

        # Start building tokens and speech_input_mask
        full_tokens = []
        speech_input_mask = []
        voice_audio = None

        # System prompt tokens
        system_tokens = self.tokenizer.encode(system_prompt, add_special_tokens=False)
        full_tokens.extend(system_tokens)
        speech_input_mask.extend([False] * len(system_tokens))

        # Load voice cache from registry if voice name is provided
        loaded_voice_cache = voice_cache
        if voice is not None and voice_cache is None:
            loaded_voice_cache = self.load_voice_cache(voice)

        # Process voice_prompt (raw audio) if provided
        if voice_prompt is not None:
            if isinstance(voice_prompt, str):
                voice_audio = self.audio_processor._load_from_path(voice_prompt)
                if self.db_normalize and self.audio_normalizer:
                    voice_audio = self.audio_normalizer(voice_audio)
            elif isinstance(voice_prompt, np.ndarray):
                voice_audio = voice_prompt.astype(np.float32)
            elif isinstance(voice_prompt, torch.Tensor):
                voice_audio = voice_prompt.cpu().numpy()
                if voice_audio.ndim > 1:
                    voice_audio = voice_audio.squeeze()
                voice_audio = voice_audio.astype(np.float32)

            # Voice input section with placeholder tokens
            voice_input_tokens = self.tokenizer.encode(" Voice input:\n", add_special_tokens=False)
            full_tokens.extend(voice_input_tokens)
            speech_input_mask.extend([False] * len(voice_input_tokens))

            # Speaker prefix for voice
            speaker_prefix = self.tokenizer.encode(" Speaker 0:", add_special_tokens=False)
            full_tokens.extend(speaker_prefix)
            speech_input_mask.extend([False] * len(speaker_prefix))

            # Calculate number of VAE tokens needed based on audio length
            num_voice_tokens = math.ceil(len(voice_audio) / self.speech_compression_ratio)

            # Add placeholder VAE tokens
            full_tokens.extend([speech_diffusion_id] * num_voice_tokens)
            speech_input_mask.extend([True] * num_voice_tokens)

            # Newline after voice
            newline_tokens = self.tokenizer.encode("\n", add_special_tokens=False)
            full_tokens.extend(newline_tokens)
            speech_input_mask.extend([False] * len(newline_tokens))

        # Process pre-encoded voice if available (and no raw voice_prompt)
        elif loaded_voice_cache is not None:
            acoustic_mean = loaded_voice_cache["acoustic_mean"]
            # Number of tokens is determined by the acoustic_mean time dimension
            if acoustic_mean.dim() == 3:
                num_voice_tokens = acoustic_mean.shape[1]
            elif acoustic_mean.dim() == 2:
                num_voice_tokens = acoustic_mean.shape[0]
            else:
                num_voice_tokens = 1

            # Voice input section with placeholder tokens
            voice_input_tokens = self.tokenizer.encode(" Voice input:\n", add_special_tokens=False)
            full_tokens.extend(voice_input_tokens)
            speech_input_mask.extend([False] * len(voice_input_tokens))

            # Speaker prefix for voice
            speaker_prefix = self.tokenizer.encode(" Speaker 0:", add_special_tokens=False)
            full_tokens.extend(speaker_prefix)
            speech_input_mask.extend([False] * len(speaker_prefix))

            # Add placeholder VAE tokens that will be replaced with speech embeddings
            full_tokens.extend([speech_diffusion_id] * num_voice_tokens)
            speech_input_mask.extend([True] * num_voice_tokens)

            # Newline after voice
            newline_tokens = self.tokenizer.encode("\n", add_special_tokens=False)
            full_tokens.extend(newline_tokens)
            speech_input_mask.extend([False] * len(newline_tokens))

        # Text input section
        text_input_tokens = self.tokenizer.encode(" Text input:\n", add_special_tokens=False)
        full_tokens.extend(text_input_tokens)
        speech_input_mask.extend([False] * len(text_input_tokens))

        # Speaker text
        speaker_text_tokens = self.tokenizer.encode(
            f" {formatted_text}\n", add_special_tokens=False
        )
        full_tokens.extend(speaker_text_tokens)
        speech_input_mask.extend([False] * len(speaker_text_tokens))

        # Speech output section
        speech_output_tokens = self.tokenizer.encode(" Speech output:\n", add_special_tokens=False)
        full_tokens.extend(speech_output_tokens)
        speech_input_mask.extend([False] * len(speech_output_tokens))

        # Add speech_start token
        full_tokens.append(speech_start_id)
        speech_input_mask.append(False)

        result = BatchEncoding()
        result["text_ids"] = full_tokens
        result["speech_input_mask"] = speech_input_mask

        if return_tensors == "pt":
            result["text_ids"] = torch.tensor([full_tokens], dtype=torch.long)
            result["speech_input_mask"] = torch.tensor([speech_input_mask], dtype=torch.bool)

        # Include voice data for the model
        if voice_audio is not None:
            # Single speaker raw audio
            if return_tensors == "pt":
                result["speech_tensors"] = torch.tensor(voice_audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                num_frames = math.ceil(len(voice_audio) / self.speech_compression_ratio)
                result["speech_masks"] = torch.ones(1, num_frames, dtype=torch.bool)
            else:
                result["speech_tensors"] = voice_audio
                num_frames = math.ceil(len(voice_audio) / self.speech_compression_ratio)
                result["speech_masks"] = [True] * num_frames
        elif loaded_voice_cache is not None:
            # Pre-encoded path
            result["voice_cache"] = loaded_voice_cache

        return result

    def batch_decode(self, *args, **kwargs):
        """Decode token IDs to text."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """Decode token IDs to text."""
        return self.tokenizer.decode(*args, **kwargs)

    def save_audio(self, audio, output_path: str = "output.wav", **kwargs) -> List[str]:
        """Save generated audio to file."""
        return self.audio_processor.save_audio(audio, output_path, **kwargs)

    @property
    def model_input_names(self) -> List[str]:
        """Return list of model input names."""
        tokenizer_names = getattr(self.tokenizer, "model_input_names", [])
        audio_names = getattr(self.audio_processor, "model_input_names", [])
        return list(
            dict.fromkeys(tokenizer_names + audio_names + ["speech_inputs", "speech_input_mask"])
        )
