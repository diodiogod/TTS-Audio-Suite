"""
Qwen3-TTS Engine Wrapper

Wraps official Qwen3-TTS implementation with ComfyUI integration:
- .to() method for VRAM management
- Device checking before generation
- Support for all 3 model variants (CustomVoice, VoiceDesign, Base)
- Audio format conversion
"""

import sys
import os
import torch
import numpy as np
import folder_paths
from typing import Optional, Tuple

from utils.models.unified_model_interface import unified_model_interface
from utils.models.factory_config import ModelLoadConfig
from utils.models.extra_paths import find_model_in_paths, get_preferred_download_path, get_all_tts_model_paths

# Add bundled implementation to path
IMPL_DIR = os.path.join(os.path.dirname(__file__), "impl")
if IMPL_DIR not in sys.path:
    sys.path.insert(0, IMPL_DIR)

# Import bundled Qwen3-TTS
try:
    from qwen_tts import Qwen3TTSModel
except ImportError as e:
    print(f"⚠️ Failed to import Qwen3-TTS: {e}")
    print(f"   Make sure official implementation is in {IMPL_DIR}")
    Qwen3TTSModel = None


class Qwen3TTSEngine:
    """
    Unified Qwen3-TTS engine supporting all 3 model variants.

    Model variants are loaded dynamically based on usage:
    - CustomVoice: 9 preset speakers with optional instruction
    - VoiceDesign: Text-to-voice design via natural language
    - Base: Zero-shot voice cloning from reference audio

    Critical features:
    - .to() method for ComfyUI model management
    - Device checking before generation (Clear VRAM support)
    - 24kHz audio output
    - NO __del__ destructor (keeps model loaded)
    """

    def __init__(self, model_name: str, device: str = "auto", dtype: Optional[str] = None,
                 attn_implementation: str = "auto", model_dir: Optional[str] = None, **kwargs):
        """
        Initialize Qwen3-TTS engine.

        Args:
            model_name: Model variant name (e.g., "Qwen3-TTS-12Hz-1.7B-CustomVoice")
            device: Device to load on ("auto", "cuda", "cpu")
            dtype: Model precision (bfloat16, float16, float32, or "auto")
            attn_implementation: Attention mechanism ("auto", "flash_attention_2", "sdpa", "eager")
            model_dir: Optional model directory path (auto-resolved if not provided)
        """
        self.model_name = model_name
        self.model_dir = model_dir or self._find_model_directory(model_name)
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.attn_implementation = attn_implementation
        self.kwargs = kwargs

        self._model = None
        self._model_config = None

        print(f"✓ Qwen3TTSEngine initialized: {model_name}")
        print(f"  Device: {self.device}, Dtype: {self.dtype}, Attention: {attn_implementation}")

    def _find_model_directory(self, model_identifier: str) -> str:
        """Find Qwen3-TTS model directory using extra_model_paths configuration."""
        try:
            # Handle local: prefix
            if model_identifier.startswith("local:"):
                model_name = model_identifier[6:]  # Remove "local:" prefix

                # Search in all configured TTS paths
                all_tts_paths = get_all_tts_model_paths('TTS')
                for base_path in all_tts_paths:
                    # Check direct path (models/TTS/Qwen3-TTS-12Hz-1.7B-CustomVoice)
                    direct_path = os.path.join(base_path, model_name)
                    if os.path.exists(direct_path):
                        return direct_path

                    # Check organized path (models/TTS/qwen3_tts/Qwen3-TTS-12Hz-1.7B-CustomVoice)
                    organized_path = os.path.join(base_path, "qwen3_tts", model_name)
                    if os.path.exists(organized_path):
                        return organized_path

                raise FileNotFoundError(f"Local Qwen3-TTS model '{model_name}' not found in any configured path")

            else:
                # Auto-download case - return preferred download path with model name appended
                base_path = get_preferred_download_path(model_type='TTS', engine_name='qwen3_tts')
                model_path = os.path.join(base_path, model_identifier)

                # Check if model exists
                if not os.path.exists(model_path):
                    print(f"📥 Qwen3-TTS model directory not found, will trigger auto-download...")

                return model_path

        except Exception:
            # Fallback to default path
            model_name = model_identifier.replace("local:", "") if model_identifier.startswith("local:") else model_identifier
            return os.path.join(folder_paths.models_dir, "TTS", "qwen3_tts", model_name)

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        from utils.device import resolve_torch_device
        resolved = resolve_torch_device(device)
        return resolved

    def _resolve_dtype(self, dtype_str: Optional[str]) -> torch.dtype:
        """Resolve dtype string to torch.dtype with GPU capability detection."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }

        if dtype_str in dtype_map:
            return dtype_map[dtype_str]

        # "auto" mode: detect GPU compute capability
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            # bf16 requires SM 8.0+ (Ampere: RTX 30xx, A100, etc.)
            supports_bf16 = (major >= 8)
            return torch.bfloat16 if supports_bf16 else torch.float16

        # CPU fallback to float16
        return torch.float16

    def _ensure_model_loaded(self):
        """Load the Qwen3-TTS model using unified model interface."""
        if self._model is not None:
            return

        if Qwen3TTSModel is None:
            raise ImportError("Qwen3-TTS not available - check bundled implementation")

        # Create model configuration
        self._model_config = ModelLoadConfig(
            engine_name="qwen3_tts",
            model_type="tts",
            model_name=self.model_name,
            device=self.device,
            model_path=self.model_dir,
            additional_params={
                "torch_dtype": self.dtype,
                "attn_implementation": self.attn_implementation,
                **self.kwargs
            }
        )

        # Load via unified interface
        print(f"🔄 Qwen3-TTS: Loading {self.model_name}...")
        self._model = unified_model_interface.load_model(self._model_config)
        print(f"✅ Model loaded on {self.device}")

    def to(self, device):
        """
        Move all model components to the specified device.

        CRITICAL for ComfyUI model management - ensures all components move together
        when models are detached to CPU and later reloaded to CUDA.

        Qwen3-TTS structure:
        - _model (Qwen3TTSModel wrapper)
          - _model.model (Qwen3TTSForConditionalGeneration)
            - _model.model.talker (Qwen3TTSTalkerForConditionalGeneration)
            - _model.model.speaker_encoder (Qwen3TTSSpeakerEncoder, Base only)
            - _model.model.speech_tokenizer (optional)
          - _model.processor (Qwen3TTSProcessor)

        Args:
            device: Target device ("cuda", "cpu", or torch.device)
        """
        self.device = device if isinstance(device, str) else str(device)

        # Move the underlying model if loaded
        if self._model is not None:
            print(f"🔄 Moving Qwen3-TTS model to {device}")

            try:
                # Move top-level wrapper (should recursively move most components)
                if hasattr(self._model, 'to'):
                    self._model = self._model.to(device)

                # Explicitly move nested Qwen3TTSForConditionalGeneration model
                if hasattr(self._model, 'model') and hasattr(self._model.model, 'to'):
                    self._model.model = self._model.model.to(device)

                    # Move talker (main TTS component)
                    if hasattr(self._model.model, 'talker') and hasattr(self._model.model.talker, 'to'):
                        self._model.model.talker = self._model.model.talker.to(device)

                    # Move speaker_encoder (Base model only, may be None)
                    if hasattr(self._model.model, 'speaker_encoder') and self._model.model.speaker_encoder is not None:
                        if hasattr(self._model.model.speaker_encoder, 'to'):
                            self._model.model.speaker_encoder = self._model.model.speaker_encoder.to(device)

                    # Move speech_tokenizer (optional, may be None)
                    if hasattr(self._model.model, 'speech_tokenizer') and self._model.model.speech_tokenizer is not None:
                        if hasattr(self._model.model.speech_tokenizer, 'to'):
                            self._model.model.speech_tokenizer = self._model.model.speech_tokenizer.to(device)

                # Update device attribute on wrapper
                if hasattr(self._model, 'device'):
                    self._model.device = torch.device(device) if isinstance(device, str) else device

                print(f"✓ Qwen3-TTS model moved to {device}")

            except Exception as e:
                print(f"⚠️ Error moving model to {device}: {e}")
                # Try to re-register with ComfyUI if move failed
                self._register_with_comfyui(device)

        return self

    def _check_and_reload_device(self):
        """
        Check if model was offloaded to CPU and reload to CUDA if needed.

        CRITICAL for Clear VRAM support - ComfyUI moves models to CPU, we need to reload.
        """
        if self._model is None:
            return

        from utils.device import resolve_torch_device
        target_device = resolve_torch_device("auto")

        try:
            # Check current device of the actual model (not wrapper)
            # The wrapper is Qwen3TTSModel, the actual model is self._model.model (Qwen3TTSForConditionalGeneration)
            actual_model = self._model.model if hasattr(self._model, 'model') else self._model

            if hasattr(actual_model, 'parameters'):
                try:
                    first_param = next(actual_model.parameters())
                    current_device = str(first_param.device)

                    if current_device != target_device:
                        print(f"🔄 Reloading Qwen3-TTS from {current_device} to {target_device}")

                        # Use unified model manager for device movement
                        try:
                            from utils.models.comfyui_model_wrapper.model_manager import tts_model_manager
                            if not tts_model_manager.ensure_device("qwen3_tts", target_device):
                                # Fallback to direct movement if not in manager cache
                                self.to(target_device)
                        except Exception as e:
                            # Fallback to direct movement if manager not available
                            print(f"⚠️ Unified manager not available, using direct device movement: {e}")
                            self.to(target_device)

                except StopIteration:
                    pass  # No parameters found

        except Exception as e:
            print(f"⚠️ Device check failed: {e}")

    def generate_custom_voice(self, text, language, speaker, instruct=None, progress_bar=None, **kwargs):
        """
        Generate audio using CustomVoice model with preset speakers.

        Args:
            text: Text to synthesize (string or list of strings)
            language: Language code ("Chinese", "English", etc. or "Auto")
            speaker: Preset speaker name ("Vivian", "Ryan", etc.)
            instruct: Optional instruction for emotion/style control
            **kwargs: Generation parameters (top_k, temperature, etc.)

        Returns:
            tuple: (audio_array, sample_rate)
                - audio_array: numpy array or list of numpy arrays
                - sample_rate: int (24000)
        """
        self._ensure_model_loaded()
        self._check_and_reload_device()

        print(f"🌍 Qwen3-TTS CustomVoice: Using language='{language}' with speaker='{speaker}'")

        try:
            # Create progress streamer if progress_bar provided
            streamer = None
            if progress_bar is not None:
                from .progress_callback import Qwen3TTSProgressStreamer
                max_tokens = kwargs.get('max_new_tokens', 2048)
                streamer = Qwen3TTSProgressStreamer(max_tokens, progress_bar)
                kwargs['streamer'] = streamer

            wavs, sr = self._model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct if instruct else None,
                **kwargs
            )

            return wavs, sr

        except Exception as e:
            print(f"✗ CustomVoice generation failed: {e}")
            raise

    def generate_voice_design(self, text, language, instruct, progress_bar=None, **kwargs):
        """
        Generate audio using VoiceDesign model with text description.

        Args:
            text: Text to synthesize (string or list of strings)
            language: Language code ("Chinese", "English", etc. or "Auto")
            instruct: Natural language voice description (REQUIRED)
            **kwargs: Generation parameters (top_k, temperature, etc.)

        Returns:
            tuple: (audio_array, sample_rate)
                - audio_array: numpy array or list of numpy arrays
                - sample_rate: int (24000)
        """
        self._ensure_model_loaded()
        self._check_and_reload_device()

        print(f"🌍 Qwen3-TTS VoiceDesign: Using language='{language}'")

        try:
            # Create progress streamer if progress_bar provided
            streamer = None
            if progress_bar is not None:
                from .progress_callback import Qwen3TTSProgressStreamer
                max_tokens = kwargs.get('max_new_tokens', 2048)
                streamer = Qwen3TTSProgressStreamer(max_tokens, progress_bar)
                kwargs['streamer'] = streamer

            wavs, sr = self._model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
                **kwargs
            )

            return wavs, sr

        except Exception as e:
            print(f"✗ VoiceDesign generation failed: {e}")
            raise

    def generate_voice_clone(self, text, language, ref_audio, ref_text=None, x_vector_only_mode=False, voice_clone_prompt=None, progress_bar=None, **kwargs):
        """
        Generate audio using Base model with zero-shot voice cloning.

        Args:
            text: Text to synthesize (string or list of strings)
            language: Language code ("Chinese", "English", etc. or "Auto")
            ref_audio: Reference audio (path, URL, numpy array, or ComfyUI audio dict)
            ref_text: Reference transcript (recommended for better quality)
            x_vector_only_mode: If True, only use speaker embedding (faster, lower quality)
            voice_clone_prompt: Pre-computed voice prompt (from create_voice_clone_prompt)
            **kwargs: Generation parameters (top_k, temperature, etc.)

        Returns:
            tuple: (audio_array, sample_rate)
                - audio_array: numpy array or list of numpy arrays
                - sample_rate: int (24000)
        """
        self._ensure_model_loaded()
        self._check_and_reload_device()

        print(f"🌍 Qwen3-TTS VoiceClone: Using language='{language}' (x_vector_only={x_vector_only_mode})")

        try:
            # Convert ComfyUI audio dict to tuple if needed
            ref_audio = self._convert_audio_input(ref_audio)

            # Create progress streamer if progress_bar provided
            streamer = None
            if progress_bar is not None:
                from .progress_callback import Qwen3TTSProgressStreamer
                max_tokens = kwargs.get('max_new_tokens', 2048)
                streamer = Qwen3TTSProgressStreamer(max_tokens, progress_bar)
                kwargs['streamer'] = streamer

            # Use pre-computed prompt if provided, otherwise compute on the fly
            if voice_clone_prompt is not None:
                wavs, sr = self._model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=voice_clone_prompt,
                    **kwargs
                )
            else:
                wavs, sr = self._model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                    **kwargs
                )

            return wavs, sr

        except Exception as e:
            print(f"✗ VoiceClone generation failed: {e}")
            raise

    def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False):
        """
        Create reusable voice clone prompt for efficient batch processing.

        Args:
            ref_audio: Reference audio (path, URL, numpy array, or ComfyUI audio dict)
            ref_text: Reference transcript (recommended)
            x_vector_only_mode: If True, only extract speaker embedding

        Returns:
            Voice clone prompt items (can be passed to generate_voice_clone)
        """
        self._ensure_model_loaded()
        self._check_and_reload_device()

        # Convert ComfyUI audio dict to tuple if needed
        ref_audio = self._convert_audio_input(ref_audio)

        print(f"Creating voice clone prompt: x_vector_only={x_vector_only_mode}")

        try:
            prompt_items = self._model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode
            )

            return prompt_items

        except Exception as e:
            print(f"✗ Voice clone prompt creation failed: {e}")
            raise

    def _convert_audio_input(self, audio):
        """
        Convert various audio input formats to format expected by Qwen3-TTS.

        Args:
            audio: Various formats:
                - File path (string)
                - URL (string)
                - ComfyUI audio dict: {"waveform": tensor, "sample_rate": int}
                - Tuple: (numpy_array, sample_rate)
                - Numpy array (assumes 24kHz)

        Returns:
            Format accepted by Qwen3-TTS (path, URL, or tuple)
        """
        # If string, assume it's a path or URL
        if isinstance(audio, str):
            return audio

        # If dict (ComfyUI format), extract waveform and sample rate
        if isinstance(audio, dict):
            waveform = audio.get('waveform')
            sample_rate = audio.get('sample_rate', 24000)

            if waveform is not None:
                # Convert tensor to numpy
                if torch.is_tensor(waveform):
                    # Remove batch and channel dimensions if present
                    if waveform.ndim == 3:  # [batch, channels, samples]
                        waveform = waveform[0, 0, :]  # Take first batch, first channel
                    elif waveform.ndim == 2:  # [channels, samples]
                        waveform = waveform[0, :]  # Take first channel

                    # Convert to CPU and numpy
                    waveform = waveform.cpu().numpy()

                # Return as tuple
                return (waveform.astype(np.float32), int(sample_rate))

        # If tuple, assume it's (waveform, sample_rate)
        if isinstance(audio, tuple) and len(audio) == 2:
            waveform, sample_rate = audio

            # Convert tensor to numpy if needed
            if torch.is_tensor(waveform):
                waveform = waveform.cpu().numpy()

            return (waveform.astype(np.float32), int(sample_rate))

        # If numpy array, assume 24kHz
        if isinstance(audio, np.ndarray):
            return (audio.astype(np.float32), 24000)

        # If tensor, convert to numpy and assume 24kHz
        if torch.is_tensor(audio):
            return (audio.cpu().numpy().astype(np.float32), 24000)

        # Unknown format
        raise ValueError(f"Unsupported audio input format: {type(audio)}")

    def get_sample_rate(self) -> int:
        """Get the native sample rate of the engine."""
        return 24000  # Qwen3-TTS native sample rate

    def unload(self):
        """Unload the model to free memory."""
        if self._model_config:
            unified_model_interface.unload_model(self._model_config)
        self._model = None
        self._model_config = None

    # NOTE: NO __del__ method - see fail doc line 59-62
    # Models should stay loaded for reuse, only unload on explicit Clear VRAM
