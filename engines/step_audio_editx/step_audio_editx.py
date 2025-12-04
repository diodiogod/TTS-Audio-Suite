import os
import sys
import torch
import torchaudio
import tempfile
import folder_paths
import numpy as np
from typing import Optional, Tuple
import warnings

from utils.models.unified_model_interface import unified_model_interface
from utils.models.factory_config import ModelLoadConfig
from utils.models.extra_paths import find_model_in_paths, get_preferred_download_path, get_all_tts_model_paths


class StepAudioEditXEngine:
    """
    Step Audio EditX Engine wrapper for TTS Audio Suite integration.

    Supports:
    - Zero-shot voice cloning (Mandarin, English, Sichuanese, Cantonese, Japanese, Korean)
    - Emotion editing (14 emotions: happy, sad, angry, excited, etc.)
    - Style editing (32 styles: whisper, serious, child, etc.)
    - Speed control (faster, slower, more faster, more slower)
    - Paralinguistic effects (10 types: Laughter, Breathing, Sigh, etc.)
    - Denoising and VAD
    - Iterative editing (1-5 iterations for stronger effects)
    """

    # Edit type configurations from original
    EMOTION_OPTIONS = [
        "happy", "sad", "angry", "excited", "calm", "fearful", "surprised", "disgusted",
        "confusion", "empathy", "embarrass", "depressed", "coldness", "admiration"
    ]

    STYLE_OPTIONS = [
        "whisper", "serious", "child", "older", "girl", "pure", "sister", "sweet",
        "exaggerated", "ethereal", "generous", "recite", "act_coy", "warm", "shy",
        "comfort", "authority", "chat", "radio", "soulful", "gentle", "story", "vivid",
        "program", "news", "advertising", "roar", "murmur", "shout", "deeply", "loudly",
        "arrogant", "friendly"
    ]

    SPEED_OPTIONS = ["faster", "slower", "more faster", "more slower"]

    PARALINGUISTIC_OPTIONS = [
        "[Breathing]", "[Laughter]", "[Surprise-oh]", "[Confirmation-en]",
        "[Uhm]", "[Surprise-ah]", "[Surprise-wa]", "[Sigh]",
        "[Question-ei]", "[Dissatisfaction-hnn]"
    ]

    def __init__(self, model_dir: str = "Step-Audio-EditX", device: str = "auto",
                 torch_dtype: str = "bfloat16", quantization: Optional[str] = None):
        """
        Initialize Step Audio EditX engine.

        Args:
            model_dir: Model identifier (following pattern: "local:ModelName" or "ModelName")
            device: Device to use ("auto", "cuda", "cpu")
            torch_dtype: Data type ("bfloat16", "float16", "float32", "auto")
            quantization: Quantization mode ("int4", "int8", or None)
        """
        # Resolve model directory using extra_model_paths
        self.model_dir = self._find_model_directory(model_dir)

        self.device = self._resolve_device(device)
        self.torch_dtype = self._resolve_dtype(torch_dtype)
        self.quantization = quantization

        self._tts_engine = None
        self._tokenizer = None
        self._model_config = None

    def _find_model_directory(self, model_identifier: str) -> str:
        """Find Step Audio EditX model directory using extra_model_paths configuration."""
        try:
            # Handle local: prefix
            if model_identifier.startswith("local:"):
                model_name = model_identifier[6:]  # Remove "local:" prefix

                # Search in all configured TTS paths
                all_tts_paths = get_all_tts_model_paths('TTS')
                for base_path in all_tts_paths:
                    # Check direct path (models/TTS/Step-Audio-EditX)
                    direct_path = os.path.join(base_path, model_name)
                    if os.path.exists(direct_path):
                        return direct_path

                    # Check organized path (models/TTS/step_audio_editx/Step-Audio-EditX)
                    organized_path = os.path.join(base_path, "step_audio_editx", model_name)
                    if os.path.exists(organized_path):
                        return organized_path

                raise FileNotFoundError(f"Local Step Audio EditX model '{model_name}' not found in any configured path")

            else:
                # Auto-download case - return preferred download path with model name appended
                base_path = get_preferred_download_path(model_type='TTS', engine_name='step_audio_editx')
                model_path = os.path.join(base_path, model_identifier)

                # Check if model exists and is complete, if not trigger auto-download
                needs_download = False
                if not os.path.exists(model_path):
                    needs_download = True
                    print(f"📥 Step Audio EditX model directory not found, triggering auto-download...")
                else:
                    # Check model completeness
                    try:
                        from engines.step_audio_editx.step_audio_editx_downloader import StepAudioEditXDownloader
                        downloader = StepAudioEditXDownloader()
                        downloader._verify_model(model_path, model_identifier)
                    except Exception as verify_error:
                        needs_download = True
                        print(f"📥 Step Audio EditX model incomplete (missing files), triggering re-download...")
                        print(f"    Verification error: {verify_error}")

                if needs_download:
                    try:
                        if 'downloader' not in locals():
                            from engines.step_audio_editx.step_audio_editx_downloader import StepAudioEditXDownloader
                            downloader = StepAudioEditXDownloader()
                        downloaded_path = downloader.download_model(model_identifier)
                        print(f"✅ Step Audio EditX auto-download completed: {downloaded_path}")
                        return downloaded_path
                    except Exception as download_error:
                        raise RuntimeError(f"Step Audio EditX model not found/incomplete and auto-download failed: {download_error}")

                return model_path

        except Exception:
            # Fallback to default path
            model_name = model_identifier.replace("local:", "") if model_identifier.startswith("local:") else model_identifier
            return os.path.join(folder_paths.models_dir, "TTS", "step_audio_editx", model_name)

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        from utils.device import resolve_torch_device
        resolved = resolve_torch_device(device)
        return resolved

    def _resolve_dtype(self, dtype_str: str) -> torch.dtype:
        """Resolve dtype string to torch.dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        }
        return dtype_map.get(dtype_str, torch.bfloat16)

    def _ensure_model_loaded(self):
        """Load the Step Audio EditX model using unified model interface."""
        if self._tts_engine is not None:
            return

        # Create model configuration
        self._model_config = ModelLoadConfig(
            engine_name="step_audio_editx",
            model_type="tts",
            model_name="Step-Audio-EditX",
            device=self.device,
            model_path=self.model_dir,
            additional_params={
                "torch_dtype": self.torch_dtype,
                "quantization": self.quantization
            }
        )

        # Load via unified interface with progress indication
        print("🔄 Step Audio EditX: Initializing engine (first run: 2-3 min)...")
        self._tts_engine = unified_model_interface.load_model(self._model_config)
        print(f"✅ Engine loaded on {self.device}")

    def clone(
        self,
        prompt_wav_path: str,
        prompt_text: str,
        target_text: str,
        temperature: float = 0.7,
        do_sample: bool = True,
        max_new_tokens: int = 8192,
        progress_bar=None
    ) -> torch.Tensor:
        """
        Generate speech using zero-shot voice cloning.

        Args:
            prompt_wav_path: Reference audio file for voice cloning
            prompt_text: Text content of reference audio
            target_text: Text to synthesize with cloned voice
            temperature: Sampling temperature (default: 0.7, hardcoded in original)
            do_sample: Use sampling (default: True, hardcoded in original)
            max_new_tokens: Maximum tokens to generate (default: 8192, hardcoded in original)
            progress_bar: ComfyUI progress bar for generation tracking

        Returns:
            Generated audio as torch.Tensor with shape [1, samples]
        """
        self._ensure_model_loaded()

        # CRITICAL: Reload model to correct device if it was offloaded
        from utils.device import resolve_torch_device
        target_device = resolve_torch_device("auto")

        # Check if model was offloaded to CPU and needs to be reloaded
        if self._tts_engine is not None and hasattr(self._tts_engine, 'llm'):
            if hasattr(self._tts_engine.llm, 'parameters'):
                try:
                    first_param = next(self._tts_engine.llm.parameters())
                    current_device = str(first_param.device)
                    if current_device != target_device:
                        # Find and call wrapper's model_load() to keep ComfyUI tracking in sync
                        try:
                            from utils.models.unified_model_interface import unified_model_interface

                            wrapper_found = False
                            if hasattr(unified_model_interface, 'model_manager'):
                                for cache_key, wrapper in unified_model_interface.model_manager._model_cache.items():
                                    model = wrapper.model if hasattr(wrapper, 'model') else None
                                    if model is self:
                                        wrapper.model_load(target_device)
                                        wrapper_found = True
                                        break
                                    elif hasattr(model, 'model') and model.model is self:
                                        wrapper.model_load(target_device)
                                        wrapper_found = True
                                        break

                            if not wrapper_found:
                                # Fallback: direct .to()
                                self.to(target_device)
                        except Exception:
                            self.to(target_device)
                except StopIteration:
                    pass

        # Call original implementation with progress bar
        audio_tensor, sample_rate = self._tts_engine.clone(
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            target_text=target_text,
            progress_bar=progress_bar,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample
        )

        # Ensure output is [1, samples] format
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        return audio_tensor

    def edit(
        self,
        input_audio_path: str,
        audio_text: str,
        edit_type: str,
        edit_info: Optional[str] = None,
        text: Optional[str] = None,
        n_edit_iterations: int = 1
    ) -> torch.Tensor:
        """
        Edit audio with specified modification.

        Args:
            input_audio_path: Path to input audio file (0.5-30s)
            audio_text: Transcript of input audio (REQUIRED)
            edit_type: Type of edit (emotion, style, speed, paralinguistic, denoising)
            edit_info: Specific edit value (e.g., 'happy', 'whisper', 'faster')
            text: Text for paralinguistic mode (where to insert effect)
            n_edit_iterations: Number of iterative edits (1-5, default: 1)

        Returns:
            Edited audio as torch.Tensor with shape [1, samples]
        """
        self._ensure_model_loaded()

        # CRITICAL: Reload model if offloaded (same as clone method)
        from utils.device import resolve_torch_device
        target_device = resolve_torch_device("auto")

        if self._tts_engine is not None and hasattr(self._tts_engine, 'llm'):
            if hasattr(self._tts_engine.llm, 'parameters'):
                try:
                    first_param = next(self._tts_engine.llm.parameters())
                    current_device = str(first_param.device)
                    if current_device != target_device:
                        try:
                            from utils.models.unified_model_interface import unified_model_interface
                            wrapper_found = False
                            if hasattr(unified_model_interface, 'model_manager'):
                                for cache_key, wrapper in unified_model_interface.model_manager._model_cache.items():
                                    model = wrapper.model if hasattr(wrapper, 'model') else None
                                    if model is self:
                                        wrapper.model_load(target_device)
                                        wrapper_found = True
                                        break
                                    elif hasattr(model, 'model') and model.model is self:
                                        wrapper.model_load(target_device)
                                        wrapper_found = True
                                        break
                            if not wrapper_found:
                                self.to(target_device)
                        except Exception:
                            self.to(target_device)
                except StopIteration:
                    pass

        # Perform iterative editing
        current_audio_path = input_audio_path
        for i in range(n_edit_iterations):
            audio_tensor, sample_rate = self._tts_engine.edit(
                input_audio_path=current_audio_path,
                audio_text=audio_text,
                edit_type=edit_type,
                edit_info=edit_info,
                text=text
            )

            # For next iteration, save current result to temp file
            if i < n_edit_iterations - 1:
                comfyui_temp_dir = folder_paths.get_temp_directory()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=comfyui_temp_dir) as tmp_file:
                    # Save for next iteration
                    torchaudio.save(tmp_file.name, audio_tensor.unsqueeze(0) if audio_tensor.dim() == 1 else audio_tensor, sample_rate)
                    current_audio_path = tmp_file.name

        # Ensure output is [1, samples] format
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        return audio_tensor

    def get_sample_rate(self) -> int:
        """Get the native sample rate of the engine."""
        return 24000  # CosyVoice native sample rate

    def to(self, device):
        """
        Move all model components to the specified device.

        Critical for ComfyUI model management - ensures all components move together
        when models are detached to CPU and later reloaded to CUDA.

        Note: Quantized models (int4/int8) cannot be moved with .to() - they're already
        on the correct device from loading.
        """
        self.device = device

        # Move the underlying TTS engine if loaded
        if self._tts_engine is not None:
            # Step Audio has multiple components: LLM, tokenizer, vocoder
            # Move each component, but skip for quantized models
            if hasattr(self._tts_engine, 'llm') and hasattr(self._tts_engine.llm, 'to'):
                # Check if model is quantized (bitsandbytes)
                # Store flag to avoid repeated warnings
                if not hasattr(self, '_quantized_move_warning_shown'):
                    self._quantized_move_warning_shown = False

                try:
                    # Try to move - will fail for quantized models
                    self._tts_engine.llm = self._tts_engine.llm.to(device)
                except ValueError as e:
                    if "is not supported for" in str(e) and ("8-bit" in str(e) or "4-bit" in str(e)):
                        # Quantized model - can't be moved, skip
                        if not self._quantized_move_warning_shown:
                            print(f"⚠️ Skipping device move for quantized model: {e}")
                            self._quantized_move_warning_shown = True
                    else:
                        # Different error - re-raise
                        raise

            # CosyVoice vocoder can still be moved (usually not quantized)
            if hasattr(self._tts_engine, 'cosy_model') and hasattr(self._tts_engine.cosy_model, 'to'):
                self._tts_engine.cosy_model = self._tts_engine.cosy_model.to(device)

            # Update device attribute on engine
            if hasattr(self._tts_engine, 'device'):
                self._tts_engine.device = torch.device(device) if isinstance(device, str) else device

        return self

    def unload(self):
        """Unload the model to free memory."""
        if self._model_config:
            unified_model_interface.unload_model(self._model_config)
        self._tts_engine = None
        self._tokenizer = None
        self._model_config = None

    def __del__(self):
        """Cleanup on deletion."""
        self.unload()
