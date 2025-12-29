"""
CosyVoice3 Engine Wrapper for TTS Audio Suite

Fun-CosyVoice3-0.5B integration providing:
- 9-language zero-shot voice cloning (Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian)
- 18+ Chinese dialects support
- Instruct mode for emotions, speed, and dialects
- Cross-lingual voice cloning with fine-grained control
"""

import os
import sys
import torch
import torchaudio
import tempfile
import folder_paths
import numpy as np
from typing import Optional, List, Dict, Any, Generator

from utils.models.unified_model_interface import unified_model_interface
from utils.models.factory_config import ModelLoadConfig
from utils.models.extra_paths import find_model_in_paths, get_preferred_download_path, get_all_tts_model_paths


class CosyVoiceEngine:
    """
    CosyVoice3 Engine wrapper for TTS Audio Suite integration.
    
    Supports:
    - Zero-shot voice cloning (9 languages)
    - Instruct mode for dialects, emotions, speed
    - Cross-lingual voice cloning
    - Fine-grained control with [breath] and other tags
    """
    
    # Supported languages
    SUPPORTED_LANGUAGES = [
        "chinese", "english", "japanese", "korean", 
        "german", "spanish", "french", "italian", "russian"
    ]
    
    # Supported Chinese dialects (for instruct mode)
    SUPPORTED_DIALECTS = [
        "guangdong", "minnan", "sichuan", "dongbei", "shan3xi", "shan1xi",
        "shanghai", "tianjin", "shandong", "ningxia", "gansu"
    ]
    
    # Generation modes
    MODES = ["zero_shot", "instruct", "cross_lingual"]
    
    def __init__(self, model_dir: str = "Fun-CosyVoice3-0.5B", device: str = "auto",
                 use_fp16: bool = True, load_trt: bool = False, load_vllm: bool = False):
        """
        Initialize CosyVoice3 engine.

        Args:
            model_dir: Model identifier ("Fun-CosyVoice3-0.5B" or "local:ModelName")
            device: Device to use ("auto", "cuda", "cpu")
            use_fp16: Use FP16 for faster inference
            load_trt: Load TensorRT engine (optional optimization)
            load_vllm: Load vLLM engine (optional optimization)
        """
        # Resolve model directory using extra_model_paths
        self.model_dir = self._find_model_directory(model_dir)
        
        self.device = self._resolve_device(device)
        self.use_fp16 = use_fp16 and self.device != "cpu"
        self.load_trt = load_trt
        self.load_vllm = load_vllm
        
        self._cosyvoice = None
        self._model_config = None
        self._quantized_move_warning_shown = False
        
    def _find_model_directory(self, model_identifier: str) -> str:
        """Find CosyVoice model directory using extra_model_paths configuration."""
        try:
            # Handle local: prefix
            if model_identifier.startswith("local:"):
                model_name = model_identifier[6:]  # Remove "local:" prefix
                
                # Search in all configured TTS paths
                all_tts_paths = get_all_tts_model_paths('TTS')
                for base_path in all_tts_paths:
                    # Check direct path (models/TTS/Fun-CosyVoice3-0.5B)
                    direct_path = os.path.join(base_path, model_name)
                    if os.path.exists(os.path.join(direct_path, "cosyvoice3.yaml")):
                        return direct_path
                    
                    # Check organized path (models/TTS/CosyVoice/Fun-CosyVoice3-0.5B)
                    organized_path = os.path.join(base_path, "CosyVoice", model_name)
                    if os.path.exists(os.path.join(organized_path, "cosyvoice3.yaml")):
                        return organized_path
                
                raise FileNotFoundError(f"Local CosyVoice model '{model_name}' not found in any configured path")
            
            else:
                # Auto-download case - return preferred download path with model name appended
                base_path = get_preferred_download_path(model_type='TTS', engine_name='CosyVoice')
                model_path = os.path.join(base_path, model_identifier)
                
                # Check if model exists and is complete, if not trigger auto-download
                needs_download = False
                if not os.path.exists(model_path):
                    needs_download = True
                    print(f"📥 CosyVoice3 model directory not found, triggering auto-download...")
                else:
                    # Check model completeness
                    try:
                        from engines.cosyvoice.cosyvoice_downloader import CosyVoiceDownloader
                        downloader = CosyVoiceDownloader()
                        downloader._verify_model(model_path, model_identifier, verbose=False)
                    except Exception as verify_error:
                        needs_download = True
                        print(f"📥 CosyVoice3 model incomplete (missing files), triggering re-download...")
                        print(f"    Verification error: {verify_error}")
                
                if needs_download:
                    try:
                        if 'downloader' not in locals():
                            from engines.cosyvoice.cosyvoice_downloader import CosyVoiceDownloader
                            downloader = CosyVoiceDownloader()
                        downloaded_path = downloader.download_model(model_identifier)
                        print(f"✅ CosyVoice3 auto-download completed: {downloaded_path}")
                        return downloaded_path
                    except Exception as download_error:
                        raise RuntimeError(f"CosyVoice3 model not found/incomplete and auto-download failed: {download_error}")
                
                return model_path
                
        except Exception:
            # Fallback to default path
            model_name = model_identifier.replace("local:", "") if model_identifier.startswith("local:") else model_identifier
            return os.path.join(folder_paths.models_dir, "TTS", "CosyVoice", model_name)
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        from utils.device import resolve_torch_device
        resolved = resolve_torch_device(device)
        return resolved
    
    def _ensure_model_loaded(self):
        """Load the CosyVoice3 model using unified model interface."""
        if self._cosyvoice is not None:
            return
        
        # Create model configuration
        self._model_config = ModelLoadConfig(
            engine_name="cosyvoice",
            model_type="tts",
            model_name="Fun-CosyVoice3-0.5B",
            device=self.device,
            model_path=self.model_dir,
            additional_params={
                "use_fp16": self.use_fp16,
                "load_trt": self.load_trt,
                "load_vllm": self.load_vllm
            }
        )
        
        # Load via unified interface with progress indication
        print("🔄 CosyVoice3: Initializing engine (first run may take 1-2 minutes)...")
        print("   Loading: LLM → Flow → HiFT → Speech Tokenizer → CampPlus...")
        self._cosyvoice = unified_model_interface.load_model(self._model_config)
        
        print(f"✅ CosyVoice3 engine loaded via unified interface on {self.device}")
        print(f"   Sample rate: {self.get_sample_rate()} Hz")
        print("⚡ Next generations will be much faster (models cached in VRAM)")
    
    def _ensure_device_loaded(self):
        """Check and reload model if it was offloaded to CPU."""
        from utils.device import resolve_torch_device
        target_device = resolve_torch_device("auto")
        
        if self._cosyvoice is None:
            return
        
        # Check if model was offloaded - look for a model component
        model_component = None
        if hasattr(self._cosyvoice, 'model') and hasattr(self._cosyvoice.model, 'llm'):
            model_component = self._cosyvoice.model.llm
        elif hasattr(self._cosyvoice, 'llm'):
            model_component = self._cosyvoice.llm
        
        if model_component is not None and hasattr(model_component, 'parameters'):
            try:
                first_param = next(model_component.parameters())
                current_device = str(first_param.device)
                
                if current_device != target_device:
                    # Use unified model manager for device movement
                    try:
                        from utils.models.comfyui_model_wrapper.model_manager import tts_model_manager
                        if not tts_model_manager.ensure_device("cosyvoice", target_device):
                            # Fallback to direct movement
                            self.to(target_device)
                    except Exception:
                        self.to(target_device)
            except StopIteration:
                pass
    
    def generate_zero_shot(
        self,
        text: str,
        prompt_wav: str,
        prompt_text: str,
        speed: float = 1.0,
        stream: bool = False,
        text_frontend: bool = False,  # False for English, True for Chinese
        progress_bar=None
    ) -> torch.Tensor:
        """
        Generate speech using zero-shot voice cloning.
        
        Args:
            text: Text to synthesize
            prompt_wav: Reference audio file for voice cloning
            prompt_text: Transcript of reference audio (REQUIRED)
            speed: Speech speed multiplier (0.5-2.0)
            stream: Enable streaming output
            text_frontend: Use text normalization frontend (False for English, True for Chinese)
            progress_bar: ComfyUI progress bar for tracking
            
        Returns:
            Generated audio as torch.Tensor [1, samples] at 24000 Hz
        """
        self._ensure_model_loaded()
        self._ensure_device_loaded()

        # CosyVoice3 REQUIRES the prompt_text to be in format:
        # 'You are a helpful assistant.<|endofprompt|>' + actual_transcript
        # This is critical for correct audio generation!
        formatted_prompt_text = prompt_text
        if not prompt_text.startswith('You are a helpful assistant.'):
            formatted_prompt_text = f'You are a helpful assistant.<|endofprompt|>{prompt_text}'
        elif '<|endofprompt|>' not in prompt_text:
            # Has prefix but missing delimiter
            formatted_prompt_text = prompt_text.replace('You are a helpful assistant.', 'You are a helpful assistant.<|endofprompt|>')
        
        # Collect all audio chunks
        audio_chunks = []
        
        # DEBUG: Show what's being passed to inference
        print(f"🔍 CosyVoice3 zero_shot DEBUG:")
        print(f"   tts_text: {text[:50]}...")
        print(f"   prompt_text: {formatted_prompt_text[:100]}...")
        print(f"   prompt_wav: {prompt_wav}")
        
        for i, output in enumerate(self._cosyvoice.inference_zero_shot(
            tts_text=text,
            prompt_text=formatted_prompt_text,
            prompt_wav=prompt_wav,
            stream=stream,
            speed=speed,
            text_frontend=text_frontend
        )):
            audio_chunk = output['tts_speech']
            
            # DEBUG: Log raw tensor statistics
            print(f"🔍 CosyVoice3 raw output chunk {i}:")
            print(f"   shape: {audio_chunk.shape}")
            print(f"   dtype: {audio_chunk.dtype}")
            print(f"   device: {audio_chunk.device}")
            print(f"   min: {audio_chunk.min().item():.6f}, max: {audio_chunk.max().item():.6f}")
            print(f"   mean: {audio_chunk.mean().item():.6f}, std: {audio_chunk.std().item():.6f}")
            
            audio_chunks.append(audio_chunk)
            
            if progress_bar is not None:
                progress_bar.update(1)
        
        # Combine all chunks
        if audio_chunks:
            audio_tensor = torch.cat(audio_chunks, dim=-1)
        else:
            # Return silence if no chunks
            audio_tensor = torch.zeros(1, self.get_sample_rate(), dtype=torch.float32)
        
        # DEBUG: Save raw audio DIRECTLY to bypass all wrapper layers
        # Normalize dimensions to [1, samples]
        audio_tensor = self._normalize_audio_dims(audio_tensor)
        
        return audio_tensor
    
    def generate_instruct(
        self,
        text: str,
        prompt_wav: str,
        instruct_text: str,
        speed: float = 1.0,
        stream: bool = False,
        text_frontend: bool = True,
        progress_bar=None
    ) -> torch.Tensor:
        """
        Generate speech using instruct mode for controlling emotions, dialects, speed.
        
        Args:
            text: Text to synthesize
            prompt_wav: Reference audio file for voice cloning
            instruct_text: Instruction text (e.g., "请用广东话表达。<|endofprompt|>")
            speed: Speech speed multiplier (0.5-2.0)
            stream: Enable streaming output
            text_frontend: Use text normalization frontend
            progress_bar: ComfyUI progress bar for tracking

        Returns:
            Generated audio as torch.Tensor [1, samples] at 24000 Hz
        """
        self._ensure_model_loaded()
        self._ensure_device_loaded()

        # CosyVoice3 instruct format MUST be: 'You are a helpful assistant. {instruction}<|endofprompt|>'
        # Example: 'You are a helpful assistant. 请用广东话表达。<|endofprompt|>'
        formatted_instruct = instruct_text
        if not instruct_text.endswith('<|endofprompt|>'):
            if instruct_text.startswith('You are a helpful assistant'):
                formatted_instruct = f'{instruct_text}<|endofprompt|>'
            else:
                formatted_instruct = f'You are a helpful assistant. {instruct_text}<|endofprompt|>'
        elif not instruct_text.startswith('You are a helpful assistant'):
            # Has <|endofprompt|> but missing prefix
            formatted_instruct = f'You are a helpful assistant. {instruct_text}'
        
        # Collect all audio chunks
        audio_chunks = []
        
        for i, output in enumerate(self._cosyvoice.inference_instruct2(
            tts_text=text,
            instruct_text=formatted_instruct,
            prompt_wav=prompt_wav,
            stream=stream,
            speed=speed,
            text_frontend=text_frontend
        )):
            audio_chunk = output['tts_speech']
            audio_chunks.append(audio_chunk)
            
            if progress_bar is not None:
                progress_bar.update(1)
        
        # Combine all chunks
        if audio_chunks:
            audio_tensor = torch.cat(audio_chunks, dim=-1)
        else:
            audio_tensor = torch.zeros(1, self.get_sample_rate(), dtype=torch.float32)
        
        # Normalize dimensions
        audio_tensor = self._normalize_audio_dims(audio_tensor)
        
        return audio_tensor
    
    def generate_cross_lingual(
        self,
        text: str,
        prompt_wav: str,
        speed: float = 1.0,
        stream: bool = False,
        text_frontend: bool = True,
        progress_bar=None
    ) -> torch.Tensor:
        """
        Generate speech using cross-lingual mode with fine-grained control.
        
        Supports embedded control tags like [breath] in the text.
        
        Args:
            text: Text to synthesize (can include [breath] and other tags)
            prompt_wav: Reference audio file for voice cloning
            speed: Speech speed multiplier (0.5-2.0)
            stream: Enable streaming output
            text_frontend: Use text normalization frontend
            progress_bar: ComfyUI progress bar for tracking

        Returns:
            Generated audio as torch.Tensor [1, samples] at 24000 Hz
        """
        self._ensure_model_loaded()
        self._ensure_device_loaded()

        # CosyVoice3 cross_lingual also requires the system prompt prefix in the text
        # Example: 'You are a helpful assistant.<|endofprompt|>[breath]...'
        formatted_text = text
        if not text.startswith('You are a helpful assistant.'):
            formatted_text = f'You are a helpful assistant.<|endofprompt|>{text}'
        elif '<|endofprompt|>' not in text:
            formatted_text = text.replace('You are a helpful assistant.', 'You are a helpful assistant.<|endofprompt|>')
        
        # Collect all audio chunks
        audio_chunks = []
        
        for i, output in enumerate(self._cosyvoice.inference_cross_lingual(
            tts_text=formatted_text,
            prompt_wav=prompt_wav,
            stream=stream,
            speed=speed,
            text_frontend=text_frontend
        )):
            audio_chunk = output['tts_speech']
            audio_chunks.append(audio_chunk)
            
            if progress_bar is not None:
                progress_bar.update(1)
        
        # Combine all chunks
        if audio_chunks:
            audio_tensor = torch.cat(audio_chunks, dim=-1)
        else:
            audio_tensor = torch.zeros(1, self.get_sample_rate(), dtype=torch.float32)
        
        # Normalize dimensions
        audio_tensor = self._normalize_audio_dims(audio_tensor)
        
        return audio_tensor
    
    def generate(
        self,
        text: str,
        prompt_wav: str,
        prompt_text: Optional[str] = None,
        mode: str = "zero_shot",
        instruct_text: Optional[str] = None,
        speed: float = 1.0,
        stream: bool = False,
        text_frontend: bool = False,  # False for English, True for Chinese
        progress_bar=None,
        **kwargs
    ) -> torch.Tensor:
        """
        Unified generation method that routes to appropriate mode.
        
        Args:
            text: Text to synthesize
            prompt_wav: Reference audio file for voice cloning
            prompt_text: Transcript of reference audio (required for zero_shot mode)
            mode: Generation mode ("zero_shot", "instruct", "cross_lingual")
            instruct_text: Instruction for instruct mode
            speed: Speech speed multiplier (0.5-2.0)
            stream: Enable streaming output
            text_frontend: Use text normalization frontend
            progress_bar: ComfyUI progress bar for tracking
            **kwargs: Additional parameters (ignored for compatibility)

        Returns:
            Generated audio as torch.Tensor [1, samples] at 24000 Hz
        """
        if mode == "zero_shot":
            if not prompt_text:
                raise ValueError("prompt_text is required for zero_shot mode. "
                               "Provide a transcript of the reference audio.")
            return self.generate_zero_shot(
                text=text,
                prompt_wav=prompt_wav,
                prompt_text=prompt_text,
                speed=speed,
                stream=stream,
                text_frontend=text_frontend,
                progress_bar=progress_bar
            )
        elif mode == "instruct":
            if not instruct_text:
                instruct_text = "You are a helpful assistant.<|endofprompt|>"
            return self.generate_instruct(
                text=text,
                prompt_wav=prompt_wav,
                instruct_text=instruct_text,
                speed=speed,
                stream=stream,
                text_frontend=text_frontend,
                progress_bar=progress_bar
            )
        elif mode == "cross_lingual":
            return self.generate_cross_lingual(
                text=text,
                prompt_wav=prompt_wav,
                speed=speed,
                stream=stream,
                text_frontend=text_frontend,
                progress_bar=progress_bar
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Supported: {self.MODES}")
    
    def _normalize_audio_dims(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize audio tensor to [1, samples] format."""
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # [samples] -> [1, samples]
        elif audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(0)    # [1, 1, samples] -> [1, samples]
        return audio_tensor
    
    def get_sample_rate(self) -> int:
        """Get the native sample rate of the engine."""
        # CosyVoice3 uses 24000 Hz (from cosyvoice3.yaml)
        if self._cosyvoice is not None and hasattr(self._cosyvoice, 'sample_rate'):
            return self._cosyvoice.sample_rate
        return 24000
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.SUPPORTED_LANGUAGES.copy()
    
    def get_supported_dialects(self) -> List[str]:
        """Get list of supported Chinese dialects."""
        return self.SUPPORTED_DIALECTS.copy()
    
    def to(self, device):
        """
        Move all model components to the specified device.

        Critical for ComfyUI model management - ensures all components move together
        when models are detached to CPU and later reloaded to CUDA.
        """
        self.device = device

        # Move the underlying CosyVoice model if loaded
        if self._cosyvoice is not None:
            # CosyVoice has model, frontend components
            if hasattr(self._cosyvoice, 'model') and hasattr(self._cosyvoice.model, 'to'):
                try:
                    self._cosyvoice.model = self._cosyvoice.model.to(device)
                except ValueError as e:
                    if "is not supported for" in str(e) and ("8-bit" in str(e) or "4-bit" in str(e)):
                        if not self._quantized_move_warning_shown:
                            print(f"⚠️ Skipping device move for quantized model: {e}")
                            self._quantized_move_warning_shown = True
                    else:
                        raise
            
            # Update device attribute
            if hasattr(self._cosyvoice, 'device'):
                self._cosyvoice.device = torch.device(device) if isinstance(device, str) else device

        return self

    def unload(self):
        """Unload the model to free memory."""
        if self._model_config:
            unified_model_interface.unload_model(self._model_config)
        self._cosyvoice = None
        self._model_config = None
