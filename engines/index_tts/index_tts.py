import os
import sys
import torch
import torchaudio
import tempfile
import folder_paths
from typing import Optional, Union, List, Dict, Any
import warnings

from utils.models.unified_model_interface import unified_model_interface, UnifiedModelConfig
from utils.models.extra_paths import find_model_in_paths, get_preferred_download_path, get_all_tts_model_paths


class IndexTTSEngine:
    """
    IndexTTS-2 Engine wrapper for TTS Audio Suite integration.
    
    Supports:
    - Zero-shot voice cloning
    - Emotion disentanglement (separate speaker and emotion control)  
    - Duration-controlled generation
    - Multi-modal emotion control (audio, text, vectors)
    - High-quality emotional expression
    """
    
    EMOTION_LABELS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
    
    def __init__(self, model_dir: str = "IndexTTS-2", device: str = "auto",
                 use_fp16: bool = True, use_cuda_kernel: Optional[bool] = None,
                 use_deepspeed: bool = False):
        """
        Initialize IndexTTS-2 engine.

        Args:
            model_dir: Model identifier (following F5TTS pattern: "local:ModelName" or "ModelName")
            device: Device to use ("auto", "cuda", "cpu", etc.)
            use_fp16: Whether to use FP16 for faster inference
            use_cuda_kernel: Use BigVGAN CUDA kernels (auto-detect if None)
            use_deepspeed: Use DeepSpeed for optimization
        """
        # Resolve model directory using extra_model_paths
        self.model_dir = self._find_model_directory(model_dir)

        self.device = self._resolve_device(device)
        self.use_fp16 = use_fp16 and self.device != "cpu"
        self.use_cuda_kernel = use_cuda_kernel
        self.use_deepspeed = use_deepspeed

        self._tts_engine = None
        self._model_config = None

    def _find_model_directory(self, model_identifier: str) -> str:
        """Find IndexTTS-2 model directory using extra_model_paths configuration."""
        try:
            # Handle local: prefix (following F5TTS pattern)
            if model_identifier.startswith("local:"):
                model_name = model_identifier[6:]  # Remove "local:" prefix

                # Search in all configured TTS paths
                all_tts_paths = get_all_tts_model_paths('TTS')
                for base_path in all_tts_paths:
                    # Check direct path (models/TTS/IndexTTS-2)
                    direct_path = os.path.join(base_path, model_name)
                    if os.path.exists(os.path.join(direct_path, "config.yaml")):
                        return direct_path

                    # Check organized path (models/TTS/IndexTTS/IndexTTS-2)
                    organized_path = os.path.join(base_path, "IndexTTS", model_name)
                    if os.path.exists(os.path.join(organized_path, "config.yaml")):
                        return organized_path

                raise FileNotFoundError(f"Local IndexTTS model '{model_name}' not found in any configured path")

            else:
                # Auto-download case - return preferred download path with model name appended
                base_path = get_preferred_download_path(model_type='TTS', engine_name='IndexTTS')
                model_path = os.path.join(base_path, model_identifier)

                # Check if model exists and is complete, if not trigger auto-download
                needs_download = False
                if not os.path.exists(model_path):
                    needs_download = True
                    print(f"📥 IndexTTS-2 model directory not found, triggering auto-download...")
                else:
                    # Check model completeness using downloader's verification
                    try:
                        from engines.index_tts.index_tts_downloader import IndexTTSDownloader
                        downloader = IndexTTSDownloader()
                        downloader._verify_model(model_path, model_identifier)
                    except Exception as verify_error:
                        needs_download = True
                        print(f"📥 IndexTTS-2 model incomplete (missing files), triggering re-download...")
                        print(f"    Verification error: {verify_error}")

                if needs_download:
                    try:
                        if 'downloader' not in locals():
                            from engines.index_tts.index_tts_downloader import IndexTTSDownloader
                            downloader = IndexTTSDownloader()
                        downloaded_path = downloader.download_model(model_identifier)
                        print(f"✅ IndexTTS-2 auto-download completed: {downloaded_path}")
                        return downloaded_path
                    except Exception as download_error:
                        raise RuntimeError(f"IndexTTS-2 model not found/incomplete and auto-download failed: {download_error}")

                return model_path

        except Exception:
            # Fallback to default path
            model_name = model_identifier.replace("local:", "") if model_identifier.startswith("local:") else model_identifier
            return os.path.join(folder_paths.models_dir, "TTS", "IndexTTS", model_name)

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
        
    def _ensure_model_loaded(self):
        """Load the IndexTTS-2 model using unified model interface."""
        if self._tts_engine is not None:
            return
            
        # Create model configuration
        self._model_config = UnifiedModelConfig(
            engine_name="index_tts",
            model_type="tts",
            model_name="IndexTTS-2",
            device=self.device,
            model_path=self.model_dir,
            additional_params={
                "use_fp16": self.use_fp16,
                "use_cuda_kernel": self.use_cuda_kernel,
                "use_deepspeed": self.use_deepspeed
            }
        )
        
        # Load via unified interface with progress indication
        print("🔄 IndexTTS-2: Initializing engine (first run may take 2-3 minutes to load models)...")
        print("   Loading: QwenEmotion → GPT → Semantic Codec → S2Mel → CampPlus → BigVGAN...")
        self._tts_engine = unified_model_interface.load_model(self._model_config)
        
        print(f"✅ IndexTTS-2 engine loaded via unified interface on {self.device}")
        print("⚡ Next generations will be much faster (models cached in VRAM)")
        
        # Performance warning for non-Python 3.13 environments
        import sys
        if sys.version_info[:2] != (3, 13):
            print("⚠️ Performance warning: IndexTTS-2 tested on Python 3.13 performs smoothly")
            print("⚠️ Our Python 3.12 tests showed HIGH VRAM spikes during generation")
    
    def generate(
        self,
        text: str,
        speaker_audio: str,
        emotion_audio: Optional[str] = None,
        emotion_alpha: float = 1.0,
        emotion_vector: Optional[List[float]] = None,
        use_emotion_text: bool = False,
        emotion_text: Optional[str] = None,
        use_random: bool = False,
        interval_silence: int = 200,
        max_text_tokens_per_segment: int = 120,
        # Generation parameters
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 30,
        length_penalty: float = 0.0,
        num_beams: int = 3,
        repetition_penalty: float = 10.0,
        max_mel_tokens: int = 1500,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate speech using IndexTTS-2.
        
        Args:
            text: Text to synthesize
            speaker_audio: Reference audio file for speaker voice
            emotion_audio: Reference audio file for emotion (optional)
            emotion_alpha: Blend factor for emotion (0.0-1.0)
            emotion_vector: Manual emotion vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
            use_emotion_text: Use text-based emotion extraction
            emotion_text: Custom emotion description text
            use_random: Enable random sampling for variation
            interval_silence: Silence between segments (ms)
            max_text_tokens_per_segment: Max tokens per segment
            do_sample: Use sampling for generation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            length_penalty: Length penalty for beam search
            num_beams: Number of beams for beam search
            repetition_penalty: Repetition penalty
            max_mel_tokens: Maximum mel tokens to generate
            
        Returns:
            Generated audio as torch.Tensor with shape [1, samples]
        """
        self._ensure_model_loaded()

        # CRITICAL FIX: Reload model to correct device if it was offloaded
        # IMPORTANT: Always check against the INTENDED device (cuda if available), not self.device which gets updated to CPU
        target_device = "cuda" if torch.cuda.is_available() else "cpu"

        # Check if model was offloaded to CPU and needs to be reloaded
        if self._tts_engine is not None and hasattr(self._tts_engine, 'semantic_model'):
            if hasattr(self._tts_engine.semantic_model, 'parameters'):
                try:
                    first_param = next(self._tts_engine.semantic_model.parameters())
                    current_device = str(first_param.device)
                    # print(f"🔧 Index-TTS device check: current={current_device}, target={target_device}")
                    if current_device != target_device:
                        # print(f"🔄 Reloading Index-TTS model from {current_device} to {target_device} via wrapper")

                        # Find and call wrapper's model_load() to keep ComfyUI tracking in sync
                        try:
                            from utils.models.unified_model_interface import unified_model_interface

                            # Index-TTS is loaded via unified interface, search its cache
                            wrapper_found = False
                            if hasattr(unified_model_interface, 'model_manager'):
                                for cache_key, wrapper in unified_model_interface.model_manager._model_cache.items():
                                    # Check if wrapper.model is self, or if it's wrapped in SimpleModelWrapper
                                    model = wrapper.model if hasattr(wrapper, 'model') else None
                                    if model is self:
                                        wrapper.model_load(target_device)
                                        # print(f"✅ Reloaded Index-TTS via wrapper - ComfyUI management stays in sync")
                                        wrapper_found = True
                                        break
                                    # Also check through SimpleModelWrapper if present
                                    elif hasattr(model, 'model') and model.model is self:
                                        wrapper.model_load(target_device)
                                        # print(f"✅ Reloaded Index-TTS via wrapper (unwrapped SimpleModelWrapper) - ComfyUI management stays in sync")
                                        wrapper_found = True
                                        break

                            if not wrapper_found:
                                # Fallback: direct .to() and check if already registered before re-registering
                                self.to(target_device)

                                # Check if we're already in ComfyUI's current_loaded_models (from previous re-registration)
                                try:
                                    import comfy.model_management as model_management

                                    already_registered = False
                                    if hasattr(model_management, 'current_loaded_models'):
                                        for wrapper in model_management.current_loaded_models:
                                            if hasattr(wrapper, 'model_info') and wrapper.model_info.engine == "index_tts":
                                                # Found an index_tts wrapper, just reload it
                                                wrapper.model_load(target_device)
                                                print(f"✅ Reloaded Index-TTS via existing ComfyUI wrapper")
                                                already_registered = True
                                                break

                                    if not already_registered:
                                        # Re-register with ComfyUI after direct reload
                                        from utils.models.comfyui_model_wrapper.base_wrapper import ComfyUIModelWrapper, ModelInfo, SimpleModelWrapper

                                        # Estimate model size
                                        model_size = ComfyUIModelWrapper.calculate_model_memory(self)

                                        # Wrap with SimpleModelWrapper to add .model attribute for ComfyUI logging
                                        wrapped_model = SimpleModelWrapper(self)

                                        # Create new wrapper and register
                                        model_info = ModelInfo(
                                            model=wrapped_model,
                                            model_type="tts",
                                            engine="index_tts",
                                            device=target_device,
                                            memory_size=model_size,
                                            load_device=target_device
                                        )
                                        new_wrapper = ComfyUIModelWrapper(wrapped_model, model_info)

                                        model_management.current_loaded_models.append(new_wrapper)
                                        print(f"✅ Re-registered Index-TTS with ComfyUI model management")
                                except Exception as reg_error:
                                    print(f"⚠️ Re-registration failed: {reg_error}")

                        except Exception as e:
                            # Fallback to direct .to()
                            print(f"⚠️ Wrapper reload failed ({e}), using direct .to()")
                            self.to(target_device)
                except StopIteration:
                    pass

        # Validate emotion vector if provided
        if emotion_vector is not None:
            if len(emotion_vector) != 8:
                raise ValueError(f"Emotion vector must have 8 values for {self.EMOTION_LABELS}")
            # Normalize to valid range
            emotion_vector = [max(0.0, min(1.2, v)) for v in emotion_vector]
        
        # Create temporary output file in ComfyUI temp directory
        comfyui_temp_dir = folder_paths.get_temp_directory()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=comfyui_temp_dir) as tmp_file:
            output_path = tmp_file.name
            
        try:
            # Call IndexTTS-2 inference
            result = self._tts_engine.infer(
                spk_audio_prompt=speaker_audio,
                text=text,
                output_path=output_path,
                emo_audio_prompt=emotion_audio,
                emo_alpha=emotion_alpha,
                emo_vector=emotion_vector,
                use_emo_text=use_emotion_text,
                emo_text=emotion_text,
                use_random=use_random,
                interval_silence=interval_silence,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                # Generation kwargs
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                length_penalty=length_penalty,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                max_mel_tokens=max_mel_tokens,
                **kwargs
            )
            
            # Load generated audio
            audio, sample_rate = torchaudio.load(output_path)
            
            # Convert to expected format [1, samples] at 22050 Hz
            if sample_rate != 22050:
                resampler = torchaudio.transforms.Resample(sample_rate, 22050)
                audio = resampler(audio)
                
            if audio.shape[0] != 1:
                audio = audio.mean(dim=0, keepdim=True)  # Convert to mono
                
            return audio
            
        finally:
            # Clean up temporary file
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except OSError:
                    pass
    
    def get_sample_rate(self) -> int:
        """Get the native sample rate of the engine."""
        return 22050
    
    def get_supported_formats(self) -> List[str]:
        """Get supported audio formats."""
        return ["wav", "mp3", "flac", "ogg"]
    
    def get_emotion_labels(self) -> List[str]:
        """Get supported emotion labels."""
        return self.EMOTION_LABELS.copy()
    
    def create_emotion_vector(self, **emotions) -> List[float]:
        """
        Create emotion vector from keyword arguments.
        
        Args:
            **emotions: Emotion intensities (e.g., happy=0.8, angry=0.2)
            
        Returns:
            List of 8 emotion values
        """
        vector = [0.0] * 8
        for i, label in enumerate(self.EMOTION_LABELS):
            if label in emotions:
                vector[i] = max(0.0, min(1.2, float(emotions[label])))
        return vector
    
    def to(self, device):
        """
        Move all model components to the specified device.

        Critical for ComfyUI model management - ensures all components move together
        when models are detached to CPU and later reloaded to CUDA.
        """
        self.device = device

        # Move the underlying TTS engine if loaded
        if self._tts_engine is not None:
            # Index-TTS has deeply nested components - use recursive approach
            # Call .to() on the engine itself which should recursively move all PyTorch modules
            if hasattr(self._tts_engine, 'to'):
                self._tts_engine = self._tts_engine.to(device)
            else:
                # Fallback: manually move known components
                for attr_name in ['semantic_model', 'semantic_codec', 'gpt_model', 'gpt',
                                  's2mel', 'campplus_model', 'bigvgan', 'qwen_model']:
                    if hasattr(self._tts_engine, attr_name):
                        component = getattr(self._tts_engine, attr_name)
                        if hasattr(component, 'to'):
                            setattr(self._tts_engine, attr_name, component.to(device))

            # Update device attribute on engine
            if hasattr(self._tts_engine, 'device'):
                self._tts_engine.device = torch.device(device) if isinstance(device, str) else device

        return self

    def unload(self):
        """Unload the model to free memory."""
        if self._model_config:
            unified_model_interface.unload_model(self._model_config)
        self._tts_engine = None
        self._model_config = None

    def __del__(self):
        """Cleanup on deletion."""
        self.unload()