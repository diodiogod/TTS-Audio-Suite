import os
import sys
import torch
import torchaudio
import tempfile
from typing import Optional, Union, List, Dict, Any
import warnings

from utils.models.comfyui_model_wrapper import ComfyUIModelWrapper
from utils.downloads.unified_downloader import UnifiedDownloader


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
    
    def __init__(self, model_dir: str = "models/TTS/IndexTTS-2", device: str = "auto", 
                 use_fp16: bool = True, use_cuda_kernel: Optional[bool] = None,
                 use_deepspeed: bool = False):
        """
        Initialize IndexTTS-2 engine.
        
        Args:
            model_dir: Directory containing IndexTTS-2 models
            device: Device to use ("auto", "cuda", "cpu", etc.)
            use_fp16: Whether to use FP16 for faster inference
            use_cuda_kernel: Use BigVGAN CUDA kernels (auto-detect if None)
            use_deepspeed: Use DeepSpeed for optimization
        """
        self.model_dir = model_dir
        self.device = self._resolve_device(device)
        self.use_fp16 = use_fp16 and self.device != "cpu"
        self.use_cuda_kernel = use_cuda_kernel
        self.use_deepspeed = use_deepspeed
        
        self._tts_engine = None
        self._model_wrapper = None
        
        # Cache for reference audio processing
        self._spk_cache = {}
        self._emo_cache = {}
        
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
        """Lazy load the IndexTTS-2 model when first needed."""
        if self._tts_engine is not None:
            return
            
        try:
            # Add IndexTTS module to path if needed
            index_tts_path = os.path.join(self.model_dir, "index-tts")
            if os.path.exists(index_tts_path) and index_tts_path not in sys.path:
                sys.path.insert(0, index_tts_path)
                
            from indextts.infer_v2 import IndexTTS2
            
            # Initialize IndexTTS-2 engine
            config_path = os.path.join(self.model_dir, "config.yaml")
            
            self._tts_engine = IndexTTS2(
                cfg_path=config_path,
                model_dir=self.model_dir,
                device=self.device,
                use_fp16=self.use_fp16,
                use_cuda_kernel=self.use_cuda_kernel,
                use_deepspeed=self.use_deepspeed
            )
            
            # Wrap for ComfyUI model management
            self._model_wrapper = ComfyUIModelWrapper(
                self._tts_engine,
                model_name=f"IndexTTS-2-{self.device}",
                memory_required_mb=4000  # Estimate based on model size
            )
            
            print(f"IndexTTS-2 engine loaded on {self.device}")
            
        except ImportError as e:
            raise ImportError(
                f"IndexTTS-2 dependencies not available. Please install IndexTTS-2 first. Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load IndexTTS-2 model: {e}")
    
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
        
        # Validate emotion vector if provided
        if emotion_vector is not None:
            if len(emotion_vector) != 8:
                raise ValueError(f"Emotion vector must have 8 values for {self.EMOTION_LABELS}")
            # Normalize to valid range
            emotion_vector = [max(0.0, min(1.2, v)) for v in emotion_vector]
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
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
    
    def unload(self):
        """Unload the model to free memory."""
        if self._model_wrapper:
            self._model_wrapper.unload()
        self._tts_engine = None
        self._model_wrapper = None
        self._spk_cache.clear()
        self._emo_cache.clear()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.unload()