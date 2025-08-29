"""
VibeVoice Engine - Main TTS engine wrapper for ComfyUI integration
Provides multi-speaker text-to-speech with long-form generation capabilities
Based on Microsoft VibeVoice model
"""

import torch
import numpy as np
import os
import sys
import gc
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# Add parent directory for imports
current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import utilities
from utils.audio.processing import AudioProcessingUtils
from utils.audio.cache import CacheKeyGenerator, get_audio_cache
from utils.text.chunking import ImprovedChatterBoxChunker
from .vibevoice_downloader import VibeVoiceDownloader, VIBEVOICE_MODELS
import folder_paths

# Import unified model interface for ComfyUI integration
from utils.models.unified_model_interface import load_tts_model

# Setup logging
logger = logging.getLogger("VibeVoice")

# Convert token-based limits to character-based for unified chunker
def tokens_to_chars(max_tokens: int) -> int:
    """Convert VibeVoice token limit to character limit for unified chunker"""
    # VibeVoice uses ~4 chars per token, but we use conservative 3.5 for safety
    return int(max_tokens * 3.5)


class VibeVoiceEngine:
    """
    Main VibeVoice engine wrapper for ComfyUI
    Handles model loading, text generation, and multi-speaker support
    """
    
    def __init__(self):
        """Initialize VibeVoice engine"""
        self.model = None
        self.processor = None
        self.model_path = None
        self.device = None
        self.current_model_name = None
        
        # Use global shared cache
        self.cache = get_audio_cache()
        self.downloader = VibeVoiceDownloader()
        
        # Chunking support
        self.chunker = ImprovedChatterBoxChunker()
        
        # Track if package is available
        self._package_available = None
    
    def _ensure_package(self) -> bool:
        """Ensure VibeVoice package is installed"""
        if self._package_available is not None:
            return self._package_available
        
        self._package_available = self.downloader.ensure_vibevoice_package()
        return self._package_available
    
    def get_available_models(self) -> List[str]:
        """Get list of available VibeVoice models"""
        return self.downloader.get_available_models()
    
    def initialize_engine(self, 
                         model_name: str = "vibevoice-1.5B",
                         device: str = "auto") -> None:
        """
        Initialize VibeVoice engine with specified model.
        
        Args:
            model_name: Model to load ("vibevoice-1.5B" or "vibevoice-7B")
            device: Device to use ("auto", "cuda", or "cpu")
        """
        # Check if already loaded
        if self.model is not None and self.current_model_name == model_name:
            print(f"💾 VibeVoice model '{model_name}' already loaded")
            return
        
        # Ensure package is installed
        if not self._ensure_package():
            raise RuntimeError("VibeVoice package not available. Please install it manually.")
        
        # Import VibeVoice modules
        try:
            import vibevoice
            from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        except ImportError as e:
            raise RuntimeError(f"VibeVoice package not installed. Please install with: pip install git+https://github.com/microsoft/VibeVoice.git\nError: {e}")
        
        # Get model path (downloads if necessary)
        model_path = self.downloader.get_model_path(model_name)
        if not model_path:
            raise RuntimeError(f"Failed to get VibeVoice model '{model_name}'")
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔄 Loading VibeVoice model '{model_name}' on {device}...")
        
        try:
            # Load model using unified interface
            config = {
                "engine_name": "vibevoice",
                "model_type": "tts",
                "model_name": model_name,
                "device": device,
                "model_path": model_path
            }
            
            # For now, load directly until unified interface is updated
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=device if device != "auto" else None
            )
            
            # Load processor
            self.processor = VibeVoiceProcessor.from_pretrained(model_path)
            
            # Move to device if needed
            if device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
            
            self.model_path = model_path
            self.device = device
            self.current_model_name = model_name
            
            print(f"✅ VibeVoice model '{model_name}' loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load VibeVoice model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _create_synthetic_voice_sample(self, speaker_idx: int) -> np.ndarray:
        """
        Create synthetic voice sample for a specific speaker.
        Based on reference implementation but with our own characteristics.
        
        Args:
            speaker_idx: Speaker index (0-3)
            
        Returns:
            Numpy array with synthetic voice sample
        """
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)
        
        t = np.linspace(0, duration, samples, False)
        
        # Create realistic voice-like characteristics for each speaker
        # Use different base frequencies for different speaker types
        base_frequencies = [120, 180, 140, 200]  # Mix of male/female-like frequencies
        base_freq = base_frequencies[speaker_idx % len(base_frequencies)]
        
        # Create vowel-like formants (like "ah" sound) - unique per speaker
        formant1 = 800 + speaker_idx * 100  # First formant
        formant2 = 1200 + speaker_idx * 150  # Second formant
        
        # Generate more voice-like waveform
        voice_sample = (
            # Fundamental with harmonics (voice-like)
            0.6 * np.sin(2 * np.pi * base_freq * t) +
            0.25 * np.sin(2 * np.pi * base_freq * 2 * t) +
            0.15 * np.sin(2 * np.pi * base_freq * 3 * t) +
            
            # Formant resonances (vowel-like characteristics)
            0.1 * np.sin(2 * np.pi * formant1 * t) * np.exp(-t * 2) +
            0.05 * np.sin(2 * np.pi * formant2 * t) * np.exp(-t * 3) +
            
            # Natural breath noise (reduced)
            0.02 * np.random.normal(0, 1, len(t))
        )
        
        # Add natural envelope (like human speech pattern)
        vibrato_freq = 4 + speaker_idx * 0.3  # Slightly different vibrato per speaker
        envelope = (np.exp(-t * 0.3) * (1 + 0.1 * np.sin(2 * np.pi * vibrato_freq * t)))
        voice_sample *= envelope * 0.08  # Lower volume
        
        return voice_sample.astype(np.float32)
    
    def _prepare_voice_samples(self, voice_refs: List[Optional[Dict]]) -> List[np.ndarray]:
        """
        Prepare voice samples from ComfyUI audio inputs or create synthetic ones.
        
        Args:
            voice_refs: List of voice reference dicts from ComfyUI (can contain None)
            
        Returns:
            List of numpy arrays with voice samples
        """
        voice_samples = []
        
        for i, voice_ref in enumerate(voice_refs):
            if voice_ref is not None and isinstance(voice_ref, dict) and "waveform" in voice_ref:
                # Extract waveform from ComfyUI audio format
                waveform = voice_ref["waveform"]
                input_sample_rate = voice_ref.get("sample_rate", 24000)
                
                # Convert to numpy
                if isinstance(waveform, torch.Tensor):
                    audio_np = waveform.cpu().numpy()
                else:
                    audio_np = np.array(waveform)
                
                # Handle different audio shapes
                if audio_np.ndim == 3:  # (batch, channels, samples)
                    audio_np = audio_np[0, 0, :]  # Take first batch, first channel
                elif audio_np.ndim == 2:  # (channels, samples)
                    audio_np = audio_np[0, :]  # Take first channel
                
                # Resample if needed
                if input_sample_rate != 24000:
                    target_length = int(len(audio_np) * 24000 / input_sample_rate)
                    audio_np = np.interp(
                        np.linspace(0, len(audio_np), target_length),
                        np.arange(len(audio_np)), 
                        audio_np
                    )
                
                # Normalize
                audio_max = np.abs(audio_np).max()
                if audio_max > 0:
                    audio_np = audio_np / max(audio_max, 1.0)
                
                voice_samples.append(audio_np.astype(np.float32))
            else:
                # Create synthetic voice sample
                voice_samples.append(self._create_synthetic_voice_sample(i))
        
        return voice_samples
    
    def generate_speech(self, 
                       text: str,
                       voice_samples: List[np.ndarray],
                       cfg_scale: float = 1.3,
                       seed: int = 42,
                       use_sampling: bool = False,
                       temperature: float = 0.95,
                       top_p: float = 0.95,
                       max_new_tokens: Optional[int] = None,
                       enable_cache: bool = True,
                       character: str = "narrator",
                       stable_audio_component: str = "") -> Dict[str, Any]:
        """
        Generate speech from text using VibeVoice.
        
        Args:
            text: Text to convert (should be formatted with Speaker labels)
            voice_samples: List of voice samples for speakers
            cfg_scale: Classifier-free guidance scale
            seed: Random seed for generation
            use_sampling: Whether to use sampling mode
            temperature: Temperature for sampling
            top_p: Top-p for sampling
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dict with "waveform" and "sample_rate"
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not initialized. Call initialize_engine first.")
        
        # Handle caching if enabled (following ChatterBox pattern)
        if enable_cache:
            from utils.audio.cache import create_cache_function
            print(f"🐛 VibeVoice ENGINE: Creating cache with audio_component='{stable_audio_component[:50]}...'")
            
            # Fix floating point precision issues by rounding to 3 decimal places
            cfg_scale_rounded = round(float(cfg_scale), 3) if isinstance(cfg_scale, (int, float)) else cfg_scale
            temperature_rounded = round(float(temperature), 3) if isinstance(temperature, (int, float)) else temperature
            top_p_rounded = round(float(top_p), 3) if isinstance(top_p, (int, float)) else top_p
            
            print(f"🐛 VibeVoice ENGINE: Cache params - character='{character}', cfg_scale={cfg_scale_rounded}, use_sampling={use_sampling}")
            cache_fn = create_cache_function(
                "vibevoice",
                character=character,
                cfg_scale=cfg_scale_rounded,
                temperature=temperature_rounded,
                top_p=top_p_rounded,
                use_sampling=use_sampling,
                seed=seed,
                model_source=self.current_model_name or "vibevoice-1.5B",
                device=self.device,
                max_new_tokens=max_new_tokens,
                audio_component=stable_audio_component
            )
            
            # Try cache first
            cached_audio = cache_fn(text)
            if cached_audio is not None:
                print(f"💾 CACHE HIT for {character}: '{text[:30]}...'")
                print(f"🐛 VibeVoice ENGINE: CACHE HIT - audio_component was '{stable_audio_component[:50]}...'")
                return {
                    "waveform": cached_audio,
                    "sample_rate": 24000
                }
            else:
                print(f"🐛 VibeVoice ENGINE: CACHE MISS - generating new audio for audio_component '{stable_audio_component[:50]}...'")
        
        try:
            # Set seeds for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            print(f"🐛 VibeVoice ENGINE: Starting generation with {len(voice_samples)} voice samples")
            print(f"🐛 VibeVoice ENGINE: Generation params - cfg_scale={cfg_scale}, use_sampling={use_sampling}, seed={seed}")
            print(f"🐛 VibeVoice ENGINE: Text length: {len(text)} chars")
            
            # Prepare inputs using processor
            print(f"🐛 VibeVoice ENGINE: Processing inputs - text='{text[:100]}...', voice_samples count={len(voice_samples)}")
            for i, vs in enumerate(voice_samples):
                if vs is not None:
                    print(f"🐛 VibeVoice ENGINE: Voice sample {i} shape: {vs.shape if hasattr(vs, 'shape') else type(vs)}")
                else:
                    print(f"🐛 VibeVoice ENGINE: Voice sample {i} is None - using synthetic")
            
            inputs = self.processor(
                [text],  # Wrap text in list
                voice_samples=[voice_samples],  # Provide voice samples
                return_tensors="pt",
                return_attention_mask=True
            )
            print(f"🐛 VibeVoice ENGINE: Processor inputs created - input_ids shape: {inputs['input_ids'].shape}")
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Generate with appropriate mode
            with torch.no_grad():
                if use_sampling:
                    # Sampling mode
                    output = self.model.generate(
                        **inputs,
                        tokenizer=self.processor.tokenizer,
                        cfg_scale=cfg_scale,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                    )
                else:
                    # Deterministic mode
                    output = self.model.generate(
                        **inputs,
                        tokenizer=self.processor.tokenizer,
                        cfg_scale=cfg_scale,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
            
            # Extract audio from output
            if hasattr(output, 'speech_outputs') and output.speech_outputs:
                speech_tensors = output.speech_outputs
                
                if isinstance(speech_tensors, list) and len(speech_tensors) > 0:
                    audio_tensor = torch.cat(speech_tensors, dim=-1)
                else:
                    audio_tensor = speech_tensors
                
                # Ensure proper format (1, 1, samples)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
                elif audio_tensor.dim() == 2:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                result = {
                    "waveform": audio_tensor.cpu(),
                    "sample_rate": 24000
                }
                
                # Cache result if enabled (following ChatterBox pattern)
                if enable_cache:
                    # Clone tensor to avoid autograd issues like ChatterBox does
                    # Cache only the waveform tensor, not the full dict
                    waveform_clone = result["waveform"].detach().clone() if result["waveform"].requires_grad else result["waveform"]
                    print(f"🐛 VibeVoice ENGINE: CACHING result for audio_component '{stable_audio_component[:50]}...'")
                    cache_fn(text, audio_result=waveform_clone)
                
                return result
            else:
                raise RuntimeError("VibeVoice failed to generate audio output")
                
        except Exception as e:
            logger.error(f"VibeVoice generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}")
    
    def generate_multi_speaker(self,
                              segments: List[Tuple[str, str]],
                              voice_mapping: Dict[str, np.ndarray],
                              **kwargs) -> Dict[str, Any]:
        """
        Generate multi-speaker dialogue.
        
        Args:
            segments: List of (character, text) tuples
            voice_mapping: Dict mapping character names to voice samples
            **kwargs: Additional generation parameters
            
        Returns:
            Dict with combined audio
        """
        # Convert segments to VibeVoice format
        speaker_map = {}
        speaker_voices = []
        formatted_lines = []
        
        for char, text in segments:
            if char not in speaker_map:
                speaker_idx = len(speaker_map)
                speaker_map[char] = speaker_idx
                speaker_voices.append(voice_mapping.get(char, 
                                     self._create_synthetic_voice_sample(speaker_idx)))
            
            speaker_idx = speaker_map[char]
            formatted_lines.append(f"Speaker {speaker_idx}: {text}")
        
        # Join with newlines for multi-speaker format
        formatted_text = "\n".join(formatted_lines)
        
        # Generate with multi-speaker text
        return self.generate_speech(formatted_text, speaker_voices, **kwargs)
    
    def cleanup(self):
        """Clean up resources"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🧹 VibeVoice engine cleaned up")
    
    def unload_models(self):
        """Unload models (called by ComfyUI's unload button)"""
        self.cleanup()
        self.current_model_name = None
        print("📤 VibeVoice models unloaded")