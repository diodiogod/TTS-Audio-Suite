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

# Global singleton instance to prevent multiple engine creations
_vibevoice_engine_instance = None

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
from .gguf_loader import gguf_loader
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

def get_vibevoice_engine():
    """Get singleton VibeVoice engine instance"""
    global _vibevoice_engine_instance
    if _vibevoice_engine_instance is None:
        _vibevoice_engine_instance = VibeVoiceEngine()
    return _vibevoice_engine_instance


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
        self._initialization_lock = False  # Prevent multiple concurrent initializations
        
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
                         device: str = "auto",
                         attention_mode: str = "auto",
                         quantize_llm_4bit: bool = False) -> None:
        """
        Initialize VibeVoice engine with specified model.
        
        Args:
            model_name: Model to load ("vibevoice-1.5B" or "vibevoice-7B")
            device: Device to use ("auto", "cuda", or "cpu")  
            attention_mode: Attention implementation ("auto", "eager", "sdpa", "flash_attention_2")
            quantize_llm_4bit: Enable 4-bit LLM quantization for VRAM savings
        """
        # Prevent multiple initializations with lock
        if self._initialization_lock:
            print(f"⏳ VibeVoice initialization already in progress, skipping")
            return
            
        # Check if already loaded with same config
        current_config = (model_name, device, attention_mode, quantize_llm_4bit)
        if (hasattr(self, '_current_config') and 
            self._current_config == current_config and
            hasattr(self, 'model') and self.model is not None):
            print(f"💾 VibeVoice model '{model_name}' already loaded with same config")
            return
            
        # Set lock to prevent concurrent initializations
        self._initialization_lock = True
        
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
        
        # Check if this is a GGUF model and handle accordingly
        is_gguf = self.downloader.is_gguf_model(model_name)
        is_quantized = self.downloader.is_quantized_model(model_name)
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔄 Loading VibeVoice model '{model_name}' on {device}...")
        if attention_mode != "auto":
            print(f"   🧠 Using {attention_mode} attention")
        if quantize_llm_4bit:
            print(f"   🗜️ 4-bit LLM quantization enabled")
        
        try:
            # Import required modules  
            from transformers import BitsAndBytesConfig
            
            # Build quantization config if requested
            # Based on wildminder/ComfyUI-VibeVoice implementation
            quant_config = None
            if quantize_llm_4bit:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                print(f"   📊 Quantization config: NF4, double_quant, bfloat16")
            
            # Determine final attention mode
            final_attention_mode = attention_mode
            if attention_mode == "auto":
                # Auto-select best available attention
                try:
                    import flash_attn
                    final_attention_mode = "flash_attention_2"
                    print(f"   ✨ Auto-selected flash_attention_2")
                except ImportError:
                    final_attention_mode = "sdpa"  # PyTorch SDPA as fallback
                    print(f"   ⚡ Auto-selected sdpa (flash_attention_2 not available)")
            
            # Build model kwargs
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if quant_config else torch.bfloat16,
            }
            
            # Set device_map based on quantization and device
            if quant_config:
                # For quantization, use explicit device mapping to avoid buffer issues
                if device == "cuda" or device == "auto":
                    model_kwargs["device_map"] = {"": 0}  # Put everything on GPU 0
                else:
                    model_kwargs["device_map"] = {"": "cpu"}
            else:
                model_kwargs["device_map"] = device if device != "auto" else None
            
            # Add attention implementation if not auto
            if final_attention_mode != "auto":
                model_kwargs["attn_implementation"] = final_attention_mode
                
            # Add quantization config if enabled
            if quant_config:
                model_kwargs["quantization_config"] = quant_config
            
            # Load model with enhanced configuration
            # Handle GGUF models differently
            if is_gguf:
                print(f"🔄 Loading GGUF model from: {model_path}")
                
                # Load GGUF model using our custom loader
                gguf_file = os.path.join(model_path, "model.gguf")
                config_file = os.path.join(model_path, "config.json")
                
                try:
                    # Try truly lazy loading first - no model construction at all
                    from .vibevoice_lazy_model import create_lazy_vibevoice_from_gguf
                    
                    print(f"💨 Attempting lazy GGUF loading (instant, no construction)...")
                    
                    # Create lazy model - should be instant
                    self.model, success = create_lazy_vibevoice_from_gguf(
                        model_path=model_path,
                        device=self.device
                    )
                    
                    if not success:
                        raise RuntimeError("Lazy GGUF loading failed")
                    
                    print(f"✅ Lazy GGUF model ready (modules load on demand)!")
                    
                except Exception as gguf_error:
                    print(f"❌ GGUF loading failed: {gguf_error}")
                    print(f"🔄 Falling back to standard loading...")
                    # Fall back to standard loading
                    is_gguf = False
            
            if not is_gguf:
                # Standard loading path
                # Credits: Based on drbaph's implementation in wildminder/ComfyUI-VibeVoice
                try:
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        model_path,
                        **model_kwargs
                    )
                    
                    # Set model to evaluation mode and mark quantization status
                    self.model.eval()
                    if quant_config:
                        setattr(self.model, "_llm_4bit", True)
                        
                except Exception as e:
                    # If quantization fails, try fallback without quantization
                    if quant_config:
                        print(f"⚠️ Quantization failed, falling back to full precision: {e}")
                        model_kwargs_fallback = model_kwargs.copy()
                        model_kwargs_fallback.pop("quantization_config", None)
                        model_kwargs_fallback["device_map"] = device if device != "auto" else None
                        
                        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            model_path,
                            **model_kwargs_fallback
                        )
                        self.model.eval()
                        quant_config = None  # Mark as no quantization for later logic
                    else:
                        raise
            
            # Load processor
            self.processor = VibeVoiceProcessor.from_pretrained(model_path)
            
            # Verify model and processor are actually set + keep strong references
            if self.model is None:
                raise RuntimeError("Model loading completed but self.model is None")
            if self.processor is None:
                raise RuntimeError("Processor loading completed but self.processor is None")
            
            # Keep additional strong references to prevent garbage collection
            self._model_ref = self.model
            self._processor_ref = self.processor
            
            print(f"✅ Engine validation: model={self.model is not None}, processor={self.processor is not None}")
            print(f"✅ Strong refs: model={self._model_ref is not None}, processor={self._processor_ref is not None}")
            
            # Move to device if needed (only if not using quantization which handles device_map)
            if not quant_config and device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
                
            # Ensure all model parameters are on the same device (fix for speech_bias_factor issue)
            if quant_config and hasattr(self.model, 'speech_bias_factor'):
                try:
                    # Find the device of the main model components
                    main_device = next(self.model.parameters()).device
                    if hasattr(self.model.speech_bias_factor, 'to'):
                        self.model.speech_bias_factor = self.model.speech_bias_factor.to(main_device)
                except Exception as device_fix_error:
                    print(f"⚠️ Device placement fix attempt failed (non-critical): {device_fix_error}")
            
            # Store configuration and model info
            self.model_path = model_path
            self.device = device
            self.current_model_name = model_name
            self._current_config = current_config
            self._attention_mode = final_attention_mode
            self._quantize_llm_4bit = quantize_llm_4bit
            self._is_gguf = is_gguf
            self._is_quantized = is_quantized
            
            print(f"✅ VibeVoice model '{model_name}' loaded successfully")
            print(f"   Device: {device}, Attention: {final_attention_mode}")
            
            if is_gguf:
                print(f"   📦 GGUF format model (experimental)")
            elif is_quantized:
                print(f"   🗜️ Pre-quantized model (VRAM optimized)")
            elif quantize_llm_4bit and quant_config:
                print(f"   🗜️ LLM quantized to 4-bit (VRAM savings expected)")
            elif quantize_llm_4bit and not quant_config:
                print(f"   ⚠️ Quantization failed, using full precision")
            
            # Release lock after successful initialization
            self._initialization_lock = False
            
        except Exception as e:
            # Release lock on error
            self._initialization_lock = False
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
            if voice_ref is not None and isinstance(voice_ref, dict):
                audio_np = None
                input_sample_rate = 24000
                
                if "waveform" in voice_ref:
                    # Extract waveform from ComfyUI audio format
                    waveform = voice_ref["waveform"]
                    input_sample_rate = voice_ref.get("sample_rate", 24000)
                    
                    # Convert to numpy
                    if isinstance(waveform, torch.Tensor):
                        audio_np = waveform.cpu().numpy()
                    else:
                        audio_np = np.array(waveform)
                
                elif "audio_path" in voice_ref and voice_ref["audio_path"]:
                    # Load audio file from path (like TTS Text does)
                    audio_path = voice_ref["audio_path"]
                    try:
                        from utils.audio.librosa_fallback import safe_load
                        audio_np, input_sample_rate = safe_load(audio_path, sr=None, mono=True)
                        print(f"🎵 VibeVoice ENGINE: Loaded audio from {audio_path} - shape: {audio_np.shape}, sr: {input_sample_rate}")
                    except Exception as e:
                        print(f"⚠️ VibeVoice ENGINE: Failed to load audio from {audio_path}: {e}")
                        audio_np = None
                
                if audio_np is not None:
                    # Handle different audio shapes and convert to mono (matches official VibeVoice)
                    if audio_np.ndim == 3:  # (batch, channels, samples)
                        audio_np = audio_np[0]  # Take first batch -> (channels, samples)
                    
                    if audio_np.ndim == 2:
                        if audio_np.shape[0] == 2:  # (2, time) - stereo
                            audio_np = np.mean(audio_np, axis=0)  # Average both channels
                        elif audio_np.shape[1] == 2:  # (time, 2) - stereo
                            audio_np = np.mean(audio_np, axis=1)  # Average both channels
                        else:
                            # If one dimension is 1, squeeze it
                            if audio_np.shape[0] == 1:
                                audio_np = audio_np.squeeze(0)
                            elif audio_np.shape[1] == 1:
                                audio_np = audio_np.squeeze(1)
                            else:
                                # Default: take first channel if not clear stereo format
                                audio_np = audio_np[0, :]
                    
                    # Resample if needed
                    if input_sample_rate != 24000:
                        from utils.audio.librosa_fallback import safe_resample
                        audio_np = safe_resample(audio_np, orig_sr=input_sample_rate, target_sr=24000)
                    
                    # Normalize using dB FS (matches official VibeVoice)
                    target_dB_FS = -25
                    eps = 1e-6
                    
                    # First: normalize to target dB FS using RMS
                    rms = np.sqrt(np.mean(audio_np**2))
                    scalar = 10 ** (target_dB_FS / 20) / (rms + eps)
                    audio_np = audio_np * scalar
                    
                    # Then: avoid clipping
                    max_val = np.abs(audio_np).max()
                    if max_val > 1.0:
                        audio_np = audio_np / (max_val + eps)
                    
                    voice_samples.append(audio_np.astype(np.float32))
                else:
                    # Create synthetic voice sample
                    voice_samples.append(self._create_synthetic_voice_sample(i))
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
                       inference_steps: int = 20,
                       max_new_tokens: Optional[int] = None,
                       enable_cache: bool = True,
                       character: str = "narrator",
                       stable_audio_component: str = "",
                       multi_speaker_mode: str = "Custom Character Switching") -> Dict[str, Any]:
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
            inference_steps: Number of diffusion inference steps (5-100)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dict with "waveform" and "sample_rate"
        """
        # Check model reference and try to recover from strong refs
        if not hasattr(self, 'model') or self.model is None:
            # Try to restore from strong reference first
            if hasattr(self, '_model_ref') and self._model_ref is not None:
                print(f"🔄 Restoring model from strong reference")
                self.model = self._model_ref
            else:
                # Try to recover from current_model_name if available
                if hasattr(self, 'current_model_name') and self.current_model_name:
                    print(f"⚠️ Model reference lost, attempting recovery for '{self.current_model_name}'")
                    try:
                        self.initialize_engine(self.current_model_name, getattr(self, 'device', 'auto'))
                    except Exception as recovery_error:
                        raise RuntimeError(f"Model not initialized and recovery failed: {recovery_error}")
                else:
                    raise RuntimeError("Model not initialized. Call initialize_engine first.")
        
        if not hasattr(self, 'processor') or self.processor is None:
            # Try to restore from strong reference first
            if hasattr(self, '_processor_ref') and self._processor_ref is not None:
                print(f"🔄 Restoring processor from strong reference")
                self.processor = self._processor_ref
            else:
                raise RuntimeError("Processor not initialized. Call initialize_engine first.")
        
        # Handle caching if enabled (following ChatterBox pattern)
        if enable_cache:
            from utils.audio.cache import create_cache_function
            # print(f"🐛 VibeVoice ENGINE: Creating cache with audio_component='{stable_audio_component[:50]}...'")
            
            # Fix floating point precision issues by rounding to 3 decimal places
            cfg_scale_rounded = round(float(cfg_scale), 3) if isinstance(cfg_scale, (int, float)) else cfg_scale
            temperature_rounded = round(float(temperature), 3) if isinstance(temperature, (int, float)) else temperature
            top_p_rounded = round(float(top_p), 3) if isinstance(top_p, (int, float)) else top_p
            
            # print(f"🐛 VibeVoice ENGINE: Cache params - character='{character}', cfg_scale={cfg_scale_rounded}, use_sampling={use_sampling}, multi_speaker_mode='{multi_speaker_mode}', attention={getattr(self, '_attention_mode', 'auto')}, steps={inference_steps}, quant={getattr(self, '_quantize_llm_4bit', False)}")
            # print(f"🐛 VibeVoice ENGINE: Original vs rounded - cfg_scale: {cfg_scale} -> {cfg_scale_rounded}, temp: {temperature} -> {temperature_rounded}, top_p: {top_p} -> {top_p_rounded}")
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
                audio_component=stable_audio_component,
                multi_speaker_mode=multi_speaker_mode,
                # New parameters that should invalidate cache
                attention_mode=getattr(self, '_attention_mode', 'auto'),
                inference_steps=inference_steps,
                quantize_llm_4bit=getattr(self, '_quantize_llm_4bit', False)
            )
            
            # Try cache first
            cached_audio = cache_fn(text)
            if cached_audio is not None:
                print(f"💾 CACHE HIT for {character}: '{text[:30]}...'")
                # print(f"🐛 VibeVoice ENGINE: CACHE HIT - audio_component was '{stable_audio_component[:50]}...'")
                return {
                    "waveform": cached_audio,
                    "sample_rate": 24000
                }
        
        try:
            # Set seeds for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            # print(f"🐛 VibeVoice ENGINE: Starting generation with {len(voice_samples)} voice samples")
            # print(f"🐛 VibeVoice ENGINE: Generation params - cfg_scale={cfg_scale}, use_sampling={use_sampling}, seed={seed}")
            # print(f"🐛 VibeVoice ENGINE: Text length: {len(text)} chars")
            
            # Prepare inputs using processor
            # print(f"🐛 VibeVoice ENGINE: Processing inputs - text='{text[:100]}...', voice_samples count={len(voice_samples)}")
            inputs = self.processor(
                [text],  # Wrap text in list
                voice_samples=[voice_samples],  # Provide voice samples
                return_tensors="pt",
                return_attention_mask=True
            )
            # print(f"🐛 VibeVoice ENGINE: Processor inputs created - input_ids shape: {inputs['input_ids'].shape}")
            
            # Use stored device instead of trying to get from model (which is None)
            device = getattr(self, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Set diffusion inference steps (based on wildminder implementation)
            # Credits: drbaph's implementation for inference steps control
            self.model.set_ddpm_inference_steps(num_steps=inference_steps)
            print(f"🔄 VibeVoice: Using {inference_steps} diffusion inference steps")
            
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
                    # print(f"🐛 VibeVoice ENGINE: CACHING result for audio_component '{stable_audio_component[:50]}...'")
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
        """Clean up resources - DISABLED to prevent model clearing during generation"""
        print("⚠️ VibeVoice cleanup called but DISABLED to prevent model clearing")
        # DON'T clear model/processor during generation
        # if self.model is not None:
        #     del self.model
        #     self.model = None
        # 
        # if self.processor is not None:
        #     del self.processor
        #     self.processor = None
    
    def unload_models(self):
        """Unload models (called by ComfyUI's unload button)"""
        self.cleanup()
        self.current_model_name = None
        print("📤 VibeVoice models unloaded")