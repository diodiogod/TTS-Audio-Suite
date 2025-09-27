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
from packaging import version

# Apply accelerate compatibility patches
try:
    from utils.compatibility.transformers_patches import TransformersPatches
    TransformersPatches.patch_accelerate_compatibility(verbose=True)
except ImportError:
    print("⚠️ Could not import transformers patches")

import transformers

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

# Check transformers version for dtype parameter compatibility
_transformers_version = version.parse(transformers.__version__)
_DTYPE_ARG_SUPPORTED = _transformers_version >= version.parse("4.56.0")

# Try importing SageAttention support
try:
    from .sage_attention_patch import (
        SAGE_ATTENTION_AVAILABLE,
        set_sage_attention,
        restore_original_attention,
        SAGE_ATTENTION_FUNCTION
    )
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False
    set_sage_attention = None
    restore_original_attention = None
    SAGE_ATTENTION_FUNCTION = None

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
    
    # Class-level cache for shared model instance (prevents reloading on each run)
    _shared_model = None
    _shared_processor = None
    _shared_config = None
    _shared_model_name = None
    
    def __init__(self):
        """Initialize VibeVoice engine"""
        # Use class-level shared model if available
        self.model = self.__class__._shared_model
        self.processor = self.__class__._shared_processor
        self.current_model_name = self.__class__._shared_model_name
        self.model_path = None
        self.device = None
        self.attention_mode = None  # Track current attention mode for re-patching

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
        # Check if already loaded with same config using class-level cache
        current_config = (model_name, device, attention_mode, quantize_llm_4bit)
        if (self.__class__._shared_model is not None and
            self.__class__._shared_config == current_config):
            print(f"💾 VibeVoice model '{model_name}' already loaded with same config (reusing cached)")
            # Reuse the cached model
            self.model = self.__class__._shared_model
            self.processor = self.__class__._shared_processor

            # Re-apply SageAttention if model was moved between devices
            if (attention_mode == "sage" and
                SAGE_ATTENTION_AVAILABLE and set_sage_attention and
                hasattr(self.model, 'device') and self.model.device.type == "cuda"):
                print(f"🔄 Re-applying SageAttention after device movement...")
                try:
                    set_sage_attention(self.model)
                    print(f"✅ SageAttention re-patched successfully")
                except Exception as e:
                    print(f"⚠️ Failed to re-patch SageAttention: {e}")
            self.current_model_name = self.__class__._shared_model_name
            self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            self._original_device = device  # Store original device setting for auto detection
            return
        
        # Ensure package is installed
        if not self._ensure_package():
            raise RuntimeError("VibeVoice package not available. Please install it manually.")
        
        # Import VibeVoice modules
        try:
            # Import VibeVoice
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
        if attention_mode != "auto":
            print(f"   🧠 Using {attention_mode} attention")
        if quantize_llm_4bit:
            print(f"   🗜️ 4-bit LLM quantization enabled")
        
        try:
            # Import required modules  
            from transformers import BitsAndBytesConfig
            
            # Determine base dtype
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                model_dtype = torch.bfloat16
            else:
                model_dtype = torch.float16
            
            # Build quantization config if requested
            quant_config = None
            final_load_dtype = model_dtype
            
            if quantize_llm_4bit:
                # Default compute dtype for quantization
                bnb_compute_dtype = model_dtype
                
                # SageAttention requires fp32 compute dtype for stability with 4-bit
                if attention_mode == 'sage':
                    print(f"   🎯 Using SageAttention with 4-bit: forcing fp32 compute dtype for stability")
                    bnb_compute_dtype = torch.float32
                    final_load_dtype = torch.float32
                else:
                    print(f"   📊 Using {attention_mode} with 4-bit: using {model_dtype} compute dtype")
                
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=bnb_compute_dtype,
                )
                print(f"   🗜️ 4-bit quantization enabled")
            
            # Determine final attention mode
            final_attention_mode = attention_mode
            if attention_mode == "auto":
                # Auto-select best available attention
                # Check SageAttention first
                if SAGE_ATTENTION_AVAILABLE and SAGE_ATTENTION_FUNCTION is not None:
                    final_attention_mode = "sage"
                    print(f"   🚀 Auto-selected SageAttention (GPU-optimized mixed-precision)")
                else:
                    # Try flash attention
                    try:
                        import flash_attn
                        final_attention_mode = "flash_attention_2"
                        print(f"   ✨ Auto-selected flash_attention_2")
                    except ImportError:
                        final_attention_mode = "sdpa"  # PyTorch SDPA as fallback
                        print(f"   ⚡ Auto-selected sdpa (flash_attention_2 not available)")
            
            # For SageAttention, we need to load with SDPA and patch later
            attn_implementation_for_load = "sdpa" if final_attention_mode == "sage" else final_attention_mode
            
            # Build model kwargs with version-compatible dtype parameter
            model_kwargs = {
                "trust_remote_code": True,
            }
            
            # Use correct dtype parameter based on transformers version
            if _DTYPE_ARG_SUPPORTED:
                model_kwargs['dtype'] = final_load_dtype
            else:
                model_kwargs['torch_dtype'] = final_load_dtype
            
            # Set device_map based on quantization and device (original logic)
            if quant_config:
                # For quantization, use explicit device mapping to avoid buffer issues
                if device == "cuda" or device == "auto":
                    model_kwargs["device_map"] = {"": 0}  # Put everything on GPU 0
                else:
                    model_kwargs["device_map"] = {"": "cpu"}
                print(f"🔧 VibeVoice: Using device_map for quantization: {model_kwargs['device_map']}")
            else:
                model_kwargs["device_map"] = device if device != "auto" else None
                print(f"🔧 VibeVoice: Using device_map: {model_kwargs.get('device_map', 'None')}")
            
            # Add attention implementation (use SDPA for SageAttention, patch later)
            if attn_implementation_for_load != "auto":
                model_kwargs["attn_implementation"] = attn_implementation_for_load
                
            # Add quantization config if enabled
            if quant_config:
                model_kwargs["quantization_config"] = quant_config
            
            # Check if this is a standalone .safetensors file
            is_standalone = model_path.endswith('.safetensors')

            # Load model with enhanced configuration
            try:
                if is_standalone:
                    print(f"🔧 Loading standalone VibeVoice model from: {model_path}")
                    # Load state_dict directly from .safetensors file
                    state_dict = self._load_standalone_state_dict(model_path, device)

                    # Load config from sidecar file or use fallback
                    config_path = self._get_standalone_config_path(model_path, model_name)

                    # Load with state_dict
                    model_kwargs["state_dict"] = state_dict
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        config_path if config_path else None,
                        **model_kwargs
                    )
                else:
                    # Regular directory-based loading
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        model_path,
                        **model_kwargs
                    )
                
                # Apply SageAttention if selected, restore original if not
                if final_attention_mode == "sage":
                    if SAGE_ATTENTION_AVAILABLE and set_sage_attention:
                        print(f"   🎯 Applying SageAttention patch to model...")
                        try:
                            set_sage_attention(self.model)
                            print(f"   ✅ SageAttention successfully applied")
                        except Exception as sage_error:
                            print(f"   ⚠️ Failed to apply SageAttention: {sage_error}")
                            print(f"   📌 Falling back to SDPA")
                            final_attention_mode = "sdpa"
                    else:
                        print(f"   ⚠️ SageAttention not available, using SDPA")
                        final_attention_mode = "sdpa"
                else:
                    # Restore original attention if switching away from SageAttention
                    if restore_original_attention:
                        print(f"   🔄 Restoring original attention (cleaning SageAttention patches)")
                        restore_original_attention(self.model)

                # Store attention mode for re-patching after device movement
                self.attention_mode = final_attention_mode
                
                # Set model to evaluation mode and mark quantization status
                self.model.eval()
                if quant_config:
                    setattr(self.model, "_llm_4bit", True)
                    
            except Exception as e:
                # Fallback logic for attention modes
                if final_attention_mode in ["sage", "flash_attention_2"]:
                    print(f"⚠️ Failed with {final_attention_mode}, trying SDPA fallback: {e}")
                    # Retry with SDPA
                    model_kwargs["attn_implementation"] = "sdpa"
                    try:
                        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            model_path,
                            **model_kwargs
                        )
                        self.model.eval()
                        if quant_config:
                            setattr(self.model, "_llm_4bit", True)
                        final_attention_mode = "sdpa"
                    except Exception as sdpa_error:
                        print(f"⚠️ SDPA also failed, trying eager: {sdpa_error}")
                        model_kwargs["attn_implementation"] = "eager"
                        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            model_path,
                            **model_kwargs
                        )
                        self.model.eval()
                        if quant_config:
                            setattr(self.model, "_llm_4bit", True)
                        final_attention_mode = "eager"
                elif quant_config:
                    # If quantization fails, try without it
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
            
            # Load processor with unified tokenizer handling
            # For standalone models, we need to pass the directory for processor configs
            processor_path = os.path.dirname(model_path) if is_standalone else model_path
            self.processor = self._load_processor_with_unified_tokenizer(processor_path, model_name)
            
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
            self._original_device = device  # Store original device setting for auto detection
            self.current_model_name = model_name
            self._current_config = current_config
            self._quantize_llm_4bit = quantize_llm_4bit
            
            # Store in class-level cache for reuse
            self.__class__._shared_model = self.model
            self.__class__._shared_processor = self.processor
            self.__class__._shared_config = current_config
            self.__class__._shared_model_name = model_name
            
            print(f"✅ VibeVoice model '{model_name}' loaded successfully")
            print(f"   Device: {device}, Attention: {final_attention_mode}")
            if quantize_llm_4bit and quant_config:
                print(f"   🗜️ LLM quantized to 4-bit (VRAM savings expected)")
            elif quantize_llm_4bit and not quant_config:
                print(f"   ⚠️ Quantization failed, using full precision")
            
            
        except Exception as e:
            logger.error(f"Failed to load VibeVoice model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def _load_processor_with_unified_tokenizer(self, model_path: str, model_name: str):
        """
        Load VibeVoice processor following our ChatterBox-style unified pattern:
        1. Check local unified folder first
        2. Check HuggingFace cache
        3. If missing, download TO unified folder
        4. Load from wherever found (no copying)
        """
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
        from huggingface_hub import hf_hub_download

        # Determine which Qwen repo to use based on model name
        if "7b" in model_name.lower() or "large" in model_name.lower():
            qwen_repo = "Qwen/Qwen2.5-7B"
        else:
            qwen_repo = "Qwen/Qwen2.5-1.5B"

        # print(f"🔍 Looking for tokenizer from {qwen_repo} using unified pattern...")  # Verbose logging

        # Check our unified TTS folder first
        tokenizer_unified_path = os.path.join(self.downloader.downloader.tts_dir, "VibeVoice", "tokenizer", qwen_repo.replace("/", "_"))
        tokenizer_file = os.path.join(tokenizer_unified_path, "tokenizer.json")

        tokenizer_source = None

        if os.path.exists(tokenizer_file):
            print(f"📁 Using unified folder tokenizer: {tokenizer_file}")
            tokenizer_source = tokenizer_unified_path
        else:
            # Check HuggingFace cache
            try:
                cached_file = hf_hub_download(repo_id=qwen_repo, filename="tokenizer.json", local_files_only=True)
                print(f"📁 Using cached tokenizer: {cached_file}")
                tokenizer_source = os.path.dirname(cached_file)
            except Exception as cache_error:
                print(f"📋 Cache check failed: {str(cache_error)[:100]}... - will download")

                # Download to unified folder
                print(f"📥 Downloading tokenizer to unified folder...")
                os.makedirs(tokenizer_unified_path, exist_ok=True)

                tokenizer_files = [
                    {"remote": "tokenizer.json", "local": "tokenizer.json"},
                    {"remote": "tokenizer_config.json", "local": "tokenizer_config.json"},
                ]

                download_path = self.downloader.download_huggingface_model(
                    repo_id=qwen_repo,
                    model_name=qwen_repo.replace("/", "_"),
                    files=tokenizer_files,
                    engine_type="VibeVoice",
                    subfolder="tokenizer"
                )

                if download_path and os.path.exists(os.path.join(download_path, "tokenizer.json")):
                    print(f"✅ Downloaded tokenizer to: {download_path}")
                    tokenizer_source = download_path
                else:
                    raise RuntimeError(f"Failed to download tokenizer from {qwen_repo}")

        # Check if we're using our unified TTS folder (not HF cache)
        is_unified_tokenizer = tokenizer_source and tokenizer_unified_path in tokenizer_source

        if is_unified_tokenizer:
            try:
                # We have tokenizer in unified folder - build processor manually to use it
                # print(f"🔧 Building processor with unified tokenizer from: {tokenizer_source}")  # Verbose logging

                from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
                from vibevoice.processor.vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor
                import json

                # Load tokenizer directly with tokenizer_file to avoid class mismatch
                tokenizer_file_path = os.path.join(tokenizer_source, "tokenizer.json")
                if os.path.exists(tokenizer_file_path):
                    tokenizer = VibeVoiceTextTokenizerFast(tokenizer_file=tokenizer_file_path)
                    # print(f"🔧 Loaded tokenizer directly from file: {tokenizer_file_path}")  # Verbose logging
                else:
                    # Fallback to from_pretrained (may show warning)
                    tokenizer = VibeVoiceTextTokenizerFast.from_pretrained(tokenizer_source)

                # Load processor config if available
                processor_config_path = os.path.join(model_path, "preprocessor_config.json")
                processor_config_data = {}
                if os.path.exists(processor_config_path):
                    with open(processor_config_path, 'r', encoding='utf-8') as f:
                        processor_config_data = json.load(f)

                # Create audio processor
                audio_processor = VibeVoiceTokenizerProcessor()

                # Build final processor
                processor = VibeVoiceProcessor(
                    tokenizer=tokenizer,
                    audio_processor=audio_processor,
                    speech_tok_compress_ratio=processor_config_data.get("speech_tok_compress_ratio", 3200),
                    db_normalize=processor_config_data.get("db_normalize", True)
                )

                # print(f"✅ Built VibeVoice processor with unified tokenizer")  # Verbose logging
                return processor

            except Exception as e:
                print(f"⚠️ Unified tokenizer processor build failed: {e}")
                print(f"🔄 Falling back to standard processor loading...")

        # For HF cached tokenizer, build processor manually to avoid class mismatch warnings
        else:
            try:
                # print(f"🔧 Building processor with HF cached tokenizer from: {tokenizer_source}")  # Verbose logging

                from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
                from vibevoice.processor.vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor
                import json

                # Load tokenizer directly with tokenizer_file to avoid class mismatch
                tokenizer_file_path = os.path.join(tokenizer_source, "tokenizer.json")
                if os.path.exists(tokenizer_file_path):
                    tokenizer = VibeVoiceTextTokenizerFast(tokenizer_file=tokenizer_file_path)
                    # print(f"🔧 Loaded tokenizer directly from file: {tokenizer_file_path}")  # Verbose logging
                else:
                    # Fallback to from_pretrained (may show warning)
                    tokenizer = VibeVoiceTextTokenizerFast.from_pretrained(tokenizer_source)

                # Load processor config if available
                processor_config_path = os.path.join(model_path, "preprocessor_config.json")
                processor_config_data = {}
                if os.path.exists(processor_config_path):
                    with open(processor_config_path, 'r', encoding='utf-8') as f:
                        processor_config_data = json.load(f)

                # Create audio processor
                audio_processor = VibeVoiceTokenizerProcessor()

                # Build final processor
                processor = VibeVoiceProcessor(
                    tokenizer=tokenizer,
                    audio_processor=audio_processor,
                    speech_tok_compress_ratio=processor_config_data.get("speech_tok_compress_ratio", 3200),
                    db_normalize=processor_config_data.get("db_normalize", True)
                )

                # print(f"✅ Built VibeVoice processor with HF cached tokenizer (no warnings)")  # Verbose logging
                return processor

            except Exception as e:
                print(f"⚠️ HF cached tokenizer processor build failed: {e}")
                print(f"🔄 Falling back to standard processor loading...")

        # Final fallback to standard loading (may show warnings)
        try:
            processor = VibeVoiceProcessor.from_pretrained(model_path)
            print(f"✅ Loaded VibeVoice processor with standard method")
            return processor
        except Exception as e:
            # Ultimate fallback with explicit repo specification
            print(f"⚠️ Standard processor loading failed: {e}")
            processor = VibeVoiceProcessor.from_pretrained(
                model_path,
                language_model_pretrained_name=qwen_repo
            )
            print(f"✅ Loaded VibeVoice processor with explicit tokenizer repo: {qwen_repo}")
            return processor

    def _load_standalone_state_dict(self, safetensors_path: str, device: str):
        """Load state_dict from standalone .safetensors file using ComfyUI's loader"""
        try:
            # Use ComfyUI's safetensors loader with device mapping
            import comfy.utils
            state_dict = comfy.utils.load_torch_file(safetensors_path, device=device)
            print(f"✅ Loaded state_dict from: {safetensors_path}")
            return state_dict
        except Exception as e:
            print(f"❌ Failed to load state_dict from {safetensors_path}: {e}")
            raise

    def _get_standalone_config_path(self, safetensors_path: str, model_name: str) -> Optional[str]:
        """Get config path for standalone model, with fallbacks"""
        # Try sidecar config file first
        base_path = os.path.splitext(safetensors_path)[0]
        sidecar_config = base_path + ".config.json"

        if os.path.exists(sidecar_config):
            print(f"📄 Using sidecar config: {sidecar_config}")
            return sidecar_config

        # Try looking for config.json in same directory
        dir_config = os.path.join(os.path.dirname(safetensors_path), "config.json")
        if os.path.exists(dir_config):
            print(f"📄 Using directory config: {dir_config}")
            return dir_config

        # Fallback to default configs based on model name
        config_name = "default_VibeVoice-Large_config.json" if "large" in model_name.lower() or "7b" in model_name.lower() else "default_VibeVoice-1.5B_config.json"
        fallback_path = os.path.join(os.path.dirname(__file__), "..", "..", "IgnoredForGitHubDocs", "For_reference", "VibeVoice-ComfyUI-wildminder", "vibevoice", "configs", config_name)

        if os.path.exists(fallback_path):
            print(f"📄 Using fallback config: {fallback_path}")
            return fallback_path

        print(f"⚠️ No config found for standalone model, using None (transformers will use defaults)")
        return None

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
    
    def _prepare_voice_samples(self, voice_refs: List[Optional[Dict]]) -> List[Optional[np.ndarray]]:
        """
        Prepare voice samples from ComfyUI audio inputs or None for zero-shot generation.

        Args:
            voice_refs: List of voice reference dicts from ComfyUI (can contain None)

        Returns:
            List of numpy arrays or None values (None enables true zero-shot generation)
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
                    
                    # Check for invalid values and clean them up
                    if np.any(np.isnan(audio_np)) or np.any(np.isinf(audio_np)):
                        print(f"⚠️ VibeVoice ENGINE: Audio contains NaN or Inf values, replacing with zeros")
                        audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Ensure audio is not completely silent or has extreme values
                    if np.all(audio_np == 0):
                        print(f"⚠️ VibeVoice ENGINE: Audio waveform is completely silent")
                    
                    # Normalize extreme values (prevents generation issues)
                    max_val = np.abs(audio_np).max()
                    if max_val > 10.0:
                        print(f"⚠️ VibeVoice ENGINE: Audio values are very large (max: {max_val}), normalizing")
                        audio_np = audio_np / max_val
                    
                    # Resample if needed
                    if input_sample_rate != 24000:
                        from utils.audio.librosa_fallback import safe_resample
                        audio_np = safe_resample(audio_np, orig_sr=input_sample_rate, target_sr=24000)
                    
                    # Final check after resampling (can introduce artifacts)
                    if np.any(np.isnan(audio_np)) or np.any(np.isinf(audio_np)):
                        print(f"⚠️ VibeVoice ENGINE: Audio contains NaN or Inf after resampling, replacing with zeros")
                        audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)
                    
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
                    # Use None for true zero-shot generation (processor will skip voice prompt)
                    voice_samples.append(None)
            else:
                # Use None for true zero-shot generation (processor will skip voice prompt)
                voice_samples.append(None)
        
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
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not initialized. Call initialize_engine first.")
        
        # Handle caching if enabled (following ChatterBox pattern)
        if enable_cache:
            from utils.audio.cache import create_cache_function
            # print(f"🐛 VibeVoice ENGINE: Creating cache with audio_component='{stable_audio_component[:50]}...'")
            
            # Fix floating point precision issues by rounding to 3 decimal places
            cfg_scale_rounded = round(float(cfg_scale), 3) if isinstance(cfg_scale, (int, float)) else cfg_scale
            temperature_rounded = round(float(temperature), 3) if isinstance(temperature, (int, float)) else temperature
            top_p_rounded = round(float(top_p), 3) if isinstance(top_p, (int, float)) else top_p
            
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
                attention_mode=getattr(self, 'attention_mode', 'auto'),
                inference_steps=inference_steps,
                quantize_llm_4bit=getattr(self, '_quantize_llm_4bit', False)
            )
            
            # Try cache first
            cached_audio = cache_fn(text)
            if cached_audio is not None:
                print(f"💾 CACHE HIT for {character}: '{text[:30]}...'")
                # print(f"🐛 VibeVoice ENGINE: CACHE HIT - audio_component was '{stable_audio_component[:50]}...'")
                # Ensure cached audio is also in Float32 for compatibility
                if hasattr(cached_audio, 'dtype') and cached_audio.dtype == torch.bfloat16:
                    cached_audio = cached_audio.to(torch.float32)
                return {
                    "waveform": cached_audio,
                    "sample_rate": 24000
                }
        
        # Ensure model is on correct device for generation (regardless of cache hit/miss)
        # This fixes the issue where "auto" device detection doesn't move model back to GPU after cache hits
        if (hasattr(self, '_original_device') and self._original_device == "auto" and
            torch.cuda.is_available() and self.model is not None):
            actual_device = next(self.model.parameters()).device
            if actual_device.type != 'cuda':
                print(f"🔄 VibeVoice: Auto mode - moving model from {actual_device} to CUDA for generation")
                self.model.to('cuda')

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
            
            # Validate inputs before moving to GPU (prevents corrupted generation)
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if torch.any(torch.isnan(value)) or torch.any(torch.isinf(value)):
                        logger.error(f"Input tensor '{key}' contains NaN or Inf values")
                        raise ValueError(f"Invalid values in input tensor: {key}")
            
            # Move inputs to device (model device already handled above)
            actual_device = next(self.model.parameters()).device
            inputs = {k: v.to(actual_device) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}
            
            # Debug inputs
            # Debug: Print input information (commented out for production)
            # print(f"🐛 Inputs keys: {list(inputs.keys())}")
            # for k, v in inputs.items():
            #     if isinstance(v, torch.Tensor):
            #         print(f"🐛 Input {k}: shape={v.shape}, dtype={v.dtype}")
            #     else:
            #         print(f"🐛 Input {k}: type={type(v)}, value={v}")
            
            # Set diffusion inference steps (based on wildminder implementation)
            # Credits: drbaph's implementation for inference steps control
            self.model.set_ddpm_inference_steps(num_steps=inference_steps)
            print(f"🔄 VibeVoice: Using {inference_steps} diffusion inference steps")
            
            # Ensure model has proper generation config with bos_token_id (silent configuration)
            if hasattr(self.model, 'generation_config'):
                if not hasattr(self.model.generation_config, 'bos_token_id') or self.model.generation_config.bos_token_id is None:
                    self.model.generation_config.bos_token_id = 151643  # Qwen2 BOS token
            elif hasattr(self.model, 'config'):
                if not hasattr(self.model.config, 'bos_token_id') or self.model.config.bos_token_id is None:
                    self.model.config.bos_token_id = 151643  # Qwen2 BOS token
            
            # Generate with appropriate mode
            with torch.no_grad():
                # Debug tokenizer bos_token_id (commented out for production)
                # if hasattr(self.processor.tokenizer, 'bos_token_id'):
                #     print(f"🐛 Tokenizer bos_token_id: {self.processor.tokenizer.bos_token_id}")
                # else:
                #     print(f"🐛 Tokenizer has no bos_token_id attribute")
                
                # Ensure proper token IDs are set (critical for stopping generation)
                generation_kwargs = {
                    "tokenizer": self.processor.tokenizer,
                    "cfg_scale": cfg_scale,
                    "max_new_tokens": max_new_tokens,
                    "bos_token_id": 151643,  # Qwen2 BOS token
                    "eos_token_id": self.processor.tokenizer.eos_token_id,  # Critical: EOS for stopping
                    "pad_token_id": self.processor.tokenizer.eos_token_id,  # Use EOS as pad
                }
                
                if use_sampling:
                    # Sampling mode
                    generation_kwargs.update({
                        "do_sample": True,
                        "temperature": temperature,
                        "top_p": top_p,
                    })
                    output = self.model.generate(**inputs, **generation_kwargs)
                else:
                    # Deterministic mode
                    generation_kwargs["do_sample"] = False
                    output = self.model.generate(**inputs, **generation_kwargs)
            
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
                
                # Ensure waveform is in Float32 for compatibility (VibeVoice may output BFloat16)
                audio_output = audio_tensor.cpu()
                if audio_output.dtype == torch.bfloat16:
                    audio_output = audio_output.to(torch.float32)

                result = {
                    "waveform": audio_output,
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