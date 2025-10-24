"""
Base F5-TTS Node - Common functionality for all F5-TTS nodes
Extends ChatterBox foundation for F5-TTS specific requirements
"""

import torch
import numpy as np
import tempfile
import os
import hashlib
from typing import Dict, Any, Optional, Tuple, List

# Use direct file imports that work when loaded via importlib
import os
import sys
import importlib.util

# Add project root directory to path for imports
# When loaded via importlib, __file__ might not resolve correctly
# So we'll check if utils is already accessible, and if not, try to find the project root
current_dir = os.path.dirname(os.path.abspath(__file__))
try:
    import utils.text.chunking
    utils_available = True
except ImportError:
    # Find project root by going up from this file location
    nodes_dir = os.path.dirname(current_dir)  # nodes/
    project_root = os.path.dirname(nodes_dir)  # project root
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    utils_available = True

# Load base_node module directly
base_node_path = os.path.join(current_dir, "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

# Import the base class
BaseChatterBoxNode = base_module.BaseChatterBoxNode

from utils.text.chunking import ImprovedChatterBoxChunker
from utils.audio.processing import AudioProcessingUtils
from utils.text.pause_processor import PauseTagProcessor
import comfy.model_management as model_management

# F5-TTS specific constants
F5TTS_SAMPLE_RATE = 24000
F5TTS_ERROR_MESSAGES = {
    "missing_ref_text": "F5-TTS requires both reference audio AND reference text. Please provide ref_text parameter.",
    "model_not_found": "F5-TTS model '{}' not found. Available models: {}",
    "audio_format_error": "Audio format not compatible with F5-TTS. Expected 24kHz, got {}kHz.",
    "memory_error": "Insufficient memory for F5-TTS generation. Try reducing chunk size or switching to CPU.",
    "import_error": "F5-TTS not available. Please install F5-TTS dependencies."
}


class BaseF5TTSNode(BaseChatterBoxNode):
    """
    Base class for F5-TTS nodes extending ChatterBox foundation.
    Handles F5-TTS specific requirements while maintaining compatibility.
    """
    
    # Node metadata
    CATEGORY = "F5-TTS Voice"
    
    def __init__(self):
        super().__init__()
        self.f5tts_model = None
        self.f5tts_sample_rate = F5TTS_SAMPLE_RATE
        self.chunker = ImprovedChatterBoxChunker()
        
        # F5-TTS availability check
        self.f5tts_available = False
        self._check_f5tts_availability()
    
    def _check_f5tts_availability(self):
        """Check if F5-TTS is available"""
        try:
            from engines.f5tts import F5TTS_AVAILABLE
            self.f5tts_available = F5TTS_AVAILABLE
        except ImportError:
            self.f5tts_available = False
    
    def load_f5tts_model(self, model_name: str = "F5TTS_Base", device: str = "auto", force_reload: bool = False):
        """
        Load F5-TTS model using universal smart model loader
        """
        if not self.f5tts_available:
            raise ImportError(F5TTS_ERROR_MESSAGES["import_error"])
        
        device = self.resolve_device(device)
        self.device = device
        
        # Normalize model name for caching consistency
        normalized_model_name = model_name.replace("local:", "") if model_name.startswith("local:") else model_name
        
        # Use unified model interface for ComfyUI integration
        from utils.models.unified_model_interface import load_tts_model
        
        try:
            # Load F5-TTS model through unified interface for ComfyUI memory management  
            wrapped_model = load_tts_model(
                engine_name="f5tts",
                model_name=normalized_model_name,
                device=device,
                language="default",  # Use default for F5-TTS since model name distinguishes models
                force_reload=force_reload
            )
            
            self.f5tts_model = wrapped_model
            was_cached = not force_reload  # Assume cached unless forced
            
        except Exception as e:
            print(f"⚠️ Failed to load F5-TTS via unified interface: {e}")
            print("🔄 Falling back to direct Smart Loader...")
            
            # Fallback to direct Smart Loader if unified interface fails
            from utils.models.smart_loader import smart_model_loader
            
            def f5tts_load_callback(device: str, model: str) -> Any:
                """Callback for F5-TTS model loading"""
                from engines.f5tts import ChatterBoxF5TTS
                
                # Try to find local models first
                model_paths = self._find_f5tts_models(model)
                
                model_loaded = False
                last_error = None
                
                for source, path in model_paths:
                    try:
                        if source == "comfyui" and path:
                            # Load from local ComfyUI models directory
                            normalized_name = model.replace("local:", "") if model.startswith("local:") else model
                            return ChatterBoxF5TTS.from_local(path, device, normalized_name)
                        else:
                            # Load from HuggingFace - but check if we have local files first
                            local_path = None
                            if model in ["F5TTS_Base", "F5TTS_v1_Base", "E2TTS_Base"]:
                                import folder_paths
                                # Try TTS path first, then legacy
                                search_paths = [
                                    os.path.join(folder_paths.models_dir, "TTS", "F5-TTS", model),
                                    os.path.join(folder_paths.models_dir, "F5-TTS", model),  # Legacy
                                    os.path.join(folder_paths.models_dir, "Checkpoints", "F5-TTS", model)  # Legacy
                                ]
                                
                                potential_path = None
                                for path in search_paths:
                                    if os.path.exists(path):
                                        potential_path = path
                                        break
                                if potential_path:
                                    local_path = potential_path
                            
                            if local_path:
                                # Use local files even for non-local model names for consistency
                                return ChatterBoxF5TTS.from_local(local_path, device, model)
                            else:
                                # True HuggingFace download
                                return ChatterBoxF5TTS.from_pretrained(device, model)
                            
                    except Exception as e:
                        print(f"⚠️ Failed to load F5-TTS model from {source}: {str(e)}")
                        last_error = e
                        continue
                
                # If we get here, all sources failed
                error_msg = f"Failed to load F5-TTS model '{model}' from any source"
                if last_error:
                    error_msg += f". Last error: {last_error}"
                raise RuntimeError(error_msg)
            
            self.f5tts_model, was_cached = smart_model_loader.load_model_if_needed(
                engine_type="f5tts",
                model_name=normalized_model_name,
                current_model=getattr(self, 'f5tts_model', None),
                device=device,
                load_callback=f5tts_load_callback,
                force_reload=force_reload
            )
            
            # Store normalized model name for cache validation
            self.current_model_name = normalized_model_name
            return self.f5tts_model
            
        except ImportError:
            raise ImportError(F5TTS_ERROR_MESSAGES["import_error"])
        except Exception as e:
            raise RuntimeError(f"Failed to load F5-TTS model: {e}")
    
    def _find_f5tts_models(self, model_name: str = None) -> List[Tuple[str, Optional[str]]]:
        """Find F5-TTS models following existing pattern"""
        try:
            from utils.models.f5tts_manager import f5tts_model_manager
            return f5tts_model_manager.find_f5tts_models(model_name)
        except ImportError:
            return [("huggingface", None)]
    
    def generate_f5tts_audio(self, text: str, ref_audio_path: str, ref_text: str, 
                            temperature: float = 0.8, speed: float = 1.0,
                            target_rms: float = 0.1, cross_fade_duration: float = 0.15,
                            nfe_step: int = 32, cfg_strength: float = 2.0, 
                            auto_phonemization: bool = True) -> torch.Tensor:
        """Generate audio using F5-TTS with reference audio+text"""
        if self.f5tts_model is None:
            raise RuntimeError("F5-TTS model not loaded. Call load_f5tts_model() first.")
        
        # Pass auto_phonemization directly through function parameters
        
        return self.f5tts_model.generate(
            text=text,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            temperature=temperature,
            speed=speed,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            auto_phonemization=auto_phonemization
        )

    def generate_f5tts_with_pause_tags(self, text: str, ref_audio_path: str, ref_text: str,
                                     enable_pause_tags: bool = True, character: str = "narrator", 
                                     seed: int = 0, enable_cache: bool = False, cache_fn=None, auto_phonemization: bool = True, **generation_kwargs) -> torch.Tensor:
        """
        Generate F5-TTS audio with pause tag support.
        
        Args:
            text: Input text potentially with pause tags
            ref_audio_path: Reference audio file path
            ref_text: Reference text
            enable_pause_tags: Whether to process pause tags
            character: Character name for cache key
            seed: Seed for reproducibility and cache key
            enable_cache: Whether to use caching
            cache_fn: Function to handle caching (cache_fn(text_content) -> cached_audio or None)
            **generation_kwargs: Additional F5-TTS generation parameters
            
        Returns:
            Generated audio tensor with pauses
        """
        # Preprocess text for pause tags
        processed_text, pause_segments = PauseTagProcessor.preprocess_text_with_pause_tags(
            text, enable_pause_tags
        )
        
        if pause_segments is None:
            # No pause tags, use regular generation with optional caching
            if enable_cache and cache_fn:
                cached_audio = cache_fn(processed_text)
                if cached_audio is not None:
                    return cached_audio
            
            audio = self.generate_f5tts_audio(
                processed_text, ref_audio_path, ref_text, auto_phonemization=auto_phonemization, **generation_kwargs
            )
            
            if enable_cache and cache_fn:
                cache_fn(processed_text, audio)  # Cache the result
            
            return audio
        
        # Generate audio with pause tags, with optional caching per text segment
        def f5tts_generate_func(text_content: str) -> torch.Tensor:
            """F5-TTS generation function for pause tag processor with optional caching"""
            if enable_cache and cache_fn:
                cached_audio = cache_fn(text_content)
                if cached_audio is not None:
                    return cached_audio
            
            audio = self.generate_f5tts_audio(
                text_content, ref_audio_path, ref_text, auto_phonemization=auto_phonemization, **generation_kwargs
            )
            
            if enable_cache and cache_fn:
                cache_fn(text_content, audio)  # Cache the result
            
            return audio
        
        return PauseTagProcessor.generate_audio_with_pauses(
            pause_segments, f5tts_generate_func, F5TTS_SAMPLE_RATE
        )
    
    def handle_f5tts_reference(self, reference_audio: Optional[Dict[str, Any]], 
                              audio_prompt_path: str, ref_text: str) -> Tuple[Optional[str], str]:
        """
        Handle F5-TTS reference audio and text requirements
        Returns (audio_path, validated_ref_text)
        """
        # Validate reference text is provided
        if not ref_text or not ref_text.strip():
            raise ValueError(F5TTS_ERROR_MESSAGES["missing_ref_text"])
        
        # Handle reference audio (same as base class but with F5-TTS validation)
        audio_prompt = self.handle_reference_audio(reference_audio, audio_prompt_path)
        
        if audio_prompt is None:
            raise ValueError("F5-TTS requires reference audio. Please provide either reference_audio input or audio_prompt_path.")
        
        return audio_prompt, ref_text.strip()
    
    def validate_f5tts_inputs(self, **inputs) -> Dict[str, Any]:
        """Validate F5-TTS specific inputs including reference text"""
        # Call the base class validation directly (from BaseChatterBoxNode)
        validated = super().validate_inputs(**inputs)
        
        # Ensure ref_text is provided and not empty
        ref_text = validated.get("ref_text", "")
        if not ref_text or not ref_text.strip():
            raise ValueError(F5TTS_ERROR_MESSAGES["missing_ref_text"])
        
        validated["ref_text"] = ref_text.strip()
        
        # Validate model name if provided
        model_name = validated.get("model", "F5TTS_Base")
        available_models = self._get_available_models()
        if model_name not in available_models:
            raise ValueError(F5TTS_ERROR_MESSAGES["model_not_found"].format(model_name, available_models))
        
        return validated
    
    def _get_available_models(self) -> List[str]:
        """Get list of available F5-TTS models"""
        try:
            from engines.f5tts import get_f5tts_models
            return get_f5tts_models()
        except ImportError:
            return ["F5TTS_Base", "F5TTS_v1_Base", "E2TTS_Base"]
    
    @classmethod
    def get_available_models_for_dropdown(cls) -> List[str]:
        """Get list of available F5-TTS models without importing heavy modules.
        Reconstructs the discovery logic using file reading only."""
        # Official models (static list)
        F5TTS_MODELS = {
            "F5TTS_Base": {"repo": "SWivid/F5-TTS"},
            "F5TTS_v1_Base": {"repo": "SWivid/F5-TTS"},
            "E2TTS_Base": {"repo": "SWivid/E2-TTS"},
            "F5-DE": {"repo": "aihpi/F5-TTS-German"},
            "F5-ES": {"repo": "jpgallegoar/F5-Spanish"},
            "F5-FR": {"repo": "RASPIAUDIO/F5-French-MixedSpeakers-reduced"},
            "F5-JP": {"repo": "Jmica/F5TTS"},
            "F5-IT": {"repo": "alien79/F5-TTS-italian"},
            "F5-TH": {"repo": "VIZINTZOR/F5-TTS-THAI"},
            "F5-PT-BR": {"repo": "firstpixel/F5-TTS-pt-br"},
            "F5-Polish": {"repo": "Gregniuki/F5-tts_English_German_Polish"},
            "F5-Hindi-Small": {"repo": "SPRINGLab/F5-Hindi-24KHz"},
        }

        models = list(F5TTS_MODELS.keys())

        try:
            import folder_paths

            # Search paths (same as original)
            search_paths = [
                os.path.join(folder_paths.models_dir, "TTS", "F5-TTS"),
                os.path.join(folder_paths.models_dir, "F5-TTS"),  # Legacy
                os.path.join(folder_paths.models_dir, "Checkpoints", "F5-TTS")  # Legacy fallback
            ]

            for models_dir in search_paths:
                try:
                    if os.path.exists(models_dir):
                        for item in os.listdir(models_dir):
                            item_path = os.path.join(models_dir, item)
                            if os.path.isdir(item_path):
                                # Check if it contains model files
                                for ext in [".safetensors", ".pt"]:
                                    if any(f.endswith(ext) for f in os.listdir(item_path)):
                                        local_model = f"local:{item}"
                                        if local_model not in models:
                                            models.append(local_model)
                                        break
                except Exception:
                    pass
        except Exception:
            pass

        return models if models else ["F5TTS_Base", "F5TTS_v1_Base", "E2TTS_Base"]
    
    def format_f5tts_audio_output(self, audio_tensor: torch.Tensor) -> Dict[str, Any]:
        """Format F5-TTS audio tensor for ComfyUI output"""
        return self.format_audio_output(audio_tensor, self.f5tts_sample_rate)
    
    def combine_f5tts_audio_chunks(self, audio_segments: List[torch.Tensor], method: str, 
                                  silence_ms: int, text_length: int, 
                                  original_text: str = "", text_chunks: List[str] = None,
                                  return_info: bool = False) -> torch.Tensor:
        """Combine F5-TTS audio segments using modular combination utility"""
        if len(audio_segments) == 1:
            if return_info:
                # Create basic info for single chunk
                chunk_info = {
                    "method_used": "none",
                    "total_chunks": 1,
                    "chunk_timings": [{"start": 0.0, "end": audio_segments[0].size(-1) / self.f5tts_sample_rate, 
                                     "text": text_chunks[0] if text_chunks else ""}],
                    "auto_selected": False
                }
                return audio_segments[0], chunk_info
            return audio_segments[0]
        
        print(f"🔗 Combining {len(audio_segments)} F5-TTS chunks using '{method}' method")
        
        # Use modular chunk combiner
        from utils.audio.chunk_combiner import ChunkCombiner
        result = ChunkCombiner.combine_chunks(
            audio_segments=audio_segments,
            method=method,
            silence_ms=silence_ms,
            crossfade_duration=0.1,
            sample_rate=self.f5tts_sample_rate,
            text_length=text_length,
            original_text=original_text,
            text_chunks=text_chunks,
            return_info=return_info
        )
        
        if return_info:
            return result  # (combined_audio, chunk_info)
        else:
            return result  # combined_audio
    
    def get_f5tts_model_info(self) -> Dict[str, Any]:
        """Get information about loaded F5-TTS model"""
        if self.f5tts_model:
            return {
                "model_name": getattr(self.f5tts_model, 'model_name', 'unknown'),
                "sample_rate": self.f5tts_sample_rate,
                "device": self.device,
                "available": self.f5tts_available,
            }
        else:
            return {
                "model_name": None,
                "sample_rate": self.f5tts_sample_rate,
                "device": self.device,
                "available": self.f5tts_available,
            }
    
    @classmethod
    def get_f5tts_input_types_base(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get base input types for F5-TTS nodes.
        Extends base input types with F5-TTS specific requirements.
        """
        base_types = cls.get_input_types_base()
        
        # Add F5-TTS specific required inputs
        base_types["required"].update({
            "ref_text": ("STRING", {
                "multiline": True, 
                "default": "This is the reference text that matches the reference audio.",
                "tooltip": "Text that corresponds to the reference audio. Required for F5-TTS voice cloning."
            }),
            "model": (cls.get_available_models_for_dropdown(), {"default": "F5TTS_v1_Base"}),
        })
        
        # Add F5-TTS specific optional inputs
        base_types["optional"].update({
            "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1}),
            "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            "target_rms": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
            "cross_fade_duration": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
            "nfe_step": ("INT", {"default": 32, "min": 1, "max": 100}),
            "cfg_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
        })
        
        return base_types