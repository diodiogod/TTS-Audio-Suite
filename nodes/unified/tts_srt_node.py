"""
Unified TTS SRT Node - Engine-agnostic SRT subtitle-aware text-to-speech for TTS Audio Suite
Replaces both ChatterBox SRT and F5-TTS SRT nodes with unified architecture
"""

import torch
import numpy as np
import tempfile
import os
import hashlib
import gc
from typing import Dict, Any, Optional, List, Tuple

# Use direct file imports that work when loaded via importlib
import os
import sys
import importlib.util

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load base_node module directly
base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

# Import the base class
BaseTTSNode = base_module.BaseTTSNode

from utils.voice.discovery import get_available_voices
from utils.audio.processing import AudioProcessingUtils
from utils.config_sanitizer import ConfigSanitizer
import comfy.model_management as model_management

# AnyType for flexible input types (accepts any data type)
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")


class UnifiedTTSSRTNode(BaseTTSNode):
    """
    Unified TTS SRT Node - Engine-agnostic SRT subtitle-aware text-to-speech generation.
    Works with any TTS engine (ChatterBox, F5-TTS, future RVC, etc.) through engine delegation.
    Replaces both ChatterBox SRT and F5-TTS SRT nodes while preserving ALL functionality.
    """
    
    @classmethod
    def NAME(cls):
        return "📺 TTS SRT"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available reference audio files from voice folders
        reference_files = get_available_voices()
        
        return {
            "required": {
                "TTS_engine": ("TTS_ENGINE", {
                    "tooltip": "TTS engine configuration from ChatterBox Engine or F5 TTS Engine nodes"
                }),
                "srt_content": ("STRING", {
                    "multiline": True,
                    "default": """1
00:00:01,000 --> 00:00:04,000
Hello! This is unified SRT TTS with character switching.

2
00:00:04,500 --> 00:00:09,500
[Alice] Hi there! I'm Alice speaking with precise timing.

3
00:00:10,000 --> 00:00:14,000
[Bob] And I'm Bob! The audio matches these exact SRT timings.""",
                    "tooltip": "The SRT subtitle content. Each entry defines a text segment and its precise start and end times. Supports character switching with [Character] tags."
                }),
                "narrator_voice": (reference_files, {
                    "default": "none",
                    "tooltip": "Fallback narrator voice from voice folders. Used when opt_narrator is not connected. Select 'none' if you only use opt_narrator input."
                }),
                "seed": ("INT", {
                    "default": 1, "min": 0, "max": 2**32 - 1,
                    "tooltip": "Seed for reproducible TTS generation. Same seed with same inputs will produce identical results. Set to 0 for random generation."
                }),
                "timing_mode": (["stretch_to_fit", "pad_with_silence", "smart_natural", "concatenate"], {
                    "default": "smart_natural",
                    "tooltip": "Determines how audio segments are aligned with SRT timings:\n🔹 stretch_to_fit: Stretches/compresses audio to exactly match SRT segment durations.\n🔹 pad_with_silence: Places natural audio at SRT start times, padding gaps with silence. May result in overlaps.\n🔹 smart_natural: Intelligently adjusts timings within 'timing_tolerance', prioritizing natural audio and shifting subsequent segments. Applies stretch/shrink within limits if needed.\n🔹 concatenate: Ignores original SRT timings, concatenates audio naturally and generates new SRT with actual timings."
                }),
            },
            "optional": {
                "opt_narrator": (any_typ, {
                    "tooltip": "Voice reference: Connect Character Voices node output OR direct audio input. Takes priority over narrator_voice dropdown when connected."
                }),
                "enable_audio_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, generated audio segments will be cached in memory to speed up subsequent runs with identical parameters."
                }),
                "fade_for_StretchToFit": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Duration (in seconds) for crossfading between audio segments in 'stretch_to_fit' mode."
                }),
                "max_stretch_ratio": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Maximum factor to slow down audio in 'smart_natural' mode. (e.g., 2.0x means audio can be twice as long). Recommend leaving at 1.0 for natural speech preservation and silence addition."
                }),
                "min_stretch_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Minimum factor to speed up audio in 'smart_natural' mode. (e.g., 0.5x means audio can be half as long). min=faster speech"
                }),
                "timing_tolerance": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Maximum allowed deviation (in seconds) for timing adjustments in 'smart_natural' mode. Higher values allow more flexibility."
                }),
                "batch_size": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Parallel processing workers. 0 = sequential (recommended), 2+ = streaming mode. Note: Streaming often slower than sequential mode. F5-TTS doesn't support streaming yet."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "generation_info", "timing_report", "Adjusted_SRT")
    FUNCTION = "generate_srt_speech"
    CATEGORY = "TTS Audio Suite/🎤 Text to Speech"

    def __init__(self):
        super().__init__()
        # Cache engine instances to prevent model reloading
        self._cached_engine_instances = {}

    def _create_proper_engine_node_instance(self, engine_data: Dict[str, Any]):
        """
        Create a proper engine SRT node instance that has all the needed functionality.
        Uses caching to reuse instances and preserve model state across segments.
        
        Args:
            engine_data: Engine configuration from TTS_engine input
            
        Returns:
            Proper engine SRT node instance with all functionality
        """
        try:
            engine_type = engine_data.get("engine_type")
            # Extract the nested config - engine_data has structure {engine_type, config, adapter_class}
            config = engine_data.get("config", engine_data)
            
            # Create cache key based only on stable parameters that affect engine instance creation
            stable_params = {
                'engine_type': config.get('engine_type'),
                'model': config.get('model'),
                'device': config.get('device'),
                'adapter_class': config.get('adapter_class')
            }
            
            # For VibeVoice, include attention_mode and quantization in cache key since they require model reload
            if engine_type == "vibevoice":
                stable_params['attention_mode'] = config.get('attention_mode', 'auto')
                stable_params['quantize_llm_4bit'] = config.get('quantize_llm_4bit', False)
                # Also include chunk_minutes if it affects behavior
                if 'chunk_minutes' in config:
                    stable_params['chunk_minutes'] = config.get('chunk_minutes', 0)

            # For ChatterBox Official 23-Lang, include model_version in cache key since v1/v2 are different models
            if engine_type == "chatterbox_official_23lang":
                stable_params['model_version'] = config.get('model_version', 'v1')

            cache_key = f"{engine_type}_{hashlib.md5(str(sorted(stable_params.items())).encode()).hexdigest()[:8]}"
            
            # Check if we have a cached instance with the same stable configuration
            if cache_key in self._cached_engine_instances:
                cached_data = self._cached_engine_instances[cache_key]
                
                # Handle both old (direct instance) and new (timestamped dict) cache formats
                if isinstance(cached_data, dict) and 'instance' in cached_data:
                    # New timestamped format
                    cached_instance = cached_data['instance']
                    cache_timestamp = cached_data['timestamp']
                    
                    # Check if cache is still valid (not invalidated by model unloading)
                    from utils.models.comfyui_model_wrapper import is_engine_cache_valid
                    if is_engine_cache_valid(cache_timestamp):
                        # CRITICAL FIX: Update the cached instance's config with ALL current parameters

                        if hasattr(cached_instance, 'update_config'):
                            cached_instance.update_config(config.copy())  # Propagate to processor
                        else:
                            cached_instance.config = config.copy()  # Fallback for other engines
                        print(f"🔄 Reusing cached {engine_type} SRT engine instance (updated with new generation parameters)")
                        return cached_instance
                    else:
                        # Cache invalidated by model unloading, remove it
                        print(f"🗑️ Removing invalidated {engine_type} SRT engine cache (models were unloaded)")
                        del self._cached_engine_instances[cache_key]
                else:
                    # Old format (direct instance) - assume invalid and remove
                    print(f"🗑️ Removing old-format {engine_type} SRT engine cache (upgrading to timestamped format)")
                    del self._cached_engine_instances[cache_key]
            
            # print(f"🔧 Creating new {engine_type} SRT engine instance")
            
            if engine_type == "chatterbox":
                # Import and create the original ChatterBox SRT node using absolute import
                chatterbox_srt_path = os.path.join(nodes_dir, "chatterbox", "chatterbox_srt_node.py")
                chatterbox_srt_spec = importlib.util.spec_from_file_location("chatterbox_srt_module", chatterbox_srt_path)
                chatterbox_srt_module = importlib.util.module_from_spec(chatterbox_srt_spec)
                chatterbox_srt_spec.loader.exec_module(chatterbox_srt_module)
                
                ChatterboxSRTTTSNode = chatterbox_srt_module.ChatterboxSRTTTSNode
                engine_instance = ChatterboxSRTTTSNode()
                # Apply configuration
                for key, value in config.items():
                    if hasattr(engine_instance, key):
                        setattr(engine_instance, key, value)
                
                # Cache the instance with timestamp
                import time
                self._cached_engine_instances[cache_key] = {
                    'instance': engine_instance,
                    'timestamp': time.time()
                }
                return engine_instance
                
            elif engine_type == "chatterbox_official_23lang":
                # ChatterBox Official 23-Lang can reuse the standard ChatterBox SRT processor architecture
                # The multilingual model works the same way for SRT processing
                # Import using absolute path like other engines in this file
                chatterbox_official_23lang_processor_path = os.path.join(nodes_dir, "chatterbox_official_23lang", "chatterbox_official_23lang_processor.py")
                chatterbox_official_23lang_spec = importlib.util.spec_from_file_location("chatterbox_official_23lang_processor_module", chatterbox_official_23lang_processor_path)
                chatterbox_official_23lang_module = importlib.util.module_from_spec(chatterbox_official_23lang_spec)
                chatterbox_official_23lang_spec.loader.exec_module(chatterbox_official_23lang_module)
                
                ChatterboxOfficial23LangTTSNode = chatterbox_official_23lang_module.ChatterboxOfficial23LangTTSNode
                
                # Create SRT processor wrapper
                chatterbox_official_23lang_srt_processor_path = os.path.join(nodes_dir, "chatterbox_official_23lang", "chatterbox_official_23lang_srt_processor.py")
                chatterbox_official_23lang_srt_spec = importlib.util.spec_from_file_location("chatterbox_official_23lang_srt_processor_module", chatterbox_official_23lang_srt_processor_path)
                chatterbox_official_23lang_srt_module = importlib.util.module_from_spec(chatterbox_official_23lang_srt_spec)
                chatterbox_official_23lang_srt_spec.loader.exec_module(chatterbox_official_23lang_srt_module)
                
                ChatterboxOfficial23LangSRTProcessor = chatterbox_official_23lang_srt_module.ChatterboxOfficial23LangSRTProcessor
                
                # Create a minimal wrapper node for the processor (following VibeVoice pattern)
                class ChatterboxOfficial23LangSRTWrapper:
                    def __init__(self, config):
                        self.config = config.copy()
                        self.tts_node = ChatterboxOfficial23LangTTSNode()
                        self.processor = ChatterboxOfficial23LangSRTProcessor(self.tts_node, config)
                        
                    def update_config(self, new_config):
                        """Update configuration for reused instances"""
                        self.config.update(new_config)
                        self.processor.config.update(new_config)
                    
                    def generate_srt_speech(self, srt_content, language, device, model, 
                                          narrator_voice, seed, temperature, exaggeration, cfg_weight,
                                          repetition_penalty, min_p, top_p, enable_audio_cache,
                                          timing_mode, fade_for_StretchToFit, max_stretch_ratio,
                                          min_stretch_ratio, timing_tolerance, batch_size=0):
                        """SRT speech generation for ChatterBox Official 23-Lang"""
                        # Use the processor's main entry point
                        voice_mapping = {"narrator": narrator_voice} if narrator_voice else {}
                        
                        timing_params = {
                            "fade_for_StretchToFit": fade_for_StretchToFit,
                            "max_stretch_ratio": max_stretch_ratio, 
                            "min_stretch_ratio": min_stretch_ratio,
                            "timing_tolerance": timing_tolerance
                        }
                        
                        # Pass current TTS parameters for proper cache invalidation
                        tts_params = {
                            'exaggeration': exaggeration,
                            'temperature': temperature,
                            'cfg_weight': cfg_weight,
                            'repetition_penalty': repetition_penalty,
                            'min_p': min_p,
                            'top_p': top_p,
                            'language': language,
                            'device': device
                        }
                        
                        return self.processor.process_srt_content(
                            srt_content=srt_content,
                            voice_mapping=voice_mapping,
                            seed=seed,
                            timing_mode=timing_mode,
                            timing_params=timing_params,
                            tts_params=tts_params
                        )
                
                engine_instance = ChatterboxOfficial23LangSRTWrapper(config)
                # Cache the instance with timestamp
                import time
                self._cached_engine_instances[cache_key] = {
                    'instance': engine_instance,
                    'timestamp': time.time()
                }
                return engine_instance
                
            elif engine_type == "f5tts":
                # Import and create the original F5-TTS SRT node using absolute import
                f5tts_srt_path = os.path.join(nodes_dir, "f5tts", "f5tts_srt_node.py")
                f5tts_srt_spec = importlib.util.spec_from_file_location("f5tts_srt_module", f5tts_srt_path)
                f5tts_srt_module = importlib.util.module_from_spec(f5tts_srt_spec)
                f5tts_srt_spec.loader.exec_module(f5tts_srt_module)
                
                F5TTSSRTNode = f5tts_srt_module.F5TTSSRTNode
                engine_instance = F5TTSSRTNode()
                # Apply configuration
                for key, value in config.items():
                    if hasattr(engine_instance, key):
                        setattr(engine_instance, key, value)
                
                # Cache the instance with timestamp
                import time
                self._cached_engine_instances[cache_key] = {
                    'instance': engine_instance,
                    'timestamp': time.time()
                }
                return engine_instance
                
            elif engine_type == "higgs_audio":
                # Create a wrapper instance for Higgs Audio using the adapter pattern
                from engines.adapters.higgs_audio_adapter import HiggsAudioEngineAdapter
                
                # Create a minimal wrapper node for the adapter
                class HiggsAudioSRTWrapper:
                    def __init__(self, config):
                        self.config = config
                        # Don't cache adapter - create fresh each time to ensure config updates
                        # Store current model name for adapter caching
                        self.current_model_name = None
                    
                    def generate_srt_audio(self, **params):
                        # SRT processing is handled by the unified node, this just handles audio generation
                        # Merge config with runtime params
                        merged_params = self.config.copy()
                        merged_params.update(params)
                        
                        # Extract required parameters for adapter
                        text = merged_params.get('text', '')
                        char_audio = merged_params.get('char_audio')
                        char_text = merged_params.get('char_text', '')
                        character = merged_params.get('character', 'narrator')
                        
                        # Remove non-adapter parameters
                        adapter_params = {k: v for k, v in merged_params.items() 
                                        if k not in ['srt_content', 'text', 'char_audio', 'char_text', 'character']}
                        
                        # Create fresh adapter instance with current config to ensure parameter updates
                        adapter = HiggsAudioEngineAdapter(self)
                        return adapter.generate_segment_audio(
                            text, char_audio, char_text, character, **adapter_params
                        )
                
                engine_instance = HiggsAudioSRTWrapper(config)
                # Cache the instance with timestamp
                import time
                self._cached_engine_instances[cache_key] = {
                    'instance': engine_instance,
                    'timestamp': time.time()
                }
                return engine_instance
                
            elif engine_type == "index_tts":
                # Import and create the IndexTTS-2 SRT processor using absolute import
                index_tts_srt_processor_path = os.path.join(nodes_dir, "index_tts", "index_tts_srt_processor.py")
                index_tts_srt_spec = importlib.util.spec_from_file_location("index_tts_srt_processor_module", index_tts_srt_processor_path)
                index_tts_srt_module = importlib.util.module_from_spec(index_tts_srt_spec)
                index_tts_srt_spec.loader.exec_module(index_tts_srt_module)

                IndexTTSSRTProcessor = index_tts_srt_module.IndexTTSSRTProcessor

                # Create a minimal wrapper node for the processor
                class IndexTTSSRTWrapper:
                    def __init__(self, config):
                        self.config = config
                        self.processor = IndexTTSSRTProcessor(self, config)

                    def update_config(self, new_config):
                        """Update configuration for both wrapper and processor"""
                        self.config = new_config.copy()
                        self.processor.update_config(new_config)

                    def process_with_error_handling(self, func):
                        """Error handling wrapper to match node interface"""
                        try:
                            return func()
                        except Exception as e:
                            raise e

                    def format_audio_output(self, audio_tensor, sample_rate):
                        """Format audio for ComfyUI output"""
                        # Move tensor to CPU if it's on CUDA
                        if audio_tensor.is_cuda:
                            audio_tensor = audio_tensor.cpu()
                        return {"waveform": audio_tensor, "sample_rate": sample_rate}

                engine_instance = IndexTTSSRTWrapper(config)

                # Cache the instance with timestamp
                import time
                self._cached_engine_instances[cache_key] = {
                    'instance': engine_instance,
                    'timestamp': time.time()
                }
                return engine_instance

            elif engine_type == "vibevoice":
                # Import and create the VibeVoice SRT processor using absolute import
                vibevoice_srt_processor_path = os.path.join(nodes_dir, "vibevoice", "vibevoice_srt_processor.py")
                vibevoice_srt_spec = importlib.util.spec_from_file_location("vibevoice_srt_processor_module", vibevoice_srt_processor_path)
                vibevoice_srt_module = importlib.util.module_from_spec(vibevoice_srt_spec)
                vibevoice_srt_spec.loader.exec_module(vibevoice_srt_module)
                
                VibeVoiceSRTProcessor = vibevoice_srt_module.VibeVoiceSRTProcessor
                
                # Create a minimal wrapper node for the processor
                class VibeVoiceSRTWrapper:
                    def __init__(self, config):
                        self.config = config
                        self.processor = VibeVoiceSRTProcessor(self, config)
                    
                    def update_config(self, new_config):
                        """Update configuration for both wrapper and processor"""
                        self.config = new_config.copy()
                        self.processor.update_config(new_config)
                    
                    def process_with_error_handling(self, func):
                        """Error handling wrapper to match node interface"""
                        try:
                            return func()
                        except Exception as e:
                            raise e
                    
                    def format_audio_output(self, audio_tensor, sample_rate):
                        """Format audio for ComfyUI output"""
                        # Move tensor to CPU if it's on CUDA
                        if audio_tensor.is_cuda:
                            audio_tensor = audio_tensor.cpu()
                        if audio_tensor.dim() == 1:
                            audio_tensor = audio_tensor.unsqueeze(0)
                        if audio_tensor.dim() == 2:
                            audio_tensor = audio_tensor.unsqueeze(0)
                        return {"waveform": audio_tensor, "sample_rate": sample_rate}
                
                engine_instance = VibeVoiceSRTWrapper(config)
                # Cache the instance with timestamp
                import time
                self._cached_engine_instances[cache_key] = {
                    'instance': engine_instance,
                    'timestamp': time.time()
                }
                return engine_instance
                
            else:
                raise ValueError(f"Unknown engine type: {engine_type}")
                
        except Exception as e:
            print(f"❌ Failed to create engine SRT node instance: {e}")
            return None

    def _get_voice_reference(self, opt_narrator, narrator_voice: str):
        """
        Get voice reference from opt_narrator input or narrator_voice dropdown.
        
        Args:
            opt_narrator: Voice data from Character Voices node OR direct audio input (priority)
            narrator_voice: Fallback voice from dropdown
            
        Returns:
            Tuple of (audio_path, audio_tensor, reference_text, character_name)
        """
        try:
            # Priority 1: opt_narrator input
            if opt_narrator is not None:
                # Check if it's a Character Voices node output (dict with specific keys)
                if isinstance(opt_narrator, dict) and "audio" in opt_narrator:
                    # Character Voices node output
                    audio = opt_narrator.get("audio")
                    audio_path = opt_narrator.get("audio_path") 
                    reference_text = opt_narrator.get("reference_text", "")
                    character_name = opt_narrator.get("character_name", "narrator")
                    
                    print(f"📺 TTS SRT: Using voice reference from Character Voices node ({character_name})")
                    return audio_path, audio, reference_text, character_name
                
                # Check if it's a direct audio input (dict with waveform and sample_rate)
                elif isinstance(opt_narrator, dict) and "waveform" in opt_narrator:
                    # Direct audio input - no reference text available
                    audio_tensor = opt_narrator
                    character_name = "narrator"
                    reference_text = ""  # No reference text available from direct audio
                    
                    print(f"📺 TTS SRT: Using direct audio input ({character_name})")
                    print(f"⚠️ TTS SRT: Direct audio input has no reference text - F5-TTS engines will fail")
                    return None, audio_tensor, reference_text, character_name
            
            # Priority 2: narrator_voice dropdown (fallback)
            elif narrator_voice != "none":
                from utils.voice.discovery import load_voice_reference
                audio_path, reference_text = load_voice_reference(narrator_voice)
                
                if audio_path and os.path.exists(audio_path):
                    # Load audio tensor
                    import torchaudio
                    waveform, sample_rate = torchaudio.load(audio_path)
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    audio_tensor = {"waveform": waveform, "sample_rate": sample_rate}
                    character_name = os.path.splitext(os.path.basename(narrator_voice))[0]
                    
                    print(f"📺 TTS SRT: Using voice reference from folder ({character_name})")
                    return audio_path, audio_tensor, reference_text or "", character_name
            
            print("⚠️ TTS SRT: No voice reference provided - this may cause issues with some engines")
            return None, None, "", "narrator"
            
        except Exception as e:
            print(f"❌ Voice reference error: {e}")
            return None, None, "", "narrator"

    def generate_srt_speech(self, TTS_engine: Dict[str, Any], srt_content: str, narrator_voice: str,
                           seed: int, timing_mode: str, opt_narrator=None, enable_audio_cache: bool = True,
                           fade_for_StretchToFit: float = 0.01, max_stretch_ratio: float = 1.0,
                           min_stretch_ratio: float = 0.5, timing_tolerance: float = 2.0,
                           batch_size: int = 0):
        """
        Generate SRT-timed speech using the selected TTS engine.
        This is a DELEGATION WRAPPER that preserves all original SRT functionality.
        
        Args:
            TTS_engine: Engine configuration from engine nodes
            srt_content: SRT subtitle content
            narrator_voice: Fallback narrator voice
            seed: Random seed
            timing_mode: How to align audio with SRT timings
            opt_narrator: Voice reference from Character Voices node
            enable_audio_cache: Enable audio caching
            fade_for_StretchToFit: Crossfade duration for stretch_to_fit mode
            max_stretch_ratio: Maximum stretch ratio for smart_natural mode
            min_stretch_ratio: Minimum stretch ratio for smart_natural mode
            timing_tolerance: Timing tolerance for smart_natural mode
            batch_size: Number of parallel workers (0=sequential, 1+=streaming parallel)
            
        Returns:
            Tuple of (audio_tensor, generation_info, timing_report, adjusted_srt)
        """
        try:
            # Apply Python 3.12 CUDNN compatibility fix before TTS generation
            from utils.comfyui_compatibility import ensure_python312_cudnn_fix
            ensure_python312_cudnn_fix()
            # Validate engine input
            if not TTS_engine or not isinstance(TTS_engine, dict):
                raise ValueError("Invalid TTS_engine input - connect a TTS engine node")

            engine_type = TTS_engine.get("engine_type")
            config = TTS_engine.get("config", {})

            # FIX: Sanitize engine config - ComfyUI JSON serialization corrupts numeric types
            # When workflow JSON is saved/loaded, floats like 10.0 become integers 10
            # This particularly affects first run after reboot or when using Any Switch
            config = ConfigSanitizer.sanitize(config)

            # Check if emotion control is connected
            if config.get('emotion_audio') is not None:
                print(f"🎭 Emotion control connected to SRT generation")

            if not engine_type:
                raise ValueError("TTS engine missing engine_type")

            print(f"📺 TTS SRT: Starting {engine_type} SRT generation")
            
            # Get voice reference (opt_narrator takes priority)
            audio_path, audio_tensor, reference_text, character_name = self._get_voice_reference(opt_narrator, narrator_voice)
            
            # Validate F5-TTS requirements: must have reference text
            if engine_type == "f5tts" and not reference_text.strip():
                raise ValueError(
                    "F5-TTS requires reference text. When using direct audio input, "
                    "please use Character Voices node instead, which provides both audio and text."
                )
            
            # Create proper engine SRT node instance to preserve ALL functionality
            engine_instance = self._create_proper_engine_node_instance(TTS_engine)
            if not engine_instance:
                raise RuntimeError("Failed to create engine SRT node instance")
            
            # Configure batch processing for ChatterBox if applicable
            if engine_type == "chatterbox" and hasattr(engine_instance, 'adapter'):
                # Configure the adapter's batch processing settings
                # Batch size configuration is handled automatically by the engine
                print(f"🔧 ChatterBox SRT batch processing configured: batch_size={batch_size}")
            
            # Prepare parameters for the original SRT node's generate_srt_speech method
            if engine_type == "chatterbox":
                # ChatterBox SRT parameters with batch processing support
                result = engine_instance.generate_srt_speech(
                    srt_content=srt_content,
                    language=config.get("language", "English"),
                    device=config.get("device", "auto"),
                    exaggeration=config.get("exaggeration", 0.5),
                    temperature=config.get("temperature", 0.8),
                    cfg_weight=config.get("cfg_weight", 0.5),
                    seed=seed,
                    timing_mode=timing_mode,
                    reference_audio=audio_tensor,
                    audio_prompt_path=audio_path or "",
                    enable_audio_cache=enable_audio_cache,
                    fade_for_StretchToFit=fade_for_StretchToFit,
                    max_stretch_ratio=max_stretch_ratio,
                    min_stretch_ratio=min_stretch_ratio,
                    timing_tolerance=timing_tolerance,
                    crash_protection_template=config.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
                    batch_size=batch_size
                )
                
            elif engine_type == "chatterbox_official_23lang":
                # ChatterBox Official 23-Lang SRT parameters with multilingual support
                # Use audio_tensor if available (direct audio input), otherwise fallback to audio_path (dropdown)
                narrator_input = audio_tensor if audio_tensor is not None else (audio_path or "")
                result = engine_instance.generate_srt_speech(
                    srt_content=srt_content,
                    language=config.get("language", "English"),  # Language name (e.g. "Turkish", "Arabic")
                    device=config.get("device", "auto"),
                    model=config.get("model", "ChatterBox Official 23-Lang"),
                    narrator_voice=narrator_input,  # Audio tensor or file path
                    seed=seed,
                    temperature=config.get("temperature", 0.8),
                    exaggeration=config.get("exaggeration", 0.5),
                    cfg_weight=config.get("cfg_weight", 0.5),
                    repetition_penalty=config.get("repetition_penalty", 1.2),
                    min_p=config.get("min_p", 0.05),
                    top_p=config.get("top_p", 1.0),
                    enable_audio_cache=enable_audio_cache,
                    timing_mode=timing_mode,
                    fade_for_StretchToFit=fade_for_StretchToFit,
                    max_stretch_ratio=max_stretch_ratio,
                    min_stretch_ratio=min_stretch_ratio,
                    timing_tolerance=timing_tolerance,
                    batch_size=batch_size
                )
                
            elif engine_type == "f5tts":
                # F5-TTS streaming warning and fallback
                if batch_size > 1:
                    print(f"⚠️ F5-TTS doesn't support streaming mode yet. Falling back to sequential processing (batch_size=0)")
                    batch_size = 0
                
                # F5-TTS SRT parameters
                # Always use the resolved audio reference from _get_voice_reference priority logic
                if audio_tensor:
                    # Use resolved audio reference (from opt_narrator OR narrator_voice dropdown)
                    opt_reference_audio = audio_tensor
                    reference_audio_file = "none"
                    opt_reference_text = reference_text
                else:
                    # Fallback to file-based reference (shouldn't happen if _get_voice_reference worked)
                    opt_reference_audio = None
                    reference_audio_file = narrator_voice
                    opt_reference_text = reference_text
                
                result = engine_instance.generate_srt_speech(
                    srt_content=srt_content,
                    reference_audio_file=reference_audio_file,
                    opt_reference_text=opt_reference_text,
                    device=config.get("device", "auto"),
                    model=config.get("model", "F5TTS_Base"),
                    seed=seed,
                    timing_mode=timing_mode,
                    opt_reference_audio=opt_reference_audio,
                    temperature=config.get("temperature", 0.8),
                    speed=config.get("speed", 1.0),
                    target_rms=config.get("target_rms", 0.1),
                    cross_fade_duration=config.get("cross_fade_duration", 0.15),
                    nfe_step=config.get("nfe_step", 32),
                    cfg_strength=config.get("cfg_strength", 2.0),
                    enable_audio_cache=enable_audio_cache,
                    fade_for_StretchToFit=fade_for_StretchToFit,
                    max_stretch_ratio=max_stretch_ratio,
                    min_stretch_ratio=min_stretch_ratio,
                    timing_tolerance=timing_tolerance,
                    auto_phonemization=config.get("auto_phonemization", True)
                )
                
            elif engine_type == "higgs_audio":
                # Import and create the Higgs Audio SRT processor
                higgs_srt_processor_path = os.path.join(nodes_dir, "higgs_audio", "higgs_audio_srt_processor.py")
                higgs_srt_spec = importlib.util.spec_from_file_location("higgs_audio_srt_processor_module", higgs_srt_processor_path)
                higgs_srt_module = importlib.util.module_from_spec(higgs_srt_spec)
                higgs_srt_spec.loader.exec_module(higgs_srt_module)
                
                HiggsAudioSRTProcessor = higgs_srt_module.HiggsAudioSRTProcessor
                srt_processor = HiggsAudioSRTProcessor(engine_instance)
                
                # Clean delegation to SRT processor
                result = srt_processor.generate_srt_speech(
                    srt_content=srt_content,
                    multi_speaker_mode=config.get("multi_speaker_mode", "Custom Character Switching"),
                    audio_tensor=audio_tensor,
                    reference_text=reference_text,
                    seed=seed,
                    timing_mode=timing_mode,
                    fade_for_StretchToFit=fade_for_StretchToFit,
                    max_stretch_ratio=max_stretch_ratio,
                    min_stretch_ratio=min_stretch_ratio,
                    timing_tolerance=timing_tolerance,
                    enable_audio_cache=enable_audio_cache
                )
                
            elif engine_type == "index_tts":
                # Use the IndexTTS-2 SRT processor from the wrapper instance
                voice_mapping = {"narrator": audio_tensor} if audio_tensor else {}

                # Prepare timing parameters
                timing_params = {
                    'fade_for_StretchToFit': fade_for_StretchToFit,
                    'max_stretch_ratio': max_stretch_ratio,
                    'min_stretch_ratio': min_stretch_ratio,
                    'timing_tolerance': timing_tolerance
                }

                # Use the processor's main entry point with IndexTTS-2 emotion control
                result = engine_instance.processor.process_srt_content(
                    srt_content=srt_content,
                    voice_mapping=voice_mapping,
                    seed=seed,
                    timing_mode=timing_mode,
                    timing_params=timing_params
                )

            elif engine_type == "vibevoice":
                # Use the VibeVoice SRT processor from the wrapper instance
                voice_mapping = {"narrator": audio_tensor} if audio_tensor else {}

                # Prepare timing parameters
                timing_params = {
                    'fade_for_StretchToFit': fade_for_StretchToFit,
                    'max_stretch_ratio': max_stretch_ratio,
                    'min_stretch_ratio': min_stretch_ratio,
                    'timing_tolerance': timing_tolerance
                }

                # Use the processor's main entry point
                result = engine_instance.processor.process_srt_content(
                    srt_content=srt_content,
                    voice_mapping=voice_mapping,
                    seed=seed,
                    timing_mode=timing_mode,
                    timing_params=timing_params
                )
                
            else:
                raise ValueError(f"Unknown engine type: {engine_type}")
            
            # The original SRT nodes return (audio, generation_info, timing_report, adjusted_srt)
            audio_output, generation_info, timing_report, adjusted_srt = result
            
            # Ensure audio tensor is moved to CPU and has correct dtype
            if audio_output and isinstance(audio_output, dict) and "waveform" in audio_output:
                waveform = audio_output["waveform"]
                if isinstance(waveform, torch.Tensor):
                    needs_update = False
                    new_waveform = waveform
                    
                    # Move to CPU if on CUDA
                    if waveform.is_cuda:
                        new_waveform = new_waveform.cpu()
                        needs_update = True
                    
                    # Convert to float32 if not already
                    if waveform.dtype != torch.float32:
                        new_waveform = new_waveform.to(torch.float32)
                        needs_update = True
                    
                    if needs_update:
                        audio_output = audio_output.copy()
                        audio_output["waveform"] = new_waveform
            
            # Add unified prefix to generation info
            unified_info = f"📺 TTS SRT (Unified) - {engine_type.upper()} Engine:\n{generation_info}"
            
            print(f"✅ TTS SRT: {engine_type} SRT generation successful")
            return (audio_output, unified_info, timing_report, adjusted_srt)
                
        except Exception as e:
            error_msg = f"❌ TTS SRT generation failed: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            # Return empty audio and error info (preserving original return structure)
            empty_audio = AudioProcessingUtils.create_silence(1.0, 24000)
            empty_comfy = AudioProcessingUtils.format_for_comfyui(empty_audio, 24000)
            
            return (empty_comfy, error_msg, "Error: No timing report available", "Error: No adjusted SRT available")


# Register the node class
NODE_CLASS_MAPPINGS = {
    "UnifiedTTSSRTNode": UnifiedTTSSRTNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnifiedTTSSRTNode": "📺 TTS SRT"
}