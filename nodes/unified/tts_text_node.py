"""
Unified TTS Text Node - Engine-agnostic text-to-speech generation for TTS Audio Suite
Replaces both ChatterBox TTS and F5-TTS nodes with unified architecture
"""

import torch
import numpy as np
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

from utils.text.chunking import ImprovedChatterBoxChunker
from utils.audio.processing import AudioProcessingUtils
from utils.voice.discovery import get_available_voices, load_voice_reference, get_available_characters, get_character_mapping
from utils.text.character_parser import parse_character_text, character_parser
from utils.voice.multilingual_engine import MultilingualEngine
import comfy.model_management as model_management

# Global audio cache for unified TTS segments
GLOBAL_AUDIO_CACHE = {}

# AnyType for flexible input types (accepts any data type)
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")


class UnifiedTTSTextNode(BaseTTSNode):
    """
    Unified TTS Text Node - Engine-agnostic text-to-speech generation.
    Works with any TTS engine (ChatterBox, F5-TTS, future RVC, etc.) through engine adapters.
    Replaces both ChatterBox TTS and F5-TTS nodes.
    """
    
    @classmethod
    def NAME(cls):
        return "🎤 TTS Text"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available reference audio files from voice folders
        reference_files = get_available_voices()
        
        return {
            "required": {
                "TTS_engine": ("TTS_ENGINE", {
                    "tooltip": "TTS engine configuration from ChatterBox Engine or F5 TTS Engine nodes"
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": """Hello! This is unified TTS with character switching support.
[Alice] Hi there! I'm Alice speaking with the selected TTS engine.
[Bob] And I'm Bob! This works with any TTS engine.
Back to the main narrator voice for the conclusion.""",
                    "tooltip": "Text to convert to speech. Use [Character] tags for voice switching. Characters not found in voice folders will use the narrator voice."
                }),
                "narrator_voice": (reference_files, {
                    "default": "none",
                    "tooltip": "Fallback narrator voice from voice folders. Used when opt_narrator is not connected. Select 'none' if you only use opt_narrator input."
                }),
                "seed": ("INT", {
                    "default": 1, "min": 0, "max": 2**32 - 1,
                    "tooltip": "Seed for reproducible TTS generation. Same seed with same inputs will produce identical results. Set to 0 for random generation."
                }),
            },
            "optional": {
                "opt_narrator": (any_typ, {
                    "tooltip": "Voice reference: Connect Character Voices node output OR direct audio input. Takes priority over narrator_voice dropdown when connected."
                }),
                "enable_chunking": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable text chunking for long texts. When enabled, long texts are split into smaller chunks for more stable generation."
                }),
                "max_chars_per_chunk": ("INT", {
                    "default": 400, "min": 100, "max": 1000, "step": 50,
                    "tooltip": "Maximum characters per chunk when chunking is enabled. Smaller chunks = more stable but potentially less coherent speech."
                }),
                "chunk_combination_method": (["auto", "concatenate", "silence_padding", "crossfade"], {
                    "default": "auto",
                    "tooltip": "Method to combine audio chunks: 'auto' chooses best method, 'concatenate' joins directly, 'silence_padding' adds silence between chunks, 'crossfade' smoothly blends chunks."
                }),
                "silence_between_chunks_ms": ("INT", {
                    "default": 100, "min": 0, "max": 500, "step": 25,
                    "tooltip": "Silence duration between chunks in milliseconds when using 'silence_padding' combination method. Longer silences = more distinct separation between chunks."
                }),
                "enable_audio_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, generated audio segments will be cached in memory to speed up subsequent runs with identical parameters."
                }),
                "enable_batch_processing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable batch processing for ChatterBox engine. Processes multiple text chunks in parallel for faster generation."
                }),
                "batch_size": ("INT", {
                    "default": 4, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Number of text chunks to process in parallel when batch processing is enabled. Higher values = faster but more memory usage. Uses overlapping parallel processing."
                }),
                "enable_adaptive_batching": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable dynamic worker adjustment. Starts with your batch_size and adapts based on real-time performance. Disable to use exact batch_size."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "generation_info")
    FUNCTION = "generate_speech"
    CATEGORY = "TTS Audio Suite"

    def __init__(self):
        super().__init__()
        self.chunker = ImprovedChatterBoxChunker()
        self._current_engine = None
        self._current_adapter = None
        # Cache engine instances to prevent model reloading
        self._cached_engine_instances = {}

    def _create_proper_engine_node_instance(self, engine_data: Dict[str, Any]):
        """
        Create a proper engine node instance that has all the needed functionality.
        Uses caching to reuse instances and preserve model state across segments.
        
        Args:
            engine_data: Engine configuration from TTS_engine input
            
        Returns:
            Proper engine node instance with all functionality
        """
        try:
            engine_type = engine_data.get("engine_type")
            config = engine_data.get("config", {})
            
            # Create cache key based on engine type and stable config
            cache_key = f"{engine_type}_{hashlib.md5(str(sorted(config.items())).encode()).hexdigest()[:8]}"
            
            # Check if we have a cached instance with the same configuration
            if cache_key in self._cached_engine_instances:
                cached_instance = self._cached_engine_instances[cache_key]
                print(f"🔄 Reusing cached {engine_type} engine instance (preserves model state)")
                return cached_instance
            
            print(f"🔧 Creating new {engine_type} engine instance")
            
            if engine_type == "chatterbox":
                # Import and create the original ChatterBox node using absolute import
                chatterbox_node_path = os.path.join(nodes_dir, "chatterbox", "chatterbox_tts_node.py")
                chatterbox_spec = importlib.util.spec_from_file_location("chatterbox_tts_module", chatterbox_node_path)
                chatterbox_module = importlib.util.module_from_spec(chatterbox_spec)
                chatterbox_spec.loader.exec_module(chatterbox_module)
                
                ChatterboxTTSNode = chatterbox_module.ChatterboxTTSNode
                engine_instance = ChatterboxTTSNode()
                # Apply configuration
                for key, value in config.items():
                    if hasattr(engine_instance, key):
                        setattr(engine_instance, key, value)
                
                # Cache the instance
                self._cached_engine_instances[cache_key] = engine_instance
                return engine_instance
                
            elif engine_type == "f5tts":
                # Import and create the original F5-TTS node using absolute import
                f5tts_node_path = os.path.join(nodes_dir, "f5tts", "f5tts_node.py")
                f5tts_spec = importlib.util.spec_from_file_location("f5tts_module", f5tts_node_path)
                f5tts_module = importlib.util.module_from_spec(f5tts_spec)
                f5tts_spec.loader.exec_module(f5tts_module)
                
                F5TTSNode = f5tts_module.F5TTSNode
                engine_instance = F5TTSNode()
                # Apply configuration
                for key, value in config.items():
                    if hasattr(engine_instance, key):
                        setattr(engine_instance, key, value)
                
                # Cache the instance
                self._cached_engine_instances[cache_key] = engine_instance
                return engine_instance
                
            else:
                raise ValueError(f"Unknown engine type: {engine_type}")
                
        except Exception as e:
            print(f"❌ Failed to create engine node instance: {e}")
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
                    
                    print(f"🎤 TTS Text: Using voice reference from Character Voices node ({character_name})")
                    return audio_path, audio, reference_text, character_name
                
                # Check if it's a direct audio input (dict with waveform and sample_rate)
                elif isinstance(opt_narrator, dict) and "waveform" in opt_narrator:
                    # Direct audio input - no reference text available
                    audio_tensor = opt_narrator
                    character_name = "narrator"
                    reference_text = ""  # No reference text available from direct audio
                    
                    print(f"🎤 TTS Text: Using direct audio input ({character_name})")
                    print(f"⚠️ TTS Text: Direct audio input has no reference text - F5-TTS engines will fail")
                    return None, audio_tensor, reference_text, character_name
            
            # Priority 2: narrator_voice dropdown (fallback)
            elif narrator_voice != "none":
                audio_path, reference_text = load_voice_reference(narrator_voice)
                
                if audio_path and os.path.exists(audio_path):
                    # Load audio tensor
                    import torchaudio
                    waveform, sample_rate = torchaudio.load(audio_path)
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    audio_tensor = {"waveform": waveform, "sample_rate": sample_rate}
                    character_name = os.path.splitext(os.path.basename(narrator_voice))[0]
                    
                    print(f"🎤 TTS Text: Using voice reference from folder ({character_name})")
                    return audio_path, audio_tensor, reference_text or "", character_name
            
            print("⚠️ TTS Text: No voice reference provided - this may cause issues with some engines")
            return None, None, "", "narrator"
            
        except Exception as e:
            print(f"❌ Voice reference error: {e}")
            return None, None, "", "narrator"

    def generate_speech(self, TTS_engine: Dict[str, Any], text: str, narrator_voice: str, seed: int,
                       opt_narrator=None, enable_chunking: bool = True, max_chars_per_chunk: int = 400,
                       chunk_combination_method: str = "auto", silence_between_chunks_ms: int = 100,
                       enable_audio_cache: bool = True, enable_batch_processing: bool = True,
                       batch_size: int = 4, enable_adaptive_batching: bool = False):
        """
        Generate speech using the selected TTS engine.
        This is a DELEGATION WRAPPER that preserves all original functionality.
        
        Args:
            TTS_engine: Engine configuration from engine nodes
            text: Text to convert to speech
            narrator_voice: Fallback narrator voice
            seed: Random seed
            opt_narrator: Voice reference from Character Voices node
            enable_chunking: Enable text chunking
            max_chars_per_chunk: Maximum characters per chunk
            chunk_combination_method: Method to combine chunks
            silence_between_chunks_ms: Silence between chunks
            enable_audio_cache: Enable audio caching
            enable_batch_processing: Enable batch processing for ChatterBox
            batch_size: Number of chunks to process in parallel
            
        Returns:
            Tuple of (audio_tensor, generation_info)
        """
        try:
            # Validate engine input
            if not TTS_engine or not isinstance(TTS_engine, dict):
                raise ValueError("Invalid TTS_engine input - connect a TTS engine node")
            
            engine_type = TTS_engine.get("engine_type")
            config = TTS_engine.get("config", {})
            
            if not engine_type:
                raise ValueError("TTS engine missing engine_type")
            
            print(f"🎤 TTS Text: Starting {engine_type} generation")
            
            # Get voice reference (opt_narrator takes priority)
            audio_path, audio_tensor, reference_text, character_name = self._get_voice_reference(opt_narrator, narrator_voice)
            
            # Validate F5-TTS requirements: must have reference text
            if engine_type == "f5tts" and not reference_text.strip():
                raise ValueError(
                    "F5-TTS requires reference text. When using direct audio input, "
                    "please use Character Voices node instead, which provides both audio and text."
                )
            
            # Create proper engine node instance to preserve ALL functionality
            engine_instance = self._create_proper_engine_node_instance(TTS_engine)
            if not engine_instance:
                raise RuntimeError("Failed to create engine node instance")
            
            # IMPORTANT: Add crash protection template to config if missing (for ChatterBox)
            if engine_type == "chatterbox" and "crash_protection_template" not in config:
                config["crash_protection_template"] = "hmm ,, {seg} hmm ,,"
            
            # Configure batch processing for ChatterBox if applicable
            if engine_type == "chatterbox" and hasattr(engine_instance, 'adapter'):
                # Configure the adapter's batch processing settings
                if hasattr(engine_instance.adapter, 'set_batch_processing'):
                    engine_instance.adapter.set_batch_processing(enable_batch_processing, batch_size)
                    print(f"🔧 ChatterBox batch processing configured: enabled={enable_batch_processing}, batch_size={batch_size}")
            
            # Prepare parameters for the original node's generate_speech method
            if engine_type == "chatterbox":
                # ChatterBox TTS parameters with batch processing support
                result = engine_instance.generate_speech(
                    text=text,
                    language=config.get("language", "English"),
                    device=config.get("device", "auto"),
                    exaggeration=config.get("exaggeration", 0.5),
                    temperature=config.get("temperature", 0.8),
                    cfg_weight=config.get("cfg_weight", 0.5),
                    seed=seed,
                    reference_audio=audio_tensor,
                    audio_prompt_path=audio_path or "",
                    enable_chunking=enable_chunking,
                    max_chars_per_chunk=max_chars_per_chunk,
                    chunk_combination_method=chunk_combination_method,
                    silence_between_chunks_ms=silence_between_chunks_ms,
                    crash_protection_template=config.get("crash_protection_template", "hmm ,, {seg} hmm ,,"),
                    enable_audio_cache=enable_audio_cache,
                    enable_batch_processing=enable_batch_processing,
                    batch_size=batch_size,
                    enable_adaptive_batching=enable_adaptive_batching
                )
                
            elif engine_type == "f5tts":
                # F5-TTS parameters
                # For F5-TTS we need to handle reference_audio_file vs opt_reference_audio differently
                if opt_narrator:
                    # Use direct reference audio from Character Voices
                    opt_reference_audio = audio_tensor
                    reference_audio_file = "none"
                    opt_reference_text = reference_text
                else:
                    # Use narrator_voice dropdown
                    opt_reference_audio = None
                    reference_audio_file = narrator_voice
                    opt_reference_text = reference_text
                
                result = engine_instance.generate_speech(
                    reference_audio_file=reference_audio_file,
                    opt_reference_text=opt_reference_text,
                    device=config.get("device", "auto"),
                    model=config.get("model", "F5TTS_Base"),
                    seed=seed,
                    text=text,
                    opt_reference_audio=opt_reference_audio,
                    temperature=config.get("temperature", 0.8),
                    speed=config.get("speed", 1.0),
                    target_rms=config.get("target_rms", 0.1),
                    cross_fade_duration=config.get("cross_fade_duration", 0.15),
                    nfe_step=config.get("nfe_step", 32),
                    cfg_strength=config.get("cfg_strength", 2.0),
                    enable_chunking=enable_chunking,
                    max_chars_per_chunk=max_chars_per_chunk,
                    chunk_combination_method=chunk_combination_method,
                    silence_between_chunks_ms=silence_between_chunks_ms,
                    enable_audio_cache=enable_audio_cache
                )
                
            else:
                raise ValueError(f"Unknown engine type: {engine_type}")
            
            # The original nodes return (audio, generation_info)
            audio_output, generation_info = result
            
            # Add unified prefix to generation info
            unified_info = f"🎤 TTS Text (Unified) - {engine_type.upper()} Engine:\n{generation_info}"
            
            print(f"✅ TTS Text: {engine_type} generation successful")
            return (audio_output, unified_info)
                
        except Exception as e:
            error_msg = f"❌ TTS Text generation failed: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            # Return empty audio and error info
            empty_audio = AudioProcessingUtils.create_silence(1.0, 24000)
            empty_comfy = AudioProcessingUtils.format_for_comfyui(empty_audio, 24000)
            
            return (empty_comfy, error_msg)


# Register the node class
NODE_CLASS_MAPPINGS = {
    "UnifiedTTSTextNode": UnifiedTTSTextNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnifiedTTSTextNode": "🎤 TTS Text"
}