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
                "batch_size": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Parallel processing workers. 0-1 = sequential (recommended for most cases), 2+ = streaming mode. Note: Streaming may be slower than sequential for small texts. F5-TTS doesn't support streaming yet."
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
            # The engine_data IS the config - not nested under "config"
            config = engine_data
            
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
                
            elif engine_type == "higgs_audio":
                # Create a wrapper instance for Higgs Audio using the adapter pattern
                from engines.adapters.higgs_audio_adapter import HiggsAudioEngineAdapter
                
                # Create a minimal wrapper node for the adapter
                class HiggsAudioWrapper:
                    def __init__(self, config):
                        self.config = config
                        self.adapter = HiggsAudioEngineAdapter(self)
                        # Store current model name for adapter caching
                        self.current_model_name = None
                    
                    def generate_tts_audio(self, text, char_audio, char_text, character="narrator", **params):
                        # Merge config with runtime params
                        merged_params = self.config.copy()
                        merged_params.update(params)
                        return self.adapter.generate_segment_audio(text, char_audio, char_text, character, **merged_params)
                
                engine_instance = HiggsAudioWrapper(config)
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
                       enable_audio_cache: bool = True, batch_size: int = 4):
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
            batch_size: Batch size (0-1=sequential, 2+=streaming parallelization)
            
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
            
            # ChatterBox will automatically determine streaming vs sequential based on batch_size
            
            # Prepare parameters for the original node's generate_speech method
            if engine_type == "chatterbox":
                # ChatterBox TTS parameters - batch_size controls everything
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
                    batch_size=batch_size
                )
                
            elif engine_type == "f5tts":
                # F5-TTS streaming warning and fallback
                if batch_size > 1:
                    print(f"⚠️ F5-TTS doesn't support streaming mode yet. Falling back to sequential processing (batch_size=0)")
                    batch_size = 0
                
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
                
            elif engine_type == "higgs_audio":
                # Higgs Audio 2 with multiple speaker modes  
                # Get the mode from the engine config, not the TTS Text node config
                multi_speaker_mode = engine_instance.config.get("multi_speaker_mode", "Custom Character Switching")
                print(f"🎭 Higgs Audio: Using mode '{multi_speaker_mode}'")
                
                if multi_speaker_mode == "Custom Character Switching":
                    # Use existing modular utilities - pause processing first, then character parsing (like ChatterBox)
                    print(f"🎭 Higgs Audio: Using character switching with pause support")
                    
                    # Import modular utilities  
                    from utils.text.pause_processor import PauseTagProcessor
                    from utils.voice.discovery import get_character_mapping
                    
                    # Discover characters and build voice mapping
                    character_segments = parse_character_text(text)
                    all_characters = set(char for char, _ in character_segments)
                    character_mapping = get_character_mapping(list(all_characters), engine_type="higgs_audio")
                    
                    print(f"🎭 Higgs Audio: Processing {len(character_segments)} character segment(s) - {', '.join(sorted(all_characters))}")
                    
                    # Build voice references - CRITICAL: Start with narrator using connected voice
                    narrator_voice_dict = None
                    if audio_tensor is not None:
                        narrator_voice_dict = {"waveform": audio_tensor["waveform"], "sample_rate": audio_tensor["sample_rate"]}
                    
                    voice_refs = {'narrator': narrator_voice_dict}
                    
                    for character in all_characters:
                        # Skip narrator - already set above with connected voice
                        if character.lower() == "narrator":
                            continue
                            
                        audio_path, _ = character_mapping.get(character, (None, None))
                        if audio_path and os.path.exists(audio_path):
                            import torchaudio
                            waveform, sample_rate = torchaudio.load(audio_path)
                            if waveform.shape[0] > 1:
                                waveform = torch.mean(waveform, dim=0, keepdim=True)
                            voice_refs[character] = {"waveform": waveform, "sample_rate": sample_rate}
                        else:
                            # Use main narrator voice as fallback
                            voice_refs[character] = narrator_voice_dict
                    
                    def tts_generate_func(text_content: str) -> torch.Tensor:
                        """TTS generation function for pause tag processor"""
                        if '[' in text_content and ']' in text_content:
                            # Handle character switching within this segment
                            char_segments = parse_character_text(text_content)
                            segment_audio_parts = []
                            
                            for character, segment_text in char_segments:
                                char_audio_dict = voice_refs.get(character)
                                char_ref_text = reference_text or ""
                                
                                segment_result = engine_instance.generate_tts_audio(
                                    text=segment_text,
                                    char_audio=char_audio_dict,
                                    char_text=char_ref_text,
                                    character=character,
                                    voice_preset=engine_instance.config.get("voice_preset", "voice_clone"),
                                    system_prompt=engine_instance.config.get("system_prompt", "Generate audio following instruction."),
                                    temperature=engine_instance.config.get("temperature", 1.0),
                                    top_p=engine_instance.config.get("top_p", 0.95),
                                    top_k=engine_instance.config.get("top_k", 50),
                                    max_new_tokens=engine_instance.config.get("max_new_tokens", 2048),
                                    seed=seed,
                                    enable_audio_cache=enable_audio_cache,
                                    max_chars_per_chunk=max_chars_per_chunk,
                                    silence_between_chunks_ms=0
                                )
                                segment_audio_parts.append(segment_result)
                            
                            # Combine character segments
                            if segment_audio_parts:
                                return torch.cat(segment_audio_parts, dim=-1)
                            else:
                                return torch.zeros(1, 0)
                        else:
                            # Simple text segment without character switching - use narrator voice
                            narrator_audio = voice_refs.get("narrator")
                            if narrator_audio is None and audio_tensor is not None:
                                narrator_audio = {"waveform": audio_tensor, "sample_rate": 24000}
                            
                            return engine_instance.generate_tts_audio(
                                text=text_content,
                                char_audio=narrator_audio,
                                char_text=reference_text or "",
                                character="narrator",
                                voice_preset=engine_instance.config.get("voice_preset", "voice_clone"),
                                system_prompt=engine_instance.config.get("system_prompt", "Generate audio following instruction."),
                                temperature=engine_instance.config.get("temperature", 1.0),
                                top_p=engine_instance.config.get("top_p", 0.95),
                                top_k=engine_instance.config.get("top_k", 50),
                                max_new_tokens=engine_instance.config.get("max_new_tokens", 2048),
                                seed=seed,
                                enable_audio_cache=enable_audio_cache,
                                max_chars_per_chunk=max_chars_per_chunk,
                                silence_between_chunks_ms=0
                            )
                    
                    # Process with pause tag handling using existing utility
                    pause_processor = PauseTagProcessor()
                    
                    # Parse text into segments (text and pause segments)
                    segments, clean_text = pause_processor.parse_pause_tags(text)
                    
                    # Generate audio with pauses
                    if segments:
                        result = pause_processor.generate_audio_with_pauses(
                            segments=segments,
                            tts_generate_func=tts_generate_func,
                            sample_rate=24000
                        )
                    else:
                        # No pause tags, just generate directly
                        result = tts_generate_func(text)
                
                else:
                    # Native multi-speaker modes - process entire conversation as single unit
                    print(f"🎭 Higgs Audio: Using native multi-speaker mode (whole conversation processing)")
                    
                    # Get second narrator audio if provided
                    opt_second_narrator = engine_instance.config.get("opt_second_narrator")
                    
                    # Prepare reference audios for native mode
                    reference_audio_dict = None
                    second_audio_dict = None
                    
                    if audio_tensor is not None:
                        reference_audio_dict = {
                            "waveform": audio_tensor,
                            "sample_rate": 24000
                        }
                    
                    if opt_second_narrator is not None:
                        second_audio_dict = opt_second_narrator
                    
                    # Process entire conversation as single unit - let Higgs Audio handle pauses and speaker transitions
                    print(f"🎭 Processing full conversation: '{text[:100]}...'")
                    
                    result = engine_instance.generate_tts_audio(
                        text=text,  # Full conversation text
                        char_audio=reference_audio_dict,
                        char_text="",  # Higgs Audio doesn't need reference text
                        character="SPEAKER0",
                        # Config parameters
                        voice_preset=engine_instance.config.get("voice_preset", "voice_clone"),
                        system_prompt=engine_instance.config.get("system_prompt", "Generate audio following instruction."),
                        temperature=engine_instance.config.get("temperature", 1.0),
                        top_p=engine_instance.config.get("top_p", 0.95),
                        top_k=engine_instance.config.get("top_k", 50),
                        max_new_tokens=engine_instance.config.get("max_new_tokens", 2048),
                        seed=seed,
                        enable_audio_cache=enable_audio_cache,
                        max_chars_per_chunk=max_chars_per_chunk,  # This might trigger chunking - unknown how it will interact
                        silence_between_chunks_ms=0,
                        # Native mode specific parameters
                        multi_speaker_mode=multi_speaker_mode,
                        second_narrator_audio=second_audio_dict,
                        second_narrator_text=""  # Higgs Audio doesn't need reference text
                    )
                
                # CRITICAL FIX: Ensure tensor has correct dimensions for ComfyUI
                if isinstance(result, torch.Tensor):
                    print(f"🔧 Higgs Audio tensor before fix: {result.shape}")
                    if result.dim() == 3 and result.size(0) == 1:
                        result = result.squeeze(0)  # Remove batch dimension [1,1,N] -> [1,N]
                        print(f"🔧 Higgs Audio tensor after fix: {result.shape}")
                    elif result.dim() == 1:
                        result = result.unsqueeze(0)  # Add channel dimension [N] -> [1,N]
                
                # Convert single tensor result to ComfyUI audio format
                if isinstance(result, torch.Tensor):
                    # Ensure correct dimensions for ComfyUI: [channels, samples]
                    if result.dim() == 1:
                        result = result.unsqueeze(0)  # Add channel dimension: [samples] -> [1, samples]
                    elif result.dim() == 3:
                        # Handle [batch, channels, samples] -> [channels, samples]
                        if result.size(0) == 1:  # batch size of 1
                            result = result.squeeze(0)  # Remove batch dimension
                        else:
                            result = result[0]  # Take first item from batch
                    
                    # Ensure we have exactly 2 dimensions: [channels, samples]
                    if result.dim() != 2:
                        print(f"⚠️ Unexpected tensor dimensions: {result.shape}, reshaping to 2D")
                        if result.dim() == 1:
                            result = result.unsqueeze(0)
                        elif result.dim() > 2:
                            # Flatten to 1D and add channel dimension
                            result = result.view(-1).unsqueeze(0)
                    
                    # Use the same format_audio_output as ChatterBox for consistency
                    final_waveform = result.cpu()
                    print(f"🔧 Higgs Audio raw tensor: {final_waveform.shape}")
                    
                    # Format using base class method (adds batch dimension like ChatterBox)
                    formatted_audio = self.format_audio_output(final_waveform, 24000)
                    print(f"🔧 After format_audio_output: {formatted_audio['waveform'].shape}")
                    
                    result = (formatted_audio, "Higgs Audio generation completed")
                
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