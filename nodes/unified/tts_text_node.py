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
from utils.config_sanitizer import ConfigSanitizer
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
    CATEGORY = "TTS Audio Suite/🎤 Text to Speech"

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
            # Extract config from engine_data - it's nested under "config"
            config = engine_data.get("config", {})
            
            # FIX: The engine_data IS the config - no nested structure
            if not config:  # If config is empty, engine_data itself is the config
                config = engine_data
            
            # Create cache key based on stable parameters that affect engine instance creation
            # For VibeVoice, include chunk_minutes since it fundamentally changes behavior
            stable_params = {
                'engine_type': engine_type,
                'model': config.get('model'),
                'device': config.get('device'),
                'adapter_class': engine_data.get('adapter_class')
            }
            
            # For VibeVoice, include chunk_minutes in cache key as it overrides all chunking
            if engine_type == "vibevoice" and 'chunk_minutes' in config:
                stable_params['chunk_minutes'] = config.get('chunk_minutes', 0)
            
            # For VibeVoice, include attention_mode and quantization in cache key since they require model reload
            if engine_type == "vibevoice":
                stable_params['attention_mode'] = config.get('attention_mode', 'auto')
                stable_params['quantize_llm_4bit'] = config.get('quantize_llm_4bit', False)

            # For ChatterBox Official 23-Lang, include model_version in cache key since v1/v2 are different models
            if engine_type == "chatterbox_official_23lang":
                stable_params['model_version'] = config.get('model_version', 'v1')

            cache_key = f"{engine_type}_{hashlib.md5(str(sorted(stable_params.items())).encode()).hexdigest()[:8]}"
            
            # Cache key now properly includes model name for correct differentiation
            
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
                        print(f"🔄 Reusing cached {engine_type} engine instance (updated with new generation parameters)")
                        return cached_instance
                    else:
                        # Cache invalidated by model unloading, remove it
                        print(f"🗑️ Removing invalidated {engine_type} engine cache (models were unloaded)")
                        del self._cached_engine_instances[cache_key]
                else:
                    # Old format (direct instance) - assume invalid and remove
                    print(f"🗑️ Removing old-format {engine_type} engine cache (upgrading to timestamped format)")
                    del self._cached_engine_instances[cache_key]
            
            # print(f"🔧 Creating new {engine_type} engine instance")
            
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
                
                # Cache the instance with timestamp
                import time
                self._cached_engine_instances[cache_key] = {
                    'instance': engine_instance,
                    'timestamp': time.time()
                }
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
                class HiggsAudioWrapper:
                    def __init__(self, config):
                        self.config = config
                        # Don't cache adapter - create fresh each time to ensure config updates
                        # Store current model name for adapter caching
                        self.current_model_name = None
                    
                    def generate_tts_audio(self, text, char_audio, char_text, character="narrator", **params):
                        # Merge config with runtime params
                        merged_params = self.config.copy()
                        merged_params.update(params)
                        # Create fresh adapter instance with current config to ensure parameter updates
                        adapter = HiggsAudioEngineAdapter(self)
                        return adapter.generate_segment_audio(text, char_audio, char_text, character, **merged_params)
                
                engine_instance = HiggsAudioWrapper(config)
                # Cache the instance with timestamp
                import time
                self._cached_engine_instances[cache_key] = {
                    'instance': engine_instance,
                    'timestamp': time.time()
                }
                return engine_instance
                
            elif engine_type == "chatterbox_official_23lang":
                # Import and create the ChatterBox Official 23-Lang node using absolute import
                chatterbox_official_23lang_node_path = os.path.join(nodes_dir, "chatterbox_official_23lang", "chatterbox_official_23lang_processor.py")
                chatterbox_official_23lang_spec = importlib.util.spec_from_file_location("chatterbox_official_23lang_processor_module", chatterbox_official_23lang_node_path)
                chatterbox_official_23lang_module = importlib.util.module_from_spec(chatterbox_official_23lang_spec)
                chatterbox_official_23lang_spec.loader.exec_module(chatterbox_official_23lang_module)
                
                ChatterboxOfficial23LangTTSNode = chatterbox_official_23lang_module.ChatterboxOfficial23LangTTSNode
                engine_instance = ChatterboxOfficial23LangTTSNode()
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
                
            elif engine_type == "vibevoice":
                # Create a wrapper instance for VibeVoice using the adapter pattern
                # Import using same pattern as other modules
                vibevoice_processor_path = os.path.join(nodes_dir, "vibevoice", "vibevoice_processor.py")
                vibevoice_processor_spec = importlib.util.spec_from_file_location("vibevoice_processor_module", vibevoice_processor_path)
                vibevoice_processor_module = importlib.util.module_from_spec(vibevoice_processor_spec)
                vibevoice_processor_spec.loader.exec_module(vibevoice_processor_module)
                
                VibeVoiceProcessor = vibevoice_processor_module.VibeVoiceProcessor
                
                # Create a minimal wrapper node for the processor
                class VibeVoiceWrapper:
                    def __init__(self, config):
                        self.config = config
                        self.current_model_name = None
                        # Create processor instance with config
                        self.processor = VibeVoiceProcessor(self, config)
                    
                    def update_config(self, new_config):
                        """Update configuration and propagate to processor."""
                        self.config = new_config
                        self.processor.update_config(new_config)
                    
                    def generate_tts_audio(self, text, char_audio, char_text, character="narrator", **params):
                        # Parse characters from text first (like F5 does)
                        from utils.text.character_parser import parse_character_text
                        from utils.voice.discovery import get_character_mapping
                        
                        character_segments = parse_character_text(text, None)  # Auto-discover characters
                        characters = list(set(char for char, _ in character_segments))
                        print(f"🎭 VibeVoice: Character switching mode - found characters: {', '.join(characters)}")
                        
                        # Get character voice mapping (VibeVoice is audio-only, doesn't use reference text)
                        character_mapping = get_character_mapping(characters, engine_type="audio_only")
                        
                        # Build voice references with fallback to main voice (like F5 does)
                        voice_mapping = {}
                        for char in characters:
                            if char == "narrator" and char_audio:
                                # Use provided narrator voice for narrator character
                                voice_mapping[char] = char_audio
                            else:
                                # Try character-specific voice first, then fallback
                                audio_path, _ = character_mapping.get(char, (None, None))
                                if audio_path:
                                    # Load character-specific audio
                                    try:
                                        import torchaudio
                                        waveform, sample_rate = torchaudio.load(audio_path)
                                        voice_mapping[char] = {"waveform": waveform, "sample_rate": sample_rate}
                                        print(f"🎭 VibeVoice: Using character-specific voice for '{char}'")
                                    except Exception as e:
                                        print(f"⚠️ Failed to load character audio for '{char}': {e}")
                                        voice_mapping[char] = char_audio  # Fallback to main voice
                                        print(f"🔄 VibeVoice: Using main voice fallback for '{char}'")
                                else:
                                    # Fallback to main voice
                                    voice_mapping[char] = char_audio
                                    print(f"🔄 VibeVoice: Using main voice fallback for '{char}'")
                        
                        # print(f"🐛 VIBEVOICE_WRAPPER: generate_tts_audio called with character='{character}'")
                        # print(f"🐛 VIBEVOICE_WRAPPER: char_audio type: {type(char_audio)}")
                        # print(f"🐛 VIBEVOICE_WRAPPER: voice_mapping: {list(voice_mapping.keys())}")
                        
                        # Get seed from params
                        seed = params.get('seed', 42)
                        enable_chunking = params.get('enable_chunking', True)
                        max_chars = params.get('max_chars_per_chunk', 400)
                        
                        # Process text and generate audio
                        audio_segments = self.processor.process_text(
                            text, voice_mapping, seed, enable_chunking, max_chars
                        )
                        
                        # Combine segments
                        if audio_segments:
                            combined = self.processor.combine_audio_segments(
                                audio_segments,
                                params.get('chunk_combination_method', 'auto'),
                                params.get('silence_between_chunks_ms', 100)
                            )
                            
                            # Format as ComfyUI audio
                            audio_output = {
                                "waveform": combined,
                                "sample_rate": 24000
                            }
                            
                            # Generate info string
                            generation_info = f"✅ VibeVoice generation complete\n📊 Generated {len(audio_segments)} segment(s)\n🎯 Combined using {params.get('chunk_combination_method', 'auto')} method"
                            
                            return (audio_output, generation_info)
                            
                        # Return empty audio if no segments
                        empty_audio = {
                            "waveform": torch.zeros(1, 1, 0),
                            "sample_rate": 24000
                        }
                        return (empty_audio, "⚠️ No audio generated")
                
                engine_instance = VibeVoiceWrapper(config)
                # Cache the instance with timestamp
                import time
                self._cached_engine_instances[cache_key] = {
                    'instance': engine_instance,
                    'timestamp': time.time()
                }
                return engine_instance
                
            elif engine_type == "index_tts":
                # Create IndexTTS processor instance using the adapter pattern
                from engines.processors.index_tts_processor import IndexTTSProcessor
                
                engine_instance = IndexTTSProcessor(config)
                
                # Cache the instance with timestamp (follow VibeVoice pattern)
                import time
                self._cached_engine_instances[cache_key] = {
                    'instance': engine_instance,
                    'timestamp': time.time()
                }
                
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
                    # print(f"🐛 TTS_TEXT: Character Voices - character_name='{character_name}', has_audio={audio is not None}")
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
                # print(f"🐛 TTS_TEXT: Trying narrator_voice dropdown: {narrator_voice}")
                audio_path, reference_text = load_voice_reference(narrator_voice)
                # print(f"🐛 TTS_TEXT: Dropdown - audio_path={audio_path}, exists={os.path.exists(audio_path) if audio_path else False}")
                
                if audio_path and os.path.exists(audio_path):
                    # Load audio tensor
                    import torchaudio
                    waveform, sample_rate = torchaudio.load(audio_path)
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    audio_tensor = {"waveform": waveform, "sample_rate": sample_rate}
                    character_name = os.path.splitext(os.path.basename(narrator_voice))[0]
                    
                    print(f"🎤 TTS Text: Using voice reference from folder ({character_name})")
                    # print(f"🐛 TTS_TEXT: Dropdown loaded - character_name='{character_name}', waveform_shape={waveform.shape}")
                    return audio_path, audio_tensor, reference_text or "", character_name
            
            print("⚠️ TTS Text: No voice reference provided - this may cause issues with some engines")
            # print(f"🐛 TTS_TEXT: Final fallback - opt_narrator={opt_narrator is not None}, narrator_voice='{narrator_voice}'")
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
            
            
            if not engine_type:
                raise ValueError("TTS engine missing engine_type")
            
            # Get voice reference (opt_narrator takes priority)
            audio_path, audio_tensor, reference_text, character_name = self._get_voice_reference(opt_narrator, narrator_voice)
            
            # Get language for consistent logging
            language = config.get("language", "English")
            
            # Determine language code for display
            if language.startswith("local:"):
                # For local models, show "local" as the language code
                lang_code = "local"
            else:
                # Standard language codes - take first 2 chars
                lang_code = language.lower()[:2]  # en, fr, de, etc.
            
            char_display = character_name if character_name else "default"
            
            print(f"🎤 Generating {engine_type.title()} for '{char_display}' (lang: {lang_code})")
            
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
                    language=language,
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
                
            elif engine_type == "chatterbox_official_23lang":
                # ChatterBox Official 23-Lang TTS parameters - includes multilingual parameters
                result = engine_instance.generate_speech(
                    text=text,
                    language=language,
                    device=config.get("device", "auto"),
                    model_version=config.get("model_version", "v2"),
                    exaggeration=config.get("exaggeration", 0.5),
                    temperature=config.get("temperature", 0.8),
                    cfg_weight=config.get("cfg_weight", 0.5),
                    repetition_penalty=config.get("repetition_penalty", 2.0),
                    min_p=config.get("min_p", 0.05),
                    top_p=config.get("top_p", 1.0),
                    seed=seed,
                    reference_audio=audio_tensor,
                    audio_prompt_path=audio_path or "",
                    enable_chunking=enable_chunking,
                    max_chars_per_chunk=max_chars_per_chunk,
                    chunk_combination_method=chunk_combination_method,
                    silence_between_chunks_ms=silence_between_chunks_ms,
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
                    auto_phonemization=config.get("auto_phonemization", True),
                    enable_chunking=enable_chunking,
                    max_chars_per_chunk=max_chars_per_chunk,
                    chunk_combination_method=chunk_combination_method,
                    silence_between_chunks_ms=silence_between_chunks_ms,
                    enable_audio_cache=enable_audio_cache
                )
                
                
            elif engine_type == "higgs_audio":
                # Import and create the Higgs Audio TTS processor
                higgs_tts_processor_path = os.path.join(nodes_dir, "higgs_audio", "higgs_audio_tts_processor.py")
                higgs_tts_spec = importlib.util.spec_from_file_location("higgs_audio_tts_processor_module", higgs_tts_processor_path)
                higgs_tts_module = importlib.util.module_from_spec(higgs_tts_spec)
                higgs_tts_spec.loader.exec_module(higgs_tts_module)
                
                HiggsAudioTTSProcessor = higgs_tts_module.HiggsAudioTTSProcessor
                tts_processor = HiggsAudioTTSProcessor(engine_instance)
                
                # Clean delegation to TTS processor
                result = tts_processor.generate_tts_speech(
                    text=text,
                    multi_speaker_mode=engine_instance.config.get("multi_speaker_mode", "Custom Character Switching"),
                    audio_tensor=audio_tensor,
                    reference_text=reference_text,
                    seed=seed,
                    enable_audio_cache=enable_audio_cache,
                    max_chars_per_chunk=max_chars_per_chunk,
                    silence_between_chunks_ms=silence_between_chunks_ms
                )
                
            elif engine_type == "vibevoice":
                # VibeVoice uses the wrapper pattern - call directly through the wrapper's method
                result = engine_instance.generate_tts_audio(
                    text=text,
                    char_audio=audio_tensor,
                    char_text=reference_text,
                    character=char_display,
                    seed=seed,
                    enable_audio_cache=enable_audio_cache,
                    enable_chunking=enable_chunking,
                    max_chars_per_chunk=max_chars_per_chunk,
                    chunk_combination_method=chunk_combination_method,
                    silence_between_chunks_ms=silence_between_chunks_ms
                )
                
            elif engine_type == "index_tts":
                # IndexTTS-2 uses processor pattern - call through processor with emotion support
                audio_result = engine_instance.process_text(
                    text=text,
                    speaker_audio=audio_tensor,
                    reference_text=reference_text,
                    seed=seed,
                    enable_chunking=enable_chunking,
                    max_chars_per_chunk=max_chars_per_chunk,
                    silence_between_chunks_ms=silence_between_chunks_ms
                )
                
                # Format as ComfyUI audio format (processor returns tensor, we need dict)
                formatted_audio = AudioProcessingUtils.format_for_comfyui(audio_result, 22050)
                generation_info = f"✅ IndexTTS-2 generation complete\n🎭 Character switching and emotion support enabled"
                result = (formatted_audio, generation_info)
                
            else:
                raise ValueError(f"Unknown engine type: {engine_type}")
            
            # The original nodes return (audio, generation_info)
            audio_output, generation_info = result
            
            # Timing info is already included in generation_info from engines
            enhanced_info = generation_info
            
            # Add unified prefix to generation info
            unified_info = f"🎤 TTS Text (Unified) - {engine_type.upper()} Engine:\n{enhanced_info}"
            
            print(f"✅ {engine_type.title()} generation complete. Default narrator: {char_display}")
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