"""
F5-TTS Node - Basic text-to-speech generation
Enhanced Text-to-Speech node using F5-TTS with reference audio + text
"""

import torch
import numpy as np
import os
import hashlib
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

# Load f5tts_base_node module directly
f5tts_base_node_path = os.path.join(nodes_dir, "base", "f5tts_base_node.py")
f5tts_base_spec = importlib.util.spec_from_file_location("f5tts_base_node_module", f5tts_base_node_path)
f5tts_base_module = importlib.util.module_from_spec(f5tts_base_spec)
sys.modules["f5tts_base_node_module"] = f5tts_base_module
f5tts_base_spec.loader.exec_module(f5tts_base_module)

# Import the base class
BaseF5TTSNode = f5tts_base_module.BaseF5TTSNode

from utils.text.chunking import ImprovedChatterBoxChunker
from utils.audio.processing import AudioProcessingUtils
from utils.voice.discovery import get_available_voices, load_voice_reference, get_available_characters, get_character_mapping
from utils.text.character_parser import parse_character_text, character_parser
import comfy.model_management as model_management

# Global audio cache for F5-TTS segments
GLOBAL_AUDIO_CACHE = {}


class F5TTSNode(BaseF5TTSNode):
    """
    Enhanced F5-TTS text-to-speech generation node.
    Requires reference audio + text for voice cloning.
    Supports character switching using [Character] tags in text.
    """
    
    @classmethod
    def NAME(cls):
        return "🎤 F5-TTS Voice Generation"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available reference audio files from both models/voices/ and voices_examples/
        reference_files = get_available_voices()
        
        # Node layout with opt_reference_text as second widget
        base_types = {
            "required": {
                "reference_audio_file": (reference_files, {
                    "default": "none",
                    "tooltip": "Reference voice from models/voices/ or voices_examples/ folders (with companion .txt/.reference.txt file). Select 'none' to use direct inputs below."
                }),
                "opt_reference_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Direct reference text input (required when using opt_reference_audio)."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run F5-TTS model on. 'auto' selects best available (GPU if available, otherwise CPU)."
                }),
                "model": (cls.get_available_models_for_dropdown(), {
                    "default": "F5TTS_v1_Base",
                    "tooltip": "F5-TTS model variant to use. F5TTS_Base is the standard model, F5TTS_v1_Base is improved version, E2TTS_Base is enhanced variant."
                }),
                "seed": ("INT", {
                    "default": 1, "min": 0, "max": 2**32 - 1,
                    "tooltip": "Seed for reproducible F5-TTS generation. Same seed with same inputs will produce identical results. Set to 0 for random generation."
                }),
                "text": ("STRING", {
                    "multiline": True,
                    "default": """Hello! This is F5-TTS with character switching support.
[Alice] Hi there! I'm Alice speaking with my voice.
[Bob] And I'm Bob! Nice to meet you both.
Back to the main narrator voice for the conclusion.""",
                    "tooltip": "Text to convert to speech. Use [Character] tags for voice switching. Characters not found in voice folders will use the main reference voice."
                }),
            },
            "optional": {
                "opt_reference_audio": ("AUDIO", {
                    "tooltip": "Direct reference audio input (used when reference_audio_file is 'none')"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1,
                    "tooltip": "Controls randomness in F5-TTS generation. Higher values = more creative/varied speech, lower values = more consistent/predictable speech."
                }),
                "speed": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1,
                    "tooltip": "F5-TTS native speech speed control. 1.0 = normal speed, 0.5 = half speed (slower), 2.0 = double speed (faster)."
                }),
                "target_rms": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Target audio volume level (Root Mean Square). Controls output loudness normalization. Higher values = louder audio output."
                }),
                "cross_fade_duration": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Duration in seconds for smooth audio transitions between F5-TTS segments. Prevents audio clicks/pops by blending segment boundaries."
                }),
                "nfe_step": ("INT", {
                    "default": 32, "min": 1, "max": 71,
                    "tooltip": "Neural Function Evaluation steps for F5-TTS inference. Higher values = better quality but slower generation. 32 is a good balance. Values above 71 may cause ODE solver issues."
                }),
                "cfg_strength": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Speech generation control. Lower values (1.0-1.5) = more natural, conversational delivery. Higher values (3.0-5.0) = crisper, more articulated speech with stronger emphasis. Default 2.0 balances naturalness and clarity."
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
                "auto_phonemization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "🦜 Enable automatic phonemization for multilingual models. When enabled, non-English text is converted to IPA phonemes. Disable if quality is poor for your language model."
                }),
            }
        }
        
        return base_types

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "generation_info")
    FUNCTION = "generate_speech"
    CATEGORY = "TTS Audio Suite/👄 F5-TTS"

    def __init__(self):
        super().__init__()

    
    @staticmethod
    def _get_companion_txt_file(audio_file_path):
        """Get the path to companion .txt file for an audio file (legacy method)"""
        from pathlib import Path
        p = Path(audio_file_path)
        return os.path.join(os.path.dirname(audio_file_path), p.stem + ".txt")
    
    def _load_reference_from_file(self, reference_audio_file):
        """Load reference audio and text from voice discovery system"""
        if reference_audio_file == "none":
            return None, None
        
        # Use the new voice discovery system
        audio_path, ref_text = load_voice_reference(reference_audio_file)
        
        if not audio_path or not ref_text:
            raise FileNotFoundError(f"Reference voice '{reference_audio_file}' not found or has no companion text file")
        
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Reference audio file not found: {audio_path}")
        
        return audio_path, ref_text
    
    def _handle_reference_with_priority_chain(self, inputs):
        """Handle reference audio and text with improved priority chain"""
        reference_audio_file = inputs.get("reference_audio_file", "none")
        opt_reference_audio = inputs.get("opt_reference_audio")
        opt_reference_text = inputs.get("opt_reference_text", "").strip()
        
        # PRIORITY 1: Check reference_audio_file first
        if reference_audio_file != "none":
            try:
                audio_path, auto_ref_text = self._load_reference_from_file(reference_audio_file)
                if audio_path and auto_ref_text:
                    print(f"✅ F5-TTS: Using reference file '{reference_audio_file}' with auto-detected text")
                    return audio_path, auto_ref_text
            except Exception as e:
                print(f"⚠️ F5-TTS: Failed to load reference file '{reference_audio_file}': {e}")
                print("🔄 F5-TTS: Falling back to manual inputs...")
        
        # PRIORITY 2: Use opt_reference_audio + opt_reference_text (both required)
        if opt_reference_audio is not None:
            # Handle the audio input to get file path
            audio_prompt = self.handle_reference_audio(opt_reference_audio, "")
            
            if audio_prompt:
                # Check if opt_reference_text is provided
                if opt_reference_text and opt_reference_text.strip():
                    print(f"📝 F5-TTS: Using direct reference audio + text inputs")
                    return audio_prompt, opt_reference_text.strip()
                
                # Error - audio provided but no text
                raise ValueError(
                    "F5-TTS requires reference text. Please connect text to opt_reference_text input."
                )
        
        # FINAL: No reference inputs provided at all
        raise ValueError(
            "F5-TTS requires reference audio and text. Please provide either:\n"
            "1. Select a reference_audio_file with companion .txt file, OR\n"
            "2. Connect opt_reference_audio input and provide opt_reference_text"
        )
    
    def validate_inputs(self, **inputs) -> Dict[str, Any]:
        """Validate and normalize inputs."""
        # Call the base class validate_inputs directly (BaseChatterBoxNode)
        validated = super(BaseF5TTSNode, self).validate_inputs(**inputs)
        
        # Skip the F5-TTS base validation since we handle reference validation in our priority chain
        # Don't call validate_f5tts_inputs since it expects ref_text which we don't have
        
        # Handle None/empty values for backward compatibility
        if validated.get("enable_chunking") is None:
            validated["enable_chunking"] = True
        if validated.get("max_chars_per_chunk") is None or validated.get("max_chars_per_chunk", 0) < 100:
            validated["max_chars_per_chunk"] = 400
        if not validated.get("chunk_combination_method"):
            validated["chunk_combination_method"] = "auto"
        if validated.get("silence_between_chunks_ms") is None:
            validated["silence_between_chunks_ms"] = 100
        
        # Validate model name if provided
        model_name = validated.get("model", "F5TTS_Base")
        try:
            available_models = self._get_available_models()
            if model_name not in available_models:
                print(f"⚠️ F5-TTS: Model '{model_name}' not in available list, but will attempt to load")
        except:
            pass  # Don't fail validation on model check
        
        return validated
    
    def _get_available_models(self):
        """Get list of available F5-TTS models"""
        try:
            from engines.f5tts import get_f5tts_models
            return get_f5tts_models()
        except ImportError:
            try:
                from engines.f5tts import get_f5tts_models
                return get_f5tts_models()
            except ImportError:
                return ["F5TTS_Base", "F5TTS_v1_Base", "E2TTS_Base"]
    
    def _generate_stable_audio_component(self, reference_audio_file: str, opt_reference_audio, audio_prompt_path: str) -> str:
        """Generate stable identifier for audio prompt to prevent cache invalidation from temp file paths."""
        if opt_reference_audio is not None:
            waveform_hash = hashlib.md5(opt_reference_audio["waveform"].cpu().numpy().tobytes()).hexdigest()
            return f"ref_audio_{waveform_hash}_{opt_reference_audio['sample_rate']}"
        elif reference_audio_file != "none":
            return f"ref_file_{reference_audio_file}"
        elif audio_prompt_path:
            return audio_prompt_path
        else:
            return ""

    def _generate_segment_cache_key(self, text: str, model_name: str, device: str,
                                   audio_component: str, ref_text: str, temperature: float,
                                   speed: float, target_rms: float, cross_fade_duration: float,
                                   nfe_step: int, cfg_strength: float, seed: int, character: str = "narrator", 
                                   auto_phonemization: bool = True) -> str:
        """Generate cache key for a single F5-TTS audio segment based on generation parameters."""
        cache_data = {
            'text': text,
            'model_name': model_name,
            'device': device,
            'audio_component': audio_component,
            'ref_text': ref_text,
            'temperature': temperature,
            'speed': speed,
            'target_rms': target_rms,
            'cross_fade_duration': cross_fade_duration,
            'nfe_step': nfe_step,
            'cfg_strength': cfg_strength,
            'seed': seed,
            'character': character,
            'auto_phonemization': auto_phonemization,
            'engine': 'f5tts'
        }
        cache_string = str(sorted(cache_data.items()))
        cache_key = hashlib.md5(cache_string.encode()).hexdigest()
        
        # Debug: Print cache key info for phonemization debugging
        if text and ('phonemization' in text.lower() or len(text) < 50):  # Only for short text or debug
            print(f"🔍 Cache key for auto_phonemization={auto_phonemization}: {cache_key[:8]}... (text: '{text[:30]}')")
            print(f"   Parameters: temp={temperature}, speed={speed}, nfe={nfe_step}, phonem={auto_phonemization}")
        return cache_key

    def _get_cached_segment_audio(self, segment_cache_key: str) -> Optional[Tuple[torch.Tensor, float]]:
        """Retrieve cached audio for a single segment if available from global cache."""
        return GLOBAL_AUDIO_CACHE.get(segment_cache_key)

    def _cache_segment_audio(self, segment_cache_key: str, audio_tensor: torch.Tensor, natural_duration: float):
        """Cache generated audio for a single segment in global cache."""
        GLOBAL_AUDIO_CACHE[segment_cache_key] = (audio_tensor.clone(), natural_duration)
    
    def generate_speech(self, reference_audio_file, text, device, model, seed,
                       opt_reference_audio=None, opt_reference_text="",
                       audio_prompt_path="", enable_chunking=True, max_chars_per_chunk=400,
                       chunk_combination_method="auto", silence_between_chunks_ms=100,
                       temperature=0.8, speed=1.0, target_rms=0.1,
                       cross_fade_duration=0.15, nfe_step=32, cfg_strength=2.0, 
                       enable_audio_cache=True, auto_phonemization=True):
        
        # Store phonemization setting for direct parameter passing
        self._current_auto_phonemization = auto_phonemization
        
        # Normalize model name for backward compatibility (case-insensitive matching)
        # Convert V1, V2, etc. to v1, v2 for consistency
        import re
        model = re.sub(r'_V(\d+)_', r'_v\1_', model)
        
        def _process():
            # Import PauseTagProcessor at the top to avoid scoping issues
            from utils.text.pause_processor import PauseTagProcessor
            
            # Validate inputs
            inputs = self.validate_inputs(
                reference_audio_file=reference_audio_file, text=text, device=device, model=model, seed=seed,
                opt_reference_audio=opt_reference_audio, opt_reference_text=opt_reference_text,
                audio_prompt_path=audio_prompt_path, enable_chunking=enable_chunking,
                max_chars_per_chunk=max_chars_per_chunk, chunk_combination_method=chunk_combination_method,
                silence_between_chunks_ms=silence_between_chunks_ms, temperature=temperature,
                speed=speed, target_rms=target_rms, cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step, cfg_strength=cfg_strength, enable_audio_cache=enable_audio_cache
            )
            
            # Set seed for reproducibility (can be done without loading model)
            self.set_seed(inputs["seed"])
            
            # Handle reference audio and text with priority chain
            main_audio_prompt, main_ref_text = self._handle_reference_with_priority_chain(inputs)
            
            # Generate stable audio component for cache consistency
            stable_audio_component = self._generate_stable_audio_component(
                inputs["reference_audio_file"], inputs.get("opt_reference_audio"), 
                inputs.get("audio_prompt_path", "")
            )
            
            # Set up character parser with available characters BEFORE parsing
            available_chars = get_available_characters()
            character_parser.set_available_characters(list(available_chars))
            
            # Reset session cache to allow fresh logging for new generation
            character_parser.reset_session_cache()
            
            # Set engine-aware default language to prevent unnecessary model switching
            character_parser.set_engine_aware_default_language(inputs["model"], "f5tts")
            
            # Parse character segments from text with language awareness and parameter support
            # NOTE: We parse characters and languages from original text, then handle pause tags within each segment
            # Use parse_text_segments to get parameters, then convert to (char, text, lang, params) format
            character_segment_objects = character_parser.parse_text_segments(inputs["text"])
            character_segments_with_lang = [(seg.character, seg.text, seg.language, seg.parameters if seg.parameters else {}) for seg in character_segment_objects]
            
            # Check if we have character switching or language switching
            characters = list(set(char for char, _, _, _ in character_segments_with_lang))
            languages = list(set(lang for _, _, lang, _ in character_segments_with_lang))
            has_multiple_characters = len(characters) > 1 or (len(characters) == 1 and characters[0] != "narrator")
            has_multiple_languages = len(languages) > 1
            # Check if ANY segment has parameters (for parameter-only tags like [seed:42])
            has_segment_parameters = any(params for _, _, _, params in character_segments_with_lang)

            # Create backward-compatible character segments for existing logic
            character_segments = [(char, segment_text) for char, segment_text, _, _ in character_segments_with_lang]

            # Validate and clamp nfe_step to prevent ODE solver issues
            safe_nfe_step = max(1, min(inputs["nfe_step"], 71))
            if safe_nfe_step != inputs["nfe_step"]:
                print(f"⚠️ F5-TTS: Clamped nfe_step from {inputs['nfe_step']} to {safe_nfe_step} to prevent ODE solver issues")

            if has_multiple_characters or has_multiple_languages or has_segment_parameters:
                # CHARACTER AND/OR LANGUAGE SWITCHING MODE
                if has_multiple_languages:
                    print(f"🌍 F5-TTS: Language switching mode - found languages: {', '.join(languages)}")
                if has_multiple_characters:
                    print(f"🎭 F5-TTS: Character switching mode - found characters: {', '.join(characters)}")
                
                # Get character voice mapping
                character_mapping = get_character_mapping(characters, engine_type="f5tts")
                
                # Build voice references with fallback to main voice
                voice_refs = {}
                character_voices = []
                main_voices = []
                
                for character in characters:
                    # PRIORITY 1: If a narrator voice was provided, ALWAYS use it for "narrator" character
                    # This ensures language-only tags like [fr:] use the selected narrator voice, not character aliases
                    if character == "narrator" and main_audio_prompt and main_ref_text:
                        voice_refs[character] = (main_audio_prompt, main_ref_text)
                        main_voices.append(character)
                        print(f"✅ Using selected narrator voice for language-only tag (priority over character map)")
                    else:
                        # PRIORITY 2: For other characters (non-narrator), try character mapping first
                        audio_path, ref_text = character_mapping.get(character, (None, None))
                        if audio_path and ref_text:
                            voice_refs[character] = (audio_path, ref_text)
                            character_voices.append(character)
                        else:
                            # PRIORITY 3: Fallback to provided narrator voice if available
                            if main_audio_prompt and main_ref_text:
                                voice_refs[character] = (main_audio_prompt, main_ref_text)
                                main_voices.append(character)
                            else:
                                # No voice available - this will cause an error later
                                print(f"⚠️ No voice reference available for character '{character}'")
                                voice_refs[character] = (None, None)
                
                # Consolidated voice summary logging
                voice_summary = []
                if character_voices:
                    voice_summary.append(f"character voices: {', '.join(character_voices)}")
                if main_voices:
                    voice_summary.append(f"main voice: {', '.join(main_voices)}")
                
                if voice_summary:
                    print(f"🎭 Voice mapping - {' | '.join(voice_summary)}")
                
                # Map language codes to F5-TTS model names
                def get_f5tts_model_for_language(lang_code: str) -> str:
                    """Map language codes to F5-TTS model names"""
                    lang_model_map = {
                        'en': 'F5TTS_v1_Base',  # English (use v1 - better quality)
                        'de': 'F5-DE',         # German
                        'es': 'F5-ES',         # Spanish  
                        'fr': 'F5-FR',         # French
                        'jp': 'F5-JP',         # Japanese
                        'ja': 'F5-JP',         # Japanese (alternative code)
                        'it': 'F5-IT',         # Italian
                        'th': 'F5-TH',         # Thai
                        'pt': 'F5-PT-BR',      # Portuguese (Brazilian)
                        'pt-br': 'F5-PT-BR',   # Portuguese (Brazilian)
                        'pt-pt': 'F5-PT-BR',   # Portuguese (European - use Brazilian model for now)
                        'hi': 'F5-Hindi-Small', # Hindi (uses Small model from IIT Madras)
                        # Note: Indian languages (as, bn, gu, kn, ml, mr, or, pa, ta, te) 
                        # fall back to base F5TTS models - IndicF5 was removed due to architecture incompatibility
                    }
                    # Use appropriate model for detected language (supports both explicit tags and character alias language switching)
                    return lang_model_map.get(lang_code.lower(), inputs["model"])
                
                # Group segments by language with original order tracking (including parameters)
                language_groups = {}
                for idx, (char, segment_text, lang, segment_params) in enumerate(character_segments_with_lang):
                    if lang not in language_groups:
                        language_groups[lang] = []
                    language_groups[lang].append((idx, char, segment_text, lang, segment_params))  # Include original index and parameters
                
                # Generate audio for each language group, tracking original positions
                audio_segments_with_order = []  # Will store (original_index, audio_tensor)
                total_segments = len(character_segments_with_lang)
                
                # Track if base model has been loaded
                base_model_loaded = False
                
                # Process each language group with appropriate model
                for lang_code, lang_segments in language_groups.items():
                    # Check if this language group contains narrator segments
                    has_narrator_segments = any(character == "narrator" for _, character, _, _, _ in lang_segments)
                    
                    if has_narrator_segments and len([seg for seg in lang_segments if seg[1] == "narrator"]) == len(lang_segments):
                        # Pure narrator language group - use selected base model
                        required_model = inputs["model"]
                    else:
                        # Mixed or character-only group - use language mapping for their specific languages
                        required_model = get_f5tts_model_for_language(lang_code)
                    
                    # Check if ALL segments in this language group are cached
                    all_segments_cached = True
                    if inputs.get("enable_audio_cache", True):
                        for original_idx, character, segment_text, segment_lang, segment_params in lang_segments:
                            # Get voice reference for this character
                            char_audio, char_text = voice_refs[character]
                            
                            # Apply chunking to check cache for each chunk
                            if inputs["enable_chunking"] and len(segment_text) > inputs["max_chars_per_chunk"]:
                                segment_chunks = self.chunker.split_into_chunks(segment_text, inputs["max_chars_per_chunk"])
                            else:
                                segment_chunks = [segment_text]
                            
                            # Check cache for each chunk, accounting for pause tag processing
                            for chunk_text in segment_chunks:
                                # Apply pause tag processing to see what text segments will actually be generated
                                processed_text, pause_segments = PauseTagProcessor.preprocess_text_with_pause_tags(chunk_text, True)
                                
                                if pause_segments is None:
                                    # No pause tags, check cache for the full processed text
                                    cache_texts = [processed_text]
                                else:
                                    # Extract only text segments (not pause segments) 
                                    cache_texts = [content for segment_type, content in pause_segments if segment_type == 'text']
                                
                                # Check cache for each text segment that will be generated
                                char_audio_component = stable_audio_component if character == "narrator" else f"char_file_{character}"
                                for cache_text in cache_texts:
                                    cache_key = self._generate_segment_cache_key(
                                        f"{character}:{cache_text}", required_model, inputs["device"], 
                                        char_audio_component, char_text, inputs["temperature"], inputs["speed"],
                                        inputs["target_rms"], inputs["cross_fade_duration"], 
                                        safe_nfe_step, inputs["cfg_strength"], inputs["seed"], character,
                                        auto_phonemization
                                    )
                                    cached_data = self._get_cached_segment_audio(cache_key)
                                    if not cached_data:
                                        all_segments_cached = False
                                        break
                                if not all_segments_cached:
                                    break
                            if not all_segments_cached:
                                break
                    else:
                        all_segments_cached = False
                    
                    # Only load model if we need to generate new audio
                    if not all_segments_cached:
                        # Load base model if it's not the currently loaded model
                        current_model = getattr(self, 'current_model_name', None)
                        if current_model != inputs["model"]:
                            print(f"🔄 Loading base F5-TTS model '{inputs['model']}' for generation")
                            self.load_f5tts_model(inputs["model"], inputs["device"])
                            base_model_loaded = True
                        elif not base_model_loaded:
                            # Model matches but base_model_loaded flag not set
                            base_model_loaded = True
                        
                        print(f"🌍 Loading F5-TTS model '{required_model}' for language '{lang_code}' ({len(lang_segments)} segments)")
                        try:
                            self.load_f5tts_model(required_model, inputs["device"])
                        except Exception as e:
                            print(f"⚠️ Failed to load model '{required_model}' for language '{lang_code}': {e}")
                            print(f"🔄 Falling back to default model '{inputs['model']}'")
                            
                            # Check if default model is already loaded before reloading
                            current_model = getattr(self, 'current_model_name', None)
                            if current_model == inputs["model"]:
                                print(f"💾 Default model '{inputs['model']}' already loaded - reusing for fallback")
                            else:
                                self.load_f5tts_model(inputs["model"], inputs["device"])
                    else:
                        print(f"💾 Skipping model load for language '{lang_code}' - all {len(lang_segments)} segments cached")
                    
                    # Process each segment in this language group
                    for original_idx, character, segment_text, segment_lang, segment_params in lang_segments:
                        segment_display_idx = original_idx + 1  # For display (1-based)

                        # Check for interruption
                        self.check_interruption(f"F5-TTS generation segment {segment_display_idx}/{total_segments} (lang: {lang_code})")

                        # Apply per-segment parameters
                        current_config = dict(inputs)
                        if segment_params:
                            from utils.text.segment_parameters import apply_segment_parameters
                            current_config = apply_segment_parameters(current_config, segment_params, "f5tts")
                            print(f"📊 Segment {segment_display_idx}: Character '{character}' with parameters {segment_params}")

                        # Apply chunking to long segments if enabled
                        if current_config["enable_chunking"] and len(segment_text) > current_config["max_chars_per_chunk"]:
                            segment_chunks = self.chunker.split_into_chunks(segment_text, current_config["max_chars_per_chunk"])
                        else:
                            segment_chunks = [segment_text]

                        # Get voice reference for this character
                        char_audio, char_text = voice_refs[character]
                        
                        # Generate audio for each chunk of this character segment
                        segment_audio_chunks = []
                        for chunk_i, chunk_text in enumerate(segment_chunks):
                            print(f"🎤 Generating F5-TTS segment {segment_display_idx}/{total_segments} chunk {chunk_i+1}/{len(segment_chunks)} for '{character}' (lang: {lang_code})...")
                            
                            # Create cache function for this character if caching is enabled (uses per-segment parameters)
                            cache_fn = None
                            if current_config.get("enable_audio_cache", True):
                                def create_cache_fn(char_name, char_audio_comp, char_ref_text):
                                    def cache_fn_impl(text_content: str, audio_result=None):
                                        cache_key = self._generate_segment_cache_key(
                                            f"{char_name}:{text_content}", required_model, current_config["device"],
                                            char_audio_comp, char_ref_text, current_config["temperature"], current_config["speed"],
                                            current_config["target_rms"], current_config["cross_fade_duration"],
                                            safe_nfe_step, current_config["cfg_strength"], current_config["seed"], char_name,
                                            auto_phonemization
                                        )
                                        if audio_result is None:
                                            # Get from cache
                                            cached_data = self._get_cached_segment_audio(cache_key)
                                            if cached_data:
                                                print(f"💾 CACHE HIT for character '{char_name}': '{text_content[:30]}...'")
                                                return cached_data[0]
                                            return None
                                        else:
                                            # Store in cache
                                            char_duration = char_audio.size(-1) / self.f5tts_sample_rate if hasattr(char_audio, 'size') else 0.0
                                            self._cache_segment_audio(cache_key, audio_result, char_duration)
                                    return cache_fn_impl

                                # Determine character audio component for cache key
                                char_audio_component = stable_audio_component if character == "narrator" else f"char_file_{character}"
                                cache_fn = create_cache_fn(character, char_audio_component, char_text)

                            chunk_audio = self.generate_f5tts_with_pause_tags(
                                text=chunk_text,
                                ref_audio_path=char_audio,
                                ref_text=char_text,
                                enable_pause_tags=True,
                                character=character,
                                seed=current_config["seed"],
                                enable_cache=current_config.get("enable_audio_cache", True),
                                cache_fn=cache_fn,
                                temperature=current_config["temperature"],
                                speed=current_config["speed"],
                                target_rms=current_config["target_rms"],
                                cross_fade_duration=current_config["cross_fade_duration"],
                                nfe_step=safe_nfe_step,
                                cfg_strength=current_config["cfg_strength"],
                                auto_phonemization=auto_phonemization
                            )
                            segment_audio_chunks.append(chunk_audio)
                        
                        # Combine chunks for this segment and store with original order
                        if segment_audio_chunks:
                            if len(segment_audio_chunks) == 1:
                                segment_audio = segment_audio_chunks[0]
                            else:
                                segment_audio = torch.cat(segment_audio_chunks, dim=-1)
                            audio_segments_with_order.append((original_idx, segment_audio))
                
                # Sort audio segments back to original order
                audio_segments_with_order.sort(key=lambda x: x[0])  # Sort by original index
                audio_segments = [audio for _, audio in audio_segments_with_order]  # Extract audio tensors
                
                # Combine all character segments with timing info
                wav, chunk_info = self.combine_f5tts_audio_chunks(
                    audio_segments, inputs["chunk_combination_method"], 
                    inputs["silence_between_chunks_ms"], len(inputs["text"]),
                    original_text=inputs["text"], text_chunks=None, return_info=True
                )
                
                # Generate info
                total_duration = wav.size(-1) / self.f5tts_sample_rate
                model_info = self.get_f5tts_model_info()
                
                language_info = ""
                if has_multiple_languages:
                    language_info = f" across {len(languages)} languages ({', '.join(languages)})"
                
                base_info = f"Generated {total_duration:.1f}s audio from {len(character_segments)} segments using {len(characters)} characters{language_info} (F5-TTS {model_info.get('model_name', 'unknown')})"
                
                # Add chunk timing info if available
                from utils.audio.chunk_timing import ChunkTimingHelper
                info = ChunkTimingHelper.enhance_generation_info(base_info, chunk_info)
                
            else:
                # SINGLE CHARACTER MODE (original behavior)
                text_length = len(inputs["text"])
                
                # Check if single character content is cached before loading model
                single_content_cached = False
                if inputs.get("enable_audio_cache", True):
                    if not inputs["enable_chunking"] or text_length <= inputs["max_chars_per_chunk"]:
                        # Check cache for single chunk
                        processed_text, pause_segments = PauseTagProcessor.preprocess_text_with_pause_tags(inputs["text"], True)
                        
                        if pause_segments is None:
                            cache_texts = [processed_text]
                        else:
                            cache_texts = [content for segment_type, content in pause_segments if segment_type == 'text']
                        
                        single_content_cached = True
                        for cache_text in cache_texts:
                            cache_key = self._generate_segment_cache_key(
                                f"narrator:{cache_text}", inputs["model"], inputs["device"], 
                                stable_audio_component, main_ref_text, inputs["temperature"], inputs["speed"],
                                inputs["target_rms"], inputs["cross_fade_duration"], 
                                safe_nfe_step, inputs["cfg_strength"], inputs["seed"], "narrator",
                                auto_phonemization
                            )
                            cached_data = self._get_cached_segment_audio(cache_key)
                            if not cached_data:
                                single_content_cached = False
                                break
                    else:
                        # Check cache for multiple chunks
                        # BUGFIX: Clean character tags from text before chunking in single character mode
                        clean_text = character_parser.remove_character_tags(inputs["text"])
                        chunks = self.chunker.split_into_chunks(clean_text, inputs["max_chars_per_chunk"])
                        single_content_cached = True
                        for chunk in chunks:
                            processed_text, pause_segments = PauseTagProcessor.preprocess_text_with_pause_tags(chunk, True)
                            
                            if pause_segments is None:
                                cache_texts = [processed_text]
                            else:
                                cache_texts = [content for segment_type, content in pause_segments if segment_type == 'text']
                            
                            for cache_text in cache_texts:
                                cache_key = self._generate_segment_cache_key(
                                    f"narrator:{cache_text}", inputs["model"], inputs["device"], 
                                    stable_audio_component, main_ref_text, inputs["temperature"], inputs["speed"],
                                    inputs["target_rms"], inputs["cross_fade_duration"], 
                                    safe_nfe_step, inputs["cfg_strength"], inputs["seed"], "narrator",
                                    auto_phonemization
                                )
                                cached_data = self._get_cached_segment_audio(cache_key)
                                if not cached_data:
                                    single_content_cached = False
                                    break
                            if not single_content_cached:
                                break
                
                # Only load model if we need to generate something
                if not single_content_cached:
                    print(f"🔄 Loading F5-TTS model '{inputs['model']}' for single character generation")
                    self.load_f5tts_model(inputs["model"], inputs["device"])
                else:
                    print(f"💾 All single character content cached - skipping model loading")
                
                if not inputs["enable_chunking"] or text_length <= inputs["max_chars_per_chunk"]:
                    # Process single chunk with caching
                    # Create cache function for narrator if caching is enabled
                    cache_fn = None
                    if inputs.get("enable_audio_cache", True):
                        def narrator_cache_fn(text_content: str, audio_result=None):
                            cache_key = self._generate_segment_cache_key(
                                f"narrator:{text_content}", inputs["model"], inputs["device"], 
                                stable_audio_component, main_ref_text, inputs["temperature"], inputs["speed"],
                                inputs["target_rms"], inputs["cross_fade_duration"], 
                                safe_nfe_step, inputs["cfg_strength"], inputs["seed"], "narrator",
                                auto_phonemization
                            )
                            if audio_result is None:
                                # Get from cache
                                cached_data = self._get_cached_segment_audio(cache_key)
                                if cached_data:
                                    print(f"💾 CACHE HIT for narrator: '{text_content[:30]}...'")
                                    return cached_data[0]
                                return None
                            else:
                                # Store in cache
                                audio_duration = audio_result.size(-1) / self.f5tts_sample_rate if hasattr(audio_result, 'size') else 0.0
                                self._cache_segment_audio(cache_key, audio_result, audio_duration)
                        cache_fn = narrator_cache_fn
                    
                    # BUGFIX: Clean character tags from text even in single character mode
                    clean_text = character_parser.remove_character_tags(inputs["text"])
                    wav = self.generate_f5tts_with_pause_tags(
                        text=clean_text,
                        ref_audio_path=main_audio_prompt,
                        ref_text=main_ref_text,
                        enable_pause_tags=True,
                        character="narrator",
                        seed=inputs["seed"],
                        enable_cache=inputs.get("enable_audio_cache", True),
                        cache_fn=cache_fn,
                        temperature=inputs["temperature"],
                        speed=inputs["speed"],
                        target_rms=inputs["target_rms"],
                        cross_fade_duration=inputs["cross_fade_duration"],
                        nfe_step=safe_nfe_step,
                        cfg_strength=inputs["cfg_strength"]
                    )
                    model_info = self.get_f5tts_model_info()
                    info = f"Generated {wav.size(-1) / self.f5tts_sample_rate:.1f}s audio from {text_length} characters (single chunk, F5-TTS {model_info.get('model_name', 'unknown')})"
                else:
                    # Split into chunks using improved chunker
                    # BUGFIX: Clean character tags from text before chunking in single character mode
                    clean_text = character_parser.remove_character_tags(inputs["text"])
                    chunks = self.chunker.split_into_chunks(clean_text, inputs["max_chars_per_chunk"])
                    
                    # Create cache function for narrator chunks if caching is enabled
                    cache_fn = None
                    if inputs.get("enable_audio_cache", True):
                        def chunk_cache_fn(text_content: str, audio_result=None):
                            cache_key = self._generate_segment_cache_key(
                                f"narrator:{text_content}", inputs["model"], inputs["device"], 
                                stable_audio_component, main_ref_text, inputs["temperature"], inputs["speed"],
                                inputs["target_rms"], inputs["cross_fade_duration"], 
                                safe_nfe_step, inputs["cfg_strength"], inputs["seed"], "narrator",
                                auto_phonemization
                            )
                            if audio_result is None:
                                # Get from cache
                                cached_data = self._get_cached_segment_audio(cache_key)
                                if cached_data:
                                    print(f"💾 CACHE HIT for narrator chunk: '{text_content[:30]}...'")
                                    return cached_data[0]
                                return None
                            else:
                                # Store in cache
                                audio_duration = audio_result.size(-1) / self.f5tts_sample_rate if hasattr(audio_result, 'size') else 0.0
                                self._cache_segment_audio(cache_key, audio_result, audio_duration)
                        cache_fn = chunk_cache_fn
                    
                    # Process each chunk
                    audio_segments = []
                    for i, chunk in enumerate(chunks):
                        # Check for interruption
                        self.check_interruption(f"F5-TTS generation chunk {i+1}/{len(chunks)}")
                        
                        # Show progress for multi-chunk generation
                        print(f"🎤 Generating F5-TTS chunk {i+1}/{len(chunks)}...")
                        
                        chunk_audio = self.generate_f5tts_with_pause_tags(
                            text=chunk,
                            ref_audio_path=main_audio_prompt,
                            ref_text=main_ref_text,
                            enable_pause_tags=True,
                            character="narrator",
                            seed=inputs["seed"],
                            enable_cache=inputs.get("enable_audio_cache", True),
                            cache_fn=cache_fn,
                            temperature=inputs["temperature"],
                            speed=inputs["speed"],
                            target_rms=inputs["target_rms"],
                            cross_fade_duration=inputs["cross_fade_duration"],
                            nfe_step=safe_nfe_step,
                            cfg_strength=inputs["cfg_strength"],
                            auto_phonemization=self._current_auto_phonemization
                        )
                        audio_segments.append(chunk_audio)
                    
                    # Combine audio segments with timing info
                    wav, chunk_info = self.combine_f5tts_audio_chunks(
                        audio_segments, inputs["chunk_combination_method"], 
                        inputs["silence_between_chunks_ms"], text_length,
                        original_text=inputs["text"], text_chunks=None, return_info=True
                    )
                    
                    # Generate info
                    total_duration = wav.size(-1) / self.f5tts_sample_rate
                    avg_chunk_size = text_length // len(chunks)
                    model_info = self.get_f5tts_model_info()
                    base_info = f"Generated {total_duration:.1f}s audio from {text_length} characters using {len(chunks)} chunks (avg {avg_chunk_size} chars/chunk, F5-TTS {model_info.get('model_name', 'unknown')})"
                    
                    # Add chunk timing info if available
                    from utils.audio.chunk_timing import ChunkTimingHelper
                    info = ChunkTimingHelper.enhance_generation_info(base_info, chunk_info)
            
            # Return audio in ComfyUI format
            return (
                self.format_f5tts_audio_output(wav),
                info
            )
        
        return _process()
    
    def _process_single_segment_for_streaming(self, original_idx, character, segment_text, 
                                               language, voice_path, inputs):
        """
        Process a single segment for the streaming processor.
        
        This method enables F5-TTS to work with the universal streaming system,
        preserving all existing functionality including pause tags, chunking, and caching.
        
        Args:
            original_idx: Original segment index for result ordering
            character: Character name for this segment
            segment_text: Text to generate
            language: Language code for this segment
            voice_path: Path to voice reference file
            inputs: Processing parameters dict
            
        Returns:
            Generated audio tensor
        """
        try:
            from utils.text.pause_processor import PauseTagProcessor
            
            # Extract parameters from inputs
            model = inputs.get('model', 'F5TTS_Base')
            device = inputs.get('device', 'auto')
            temperature = inputs.get('temperature', 0.8)
            speed = inputs.get('speed', 1.0)
            target_rms = inputs.get('target_rms', 0.1)
            cross_fade_duration = inputs.get('cross_fade_duration', 0.15)
            nfe_step = inputs.get('nfe_step', 32)
            cfg_strength = inputs.get('cfg_strength', 2.0)
            seed = inputs.get('seed', 1)
            enable_cache = inputs.get('enable_audio_cache', True)
            enable_chunking = inputs.get('enable_chunking', True)
            max_chars_per_chunk = inputs.get('max_chars_per_chunk', 400)
            
            # Get the F5-TTS model for this language using language mapper
            from utils.models.language_mapper import get_model_for_language
            required_model = get_model_for_language('f5tts', language, model)
            
            # Ensure correct model is loaded (streaming coordinator should pre-load)
            if not hasattr(self, 'f5tts') or self.current_model != required_model:
                self.load_f5tts_model(required_model, device)
            
            # Load reference audio and text for this character
            if voice_path and voice_path != 'none' and os.path.exists(voice_path):
                # Load reference from file
                ref_audio_path = voice_path
                ref_text = None
                
                # Check for companion text file
                text_file = voice_path.replace('.wav', '.txt').replace('.mp3', '.txt')
                if os.path.exists(text_file):
                    with open(text_file, 'r', encoding='utf-8') as f:
                        ref_text = f.read().strip()
                
                if not ref_text:
                    ref_text = "This is a reference voice for synthesis."
            else:
                # Use default reference if no voice path
                ref_audio_path = inputs.get('audio_prompt_path', '')
                ref_text = inputs.get('ref_text', "This is a reference voice for synthesis.")
            
            # Generate stable audio component for caching
            stable_audio_component = f"streaming_{character}_{language}_{original_idx}"
            
            # Apply chunking if needed
            if enable_chunking and len(segment_text) > max_chars_per_chunk:
                from utils.text.chunking import ImprovedChatterBoxChunker
                if not hasattr(self, 'chunker'):
                    self.chunker = ImprovedChatterBoxChunker()
                chunks = self.chunker.split_into_chunks(segment_text, max_chars_per_chunk)
            else:
                chunks = [segment_text]
            
            # Process each chunk with pause tag support
            chunk_audios = []
            for chunk in chunks:
                # Check cache if enabled
                if enable_cache:
                    cache_key = self._generate_segment_cache_key(
                        f"{character}:{chunk}", required_model, device,
                        stable_audio_component, ref_text, temperature, speed,
                        target_rms, cross_fade_duration, nfe_step, cfg_strength, seed, character,
                        auto_phonemization
                    )
                    cached_audio = self._get_cached_segment_audio(cache_key)
                    if cached_audio:
                        # Set phonemization setting for consistency
                        import os
                        os.environ['F5TTS_AUTO_PHONEMIZATION'] = str(auto_phonemization).lower()
                        chunk_audios.append(cached_audio[0])
                        continue
                
                # Generate audio for this chunk with pause tag processing
                chunk_audio = self.generate_f5tts_with_pause_tags(
                    chunk, ref_audio_path, ref_text, temperature, speed,
                    target_rms, cross_fade_duration, nfe_step, cfg_strength, seed,
                    auto_phonemization=auto_phonemization
                )
                
                # Cache the result if enabled
                if enable_cache and chunk_audio is not None:
                    GLOBAL_AUDIO_CACHE[cache_key] = (chunk_audio, chunk_audio.shape[-1] / 24000.0)
                
                chunk_audios.append(chunk_audio)
            
            # Combine chunks if multiple
            if len(chunk_audios) == 1:
                return chunk_audios[0]
            else:
                # Add silence between chunks if configured
                silence_ms = inputs.get('silence_between_chunks_ms', 100)
                if silence_ms > 0:
                    silence_samples = int((silence_ms / 1000.0) * 24000)  # F5-TTS uses 24kHz
                    silence = torch.zeros((1, silence_samples))
                    
                    combined = []
                    for i, audio in enumerate(chunk_audios):
                        combined.append(audio)
                        if i < len(chunk_audios) - 1:
                            combined.append(silence)
                    return torch.cat(combined, dim=-1)
                else:
                    return torch.cat(chunk_audios, dim=-1)
                    
        except Exception as e:
            print(f"❌ F5-TTS streaming segment {original_idx} failed: {e}")
            # Return silence as fallback
            return torch.zeros((1, 24000))  # 1 second of silence at 24kHz
        return self.process_with_error_handling(_process)