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

# Add parent directory to path for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Load f5tts_base_node module directly
f5tts_base_node_path = os.path.join(current_dir, "f5tts_base_node.py")
f5tts_base_spec = importlib.util.spec_from_file_location("f5tts_base_node_module", f5tts_base_node_path)
f5tts_base_module = importlib.util.module_from_spec(f5tts_base_spec)
sys.modules["f5tts_base_node_module"] = f5tts_base_module
f5tts_base_spec.loader.exec_module(f5tts_base_module)

# Import the base class
BaseF5TTSNode = f5tts_base_module.BaseF5TTSNode

from core.text_chunking import ImprovedChatterBoxChunker
from core.audio_processing import AudioProcessingUtils
from core.voice_discovery import get_available_voices, load_voice_reference, get_available_characters, get_character_mapping
from core.character_parser import parse_character_text, character_parser
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
                    "default": "F5TTS_Base",
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
            }
        }
        
        return base_types

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "generation_info")
    FUNCTION = "generate_speech"
    CATEGORY = "F5-TTS Voice"

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
            from chatterbox.f5tts.f5tts import get_f5tts_models
            return get_f5tts_models()
        except ImportError:
            try:
                from chatterbox.f5tts.f5tts import get_f5tts_models
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
                                   nfe_step: int, cfg_strength: float, seed: int, character: str = "narrator") -> str:
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
            'engine': 'f5tts'
        }
        cache_string = str(sorted(cache_data.items()))
        cache_key = hashlib.md5(cache_string.encode()).hexdigest()
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
                       enable_audio_cache=True):
        
        def _process():
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
            
            # Load F5-TTS model
            self.load_f5tts_model(inputs["model"], inputs["device"])
            
            # Set seed for reproducibility
            self.set_seed(inputs["seed"])
            
            # Handle reference audio and text with priority chain
            main_audio_prompt, main_ref_text = self._handle_reference_with_priority_chain(inputs)
            
            # Generate stable audio component for cache consistency
            stable_audio_component = self._generate_stable_audio_component(
                inputs["reference_audio_file"], inputs.get("opt_reference_audio"), 
                inputs.get("audio_prompt_path", "")
            )
            
            # Parse character segments from text
            # NOTE: We parse characters from original text, then handle pause tags within each segment
            character_segments = parse_character_text(inputs["text"])
            
            # Check if we have character switching
            characters = list(set(char for char, _ in character_segments))
            has_multiple_characters = len(characters) > 1 or (len(characters) == 1 and characters[0] != "narrator")
            
            # Validate and clamp nfe_step to prevent ODE solver issues
            safe_nfe_step = max(1, min(inputs["nfe_step"], 71))
            if safe_nfe_step != inputs["nfe_step"]:
                print(f"⚠️ F5-TTS: Clamped nfe_step from {inputs['nfe_step']} to {safe_nfe_step} to prevent ODE solver issues")
            
            if has_multiple_characters:
                # CHARACTER SWITCHING MODE
                print(f"🎭 F5-TTS: Character switching mode - found characters: {', '.join(characters)}")
                
                # Set up character parser with available characters
                available_chars = get_available_characters()
                character_parser.set_available_characters(list(available_chars))
                
                # Get character voice mapping
                character_mapping = get_character_mapping(characters, engine_type="f5tts")
                
                # Build voice references with fallback to main voice
                voice_refs = {}
                for character in characters:
                    audio_path, ref_text = character_mapping.get(character, (None, None))
                    if audio_path and ref_text:
                        voice_refs[character] = (audio_path, ref_text)
                        print(f"🎭 Using character voice for '{character}'")
                    else:
                        voice_refs[character] = (main_audio_prompt, main_ref_text)
                        print(f"🔄 Using main voice for character '{character}' (not found in voice folders)")
                
                # Process each character segment
                audio_segments = []
                for i, (character, segment_text) in enumerate(character_segments):
                    # Check for interruption
                    self.check_interruption(f"F5-TTS character generation segment {i+1}/{len(character_segments)}")
                    
                    # Apply chunking to long segments if enabled
                    if inputs["enable_chunking"] and len(segment_text) > inputs["max_chars_per_chunk"]:
                        segment_chunks = self.chunker.split_into_chunks(segment_text, inputs["max_chars_per_chunk"])
                    else:
                        segment_chunks = [segment_text]
                    
                    # Get voice reference for this character
                    char_audio, char_text = voice_refs[character]
                    
                    # Generate audio for each chunk of this character segment
                    for chunk_i, chunk_text in enumerate(segment_chunks):
                        print(f"🎤 Generating F5-TTS segment {i+1}/{len(character_segments)} chunk {chunk_i+1}/{len(segment_chunks)} for '{character}'...")
                        
                        # Create cache function for this character if caching is enabled
                        cache_fn = None
                        if inputs.get("enable_audio_cache", True):
                            def create_cache_fn(char_name, char_audio_comp, char_ref_text):
                                def cache_fn_impl(text_content: str, audio_result=None):
                                    cache_key = self._generate_segment_cache_key(
                                        f"{char_name}:{text_content}", inputs["model"], inputs["device"], 
                                        char_audio_comp, char_ref_text, inputs["temperature"], inputs["speed"],
                                        inputs["target_rms"], inputs["cross_fade_duration"], 
                                        safe_nfe_step, inputs["cfg_strength"], inputs["seed"], char_name
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
                        audio_segments.append(chunk_audio)
                
                # Combine all character segments
                wav = self.combine_f5tts_audio_chunks(
                    audio_segments, inputs["chunk_combination_method"], 
                    inputs["silence_between_chunks_ms"], len(inputs["text"])
                )
                
                # Generate info
                total_duration = wav.size(-1) / self.f5tts_sample_rate
                model_info = self.get_f5tts_model_info()
                info = f"Generated {total_duration:.1f}s character-switched audio from {len(character_segments)} segments using {len(characters)} characters (F5-TTS {model_info.get('model_name', 'unknown')})"
                
            else:
                # SINGLE CHARACTER MODE (original behavior)
                text_length = len(inputs["text"])
                
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
                                safe_nfe_step, inputs["cfg_strength"], inputs["seed"], "narrator"
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
                    
                    wav = self.generate_f5tts_with_pause_tags(
                        text=inputs["text"],
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
                    chunks = self.chunker.split_into_chunks(inputs["text"], inputs["max_chars_per_chunk"])
                    
                    # Create cache function for narrator chunks if caching is enabled
                    cache_fn = None
                    if inputs.get("enable_audio_cache", True):
                        def chunk_cache_fn(text_content: str, audio_result=None):
                            cache_key = self._generate_segment_cache_key(
                                f"narrator:{text_content}", inputs["model"], inputs["device"], 
                                stable_audio_component, main_ref_text, inputs["temperature"], inputs["speed"],
                                inputs["target_rms"], inputs["cross_fade_duration"], 
                                safe_nfe_step, inputs["cfg_strength"], inputs["seed"], "narrator"
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
                            cfg_strength=inputs["cfg_strength"]
                        )
                        audio_segments.append(chunk_audio)
                    
                    # Combine audio segments
                    wav = self.combine_f5tts_audio_chunks(
                        audio_segments, inputs["chunk_combination_method"], 
                        inputs["silence_between_chunks_ms"], text_length
                    )
                    
                    # Generate info
                    total_duration = wav.size(-1) / self.f5tts_sample_rate
                    avg_chunk_size = text_length // len(chunks)
                    model_info = self.get_f5tts_model_info()
                    info = f"Generated {total_duration:.1f}s audio from {text_length} characters using {len(chunks)} chunks (avg {avg_chunk_size} chars/chunk, F5-TTS {model_info.get('model_name', 'unknown')})"
            
            # Return audio in ComfyUI format
            return (
                self.format_f5tts_audio_output(wav),
                info
            )
        
        return self.process_with_error_handling(_process)