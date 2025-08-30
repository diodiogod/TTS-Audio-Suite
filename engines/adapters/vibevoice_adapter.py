"""
VibeVoice Engine Adapter - Engine-specific adapter for VibeVoice
Provides standardized interface for VibeVoice operations in unified engine
"""

import torch
import re
from typing import Dict, Any, Optional, List, Tuple
import sys
import os

# Add parent directory for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.models.language_mapper import get_model_for_language
from utils.text.pause_processor import PauseTagProcessor
from utils.text.character_parser import CharacterParser
from engines.vibevoice_engine.vibevoice_engine import VibeVoiceEngine
from engines.vibevoice_engine.vibevoice_downloader import VIBEVOICE_MODELS


class VibeVoiceEngineAdapter:
    """Engine-specific adapter for VibeVoice."""
    
    def __init__(self, node_instance):
        """
        Initialize VibeVoice adapter.
        
        Args:
            node_instance: TTS node instance using this adapter
        """
        self.node = node_instance
        self.engine_type = "vibevoice"
        self.vibevoice_engine = VibeVoiceEngine()
        self.character_parser = CharacterParser()
        self.pause_processor = PauseTagProcessor()
        
        # Track character to speaker mapping for native multi-speaker mode
        self._character_speaker_map = {}
        self._speaker_voices = []
    
    def get_model_for_language(self, lang_code: str, default_model: str) -> str:
        """
        Get VibeVoice model name for specified language.
        
        Args:
            lang_code: Language code (e.g., 'en', 'zh')
            default_model: Default model name
            
        Returns:
            VibeVoice model name for the language
            
        Note: VibeVoice models support English and Chinese
        """
        # VibeVoice models support both English and Chinese
        supported_languages = ['en', 'zh', 'zh-cn', 'chinese', 'english']
        
        if lang_code.lower() in supported_languages:
            # Both models support EN/ZH, return the configured one
            return default_model
        else:
            print(f"⚠️ VibeVoice: Language '{lang_code}' not officially supported (EN/ZH only)")
            return default_model
    
    def load_base_model(self, model_name: str, device: str):
        """
        Load base VibeVoice model.
        
        Args:
            model_name: Model name to load
            device: Device to load model on
        """
        # Check if model is already loaded
        current_model = getattr(self.node, 'current_model_name', None)
        
        if current_model == model_name and self.vibevoice_engine.model is not None:
            print(f"💾 VibeVoice adapter: Model '{model_name}' already loaded")
            return
        
        # Initialize the VibeVoice engine
        self.vibevoice_engine.initialize_engine(
            model_name=model_name,
            device=device
        )
        
        # Store current model name on node
        self.node.current_model_name = model_name
        print(f"✅ VibeVoice adapter: Model '{model_name}' loaded on {device}")
    
    def _parse_language_tags(self, text: str) -> Tuple[str, Optional[str]]:
        """
        Parse language tags like [de:Alice] from text.
        
        Args:
            text: Text with potential language tags
            
        Returns:
            Tuple of (processed_text, detected_language)
        """
        # Pattern to match [language:character] tags
        lang_pattern = r'\[([a-zA-Z\-]+):([^\]]+)\]'
        
        detected_lang = None
        processed_text = text
        
        # Find and process language tags
        matches = re.findall(lang_pattern, text)
        if matches:
            # Take the first language found
            lang_code, character = matches[0]
            detected_lang = lang_code.lower()
            
            # Replace language tags with just character tags
            for lang, char in matches:
                processed_text = processed_text.replace(f'[{lang}:{char}]', f'[{char}]')
            
            # Warn about language since VibeVoice doesn't have language control
            if detected_lang not in ['en', 'zh', 'chinese', 'english']:
                print(f"⚠️ VibeVoice: Language tag '{detected_lang}' found but model only supports EN/ZH")
        
        return processed_text, detected_lang
    
    def _convert_character_to_speaker_format(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Convert [Character] tags to Speaker N format for native multi-speaker.
        
        Args:
            text: Text with [Character] tags
            
        Returns:
            Tuple of (formatted_text, character_mapping)
        """
        # Parse character segments
        segments = self.character_parser.parse_text(text)
        
        # Build character to speaker mapping
        character_map = {}
        speaker_lines = []
        
        for segment in segments:
            char = segment.character
            
            # Assign speaker index if not already mapped
            if char not in character_map:
                speaker_idx = len(character_map)
                if speaker_idx >= 4:
                    print(f"⚠️ VibeVoice: More than 4 characters found, extra characters will use Speaker 3")
                    speaker_idx = 3
                character_map[char] = speaker_idx
            
            speaker_idx = character_map[char]
            speaker_lines.append(f"Speaker {speaker_idx}: {segment.text.strip()}")
        
        # Join with newlines for multi-speaker format
        formatted_text = "\n".join(speaker_lines)
        
        return formatted_text, character_map
    
    def generate_segment_audio(self, text: str, char_audio: str, char_text: str, 
                             character: str = "narrator", **params) -> torch.Tensor:
        """
        Generate VibeVoice audio for a text segment with caching support.
        Follows the same pattern as other engines.
        
        Args:
            text: Text to generate audio for
            char_audio: Reference audio file path or audio dict
            char_text: Reference text
            character: Character name for caching
            **params: Additional VibeVoice parameters
            
        Returns:
            Generated audio tensor
        """
        # Extract parameters
        seed = params.get("seed", 42)
        enable_cache = params.get("enable_audio_cache", True)
        model = params.get("model", "vibevoice-1.5B")
        device = params.get("device", "auto")
        
        # Initialize engine if not already done
        self.load_base_model(model, device)
        
        # Call engine with cache support
        result = self.generate_segment(text, char_audio, {
            **params,
            'enable_cache': enable_cache,
            'seed': seed
        }, character=character)
        
        # Extract tensor from result
        if isinstance(result, dict) and "waveform" in result:
            audio_tensor = result["waveform"]
            # Ensure proper dimensions for ComfyUI
            if audio_tensor.dim() == 3:
                audio_tensor = audio_tensor.squeeze(0)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            return audio_tensor
        
        return result

    def generate_segment(self, text: str, voice_ref: Optional[Dict], params: Dict, character: str = None) -> Dict:
        """
        Generate audio for a text segment.
        
        Args:
            text: Text to generate
            voice_ref: Voice reference audio
            params: Generation parameters from engine config
            character: Character name for cache isolation (required for proper caching)
            
        Returns:
            Audio dict with waveform and sample_rate
        """
        # Extract parameters
        cfg_scale = params.get('cfg_scale', 1.3)
        seed = params.get('seed', 42)
        use_sampling = params.get('use_sampling', False)
        temperature = params.get('temperature', 0.95)
        top_p = params.get('top_p', 0.95)
        max_new_tokens = params.get('max_new_tokens')
        
        # Prepare voice samples
        voice_samples = self.vibevoice_engine._prepare_voice_samples([voice_ref])
        
        # For single segment, format as Speaker 1 (VibeVoice uses 1-based indexing)
        formatted_text = f"Speaker 1: {text}"
        print(f"🎭 SINGLE SEGMENT - Formatted text for VibeVoice:")
        print(f"📝 {formatted_text}")
        print(f"🎤 Character: '{character}', Voice ref: {type(voice_ref)} {'✅' if voice_ref else '❌'}")
        
        # Extract cache parameters
        enable_cache = params.get('enable_cache', True)
        # Use provided character or fall back to params, then default
        if character is None:
            character = params.get('character', 'narrator')
        
        # Generate stable audio component for cache consistency (like ChatterBox)
        from utils.audio.audio_hash import generate_stable_audio_component
        
        # Use provided stable component, or generate from voice reference (like ChatterBox)
        audio_component = params.get("stable_audio_component")
        if not audio_component:
            # Generate stable component like ChatterBox does
            if voice_ref and isinstance(voice_ref, dict):
                if 'waveform' in voice_ref:
                    # Direct audio tensor format
                    audio_component = generate_stable_audio_component(voice_ref, None)
                elif 'audio_path' in voice_ref:
                    # File path format  
                    audio_component = generate_stable_audio_component(None, voice_ref['audio_path'])
                else:
                    audio_component = "main_reference"
            else:
                audio_component = "main_reference"
        
        # DEBUG: Print audio component to track cache invalidation
        # print(f"🐛 VibeVoice DEBUG: character='{character}', audio_component='{audio_component[:50]}...'")
        # Generate audio
        return self.vibevoice_engine.generate_speech(
            text=formatted_text,
            voice_samples=voice_samples,
            cfg_scale=cfg_scale,
            seed=seed,
            use_sampling=use_sampling,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            enable_cache=enable_cache,
            character=character,
            stable_audio_component=audio_component,
            multi_speaker_mode=params.get('multi_speaker_mode', 'Custom Character Switching')
        )
    
    def process_character_segments(self, segments: List[Tuple[str, str]], 
                                  voice_mapping: Dict[str, Any],
                                  params: Dict) -> List[Dict]:
        """
        Process multiple character segments.
        
        Args:
            segments: List of (character, text) tuples
            voice_mapping: Dict mapping character names to voice references
            params: Generation parameters
            
        Returns:
            List of audio dicts
        """
        audio_segments = []
        
        # Check if we should use native multi-speaker mode
        multi_speaker_mode = params.get('multi_speaker_mode', 'Custom Character Switching')
        
        if multi_speaker_mode == "Native Multi-Speaker" and len(segments) <= 4:
            # Use native multi-speaker generation
            audio = self._generate_native_multispeaker(segments, voice_mapping, params)
            audio_segments.append(audio)
        else:
            # Generate each segment separately (Custom Character Switching mode)
            for character, text in segments:
                # Handle pause tags
                if self.pause_processor.has_pause_tags(text):
                    pause_segments, clean_text = self.pause_processor.parse_pause_tags(text)
                    
                    for seg_type, content in pause_segments:
                        if seg_type == 'text':
                            voice_ref = voice_mapping.get(character)
                            audio = self.generate_segment(content, voice_ref, params, character=character)
                            audio_segments.append(audio)
                        elif seg_type == 'pause':
                            # Create silence segment
                            silence = self.pause_processor.create_silence_segment(
                                content, 24000, 
                                device=torch.device('cpu'),
                                dtype=torch.float32
                            )
                            audio_segments.append({
                                "waveform": silence.unsqueeze(0),
                                "sample_rate": 24000
                            })
                else:
                    # Generate normally
                    voice_ref = voice_mapping.get(character)
                    audio = self.generate_segment(text, voice_ref, params, character=character)
                    audio_segments.append(audio)
        
        return audio_segments
    
    def _generate_native_multispeaker(self, segments: List[Tuple[str, str]], 
                                     voice_mapping: Dict[str, Any],
                                     params: Dict) -> Dict:
        """
        Generate using VibeVoice's native multi-speaker mode.
        Supports both [Character] tags and manual "Speaker N:" format.
        
        Args:
            segments: List of (character, text) tuples
            voice_mapping: Dict mapping character names to voice references
            params: Generation parameters
            
        Returns:
            Combined audio dict
        """
        # Get speaker voice inputs from engine config for priority system
        # For manual Speaker format, the main narrator voice comes from the TTS Text node
        # We need to get it from a different source since voice_mapping may not have 'narrator' key
        main_narrator_voice = voice_mapping.get('narrator')  # Try narrator first
        if main_narrator_voice is None:
            # If no 'narrator' key, get it from any available voice (fallback for manual Speaker format)
            available_voices = [v for v in voice_mapping.values() if v is not None]
            main_narrator_voice = available_voices[0] if available_voices else None
            # print(f"🐛 Debug: No 'narrator' key, using fallback voice: {'✅ found' if main_narrator_voice else '❌ none available'}")
        speaker_inputs = {
            1: main_narrator_voice,  # Speaker 1 uses main narrator from TTS Text
            2: params.get('speaker2_voice'),
            3: params.get('speaker3_voice'), 
            4: params.get('speaker4_voice')
        }
        
        # print(f"🐛 Debug: speaker_inputs[1] (main narrator): {'✅ has voice' if speaker_inputs[1] else '❌ no voice'}")
        
        # Build speaker mapping and format text
        character_map = {}
        speaker_voices = []
        formatted_lines = []
        
        print(f"🎭 Native multi-speaker: Processing {len(segments)} segments with characters: {[char for char, _ in segments]}")
        print(f"🎭 Available voice_mapping keys: {list(voice_mapping.keys())}")
        print(f"🎤 Speaker inputs connected: {[f'Speaker {k}' for k, v in speaker_inputs.items() if v is not None]}")
        
        for character, text in segments:
            # Check if this is already a manual "Speaker N:" format
            manual_speaker = self._detect_manual_speaker_format(character)
            
            if manual_speaker is not None:
                # Manual "Speaker N:" format - use speaker input directly
                speaker_idx = manual_speaker - 1  # Convert to 0-based
                if speaker_idx >= 4:
                    speaker_idx = 3
                    
                # Get the appropriate voice based on speaker number
                voice = speaker_inputs.get(manual_speaker)
                if manual_speaker == 1:
                    print(f"🎤 Manual format 'Speaker {manual_speaker}' -> using {'✅ main narrator (Tony)' if voice else '❌ no narrator, using default'}")
                else:
                    print(f"🎤 Manual format 'Speaker {manual_speaker}' -> using {'✅ connected input' if voice else '❌ no input, using default'}")
                
                # Ensure we have enough speaker_voices slots
                while len(speaker_voices) <= speaker_idx:
                    speaker_voices.append(None)
                speaker_voices[speaker_idx] = voice
                
                formatted_lines.append(f"Speaker {manual_speaker}: {text.strip()}")
                
            else:
                # Character tag format - check priority system
                if character not in character_map:
                    speaker_idx = len(character_map)
                    if speaker_idx >= 4:
                        print(f"⚠️ VibeVoice: Limiting to 4 speakers, '{character}' will use Speaker 4")
                        speaker_idx = 3  # Use 0-based internally, will convert to 1-based for format
                    else:
                        character_map[character] = speaker_idx
                        
                    # Priority system: speaker inputs override character aliases
                    speaker_num = speaker_idx + 1
                    connected_voice = speaker_inputs.get(speaker_num)
                    character_voice = voice_mapping.get(character)
                    
                    if connected_voice is not None and character_voice is not None:
                        print(f"⚠️ Priority: Speaker {speaker_num} input overrides ['{character}'] alias - using connected voice")
                        voice = connected_voice
                    elif connected_voice is not None:
                        print(f"🎤 Speaker {speaker_num}: Using connected voice input")
                        voice = connected_voice
                    else:
                        print(f"🎭 Character '{character}' -> Speaker {speaker_num}, using character voice")
                        voice = character_voice
                        
                    speaker_voices.append(voice)
                
                speaker_idx = character_map.get(character, 3)
                # Use 1-based Speaker format as per VibeVoice spec (Speaker 1:, Speaker 2:, etc.)
                formatted_lines.append(f"Speaker {speaker_idx + 1}: {text.strip()}")
        
        # Join with newlines for multi-speaker format
        formatted_text = "\n".join(formatted_lines)
        print(f"🎭 NATIVE MULTI-SPEAKER - Complete formatted text for VibeVoice:")
        print("="*60)
        print(formatted_text)
        print("="*60)
        print(f"🎤 Character mapping: {character_map}")
        print(f"🎤 Using {len(speaker_voices)} voice samples for generation")
        
        # Prepare voice samples
        voice_samples = self.vibevoice_engine._prepare_voice_samples(speaker_voices)
        
        # Generate stable audio component for multi-speaker (like ChatterBox)
        from utils.audio.audio_hash import generate_stable_audio_component
        
        # For multi-speaker, use combined hash of all voices
        combined_voice_hash = []
        for voice in speaker_voices[:4]:  # Max 4 speakers
            if voice is not None and isinstance(voice, dict):
                if 'waveform' in voice:
                    # Direct audio tensor format
                    combined_voice_hash.append(generate_stable_audio_component(voice, None))
                elif 'audio_path' in voice:
                    # File path format
                    combined_voice_hash.append(generate_stable_audio_component(None, voice['audio_path']))
                else:
                    combined_voice_hash.append("unknown_voice")
            else:
                combined_voice_hash.append("no_voice")
        audio_component = f"multi_speaker_{'_'.join(combined_voice_hash)}"
        
        # Generate with multi-speaker text
        return self.vibevoice_engine.generate_speech(
            text=formatted_text,
            voice_samples=voice_samples,
            cfg_scale=params.get('cfg_scale', 1.3),
            seed=params.get('seed', 42),
            use_sampling=params.get('use_sampling', False),
            temperature=params.get('temperature', 0.95),
            top_p=params.get('top_p', 0.95),
            max_new_tokens=params.get('max_new_tokens'),
            enable_cache=params.get('enable_cache', True),
            character="multi_speaker",
            stable_audio_component=audio_component,
            multi_speaker_mode=params.get('multi_speaker_mode', 'Native Multi-Speaker')
        )
    
    def handle_pause_tags(self, text: str) -> Tuple[str, Optional[List]]:
        """
        Handle pause tags in text.
        
        Args:
            text: Text potentially containing pause tags
            
        Returns:
            Tuple of (processed_text, pause_segments)
        """
        if self.pause_processor.has_pause_tags(text):
            segments, clean_text = self.pause_processor.parse_pause_tags(text)
            return clean_text, segments
        return text, None
    
    def generate_vibevoice_with_pause_tags(self, text: str, voice_ref: Optional[Dict], params: Dict,
                                         enable_pause_tags: bool = True, character: str = "narrator") -> torch.Tensor:
        """
        Generate VibeVoice audio with pause tag support (like F5 does).
        
        Args:
            text: Input text potentially with pause tags
            voice_ref: Voice reference dict
            params: Generation parameters
            enable_pause_tags: Whether to process pause tags
            character: Character name for logging
            
        Returns:
            Generated audio tensor with pauses
        """
        from utils.text.pause_processor import PauseTagProcessor
        
        if not enable_pause_tags or not PauseTagProcessor.has_pause_tags(text):
            # No pause tags, use normal generation
            result = self.generate_segment(text, voice_ref, params, character)
            waveform = result['waveform']
            # Ensure proper tensor format
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)  # Remove batch dim
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # Add channel dim
            return waveform
        
        print(f"🎵 VibeVoice: Processing pause tags in text")
        
        # Process pause tags
        pause_segments, _ = PauseTagProcessor.parse_pause_tags(text)
        
        # TTS generation function for pause processor
        def tts_generate_func(text_content: str) -> torch.Tensor:
            result = self.generate_segment(text_content, voice_ref, params, character)
            waveform = result['waveform']
            # Ensure proper tensor format
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)  # Remove batch dim
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # Add channel dim
            return waveform
        
        # Generate audio with pauses
        combined_audio = PauseTagProcessor.generate_audio_with_pauses(
            pause_segments, tts_generate_func, 24000  # VibeVoice sample rate
        )
        
        return combined_audio
    
    def _detect_manual_speaker_format(self, character: str) -> Optional[int]:
        """
        Detect if character is already in manual "Speaker N:" format.
        
        Args:
            character: Character name to check
            
        Returns:
            Speaker number (1-4) if manual format, None otherwise
        """
        import re
        # Match "Speaker N" (case insensitive, with optional trailing colon/whitespace)
        match = re.match(r'^speaker\s*(\d+)\s*:?\s*$', character.strip(), re.IGNORECASE)
        if match:
            speaker_num = int(match.group(1))
            if 1 <= speaker_num <= 4:
                return speaker_num
        return None
    
    def cleanup(self):
        """Clean up resources"""
        if self.vibevoice_engine:
            self.vibevoice_engine.cleanup()
        self._character_speaker_map.clear()
        self._speaker_voices.clear()