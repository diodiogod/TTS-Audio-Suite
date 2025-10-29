"""
IndexTTS-2 Processor - Handles TTS generation orchestration
Called by unified TTS nodes when using IndexTTS-2 engine
"""

import torch
from typing import Dict, Any, Optional, List, Tuple
import os
import sys
import tempfile
import torchaudio

# Add project root to path
current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.text.chunking import ImprovedChatterBoxChunker
from utils.audio.processing import AudioProcessingUtils
from utils.text.character_parser import CharacterParser
from utils.text.pause_processor import PauseTagProcessor
from utils.text.segment_parameters import apply_segment_parameters
from utils.voice.discovery import get_character_mapping
from engines.adapters.index_tts_adapter import IndexTTSAdapter


class IndexTTSProcessor:
    """
    Internal processor for IndexTTS-2 TTS generation.
    Handles emotion control, character processing, and generation orchestration.
    """
    
    def __init__(self, engine_config: Dict[str, Any]):
        """
        Initialize IndexTTS-2 processor.
        
        Args:
            engine_config: Engine configuration from IndexTTS-2 Engine node
        """
        self.config = engine_config
        self.adapter = IndexTTSAdapter()
        self.character_parser = CharacterParser()
        self.pause_processor = PauseTagProcessor()
        self.sample_rate = 22050  # IndexTTS-2 native sample rate

        # Set up character parser with available characters
        self._setup_character_parser()
        
        # Initialize adapter with engine config
        self.adapter.initialize_engine(
            model_path=engine_config.get('model_path'),
            device=engine_config.get('device', 'auto'),
            use_fp16=engine_config.get('use_fp16', True),
            use_cuda_kernel=engine_config.get('use_cuda_kernel'),
            use_deepspeed=engine_config.get('use_deepspeed', False)
        )

    def _setup_character_parser(self):
        """Set up character parser with available characters and aliases."""
        from utils.voice.discovery import get_available_characters, voice_discovery

        # Get available characters and aliases
        available_chars = get_available_characters()
        character_aliases = voice_discovery.get_character_aliases()

        # Build complete available set
        all_available = set()
        if available_chars:
            all_available.update(available_chars)
        for alias, target in character_aliases.items():
            all_available.add(alias.lower())
            all_available.add(target.lower())

        self.character_parser.set_available_characters(list(all_available))

        # Set language defaults
        char_lang_defaults = voice_discovery.get_character_language_defaults()
        for char, lang in char_lang_defaults.items():
            self.character_parser.set_character_language_default(char, lang)

    def _process_dynamic_emotion_template(self, emotion_text: str, segment_text: str) -> str:
        """
        Process dynamic emotion template by replacing {seg} with actual segment text.

        Args:
            emotion_text: Template with {seg} placeholder
            segment_text: Actual text segment content

        Returns:
            Processed emotion text for QwenEmotion analysis
        """
        if "{seg}" in emotion_text:
            processed_text = emotion_text.replace("{seg}", segment_text)
            print(f"🌈 Dynamic Text Emotion: '{segment_text[:30]}...' → '{processed_text[:50]}...'")
            return processed_text
        return emotion_text

    def process_text(self,
                    text: str,
                    speaker_audio: Optional[Dict] = None,
                    reference_text: str = "",
                    seed: int = 1,
                    enable_chunking: bool = True,
                    max_chars_per_chunk: int = 400,
                    silence_between_chunks_ms: int = 100) -> torch.Tensor:
        """
        Process text and generate audio with IndexTTS-2.

        Args:
            text: Input text with potential character tags and emotions
            speaker_audio: Speaker reference audio tensor dict
            reference_text: Reference text for voice cloning
            seed: Random seed for generation
            enable_chunking: Whether to chunk long text (may be disabled for IndexTTS-2)
            max_chars_per_chunk: Maximum characters per chunk
            silence_between_chunks_ms: Silence between segments

        Returns:
            Generated audio tensor
        """
        try:
            # Validate text input - prevent None text which causes NoneType errors
            if text is None:
                raise ValueError("Text input cannot be None. Please provide valid text to generate speech.")

            if isinstance(text, str):
                text = text.strip()
            if not text:
                raise ValueError("Text input is empty. Please provide text to generate speech.")

            print(f"🤖 IndexTTS-2: Processing text with emotion support")

            # Check if IndexTTS-2's native chunking should override our chunking
            max_text_tokens_per_segment = self.config.get('max_text_tokens_per_segment', 120)
            if max_text_tokens_per_segment > 0:
                # IndexTTS-2 has its own token-based chunking, disable our character-based chunking
                print(f"📝 IndexTTS-2: Using native token chunking ({max_text_tokens_per_segment} tokens), disabling character chunking")
                enable_chunking = False

            # Parse character segments with emotion support and parameters
            character_segment_objects = self.character_parser.parse_text_segments(text)
            character_segments = [(seg.character, seg.text, seg.language, seg.emotion) for seg in character_segment_objects]
            all_characters = set(char for char, _, _, _ in character_segments)
            all_characters.add("narrator")
            
            # Build character mapping for emotion references (IndexTTS only needs audio files)
            character_mapping = get_character_mapping(list(all_characters), engine_type="audio_only")
            
            print(f"🎭 IndexTTS-2: Processing {len(character_segments)} character segment(s) - {', '.join(sorted(all_characters))}")
            
            # Build voice references with narrator fallback
            narrator_voice_dict = None
            narrator_ref_text = ""
            
            if speaker_audio is not None:
                # Handle ComfyUI audio tensor dimensions (3D -> 2D)
                waveform = speaker_audio["waveform"]
                if waveform.dim() == 3:
                    waveform = waveform.squeeze(0)  # Remove batch dimension
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert stereo to mono
                narrator_voice_dict = {"waveform": waveform, "sample_rate": speaker_audio["sample_rate"]}
                narrator_ref_text = reference_text or ""
                print(f"📖 Using connected narrator voice | Ref: '{narrator_ref_text[:50]}...'")
            else:
                # Check mapped narrator
                mapped_narrator = character_mapping.get("narrator", (None, None))
                if mapped_narrator[0] and os.path.exists(mapped_narrator[0]):
                    import torchaudio
                    waveform, sample_rate = torchaudio.load(mapped_narrator[0])
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    narrator_voice_dict = {"waveform": waveform, "sample_rate": sample_rate}
                    narrator_ref_text = mapped_narrator[1] or ""
                    print(f"📖 Using mapped narrator voice: {mapped_narrator[0]} | Ref: '{narrator_ref_text[:50]}...'")
                else:
                    print(f"⚠️ No narrator voice available - using IndexTTS-2 without speaker reference")
            
            # Build voice references for all characters  
            voice_refs = {}
            if narrator_voice_dict and 'waveform' in narrator_voice_dict:
                voice_refs['narrator'] = narrator_voice_dict
            
            for character in all_characters:
                if character.lower() == "narrator":
                    continue
                    
                audio_path, char_ref_text = character_mapping.get(character, (None, None))
                if audio_path and os.path.exists(audio_path):
                    import torchaudio
                    waveform, sample_rate = torchaudio.load(audio_path)
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    voice_refs[character] = {"waveform": waveform, "sample_rate": sample_rate}
                    print(f"🎭 {character}: Loaded voice reference")
                else:
                    # Fallback to narrator voice
                    voice_refs[character] = narrator_voice_dict
                    if narrator_voice_dict:
                        print(f"🎭 {character}: Using narrator voice fallback")
                    else:
                        print(f"⚠️ {character}: No voice available")
            
            # Define TTS generation function for pause processor
            def tts_generate_func(text_content: str, segment_params: Optional[Dict[str, Any]] = None) -> torch.Tensor:
                # Import references for nested function scope
                import torchaudio as ta
                import tempfile as tf
                """TTS generation function for pause tag processor with per-segment parameters"""
                # Apply segment parameters if provided
                current_config = dict(self.config)
                if segment_params:
                    current_config = apply_segment_parameters(current_config, segment_params, "index_tts")

                if '[' in text_content and ']' in text_content:
                    # Handle character switching with emotion parsing using modularized parser
                    char_segments = self.character_parser.split_by_character_with_emotions(text_content)
                    segment_audio_parts = []
                    
                    for character, segment_text, language, emotion in char_segments:
                        # Character segment processing (debug info removed for cleaner logs)
                        # Get character voice reference
                        char_audio_dict = voice_refs.get(character)
                        speaker_audio_path = None
                        if char_audio_dict and 'waveform' in char_audio_dict:
                            # Convert tensor back to temporary file for IndexTTS-2
                            # IndexTTS-2 adapter expects file paths, not tensors
                            with tf.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                                waveform = char_audio_dict['waveform']
                                # Ensure 2D tensor for torchaudio.save (channels, samples)
                                if waveform.dim() == 3:
                                    waveform = waveform.squeeze(0)  # Remove batch dimension
                                ta.save(tmp_file.name, waveform, char_audio_dict['sample_rate'])
                                speaker_audio_path = tmp_file.name
                        elif character.lower() == "narrator" and narrator_voice_dict and 'waveform' in narrator_voice_dict:
                            # Fallback to narrator voice for narrator character
                            with tf.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                                waveform = narrator_voice_dict['waveform']
                                # Ensure 2D tensor for torchaudio.save (channels, samples)
                                if waveform.dim() == 3:
                                    waveform = waveform.squeeze(0)  # Remove batch dimension
                                ta.save(tmp_file.name, waveform, narrator_voice_dict['sample_rate'])
                                speaker_audio_path = tmp_file.name
                                print(f"✅ Using fallback narrator voice for character: {character}")
                        
                        # Check for emotion reference from character mapping or config
                        emotion_audio_path = None
                        if emotion:
                            # Emotion from character tag like [Alice:Bob] - use character parser's emotion resolution
                            emotion_audio_path = self.character_parser.get_emotion_voice_path(emotion)
                            if emotion_audio_path:
                                print(f"😊 Using emotion reference from tag: {emotion} -> {emotion_audio_path}")
                            else:
                                print(f"🐛 Could not resolve emotion reference '{emotion}'")

                        # Fall back to config emotion_audio if no tag emotion
                        if not emotion_audio_path:
                            emotion_from_config = self.config.get('emotion_audio')
                            # Process emotion audio for this character
                            if emotion_from_config:
                                if isinstance(emotion_from_config, dict) and 'waveform' in emotion_from_config:
                                    # Convert tensor to temporary file
                                    with tf.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                                        # Ensure correct tensor dimensions for torchaudio.save
                                        emotion_waveform = emotion_from_config['waveform']
                                        if emotion_waveform.dim() == 3:
                                            emotion_waveform = emotion_waveform.squeeze(0)
                                        elif emotion_waveform.dim() == 1:
                                            emotion_waveform = emotion_waveform.unsqueeze(0)
                                        ta.save(tmp_file.name, emotion_waveform, emotion_from_config['sample_rate'])
                                        emotion_audio_path = tmp_file.name
                                    print(f"🎭 Using connected engine emotion audio for character: {character}")
                                else:
                                    emotion_audio_path = emotion_from_config
                                    print(f"🎭 Using connected engine emotion audio for character: {character}")
                            else:
                                print(f"🎭 No emotion audio for character: {character} (no tag emotion, no connected engine emotion)")
                        
                        # Prioritize character emotion reference over global emotion controls
                        # If character has specific emotion ref, disable global emotion controls
                        if emotion_audio_path:
                            # Character has specific emotion - use only that emotion reference
                            character_emotion_vector = None
                            character_use_emotion_text = False
                            character_emotion_text = None
                        else:
                            # No character emotion - use global emotion settings (from current_config with segment params)
                            character_emotion_vector = current_config.get('emotion_vector')
                            character_use_emotion_text = current_config.get('use_emotion_text', False)
                            character_emotion_text = current_config.get('emotion_text')

                            # Handle dynamic QwenEmotion template
                            if character_use_emotion_text and character_emotion_text and current_config.get('is_dynamic_template', False):
                                character_emotion_text = self._process_dynamic_emotion_template(character_emotion_text, segment_text)

                        # Generate audio for this character segment (use parameters from current_config with segment overrides)
                        segment_result = self.adapter.generate(
                            text=segment_text,
                            speaker_audio=speaker_audio_path,
                            emotion_audio=emotion_audio_path,
                            emotion_alpha=current_config.get('emotion_alpha', 1.0),
                            emotion_vector=character_emotion_vector,
                            use_emotion_text=character_use_emotion_text,
                            emotion_text=character_emotion_text,
                            use_random=current_config.get('use_random', False),
                            interval_silence=current_config.get('interval_silence', 200),
                            max_text_tokens_per_segment=current_config.get('max_text_tokens_per_segment', 120),
                            seed=current_config.get('seed', seed),
                            temperature=current_config.get('temperature', 0.8),
                            top_p=current_config.get('top_p', 0.8),
                            top_k=current_config.get('top_k', 30),
                            length_penalty=current_config.get('length_penalty', 0.0),
                            num_beams=current_config.get('num_beams', 3),
                            repetition_penalty=current_config.get('repetition_penalty', 10.0),
                            max_mel_tokens=current_config.get('max_mel_tokens', 1500)
                        )
                        
                        # Clean up temp files
                        if speaker_audio_path and os.path.exists(speaker_audio_path):
                            os.unlink(speaker_audio_path)
                        if emotion_audio_path and isinstance(emotion_audio_path, str) and os.path.exists(emotion_audio_path) and emotion_audio_path.startswith(tempfile.gettempdir()):
                            os.unlink(emotion_audio_path)
                        
                        # Ensure correct dimensions
                        if segment_result.dim() == 1:
                            segment_result = segment_result.unsqueeze(0)
                        elif segment_result.dim() == 3:
                            segment_result = segment_result.squeeze(0)
                        
                        segment_audio_parts.append(segment_result)
                    
                    # Combine character segments
                    if segment_audio_parts:
                        return torch.cat(segment_audio_parts, dim=-1)
                    else:
                        return torch.zeros(1, 0)
                else:
                    # Simple text segment without character switching - use narrator voice
                    narrator_audio = voice_refs.get("narrator")
                    speaker_audio_path = None
                    if narrator_audio and 'waveform' in narrator_audio:
                        with tf.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                            ta.save(tmp_file.name, narrator_audio['waveform'], narrator_audio['sample_rate'])
                            speaker_audio_path = tmp_file.name
                    
                    # Ensure we have a speaker audio path - always use narrator if available  
                    if not speaker_audio_path:
                        if narrator_voice_dict and 'waveform' in narrator_voice_dict:
                            with tf.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                                ta.save(tmp_file.name, narrator_voice_dict['waveform'], narrator_voice_dict['sample_rate'])
                                speaker_audio_path = tmp_file.name
                    
                    # Handle emotion_audio - convert tensor to file path if needed
                    emotion_audio_path = self.config.get('emotion_audio')
                    # Process emotion audio from config
                    if emotion_audio_path and isinstance(emotion_audio_path, dict) and 'waveform' in emotion_audio_path:
                        # Convert tensor to temporary file
                        with tf.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                            # Ensure correct tensor dimensions for torchaudio.save
                            emotion_waveform = emotion_audio_path['waveform']
                            if emotion_waveform.dim() == 3:
                                emotion_waveform = emotion_waveform.squeeze(0)
                            elif emotion_waveform.dim() == 1:
                                emotion_waveform = emotion_waveform.unsqueeze(0)
                            ta.save(tmp_file.name, emotion_waveform, emotion_audio_path['sample_rate'])
                            emotion_audio_path = tmp_file.name
                        print(f"🎭 Using connected engine emotion audio for simple text segment")
                    elif emotion_audio_path:
                        print(f"🎭 Using connected engine emotion audio for simple text segment")
                    else:
                        print(f"🎭 No emotion audio for simple text segment (no connected engine emotion)")

                    # Prioritize connected emotion_audio over global emotion controls
                    # For simple text, use emotion_audio from config if available, else use global settings (with segment params)
                    if emotion_audio_path:
                        # Engine emotion_audio connected - use only that
                        simple_emotion_vector = None
                        simple_use_emotion_text = False
                        simple_emotion_text = None
                    else:
                        # No engine emotion_audio - use global emotion settings (from current_config with segment params)
                        simple_emotion_vector = current_config.get('emotion_vector')
                        simple_use_emotion_text = current_config.get('use_emotion_text', False)
                        simple_emotion_text = current_config.get('emotion_text')

                        # Handle dynamic QwenEmotion template
                        if simple_use_emotion_text and simple_emotion_text and current_config.get('is_dynamic_template', False):
                            simple_emotion_text = self._process_dynamic_emotion_template(simple_emotion_text, text_content)

                    result = self.adapter.generate(
                        text=text_content,
                        speaker_audio=speaker_audio_path,
                        emotion_audio=emotion_audio_path,
                        emotion_alpha=current_config.get('emotion_alpha', 1.0),
                        emotion_vector=simple_emotion_vector,
                        use_emotion_text=simple_use_emotion_text,
                        emotion_text=simple_emotion_text,
                        use_random=current_config.get('use_random', False),
                        interval_silence=current_config.get('interval_silence', 200),
                        max_text_tokens_per_segment=current_config.get('max_text_tokens_per_segment', 120),
                        seed=current_config.get('seed', seed),
                        temperature=current_config.get('temperature', 0.8),
                        top_p=current_config.get('top_p', 0.8),
                        top_k=current_config.get('top_k', 30),
                        length_penalty=current_config.get('length_penalty', 0.0),
                        num_beams=current_config.get('num_beams', 3),
                        repetition_penalty=current_config.get('repetition_penalty', 10.0),
                        max_mel_tokens=current_config.get('max_mel_tokens', 1500)
                    )
                    
                    # Clean up temp files
                    if speaker_audio_path and os.path.exists(speaker_audio_path):
                        os.unlink(speaker_audio_path)
                    if emotion_audio_path and isinstance(emotion_audio_path, str) and os.path.exists(emotion_audio_path) and emotion_audio_path.startswith(tempfile.gettempdir()):
                        os.unlink(emotion_audio_path)
                    
                    return result
            
            # Check if we have multiple character segments with different parameters
            # If so, generate each character segment separately to preserve per-segment parameters
            has_parameter_changes = False
            if len(character_segment_objects) > 1:
                for i in range(1, len(character_segment_objects)):
                    if character_segment_objects[i].parameters != character_segment_objects[i-1].parameters:
                        has_parameter_changes = True
                        break

            if has_parameter_changes:
                # Generate each character segment separately with its parameters
                # Need to reconstruct text with character tags to preserve character switching
                audio_parts = []
                for seg_idx, seg_obj in enumerate(character_segment_objects):
                    # Reconstruct segment with character tag for proper character switching
                    if seg_obj.character and seg_obj.character != "narrator":
                        segment_text_with_tag = f"[{seg_obj.character}] {seg_obj.text}"
                    else:
                        segment_text_with_tag = seg_obj.text

                    print(f"  📊 Segment {seg_idx + 1}: Character '{seg_obj.character}' with params {seg_obj.parameters}")
                    segment_audio = tts_generate_func(segment_text_with_tag, seg_obj.parameters)
                    if isinstance(segment_audio, torch.Tensor) and segment_audio.numel() > 0:
                        audio_parts.append(segment_audio)

                if audio_parts:
                    # Ensure all audio parts have correct dimensions and concatenate
                    audio_parts_normalized = []
                    for audio in audio_parts:
                        if audio.dim() == 1:
                            audio = audio.unsqueeze(0)
                        elif audio.dim() == 3:
                            audio = audio.squeeze(0)
                        audio_parts_normalized.append(audio)
                    result = torch.cat(audio_parts_normalized, dim=-1)
                else:
                    result = torch.zeros(1, self.sample_rate)
            else:
                # No parameter changes between segments, use pause processor for pause tag handling
                segments, clean_text = self.pause_processor.parse_pause_tags(text)

                # Create a wrapper that handles pause segments
                def tts_generate_with_params(text_content: str) -> torch.Tensor:
                    """Wrapper for pause processor that passes segment parameters"""
                    # For pause-based processing, use first segment's parameters
                    segment_params = character_segment_objects[0].parameters if character_segment_objects and character_segment_objects[0].parameters else None
                    return tts_generate_func(text_content, segment_params)

                # Generate audio with pauses
                if segments:
                    result = self.pause_processor.generate_audio_with_pauses(
                        segments=segments,
                        tts_generate_func=tts_generate_with_params,
                        sample_rate=self.sample_rate
                    )
                else:
                    # No pause tags, generate directly
                    segment_params = character_segment_objects[0].parameters if character_segment_objects and character_segment_objects[0].parameters else None
                    result = tts_generate_func(text, segment_params)
            
            # Ensure correct tensor format
            if isinstance(result, torch.Tensor):
                if result.dim() == 1:
                    result = result.unsqueeze(0)  # Add channel dimension
                elif result.dim() == 3:
                    result = result.squeeze(0)  # Remove batch dimension
            
            return result
            
        except Exception as e:
            print(f"❌ IndexTTS-2 processor error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return silence on error
            return torch.zeros(1, self.sample_rate)  # 1 second of silence
    
    def cleanup(self):
        """Clean up resources"""
        if self.adapter:
            self.adapter.unload()