"""
Qwen3-TTS Voice Designer Node

Creates custom voices from natural language descriptions using the VoiceDesign model.
This is a unique feature of Qwen3-TTS - no other engine has text-to-voice design capability.
"""

import os
import sys
import torch
import folder_paths
from typing import Dict, Any, Optional, Tuple

# Add project root for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.audio.processing import AudioProcessingUtils


class Qwen3TTSVoiceDesignerNode:
    """
    Qwen3-TTS Voice Designer - Create custom voices from text descriptions.

    Uses the VoiceDesign model (1.7B only) to generate voices based on natural
    language descriptions. Can save designed voices for reuse in character voices system.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "TTS_engine": ("TTS_ENGINE", {
                    "tooltip": "Qwen3-TTS Engine configuration (from ⚙️ Qwen3-TTS Engine node)"
                }),
                "voice_description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Natural language description of desired voice characteristics.\n\nExamples:\n• 'Professional male narrator, deep voice, slow pacing'\n• 'Young female, cheerful and energetic'\n• 'Elderly gentleman, warm and calm'\n• 'Child voice, playful and curious'"
                }),
                "reference_text": ("STRING", {
                    "multiline": True,
                    "default": "Welcome to the TTS Audio Suite. This advanced text-to-speech system brings your stories to life with natural, expressive voices. Whether you're creating audiobooks, videos, or interactive experiences, our technology delivers exceptional quality and versatility. What will you create today?",
                    "tooltip": "Reference text to generate preview audio (10+ seconds recommended).\nShould include varied intonation, questions, and technical terms for best voice evaluation."
                }),
                "language": (["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"], {
                    "default": "Auto",
                    "tooltip": "Language for test audio generation"
                }),
            },
            "optional": {
                "character_name": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: Save this voice with a character name.\n• If provided: Saves to models/voices/{character_name}/\n• If blank: Temporary preview only (not saved)\n\nSaved voices can be used in character switching and TTS Text node."
                }),
                "overwrite_character": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Overwrite existing character files instead of creating versioned copies.\n• False (default): Creates name_1, name_2, etc. if character exists\n• True: Overwrites existing character files (useful for refining voices)"
                }),
            }
        }

    RETURN_TYPES = ("NARRATOR_VOICE", "AUDIO", "STRING")
    RETURN_NAMES = ("opt_narrator", "preview_audio", "voice_info")
    FUNCTION = "design_voice"
    CATEGORY = "TTS Audio Suite/🎨 Qwen3-TTS"

    def design_voice(self,
                    TTS_engine: Dict[str, Any],
                    voice_description: str,
                    reference_text: str,
                    language: str,
                    character_name: str = "",
                    overwrite_character: bool = False) -> Tuple[Dict, Dict, str]:
        """
        Design a custom voice from text description.

        Args:
            TTS_engine: Engine configuration from Qwen3-TTS Engine node
            voice_description: Natural language voice description
            reference_text: Reference text for preview generation
            language: Target language
            character_name: Optional character name to save voice
            overwrite_character: If True, overwrite existing character files instead of versioning

        Returns:
            Tuple of (narrator_voice, preview_audio, voice_info)
        """
        # Validate inputs
        if not voice_description or not voice_description.strip():
            raise ValueError("Voice description cannot be empty")

        if len(reference_text.strip()) < 10:
            raise ValueError("Reference text must be at least 10 characters (30+ words recommended for quality evaluation)")

        # Validate engine type
        engine_type = TTS_engine.get("engine_type", "")
        if engine_type != "qwen3_tts":
            raise ValueError(
                f"Wrong engine type: '{engine_type}'\n"
                f"This node only works with Qwen3-TTS.\n"
                f"Connect '⚙️ Qwen3-TTS Engine' node."
            )

        # Extract config from nested structure (matching unified TTS node pattern)
        config = TTS_engine.get('config', TTS_engine)

        # Force VoiceDesign model and 1.7B
        model_size = config.get('model_size', '1.7B')
        if model_size == "0.6B":
            print("⚠️ VoiceDesign requires 1.7B model, auto-switching from 0.6B")
            model_size = "1.7B"

        # Check if character already exists on disk with same voice description
        # This avoids regenerating when user has already saved this exact voice
        import os
        import folder_paths

        if character_name and character_name.strip():
            char_name = character_name.strip()
            voices_dir = os.path.join(folder_paths.models_dir, "voices")
            existing_audio = os.path.join(voices_dir, f"{char_name}.wav")
            existing_metadata = os.path.join(voices_dir, f"{char_name}.txt")

            if os.path.exists(existing_audio) and os.path.exists(existing_metadata):
                # Check if BOTH voice description AND reference text match
                try:
                    with open(existing_metadata, 'r', encoding='utf-8') as f:
                        existing_content = f.read()

                    # Check voice description
                    description_matches = f"Voice Description:\n{voice_description}\n" in existing_content
                    
                    # Check reference text from .reference.txt file
                    existing_ref_text_path = os.path.join(voices_dir, f"{char_name}.reference.txt")
                    reference_matches = False
                    if os.path.exists(existing_ref_text_path):
                        with open(existing_ref_text_path, 'r', encoding='utf-8') as f:
                            saved_ref_text = f.read().strip()
                        reference_matches = (saved_ref_text == reference_text.strip())
                    
                    # Only load from disk if BOTH match (prevents stale audio)
                    if description_matches and reference_matches:
                        print(f"💾 Character '{char_name}' already exists with identical voice - loading from disk (skipped model loading)")

                        # Load existing audio
                        import torchaudio
                        waveform, sr = torchaudio.load(existing_audio)

                        # Ensure 3D shape [batch, channels, samples] for ComfyUI
                        if waveform.dim() == 2:
                            waveform = waveform.unsqueeze(0)  # [1, channels, samples]

                        # Cache the loaded audio for same-session reuse
                        from utils.audio.cache import audio_cache
                        import hashlib
                        cache_key_data = f"{voice_description}|{reference_text}|{language}|{config.get('temperature', 0.9)}|{config.get('top_k', 50)}|{config.get('top_p', 1.0)}"
                        cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()
                        duration = waveform.shape[-1] / float(sr)
                        audio_cache.cache_audio(f"voice_designer_{cache_key}", waveform, duration)

                        # Create preview audio dict
                        preview_audio = {
                            "waveform": waveform,
                            "sample_rate": sr
                        }

                        # Create narrator voice reference (unified format matching Character Voices)
                        narrator_voice = {
                            "audio": preview_audio,
                            "audio_path": existing_audio,
                            "reference_text": reference_text,
                            "character_name": char_name,
                            "source": "voice_design",
                            # Extra metadata for VoiceDesign
                            "description": voice_description,
                            "language": language,
                            "engine": "qwen3_tts"
                        }

                        voice_info = f"Voice loaded from disk\nDescription: {voice_description}\nTest text: {reference_text}\n\n💾 Character '{char_name}' already exists with identical voice"

                        return (narrator_voice, preview_audio, voice_info)
                    elif description_matches and not reference_matches:
                        print(f"⚠️ Character '{char_name}' exists but reference text changed - regenerating voice")
                except Exception as e:
                    # If can't read metadata, proceed with generation
                    print(f"⚠️ Could not verify existing character: {e}")
                    pass

        # Generate cache key for runtime caching (same session regeneration)
        import hashlib
        cache_key_data = f"{voice_description}|{reference_text}|{language}|{config.get('temperature', 0.9)}|{config.get('top_k', 50)}|{config.get('top_p', 1.0)}"
        cache_key = hashlib.md5(cache_key_data.encode()).hexdigest()

        # Check audio cache (for same-session regeneration optimization)
        from utils.audio.cache import audio_cache
        cached_audio = audio_cache.get_cached_audio(f"voice_designer_{cache_key}")

        if cached_audio:
            print(f"💾 Using cached voice design audio (skipped model loading)")
            audio_tensor = cached_audio[0]
        else:
            # Load engine via unified interface with VoiceDesign context
            from utils.models.unified_model_interface import unified_model_interface, ModelLoadConfig
            from utils.device import resolve_torch_device

            model_path = config.get('model_path', f'Qwen3-TTS-12Hz-{model_size}-VoiceDesign')
            device = config.get('device', 'auto')
            dtype = config.get('dtype', 'auto')
            attn_implementation = config.get('attn_implementation', 'auto')

            # Build config with VoiceDesign context
            model_config = ModelLoadConfig(
                engine_name="qwen3_tts",
                model_type="tts",
                model_name=f"Qwen3-TTS-12Hz-{model_size}-VoiceDesign",
                model_path=model_path,
                device=resolve_torch_device(device),
                additional_params={
                    "dtype": dtype,
                    "attn_implementation": attn_implementation,
                    "context": {
                        "node_type": "voice_designer",  # Force VoiceDesign model
                        "model_size": model_size
                    }
                }
            )

            print(f"🎨 Loading Qwen3-TTS VoiceDesign model...")
            engine = unified_model_interface.load_model(model_config)
            # Generate preview audio with voice description
            print(f"🎨 Generating preview audio with voice description...")
            print(f"   Language: {language}")
            print(f"   Description: {voice_description[:80]}...")
            print(f"   Test text: {reference_text[:50]}...")

            # Create ComfyUI progress bar (same pattern as processor)
            import comfy.utils
            max_new_tokens = config.get('max_new_tokens', 2048)
            pbar = comfy.utils.ProgressBar(max_new_tokens)

            # Convert to transformers streamer
            from engines.qwen3_tts.progress_callback import Qwen3TTSProgressStreamer
            streamer = Qwen3TTSProgressStreamer(max_new_tokens, pbar, text_input=reference_text)

            try:
                wavs, sr = engine.generate_voice_design(
                    text=reference_text,
                    language=language,
                    instruct=voice_description,
                    top_k=config.get('top_k', 50),
                    top_p=config.get('top_p', 1.0),
                    temperature=config.get('temperature', 0.9),
                    repetition_penalty=config.get('repetition_penalty', 1.05),
                    max_new_tokens=max_new_tokens,
                    streamer=streamer
                )
            except Exception as e:
                raise RuntimeError(f"Voice design generation failed: {e}")

            # Convert output to tensor
            import numpy as np
            if isinstance(wavs, list):
                wavs = wavs[0] if len(wavs) == 1 else np.concatenate(wavs, axis=-1)

            if isinstance(wavs, np.ndarray):
                audio_tensor = torch.from_numpy(wavs).float()
            elif torch.is_tensor(wavs):
                audio_tensor = wavs.float()
            else:
                raise ValueError(f"Unsupported audio output type: {type(wavs)}")

            # Ensure 3D shape [batch, channels, samples] for ComfyUI
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
            elif audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)  # [1, channels, samples]

            # Cache the generated audio
            duration = audio_tensor.shape[-1] / 24000.0
            audio_cache.cache_audio(f"voice_designer_{cache_key}", audio_tensor, duration)
            print(f"💾 Cached voice design audio ({duration:.1f}s)")

        # Create preview audio dict
        preview_audio = {
            "waveform": audio_tensor,
            "sample_rate": 24000
        }

        # Determine audio path and character name
        audio_path = None
        char_name = character_name.strip() if character_name and character_name.strip() else "voice_design"

        # Create narrator voice reference (unified format matching Character Voices)
        narrator_voice = {
            "audio": preview_audio,
            "audio_path": audio_path,  # Will be set after save if character_name provided
            "reference_text": reference_text,
            "character_name": char_name,
            "source": "voice_design",
            # Extra metadata for VoiceDesign
            "description": voice_description,
            "language": language,
            "engine": "qwen3_tts"
        }

        # Save to character voices if name provided
        voice_info = f"Voice designed successfully\nDescription: {voice_description}\nTest text: {reference_text}"

        if character_name and character_name.strip():
            # Check if character already exists with the same voice description
            char_name = character_name.strip()
            voices_dir = os.path.join(folder_paths.models_dir, "voices")
            existing_metadata = os.path.join(voices_dir, f"{char_name}.txt")

            should_save = True
            # When overwrite_character is True, always save (skip duplicate check)
            if not overwrite_character and os.path.exists(existing_metadata):
                # Read existing metadata to check if it's the same voice
                try:
                    with open(existing_metadata, 'r', encoding='utf-8') as f:
                        existing_content = f.read()
                    # Check if BOTH voice description AND reference text match
                    description_matches = f"Voice Description:\n{voice_description}\n" in existing_content
                    
                    existing_ref_text_path = os.path.join(voices_dir, f"{char_name}.reference.txt")
                    reference_matches = False
                    if os.path.exists(existing_ref_text_path):
                        with open(existing_ref_text_path, 'r', encoding='utf-8') as f:
                            saved_ref_text = f.read().strip()
                        reference_matches = (saved_ref_text == reference_text.strip())
                    
                    if description_matches and reference_matches:
                        print(f"💾 Character '{char_name}' already exists with identical voice - skipping save")
                        voice_info += f"\n\n💾 Character '{char_name}' already exists with identical voice"
                        should_save = False
                except:
                    pass  # If can't read metadata, proceed with save

            if should_save:
                save_path, final_char_name = self._save_voice_to_character(
                    char_name,
                    voice_description,
                    preview_audio,
                    reference_text,
                    language,
                    overwrite_character
                )
                # Update narrator_voice with saved audio path
                saved_audio_path = os.path.join(save_path, f"{final_char_name}.wav")
                narrator_voice["audio_path"] = saved_audio_path
                narrator_voice["character_name"] = final_char_name

                voice_info += f"\n\n✅ Saved to: {save_path}"
                if final_char_name != char_name:
                    voice_info += f"\n   (saved as '{final_char_name}' - original name already existed)"
                voice_info += f"\n💡 Press 'R' to refresh Character Voices list"
                print(f"✅ Voice saved to character: {final_char_name}")
                print(f"💡 Press 'R' to refresh Character Voices dropdown")

            # Invalidate cache so new character is available on next refresh
            from utils.voice.discovery import invalidate_voice_cache
            invalidate_voice_cache()
        else:
            voice_info += "\n\n💡 Tip: Provide a character_name to save this voice for reuse"

        return (narrator_voice, preview_audio, voice_info)

    def _save_voice_to_character(self,
                                character_name: str,
                                voice_description: str,
                                audio: Dict,
                                reference_text: str,
                                language: str,
                                overwrite: bool = False) -> Tuple[str, str]:
        """
        Save designed voice to models/voices/ for character system.

        Saves in standard character voice format:
        - {character_name}.wav: Voice audio (generated by VoiceDesign)
        - {character_name}.reference.txt: Transcription of the audio (reference_text)
        - {character_name}.txt: Metadata about voice creation

        Args:
            character_name: Character name
            voice_description: Voice description used for generation
            audio: Audio dict with waveform and sample_rate
            reference_text: Test text used (transcription of audio)
            language: Language used
            overwrite: If True, overwrite existing files; if False, create versioned names

        Returns:
            Tuple of (voices_dir path, final_character_name)
        """
        # Create voices directory if it doesn't exist
        voices_dir = os.path.join(folder_paths.models_dir, "voices")
        os.makedirs(voices_dir, exist_ok=True)

        # Handle existing character names based on overwrite setting
        if not overwrite:
            # Versioning mode: append number if needed
            original_name = character_name
            counter = 1
            while os.path.exists(os.path.join(voices_dir, f"{character_name}.wav")):
                character_name = f"{original_name}_{counter}"
                counter += 1

            if character_name != original_name:
                print(f"⚠️ Character '{original_name}' already exists - saving as '{character_name}'")
        else:
            # Overwrite mode: use name directly
            if os.path.exists(os.path.join(voices_dir, f"{character_name}.wav")):
                print(f"♻️ Overwriting existing character '{character_name}'")

        # Save voice audio as {character_name}.wav
        audio_path = os.path.join(voices_dir, f"{character_name}.wav")
        import torchaudio
        waveform = audio["waveform"]
        # Ensure 2D for torchaudio [channels, samples]
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        torchaudio.save(audio_path, waveform, audio["sample_rate"])

        # Save reference text as {character_name}.reference.txt (transcription)
        reference_path = os.path.join(voices_dir, f"{character_name}.reference.txt")
        with open(reference_path, 'w', encoding='utf-8') as f:
            f.write(reference_text)

        # Save metadata as {character_name}.txt (description of where voice came from)
        metadata_text = f"""Voice created with Qwen3-TTS Voice Designer

Voice Description:
{voice_description}

Language: {language}
Test Text: {reference_text}

Generated by: Qwen3-TTS Voice Designer (VoiceDesign model)
              via TTS Audio Suite for ComfyUI
              https://github.com/FahimF/ComfyUI_TTS_Audio_Suite
"""
        metadata_path = os.path.join(voices_dir, f"{character_name}.txt")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(metadata_text)

        return voices_dir, character_name
