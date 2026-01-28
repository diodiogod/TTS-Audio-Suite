"""
Qwen3-TTS Engine Adapter

Provides standardized interface for Qwen3-TTS integration with TTS Audio Suite.
Handles intelligent model selection, parameter mapping, voice references, and caching.
"""

import os
import sys
import torch
from typing import Dict, Any, Optional, List, Union

# Add project root to path
current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engines.qwen3_tts.qwen3_tts import Qwen3TTSEngine
from utils.text.pause_processor import PauseTagProcessor
from utils.audio.cache import get_audio_cache
import folder_paths


class Qwen3TTSEngineAdapter:
    """
    Adapter for Qwen3-TTS engine providing unified interface compatibility.

    Handles:
    - Intelligent model selection (CustomVoice/VoiceDesign/Base)
    - Parameter mapping between unified interface and Qwen3-TTS
    - Voice reference processing (Base model)
    - Pause tag processing and timing
    - Caching integration
    - Model management
    """

    def __init__(self, node_instance):
        """
        Initialize the Qwen3-TTS adapter.

        Args:
            node_instance: Parent node instance for context
        """
        self.node = node_instance
        self.engine = None
        self.audio_cache = get_audio_cache()
        self.current_model_type = None  # Track which model variant is loaded

        # Job-level timing tracker (persists across blocks)
        self.job_tracker = None

    def start_job(self, total_blocks: int, block_texts: list):
        """
        Initialize job tracker for time estimation across all blocks.

        Args:
            total_blocks: Number of blocks to process
            block_texts: List of text lengths for each block (for weighted progress)
        """
        import time
        total_text = sum(block_texts)
        self.job_tracker = {
            'start_time': time.time(),
            'total_blocks': total_blocks,
            'blocks_completed': 0,
            'block_texts': block_texts,
            'total_text': total_text,
            'text_completed': 0,
            'current_block_text': 0
        }

    def set_current_block(self, block_idx: int):
        """Set the current block being processed."""
        if self.job_tracker:
            self.job_tracker['current_block_text'] = self.job_tracker['block_texts'][block_idx]

    def complete_block(self):
        """Mark current block as completed."""
        if self.job_tracker:
            self.job_tracker['blocks_completed'] += 1
            self.job_tracker['text_completed'] += self.job_tracker['current_block_text']
            # Track total tokens from completed blocks
            self.job_tracker['total_tokens'] = self.job_tracker.get('total_tokens', 0) + self.job_tracker.get('current_block_tokens', 0)

    def end_job(self):
        """Clear job tracker after job completion."""
        self.job_tracker = None

    def _determine_model_type(self, context: Dict[str, Any]) -> str:
        """
        Auto-select model variant based on context.

        Priority:
        1. Voice Designer node → VoiceDesign
        2. Preset voice selected → CustomVoice
        3. Default → Base (zero-shot)

        Args:
            context: Context dict with node_type, voice_preset, etc.

        Returns:
            Model type string: "CustomVoice", "VoiceDesign", or "Base"
        """
        # Priority 1: Voice Designer node
        if context.get("node_type") == "voice_designer":
            return "VoiceDesign"

        # Priority 2: Preset voice selected
        voice_preset = context.get("voice_preset")
        if voice_preset and voice_preset != "None (Zero-shot / Custom)":
            return "CustomVoice"

        # Priority 3: Default to Base (zero-shot)
        return "Base"

    def load_base_model(self,
                       model_path: str,
                       device: str = "auto",
                       dtype: str = "auto",
                       model_size: str = "1.7B",
                       attn_implementation: str = "auto",
                       context: Optional[Dict[str, Any]] = None):
        """
        Load Qwen3-TTS engine via unified interface with intelligent model selection.

        Args:
            model_path: Model identifier (local:ModelName or ModelName for auto-download)
            device: Target device (auto/cuda/cpu)
            dtype: Model precision (bfloat16/float16/float32/auto)
            model_size: Model size (0.6B/1.7B)
            attn_implementation: Attention mechanism (auto/flash_attention_2/sdpa/eager)
            context: Context dict for model type determination
        """
        from utils.models.unified_model_interface import unified_model_interface, ModelLoadConfig
        from utils.device import resolve_torch_device

        # Determine which model variant to load
        context = context or {}
        model_type = self._determine_model_type(context)

        # VoiceDesign only supports 1.7B
        if model_type == "VoiceDesign" and model_size == "0.6B":
            model_size = "1.7B"
            print("⚠️ VoiceDesign requires 1.7B model, auto-switching from 0.6B")

        # Build model name
        model_name = f"Qwen3-TTS-12Hz-{model_size}-{model_type}"
        self.current_model_type = model_type

        # Loading message is printed by unified_model_interface

        # Create config and load via unified interface
        config = ModelLoadConfig(
            engine_name="qwen3_tts",
            model_type="tts",
            model_name=model_name,
            model_path=model_path if model_path else model_name,
            device=resolve_torch_device(device),
            additional_params={
                "dtype": dtype,
                "attn_implementation": attn_implementation
            }
        )

        self.engine = unified_model_interface.load_model(config)

    def generate_with_pause_tags(self,
                                 text: str,
                                 voice_ref: Optional[Dict[str, Any]],
                                 params: Dict[str, Any],
                                 process_pauses: bool = True,
                                 character_name: Optional[str] = None) -> torch.Tensor:
        """
        Generate speech with pause tag processing (TTS Suite integration).

        Args:
            text: Input text (may contain [pause:X] tags)
            voice_ref: Voice reference dict (for Base model) or None (for CustomVoice/VoiceDesign)
            params: Generation parameters (language, speaker, instruct, etc.)
            process_pauses: Whether to process pause tags
            character_name: Character name for logging

        Returns:
            Generated audio tensor [1, samples] at 24000 Hz
        """
        if self.engine is None:
            raise RuntimeError("Engine not loaded. Call load_base_model() first.")

        # Extract seed from params and set global torch seed for reproducibility
        seed = params.get('seed', 0)
        if seed > 0:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Check if text has pause tags
        if process_pauses and PauseTagProcessor.has_pause_tags(text):
            # Process pause tags and generate segments
            return self._generate_with_pauses(text, voice_ref, params, character_name)
        else:
            # Direct generation without pause processing
            return self._generate_direct(text, voice_ref, params, character_name)

    def _generate_direct(self,
                        text: str,
                        voice_ref: Optional[Dict[str, Any]],
                        params: Dict[str, Any],
                        character_name: Optional[str] = None) -> torch.Tensor:
        """
        Direct generation without pause processing.

        Args:
            text: Input text (clean, no pause tags)
            voice_ref: Voice reference dict (for Base model)
            params: Generation parameters
            character_name: Character name for logging

        Returns:
            Audio tensor [1, samples]
        """
        # Determine generation method based on current model type
        if self.current_model_type == "CustomVoice":
            return self._generate_custom_voice(text, params, character_name)
        elif self.current_model_type == "VoiceDesign":
            return self._generate_voice_design(text, params, character_name)
        else:  # Base
            return self._generate_voice_clone(text, voice_ref, params, character_name)

    def _generate_custom_voice(self,
                              text: str,
                              params: Dict[str, Any],
                              character_name: Optional[str] = None) -> torch.Tensor:
        """
        Generate with CustomVoice model using preset speakers.

        Args:
            text: Input text
            params: Must include 'speaker' and 'language'
            character_name: Character name for logging

        Returns:
            Audio tensor [1, samples]
        """
        speaker = params.get('voice_preset') or params.get('speaker', 'Vivian')
        language = params.get('language', 'Auto')
        instruct = params.get('instruct')  # Optional instruction

        # Generate cache key (include attn_implementation since it affects output quality)
        cache_key = self.audio_cache.generate_cache_key(
            'qwen3_tts',
            text=text,
            model_type='CustomVoice',
            speaker=speaker,
            language=language,
            instruct=instruct,
            top_k=params.get('top_k', 50),
            top_p=params.get('top_p', 1.0),
            temperature=params.get('temperature', 0.9),
            repetition_penalty=params.get('repetition_penalty', 1.05),
            max_new_tokens=params.get('max_new_tokens', 2048),
            seed=params.get('seed', 0),
            attn_implementation=params.get('attn_implementation', 'auto'),
            character=character_name or 'narrator'
        )

        # Check cache
        cached_audio = self.audio_cache.get_cached_audio(cache_key)
        if cached_audio:
            char_desc = character_name or 'narrator'
            print(f"💾 Using cached Qwen3-TTS audio for '{char_desc}': '{text[:30]}...'")
            return cached_audio[0]

        # Create ComfyUI progress bar and convert to transformers streamer
        max_new_tokens = params.get('max_new_tokens', 2048)
        progress_bar = self._create_progress_bar(max_new_tokens)

        # Convert progress_bar to transformers streamer for bundled model
        streamer = None
        if progress_bar is not None:
            from engines.qwen3_tts.progress_callback import Qwen3TTSProgressStreamer
            streamer = Qwen3TTSProgressStreamer(max_new_tokens, progress_bar, text_input=text)

        # Generate
        wavs, sr = self.engine.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            top_k=params.get('top_k', 50),
            top_p=params.get('top_p', 1.0),
            temperature=params.get('temperature', 0.9),
            repetition_penalty=params.get('repetition_penalty', 1.05),
            max_new_tokens=max_new_tokens,
            streamer=streamer  # Use transformers streamer kwarg
        )

        # Convert to tensor
        audio_tensor = self._convert_output_to_tensor(wavs, sr)

        # Cache the result
        duration = self.audio_cache._calculate_duration(audio_tensor, 'qwen3_tts')
        self.audio_cache.cache_audio(cache_key, audio_tensor, duration)

        return audio_tensor

    def _generate_voice_design(self,
                              text: str,
                              params: Dict[str, Any],
                              character_name: Optional[str] = None) -> torch.Tensor:
        """
        Generate with VoiceDesign model using text description.

        Args:
            text: Input text
            params: Must include 'instruct' (voice description) and 'language'
            character_name: Character name for logging

        Returns:
            Audio tensor [1, samples]
        """
        instruct = params.get('instruct') or params.get('voice_description')
        if not instruct:
            raise ValueError("VoiceDesign requires 'instruct' parameter with voice description")

        language = params.get('language', 'Auto')

        # Generate cache key (include attn_implementation since it affects output quality)
        cache_key = self.audio_cache.generate_cache_key(
            'qwen3_tts',
            text=text,
            model_type='VoiceDesign',
            instruct=instruct,
            language=language,
            top_k=params.get('top_k', 50),
            top_p=params.get('top_p', 1.0),
            temperature=params.get('temperature', 0.9),
            repetition_penalty=params.get('repetition_penalty', 1.05),
            max_new_tokens=params.get('max_new_tokens', 2048),
            seed=params.get('seed', 0),
            attn_implementation=params.get('attn_implementation', 'auto'),
            character=character_name or 'narrator'
        )

        # Check cache
        cached_audio = self.audio_cache.get_cached_audio(cache_key)
        if cached_audio:
            char_desc = character_name or 'narrator'
            print(f"💾 Using cached Qwen3-TTS VoiceDesign audio for '{char_desc}': '{text[:30]}...'")
            return cached_audio[0]

        # Create ComfyUI progress bar and convert to transformers streamer
        max_new_tokens = params.get('max_new_tokens', 2048)
        progress_bar = self._create_progress_bar(max_new_tokens)

        # Convert progress_bar to transformers streamer for bundled model
        streamer = None
        if progress_bar is not None:
            from engines.qwen3_tts.progress_callback import Qwen3TTSProgressStreamer
            streamer = Qwen3TTSProgressStreamer(max_new_tokens, progress_bar, text_input=text)

        # Generate
        wavs, sr = self.engine.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
            top_k=params.get('top_k', 50),
            top_p=params.get('top_p', 1.0),
            temperature=params.get('temperature', 0.9),
            repetition_penalty=params.get('repetition_penalty', 1.05),
            max_new_tokens=max_new_tokens,
            streamer=streamer  # Use transformers streamer kwarg
        )

        # Convert to tensor
        audio_tensor = self._convert_output_to_tensor(wavs, sr)

        # Cache the result
        duration = self.audio_cache._calculate_duration(audio_tensor, 'qwen3_tts')
        self.audio_cache.cache_audio(cache_key, audio_tensor, duration)

        return audio_tensor

    def _generate_voice_clone(self,
                             text: str,
                             voice_ref: Optional[Dict[str, Any]],
                             params: Dict[str, Any],
                             character_name: Optional[str] = None) -> torch.Tensor:
        """
        Generate with Base model using zero-shot voice cloning.

        Args:
            text: Input text
            voice_ref: Voice reference dict with audio and optional transcript
            params: Generation parameters
            character_name: Character name for logging

        Returns:
            Audio tensor [1, samples]
        """
        # Extract voice reference (returns converted audio + original for cache hashing)
        ref_audio, ref_text, voice_x_vector_only, ref_audio_original = self._extract_voice_reference(voice_ref)

        language = params.get('language', 'Auto')
        # Use voice_ref x_vector_only_mode if provided, otherwise use params
        x_vector_only = voice_x_vector_only if voice_x_vector_only else params.get('x_vector_only_mode', False)

        # Generate cache key using voice_ref dict (contains waveform + sample_rate)
        # This ensures different voices generate different cache keys
        from utils.audio.audio_hash import generate_stable_audio_component
        if voice_ref and isinstance(voice_ref, dict):
            # Check if voice_ref has audio tensor or file path
            if "audio" in voice_ref:
                # Unified Character Voices format: {"audio": {"waveform": ..., "sample_rate": ...}, "audio_path": ..., ...}
                audio_dict = voice_ref.get("audio")
                audio_component = generate_stable_audio_component(reference_audio=audio_dict)
            elif ref_audio_original and isinstance(ref_audio_original, str):
                # File path format: {"audio_path": "/path/to/file.wav", "reference_text": "..."}
                audio_component = generate_stable_audio_component(audio_file_path=ref_audio_original)
            elif "waveform" in voice_ref:
                # Direct tensor format: {"waveform": tensor, "sample_rate": 24000}
                audio_component = generate_stable_audio_component(reference_audio=voice_ref)
            else:
                audio_component = "default_voice"
        elif ref_audio_original and isinstance(ref_audio_original, str):
            # File path case (voice_ref is None but ref_audio_original extracted)
            audio_component = generate_stable_audio_component(audio_file_path=ref_audio_original)
        else:
            audio_component = "default_voice"

        cache_key = self.audio_cache.generate_cache_key(
            'qwen3_tts',
            text=text,
            model_type='Base',
            audio_component=audio_component,
            ref_text=ref_text,
            language=language,
            x_vector_only=x_vector_only,
            top_k=params.get('top_k', 50),
            top_p=params.get('top_p', 1.0),
            temperature=params.get('temperature', 0.9),
            repetition_penalty=params.get('repetition_penalty', 1.05),
            max_new_tokens=params.get('max_new_tokens', 2048),
            seed=params.get('seed', 0),
            attn_implementation=params.get('attn_implementation', 'auto'),
            character=character_name or 'narrator'
        )

        # Check cache
        cached_audio = self.audio_cache.get_cached_audio(cache_key)
        if cached_audio:
            char_desc = character_name or 'narrator'
            print(f"💾 Using cached Qwen3-TTS VoiceClone audio for '{char_desc}': '{text[:30]}...'")
            return cached_audio[0]

        # Create ComfyUI progress bar and convert to transformers streamer
        max_new_tokens = params.get('max_new_tokens', 2048)
        progress_bar = self._create_progress_bar(max_new_tokens)

        # Convert progress_bar to transformers streamer for bundled model
        streamer = None
        if progress_bar is not None:
            from engines.qwen3_tts.progress_callback import Qwen3TTSProgressStreamer
            streamer = Qwen3TTSProgressStreamer(max_new_tokens, progress_bar, text_input=text)

        # Generate
        wavs, sr = self.engine.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
            top_k=params.get('top_k', 50),
            top_p=params.get('top_p', 1.0),
            temperature=params.get('temperature', 0.9),
            repetition_penalty=params.get('repetition_penalty', 1.05),
            max_new_tokens=max_new_tokens,
            streamer=streamer  # Use transformers streamer kwarg
        )

        # Convert to tensor
        audio_tensor = self._convert_output_to_tensor(wavs, sr)

        # Cache the result
        duration = self.audio_cache._calculate_duration(audio_tensor, 'qwen3_tts')
        self.audio_cache.cache_audio(cache_key, audio_tensor, duration)

        return audio_tensor

    def _generate_with_pauses(self,
                             text: str,
                             voice_ref: Optional[Dict[str, Any]],
                             params: Dict[str, Any],
                             character_name: Optional[str] = None) -> torch.Tensor:
        """
        Generate with pause tag processing.

        Args:
            text: Text with pause tags ([pause:2], [pause:1.5s], [pause:500ms])
            voice_ref: Voice reference dict (for Base model)
            params: Generation parameters
            character_name: Character name for logging

        Returns:
            Combined audio tensor [1, samples]
        """
        # Parse pause tags - returns list of tuples: ('text', content) or ('pause', duration_seconds)
        segments, clean_text = PauseTagProcessor.parse_pause_tags(text)

        print(f"🎵 Processing {len(segments)} pause-delimited segments for '{character_name or 'unknown'}'")

        audio_parts = []
        sample_rate = 24000  # Qwen3-TTS native sample rate

        for segment_type, content in segments:
            if segment_type == 'text':
                # Generate audio for text segment
                audio_tensor = self._generate_direct(content, voice_ref, params, character_name)

                # Ensure correct shape [1, samples] (2D)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)  # [samples] -> [1, samples]
                elif audio_tensor.dim() == 3:
                    audio_tensor = audio_tensor.squeeze(0)  # [1, 1, samples] -> [1, samples]

                audio_parts.append(audio_tensor)

            elif segment_type == 'pause':
                # Create silence (content is duration in seconds)
                silence = PauseTagProcessor.create_silence_segment(
                    content, sample_rate
                )
                audio_parts.append(silence)

        # Concatenate all parts
        if not audio_parts:
            return torch.zeros(1, 0)

        combined_audio = torch.cat(audio_parts, dim=-1)
        return combined_audio

    def _extract_voice_reference(self, voice_ref: Optional[Dict[str, Any]]) -> tuple:
        """
        Extract voice reference audio and text from voice_ref dict.

        Args:
            voice_ref: Voice reference dict (from voice discovery)

        Returns:
            Tuple of (ref_audio, ref_text, x_vector_only_mode)
            - ref_audio: Audio in format expected by Qwen3-TTS (string path or tuple (np.ndarray, sr))
            - ref_text: Reference transcript or None
            - x_vector_only_mode: Whether to use speaker embedding only (True) or ICL mode (False)
        """
        if voice_ref is None or not isinstance(voice_ref, dict):
            # For Base model without reference, use default behavior
            return None, None, False

        # Extract reference audio (multiple possible keys)
        ref_audio_original = (voice_ref.get('prompt_audio_path') or
                             voice_ref.get('audio_path') or
                             voice_ref.get('audio') or
                             voice_ref.get('waveform'))

        # Extract reference text (multiple possible keys)
        ref_text = (voice_ref.get('prompt_text') or
                   voice_ref.get('reference_text') or
                   voice_ref.get('text', ''))

        # Extract x_vector_only_mode flag (default to False for ICL mode)
        x_vector_only_mode = voice_ref.get('x_vector_only_mode', False)

        # Convert ref_audio to format expected by Qwen3-TTS
        ref_audio_converted = self._convert_audio_to_qwen_format(ref_audio_original, voice_ref)

        # Return: (converted_audio, ref_text, x_vector_mode, original_audio_for_cache)
        return ref_audio_converted, ref_text, x_vector_only_mode, ref_audio_original

    def _convert_audio_to_qwen_format(self, ref_audio, voice_ref: Dict[str, Any]):
        """
        Convert audio reference to format expected by Qwen3-TTS bundled implementation.

        Qwen3-TTS accepts:
        - String: File path
        - Tuple[np.ndarray, int]: (waveform, sample_rate)

        Args:
            ref_audio: Audio reference (can be torch.Tensor, dict, string, etc.)
            voice_ref: Full voice reference dict (may contain sample_rate)

        Returns:
            Audio in Qwen3-TTS expected format
        """
        import numpy as np
        import torch

        # Already a string path - return as-is
        if isinstance(ref_audio, str):
            return ref_audio

        # ComfyUI audio dict format
        if isinstance(ref_audio, dict) and 'waveform' in ref_audio:
            waveform = ref_audio['waveform']
            sample_rate = ref_audio.get('sample_rate', 24000)

            # Convert torch tensor to numpy
            if isinstance(waveform, torch.Tensor):
                # ComfyUI format is [batch, channels, samples]
                # Convert to [samples] mono
                waveform_np = waveform.squeeze().cpu().numpy()
                if waveform_np.ndim > 1:
                    # Average channels if stereo
                    waveform_np = np.mean(waveform_np, axis=0)
                return (waveform_np.astype(np.float32), int(sample_rate))

        # Raw torch.Tensor (from voice_ref['waveform'])
        if isinstance(ref_audio, torch.Tensor):
            # Get sample rate from voice_ref
            sample_rate = voice_ref.get('sample_rate', 24000)

            # Convert to numpy
            waveform_np = ref_audio.squeeze().cpu().numpy()
            if waveform_np.ndim > 1:
                # Average channels if stereo
                waveform_np = np.mean(waveform_np, axis=0)
            return (waveform_np.astype(np.float32), int(sample_rate))

        # Tuple (waveform, sr) - ensure numpy
        if isinstance(ref_audio, tuple) and len(ref_audio) == 2:
            waveform, sr = ref_audio
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.squeeze().cpu().numpy()
                if waveform.ndim > 1:
                    waveform = np.mean(waveform, axis=0)
            return (waveform.astype(np.float32), int(sr))

        # None or unsupported
        return ref_audio

    def _create_progress_bar(self, max_tokens: int):
        """
        Create ComfyUI progress bar for generation tracking.

        Args:
            max_tokens: Maximum tokens to generate

        Returns:
            Progress bar instance or None if ComfyUI utils not available
        """
        try:
            import comfy.utils
            return comfy.utils.ProgressBar(max_tokens)
        except (ImportError, AttributeError):
            return None  # ComfyUI progress not available

    def _convert_output_to_tensor(self, wavs, sample_rate: int) -> torch.Tensor:
        """
        Convert Qwen3-TTS output to tensor format.

        Args:
            wavs: Output from engine (numpy array or list of numpy arrays)
            sample_rate: Sample rate (should be 24000)

        Returns:
            Audio tensor [1, samples]
        """
        import numpy as np

        # Handle list of arrays (batch generation)
        if isinstance(wavs, list):
            if len(wavs) == 1:
                wavs = wavs[0]
            else:
                # Concatenate multiple outputs
                wavs = np.concatenate(wavs, axis=-1)

        # Convert numpy to tensor
        if isinstance(wavs, np.ndarray):
            audio_tensor = torch.from_numpy(wavs).float()
        elif torch.is_tensor(wavs):
            audio_tensor = wavs.float()
        else:
            raise ValueError(f"Unsupported audio output type: {type(wavs)}")

        # Ensure 2D shape [1, samples]
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # [samples] -> [1, samples]
        elif audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(0)  # [1, 1, samples] -> [1, samples]

        return audio_tensor

    def cleanup(self):
        """
        Clean up resources.

        Note: We don't auto-unload here - let the model stay in VRAM for reuse.
        Only explicit unload (via button) should trigger model deletion.
        """
        # Don't auto-unload - let the model stay in VRAM for reuse
        pass
