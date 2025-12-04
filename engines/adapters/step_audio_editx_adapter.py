"""
Step Audio EditX Engine Adapter

Provides standardized interface for Step Audio EditX integration with TTS Audio Suite.
Handles parameter mapping, voice cloning, pause tag processing, and seed control.
"""

import os
import sys
import torch
import tempfile
import torchaudio
from typing import Dict, Any, Optional, List, Union

# Add project root to path
current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engines.step_audio_editx.step_audio_editx import StepAudioEditXEngine
from utils.text.pause_processor import PauseTagProcessor
from utils.audio.cache import get_audio_cache
import folder_paths


class StepAudioEditXEngineAdapter:
    """
    Adapter for Step Audio EditX engine providing unified interface compatibility.

    Handles:
    - Parameter mapping between unified interface and Step Audio EditX
    - Voice cloning with prompt audio/text
    - Pause tag processing and timing
    - Seed control via global torch state
    - Caching integration
    - Model management
    """

    def __init__(self, node_instance):
        """
        Initialize the Step Audio EditX adapter.

        Args:
            node_instance: Parent node instance for context
        """
        self.node = node_instance
        self.engine = None
        self.audio_cache = get_audio_cache()

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

    def load_base_model(self,
                       model_path: str,
                       device: str = "auto",
                       torch_dtype: str = "bfloat16",
                       quantization: Optional[str] = None):
        """
        Load Step Audio EditX engine.

        Args:
            model_path: Model identifier (local:ModelName or ModelName for auto-download)
            device: Target device (auto/cuda/cpu)
            torch_dtype: Model precision (bfloat16/float16/float32/auto)
            quantization: Quantization mode (int4/int8 or None)
        """
        self.engine = StepAudioEditXEngine(
            model_dir=model_path,
            device=device,
            torch_dtype=torch_dtype,
            quantization=quantization
        )

    def generate_with_pause_tags(self,
                                 text: str,
                                 voice_ref: Optional[Dict[str, Any]],
                                 params: Dict[str, Any],
                                 process_pauses: bool = True,
                                 character_name: Optional[str] = None) -> torch.Tensor:
        """
        Generate speech with pause tag processing (TTS Suite integration).

        Args:
            text: Input text (may contain <pause_X> tags)
            voice_ref: Voice reference dict with 'prompt_audio_path' and 'prompt_text'
            params: Generation parameters (including seed)
            process_pauses: Whether to process pause tags
            character_name: Character name for logging

        Returns:
            Generated audio tensor [1, samples] at 24000 Hz
        """
        if self.engine is None:
            raise RuntimeError("Engine not loaded. Call load_base_model() first.")

        # Extract seed from params
        seed = params.get('seed', 0)

        # Set global torch seed for reproducibility (Step Audio EditX uses global torch state)
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
            voice_ref: Voice reference dict
            params: Generation parameters
            character_name: Character name for logging

        Returns:
            Audio tensor [1, samples]
        """
        # Get voice reference paths
        prompt_audio_path, prompt_text = self._extract_voice_reference(voice_ref)

        # Generate cache key and check cache
        from utils.audio.audio_hash import generate_stable_audio_component
        audio_component = generate_stable_audio_component(
            audio_file_path=prompt_audio_path
        ) if prompt_audio_path else "default_voice"

        cache_key = self.audio_cache.generate_cache_key(
            'step_audio_editx',
            text=text,
            audio_component=audio_component,
            prompt_text=prompt_text,
            temperature=params.get('temperature', 0.7),
            do_sample=params.get('do_sample', True),
            max_new_tokens=params.get('max_new_tokens', 8192),
            seed=params.get('seed', 0),
            model_path=params.get('model_path', 'Step-Audio-EditX'),
            device=params.get('device', 'auto'),
            torch_dtype=params.get('torch_dtype', 'bfloat16'),
            quantization=params.get('quantization', None),
            character=character_name or 'narrator'
        )

        # Check cache
        cached_audio = self.audio_cache.get_cached_audio(cache_key)
        if cached_audio:
            char_desc = character_name or 'narrator'
            print(f"💾 Using cached Step Audio EditX audio for '{char_desc}': '{text[:30]}...'")
            return cached_audio[0]

        # Create ComfyUI progress bar for generation tracking with time prediction
        max_new_tokens = params.get('max_new_tokens', 8192)
        progress_bar = None
        try:
            import comfy.utils
            import time

            # Get job tracker from adapter (for total job time estimation)
            job_tracker = self.job_tracker

            class TimedProgressBar:
                """Progress bar wrapper with time tracking and prediction."""
                def __init__(self, total, tracker):
                    self.total = total
                    self.current = 0
                    self.start_time = time.time()
                    self.wrapped = comfy.utils.ProgressBar(total)
                    self.last_print = 0
                    self.tracker = tracker

                def update(self, delta=1):
                    self.current += delta
                    self.wrapped.update(delta)

                def get_job_elapsed(self):
                    """Get total job elapsed time in seconds."""
                    if self.tracker:
                        return time.time() - self.tracker['start_time']
                    return None

                def get_job_remaining_str(self):
                    """Calculate and return job remaining time string based on actual data."""
                    if not self.tracker or self.tracker['total_text'] <= 0:
                        return None

                    total_elapsed = time.time() - self.tracker['start_time']
                    if total_elapsed <= 0:
                        return None

                    text_completed = self.tracker['text_completed']
                    current_block_text = self.tracker['current_block_text']
                    total_text = self.tracker['total_text']
                    total_tokens_completed = self.tracker.get('total_tokens', 0)
                    total_tokens = total_tokens_completed + self.current

                    # Update current block tokens in tracker
                    self.tracker['current_block_tokens'] = self.current

                    if total_tokens <= 0:
                        return None

                    # Wait for more data before showing prediction
                    # Need at least 1 completed chunk OR 150+ tokens in first chunk
                    if total_tokens_completed == 0 and self.current < 150:
                        return None

                    # Estimate how much of current block is done based on tokens
                    # Use completed blocks to estimate tokens per char
                    if text_completed > 0 and total_tokens_completed > 0:
                        tokens_per_char = total_tokens_completed / text_completed
                    elif current_block_text > 0 and self.current > 0:
                        # First block - use conservative estimate based on current data
                        # Typical ratio is ~0.5-1.0 tokens per char
                        tokens_per_char = 0.7
                    else:
                        tokens_per_char = 0.7

                    # Estimate expected tokens for current block
                    expected_current_tokens = current_block_text * tokens_per_char if tokens_per_char > 0 else self.current * 2

                    # Current block progress (0 to 1)
                    current_block_progress = min(self.current / expected_current_tokens, 0.99) if expected_current_tokens > 0 else 0.5

                    # Text done = completed + portion of current block
                    effective_text_done = text_completed + (current_block_text * current_block_progress)

                    # Remaining text
                    remaining_text = total_text - effective_text_done

                    # Current speed (tokens per second)
                    tokens_per_sec = total_tokens / total_elapsed

                    # Remaining tokens estimate
                    remaining_tokens = remaining_text * tokens_per_char if tokens_per_char > 0 else remaining_text

                    # Remaining time
                    if tokens_per_sec > 0 and remaining_tokens > 0:
                        remaining_secs = remaining_tokens / tokens_per_sec

                        if remaining_secs < 60:
                            return f"~{remaining_secs:.0f}s left"
                        else:
                            return f"~{remaining_secs/60:.1f}m left"

                    # If no remaining text but still processing, show nothing
                    return None

            progress_bar = TimedProgressBar(max_new_tokens, job_tracker)
        except (ImportError, AttributeError):
            pass  # ComfyUI progress not available

        # Generate using clone mode
        audio_tensor = self.engine.clone(
            prompt_wav_path=prompt_audio_path,
            prompt_text=prompt_text,
            target_text=text,
            temperature=params.get('temperature', 0.7),
            do_sample=params.get('do_sample', True),
            max_new_tokens=max_new_tokens,
            progress_bar=progress_bar
        )

        # Cache the result
        duration = self.audio_cache._calculate_duration(audio_tensor, 'step_audio_editx')
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
            voice_ref: Voice reference dict
            params: Generation parameters
            character_name: Character name for logging

        Returns:
            Combined audio tensor [1, samples]
        """
        # Parse pause tags - returns list of tuples: ('text', content) or ('pause', duration_seconds)
        segments, clean_text = PauseTagProcessor.parse_pause_tags(text)

        print(f"🎵 Processing {len(segments)} pause-delimited segments for '{character_name or 'unknown'}'")

        audio_parts = []
        sample_rate = 24000  # Step Audio EditX native sample rate

        for segment_type, content in segments:
            if segment_type == 'text':
                # Generate audio for text segment
                audio_tensor = self._generate_direct(content, voice_ref, params, character_name)

                # Ensure correct shape [1, samples]
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)

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
        Extract voice reference audio path and text from voice_ref dict.

        Args:
            voice_ref: Voice reference dict (from voice discovery)

        Returns:
            Tuple of (prompt_audio_path, prompt_text)
        """
        if voice_ref is None or not isinstance(voice_ref, dict):
            raise ValueError(
                "Step Audio EditX requires voice reference. "
                "Please provide voice_ref with 'prompt_audio_path' and 'prompt_text'"
            )

        # Extract paths
        prompt_audio_path = voice_ref.get('prompt_audio_path') or voice_ref.get('audio_path')
        prompt_text = voice_ref.get('prompt_text') or voice_ref.get('reference_text', '')

        if not prompt_audio_path:
            raise ValueError(
                "Voice reference missing 'prompt_audio_path'. "
                "Ensure your voice file is in voices/ directory and properly detected."
            )

        if not prompt_text or not prompt_text.strip():
            raise ValueError(
                f"Voice reference missing 'prompt_text' for {prompt_audio_path}. "
                "Step Audio EditX requires transcription of reference audio. "
                "Add prompt_text to your voice.json file."
            )

        return prompt_audio_path, prompt_text

    def cleanup(self):
        """Clean up resources"""
        if self.engine:
            self.engine.unload()
            self.engine = None
