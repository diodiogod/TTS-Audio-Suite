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
            return self._generate_direct(text, voice_ref, params)

    def _generate_direct(self,
                        text: str,
                        voice_ref: Optional[Dict[str, Any]],
                        params: Dict[str, Any]) -> torch.Tensor:
        """
        Direct generation without pause processing.

        Args:
            text: Input text (clean, no pause tags)
            voice_ref: Voice reference dict
            params: Generation parameters

        Returns:
            Audio tensor [1, samples]
        """
        # Get voice reference paths
        prompt_audio_path, prompt_text = self._extract_voice_reference(voice_ref)

        # Create ComfyUI progress bar for generation tracking
        max_new_tokens = params.get('max_new_tokens', 8192)
        progress_bar = None
        try:
            import comfy.utils
            progress_bar = comfy.utils.ProgressBar(max_new_tokens)
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
                audio_tensor = self._generate_direct(content, voice_ref, params)

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
