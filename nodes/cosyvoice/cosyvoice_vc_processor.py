"""
CosyVoice3 Voice Conversion Processor
Handles voice conversion using CosyVoice3 engine's inference_vc method
"""

import torch
import numpy as np
import tempfile
import torchaudio
import os
import sys
import hashlib
from typing import Dict, Any, Optional, Tuple

# Add project root to path
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.audio.processing import AudioProcessingUtils

# Global cache for VC iterations (max 10 for CosyVoice - more resilient to refinement passes)
GLOBAL_COSYVOICE_VC_CACHE = {}


class CosyVoiceVCProcessor:
    """
    Voice conversion processor for CosyVoice3 engine.
    Uses CosyVoice3's native inference_vc method for voice conversion.
    """

    SAMPLE_RATE = 24000  # CosyVoice3 native sample rate

    def __init__(self, engine_config: Dict[str, Any], existing_adapter=None):
        """
        Initialize CosyVoice3 VC processor.

        Args:
            engine_config: Engine configuration from CosyVoice3 Engine node
            existing_adapter: Optional existing CosyVoiceAdapter to reuse (avoids model reload)
        """
        self.engine_config = engine_config

        # Extract configuration
        self.model_path = engine_config.get('model_path', 'Fun-CosyVoice3-0.5B')
        self.device = engine_config.get('device', 'auto')
        self.speed = engine_config.get('speed', 1.0)
        self.use_fp16 = engine_config.get('use_fp16', True)
        self.load_trt = engine_config.get('load_trt', False)
        self.load_vllm = engine_config.get('load_vllm', False)

        # Try to reuse existing CosyVoice TTS adapter if one is loaded (avoids model thrashing)
        existing_tts_adapter = None
        if existing_adapter is None:
            try:
                # Check if there's a cached CosyVoice TTS engine in the global engine cache
                from nodes.unified.tts_text_node import UnifiedTTSTextNode
                if hasattr(UnifiedTTSTextNode, '_cached_engine_instances'):
                    for cache_key, cached_entry in UnifiedTTSTextNode._cached_engine_instances.items():
                        if 'cosyvoice' in cache_key.lower() and cached_entry.get('instance'):
                            instance = cached_entry['instance']
                            # Check if it's a CosyVoice processor with an adapter
                            if hasattr(instance, 'adapter') and instance.adapter is not None:
                                existing_tts_adapter = instance.adapter
                                print(f"🔄 CosyVoice VC: Reusing existing TTS engine adapter (prevents model reload)")
                                break
            except Exception:
                pass  # Fallback to creating new adapter

        # Reuse existing adapter if available
        if existing_adapter is not None:
            print(f"🔄 CosyVoice VC: Reusing provided adapter (no model reload)")
            self.adapter = existing_adapter
        elif existing_tts_adapter is not None:
            self.adapter = existing_tts_adapter
        else:
            # Initialize new adapter
            from engines.adapters.cosyvoice_adapter import CosyVoiceAdapter
            self.adapter = CosyVoiceAdapter()
            self.adapter.initialize_engine(
                model_path=self.model_path,
                device=self.device,
                use_fp16=self.use_fp16,
                load_trt=self.load_trt,
                load_vllm=self.load_vllm
            )

        # Track temp files for cleanup
        self._temp_files = []

    def update_config(self, engine_config: Dict[str, Any]):
        """Update processor configuration with new parameters."""
        self.engine_config = engine_config
        self.speed = engine_config.get('speed', 1.0)

    def convert_voice(
        self,
        source_audio: Dict[str, Any],
        target_audio: Dict[str, Any],
        refinement_passes: int = 1
    ) -> Tuple[Dict[str, Any], str]:
        """
        Convert source audio to match target voice using CosyVoice3.

        Args:
            source_audio: Source audio dict with 'waveform' and 'sample_rate'
            target_audio: Target/reference audio dict with 'waveform' and 'sample_rate'
            refinement_passes: Number of iterative refinement passes

        Returns:
            Tuple of (converted_audio_dict, conversion_info)
        """
        print(f"🔄 CosyVoice3 VC: Starting voice conversion with {refinement_passes} pass(es)")

        # Generate cache key for this conversion
        cache_key = self._generate_vc_cache_key(source_audio, target_audio)

        # Check for cached iterations
        cached_iterations = self._get_cached_iterations(cache_key, refinement_passes)

        # If we have the exact number of passes cached, return it immediately
        if refinement_passes in cached_iterations:
            print(f"💾 CACHE HIT: Using cached CosyVoice3 VC result for {refinement_passes} passes")
            return cached_iterations[refinement_passes]

        # CosyVoice inference_vc expects file paths, not tensors
        # Save prompt to temp file (constant across all iterations)
        prompt_path = self._save_audio_to_temp(target_audio, "prompt")

        # Start from the highest cached iteration or from beginning
        start_iteration = 0
        current_audio = source_audio

        # Find the highest cached iteration we can start from
        for i in range(refinement_passes, 0, -1):
            if i in cached_iterations:
                start_iteration = i
                current_audio = cached_iterations[i][0]  # Get audio dict from cached result
                print(f"💾 CACHE: Resuming CosyVoice3 VC from cached iteration {i}/{refinement_passes}")
                break

        # Perform remaining voice conversion iterations
        for iteration in range(start_iteration, refinement_passes):
            iteration_num = iteration + 1
            if refinement_passes > 1:
                print(f"  🔄 VC pass {iteration_num}/{refinement_passes}...")

            # Save current audio to temp file for this iteration
            source_path = self._save_audio_to_temp(current_audio, f"source_iter{iteration_num}")

            # Use CosyVoice's inference_vc method via adapter
            # Returns tensor at 24kHz
            converted_wav = self.adapter.inference_vc(
                source_wav=source_path,
                prompt_wav=prompt_path,
                speed=self.speed
            )

            # Convert output to ComfyUI format for next iteration or final result
            current_audio = self._convert_to_comfyui_format(converted_wav)

            # Cache this iteration result (only up to 10 iterations)
            if iteration_num <= 10:
                duration = current_audio['waveform'].shape[-1] / self.SAMPLE_RATE
                conversion_info = (
                    f"CosyVoice3 VC: {duration:.2f}s | "
                    f"Model: {self.model_path} | "
                    f"Speed: {self.speed}x | "
                    f"Pass: {iteration_num}/{refinement_passes} | "
                    f"Device: {self.device}"
                )
                self._cache_iteration(cache_key, iteration_num, (current_audio, conversion_info))

        # Build final conversion info
        duration = current_audio['waveform'].shape[-1] / self.SAMPLE_RATE
        cache_info = f"(resumed from cache at pass {start_iteration})" if start_iteration > 0 else "(no cache used)"
        conversion_info = (
            f"CosyVoice3 VC: {duration:.2f}s | "
            f"Model: {self.model_path} | "
            f"Speed: {self.speed}x | "
            f"Refinement passes: {refinement_passes} {cache_info} | "
            f"Device: {self.device}"
        )

        print(f"✅ CosyVoice3 VC: Conversion completed ({duration:.2f}s) {cache_info}")
        # Temp files will be cleaned up by __del__ when processor is destroyed
        return current_audio, conversion_info

    def _generate_vc_cache_key(self, source_audio: Dict[str, Any], target_audio: Dict[str, Any]) -> str:
        """Generate cache key for voice conversion iterations"""
        # Create hash from source and target audio characteristics
        source_hash = hashlib.md5(source_audio["waveform"].cpu().numpy().tobytes()).hexdigest()[:16]
        target_hash = hashlib.md5(target_audio["waveform"].cpu().numpy().tobytes()).hexdigest()[:16]

        cache_data = {
            'source_hash': source_hash,
            'target_hash': target_hash,
            'source_sr': source_audio["sample_rate"],
            'target_sr': target_audio["sample_rate"],
            'model_path': self.model_path,
            'speed': self.speed,
            'device': self.device
        }

        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _get_cached_iterations(self, cache_key: str, max_iteration: int) -> Dict[int, Tuple[Dict[str, Any], str]]:
        """Get cached iterations up to max_iteration"""
        if cache_key not in GLOBAL_COSYVOICE_VC_CACHE:
            return {}

        cached_data = GLOBAL_COSYVOICE_VC_CACHE[cache_key]
        return {i: cached_data[i] for i in cached_data if i <= max_iteration}

    def _cache_iteration(self, cache_key: str, iteration: int, result: Tuple[Dict[str, Any], str]):
        """Cache a single iteration result (limit to 10 iterations max)"""
        if cache_key not in GLOBAL_COSYVOICE_VC_CACHE:
            GLOBAL_COSYVOICE_VC_CACHE[cache_key] = {}

        # Only cache up to 10 iterations to prevent memory issues
        if iteration <= 10:
            GLOBAL_COSYVOICE_VC_CACHE[cache_key][iteration] = result

    def _save_audio_to_temp(self, audio_dict: Dict[str, Any], prefix: str) -> str:
        """
        Save audio to temporary WAV file for CosyVoice.
        CosyVoice's inference_vc expects file paths, handles resampling internally.

        Args:
            audio_dict: ComfyUI audio dict with 'waveform' and 'sample_rate'
            prefix: Prefix for temp file name (e.g., "source", "prompt")

        Returns:
            Path to temporary WAV file
        """
        waveform = audio_dict['waveform']
        sample_rate = audio_dict['sample_rate']

        # Normalize to 2D for torchaudio.save (channels, samples)
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu()
        else:
            waveform = torch.from_numpy(np.array(waveform))

        while waveform.dim() > 2:
            waveform = waveform.squeeze(0)

        # Convert to mono if stereo (CosyVoice VC expects mono)
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", prefix=f"cosyvoice_vc_{prefix}_", delete=False)
        temp_file.close()

        # Save audio (CosyVoice will resample to 16kHz internally)
        torchaudio.save(temp_file.name, waveform, sample_rate)
        self._temp_files.append(temp_file.name)

        return temp_file.name

    def _save_tensor_to_temp(self, audio_tensor: torch.Tensor, prefix: str) -> str:
        """
        Save tensor to temporary WAV file for iterative refinement.

        Args:
            audio_tensor: Audio tensor from CosyVoice (24kHz)
            prefix: Prefix for temp file name

        Returns:
            Path to temporary WAV file
        """
        # Ensure 2D shape for torchaudio.save
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        elif audio_tensor.dim() > 2:
            while audio_tensor.dim() > 2:
                audio_tensor = audio_tensor.squeeze(0)

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", prefix=f"cosyvoice_vc_{prefix}_", delete=False)
        temp_file.close()

        # Save at CosyVoice's native 24kHz
        torchaudio.save(temp_file.name, audio_tensor.cpu(), self.SAMPLE_RATE)
        self._temp_files.append(temp_file.name)

        return temp_file.name

    def _cleanup_temp_files(self):
        """Clean up all temporary files."""
        for temp_path in self._temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                print(f"⚠️ Failed to delete temp file {temp_path}: {e}")
        self._temp_files.clear()

    def _convert_to_comfyui_format(self, audio_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Convert CosyVoice output to ComfyUI audio format.

        Args:
            audio_tensor: Audio tensor from CosyVoice (various shapes)

        Returns:
            ComfyUI audio dict with 'waveform' [batch, channels, samples] and 'sample_rate'
        """
        # Ensure tensor
        if not isinstance(audio_tensor, torch.Tensor):
            audio_tensor = torch.from_numpy(np.array(audio_tensor))

        # Normalize to [batch, channels, samples]
        if audio_tensor.dim() == 1:
            # [samples] -> [1, 1, samples]
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        elif audio_tensor.dim() == 2:
            # [channels, samples] or [batch, samples] -> [1, channels, samples]
            audio_tensor = audio_tensor.unsqueeze(0)

        return {
            'waveform': audio_tensor,
            'sample_rate': self.SAMPLE_RATE
        }

    def __del__(self):
        """Cleanup temporary files when processor is destroyed."""
        self._cleanup_temp_files()

    def cleanup(self):
        """Clean up resources and temp files"""
        self._cleanup_temp_files()
        if self.adapter:
            self.adapter.cleanup()
            self.adapter = None
