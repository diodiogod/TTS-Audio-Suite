"""
Audio Content Hashing Utilities

Provides consistent content-based hashing for audio inputs across all TTS nodes
to prevent cache invalidation from temporary file paths.
"""

import hashlib
import os
from typing import Optional, Dict, Any


def generate_stable_audio_component(reference_audio: Optional[Dict[str, Any]] = None, 
                                   audio_file_path: Optional[str] = None) -> str:
    """
    Generate stable audio component identifier for cache consistency.
    
    Uses content-based hashing to ensure same audio content produces same cache key,
    regardless of temporary file names or paths.
    
    Args:
        reference_audio: Direct audio input dict with 'waveform' and 'sample_rate'
        audio_file_path: Path to audio file (from dropdown selection or direct input)
        
    Returns:
        Stable identifier string for cache key generation
    """
    if reference_audio is not None:
        # For direct audio input, hash the waveform data
        try:
            # print(f"🐛 AUDIO_HASH: Hashing reference_audio with keys: {list(reference_audio.keys())}")
            waveform_hash = hashlib.md5(reference_audio["waveform"].cpu().numpy().tobytes()).hexdigest()
            result = f"ref_audio_{waveform_hash}_{reference_audio['sample_rate']}"
            # print(f"🐛 AUDIO_HASH: Generated hash: {result}")
            return result
        except Exception as e:
            print(f"⚠️ Failed to hash reference audio: {e}")
            # print(f"🐛 AUDIO_HASH: ERROR - returning 'ref_audio_error'")
            return "ref_audio_error"
    
    elif audio_file_path and audio_file_path != "none" and os.path.exists(audio_file_path):
        # For file paths (dropdown selections or temp files), hash file content
        try:
            # print(f"🐛 AUDIO_HASH: Hashing file path: {audio_file_path}")
            import librosa
            # Load audio file and create hash from content
            waveform, sample_rate = librosa.load(audio_file_path, sr=None)
            waveform_hash = hashlib.md5(waveform.tobytes()).hexdigest()
            result = f"file_audio_{waveform_hash}_{sample_rate}"
            # print(f"🐛 AUDIO_HASH: Generated file hash: {result}")
            return result
        except Exception as e:
            # Fallback to path if file reading fails
            print(f"⚠️ Failed to hash audio file {audio_file_path}: {e}, using path fallback")
            fallback = f"path_{os.path.basename(audio_file_path)}"
            # print(f"🐛 AUDIO_HASH: Using fallback: {fallback}")
            return fallback
    
    else:
        # No voice file (default voice)
        # print(f"🐛 AUDIO_HASH: No audio provided - using 'default_voice'")
        return "default_voice"


def get_stable_audio_component_for_cache(inputs: Dict[str, Any]) -> str:
    """
    Convenience function to get stable audio component from node inputs.
    
    Args:
        inputs: Node inputs dict containing 'reference_audio' and/or 'audio_prompt_path'
        
    Returns:
        Stable identifier for cache key
    """
    return generate_stable_audio_component(
        reference_audio=inputs.get("reference_audio"),
        audio_file_path=inputs.get("audio_prompt_path", "")
    )