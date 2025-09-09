"""
Audio Content Hashing Utilities

Provides consistent content-based hashing for audio inputs across all TTS nodes
to prevent cache invalidation from temporary file paths.
"""

import hashlib
import os
import sys
from typing import Optional, Dict, Any

# 🔬 NUMBA WORKAROUND: Commented out - testing if still needed with numba 0.61.2+ and librosa 0.11.0+
# Python 3.13 compatibility: Disable numba JIT for librosa compatibility
# if sys.version_info >= (3, 13):
#     os.environ['NUMBA_DISABLE_JIT'] = '1'
#     try:
#         import numba
#         numba.config.DISABLE_JIT = True
#     except ImportError:
#         pass


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
            
            # Try multiple audio loading methods for robustness
            waveform = None
            sample_rate = None
            
            # Method 1: Try soundfile first (more reliable, no numba dependencies)
            try:
                import soundfile as sf
                waveform, sample_rate = sf.read(audio_file_path, dtype='float32')
                # print(f"🐛 AUDIO_HASH: Loaded with soundfile")
            except:
                # Method 2: Try torchaudio (also reliable, minimal numba use)
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)  # Suppress torchaudio 2.9 warnings
                        import torchaudio
                        waveform_tensor, sample_rate = torchaudio.load(audio_file_path)
                        waveform = waveform_tensor.numpy().flatten()
                    # print(f"🐛 AUDIO_HASH: Loaded with torchaudio")
                except:
                    # Method 3: Fall back to librosa with enhanced numba protection
                    try:
                        # Force disable numba compilation for this specific import
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)  # Suppress torchaudio warnings
                            
                            # Smart numba compatibility for audio loading
                            from utils.compatibility import setup_numba_compatibility
                            setup_numba_compatibility(quick_startup=True, verbose=False)
                            
                            import librosa
                            waveform, sample_rate = librosa.load(audio_file_path, sr=None)
                            
                        # print(f"🐛 AUDIO_HASH: Loaded with librosa (numba protected)")
                    except Exception as librosa_error:
                        # If all methods fail, just hash the file path + modification time for uniqueness
                        file_stat = os.stat(audio_file_path)
                        path_hash = hashlib.md5(f"{audio_file_path}_{file_stat.st_mtime}_{file_stat.st_size}".encode()).hexdigest()
                        fallback = f"path_{path_hash}"
                        print(f"⚠️ Failed to load audio file for content hashing: {librosa_error}")
                        print(f"🔄 Using path+timestamp hash instead: {fallback}")
                        return fallback
            
            if waveform is not None:
                waveform_hash = hashlib.md5(waveform.tobytes()).hexdigest()
                result = f"file_audio_{waveform_hash}_{sample_rate}"
                # print(f"🐛 AUDIO_HASH: Generated file hash: {result}")
                return result
            else:
                raise Exception("Failed to load audio with all methods")
                
        except Exception as e:
            # Enhanced fallback using file metadata for better cache consistency
            try:
                file_stat = os.stat(audio_file_path)
                path_hash = hashlib.md5(f"{audio_file_path}_{file_stat.st_mtime}_{file_stat.st_size}".encode()).hexdigest()
                fallback = f"path_{path_hash}"
                print(f"⚠️ Audio content hashing failed, using file metadata hash: {fallback}")
                return fallback
            except:
                # Final fallback - just use filename
                fallback = f"path_{os.path.basename(audio_file_path)}"
                print(f"⚠️ Failed to hash audio file {audio_file_path}: {e}, using path fallback")
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