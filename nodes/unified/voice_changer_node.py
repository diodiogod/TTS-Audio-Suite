"""
Unified Voice Changer Node - Engine-agnostic voice conversion for TTS Audio Suite
Refactored from ChatterBox VC to support multiple engines (ChatterBox now, RVC in future)
"""

import torch
import numpy as np
import tempfile
import os
import hashlib
from typing import Dict, Any, List

# Use direct file imports that work when loaded via importlib
import os
import sys
import importlib.util

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load base_node module directly
base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

# Import the base class
BaseVCNode = base_module.BaseVCNode

from utils.audio.processing import AudioProcessingUtils
from utils.config_sanitizer import ConfigSanitizer
from utils.comfyui_compatibility import ensure_python312_cudnn_fix
import comfy.model_management as model_management

# AnyType for flexible input types (accepts any data type)
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")

# Global cache for RVC iteration results (similar to ChatterBox)
GLOBAL_RVC_ITERATION_CACHE = {}


class UnifiedVoiceChangerNode(BaseVCNode):
    """
    Unified Voice Changer Node - Engine-agnostic voice conversion.
    Currently supports ChatterBox, prepared for future RVC and other voice conversion engines.
    Replaces ChatterBox VC node with engine-agnostic architecture.
    """
    
    @classmethod
    def NAME(cls):
        return "🔄 Voice Changer"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "TTS_engine": ("TTS_ENGINE", {
                    "tooltip": "TTS/VC engine configuration. Supports ChatterBox TTS Engine and RVC Engine for voice conversion."
                }),
                "source_audio": (any_typ, {
                    "tooltip": "The original voice audio you want to convert to sound like the target voice. Accepts AUDIO input or Character Voices node output."
                }),
                "narrator_target": (any_typ, {
                    "tooltip": "The reference voice audio whose characteristics will be applied to the source audio. Accepts AUDIO input or Character Voices node output."
                }),
                "refinement_passes": ("INT", {
                    "default": 1, "min": 1, "max": 30, "step": 1,
                    "tooltip": "Number of conversion iterations. Each pass refines the output to sound more like the target. Recommended: Max 5 passes - more can cause distortions. Each iteration is deterministic to reduce degradation."
                }),
                "max_chunk_duration": ("INT", {
                    "default": 30, "min": 0, "max": 300, "step": 5,
                    "tooltip": "Maximum duration (in seconds) for each audio chunk. Prevents OOM on long audio. Set to 0 to disable chunking entirely (process full audio at once). Default 30s is safe for ChatterBox. RVC users can disable (0) or use higher values (60-120s). Increase if you have high VRAM (>16GB)."
                }),
                "chunk_method": (["smart", "fixed"], {
                    "default": "smart",
                    "tooltip": "Chunking method: 'smart' splits at silences for natural boundaries, 'fixed' splits at exact time intervals. Smart mode produces better quality by avoiding mid-word cuts."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("converted_audio", "conversion_info")
    FUNCTION = "convert_voice"
    CATEGORY = "TTS Audio Suite/🎤 Text to Speech"

    def __init__(self):
        super().__init__()
        # Cache engine instances to prevent model reloading
        self._cached_engine_instances = {}

    def _extract_audio_from_input(self, audio_input, input_name: str):
        """
        Extract audio tensor using base class universal normalizer.
        Supports AUDIO dict, Character Voices output, VideoHelper LazyAudioMap, etc.
        
        Args:
            audio_input: Audio input in any supported format
            input_name: Name of input for error messages
            
        Returns:
            Standard AUDIO dict suitable for voice conversion engines
        """
        try:
            # Use base class audio normalizer (handles all formats)
            normalized_audio = self.normalize_audio_input(audio_input, input_name)
            
            # Log the source type for debugging
            if isinstance(audio_input, dict) and "audio" in audio_input:
                character_name = audio_input.get("character_name", "unknown")
                print(f"🔄 Voice Changer: Using {input_name} from Character Voices node ({character_name})")
            elif hasattr(audio_input, "get"):
                print(f"🔄 Voice Changer: Using direct audio input for {input_name}")
            else:
                print(f"🔄 Voice Changer: Using VideoHelper-compatible audio input for {input_name}")
            
            return normalized_audio
            
        except Exception as e:
            raise ValueError(f"Failed to process {input_name}: {e}")

    def _handle_rvc_conversion(self, rvc_engine, source_audio, narrator_target, refinement_passes):
        """
        Handle RVC engine voice conversion with iterative refinement support.
        
        Args:
            rvc_engine: RVCEngineAdapter instance
            source_audio: Source audio to convert
            narrator_target: Target voice characteristics (RVC_MODEL from Load RVC Character Model)
            refinement_passes: Number of conversion passes for iterative refinement
            
        Returns:
            Tuple of (converted_audio, conversion_info)
        """
        try:
            # Extract audio data from flexible inputs
            processed_source_audio = self._extract_audio_from_input(source_audio, "source_audio")
            
            # For RVC, narrator_target should be RVC_MODEL from 🎭 Load RVC Character Model
            print(f"🔄 Voice Changer: RVC conversion with {refinement_passes} refinement passes")
            
            # Check if narrator_target is an RVC_MODEL
            rvc_model = None
            if narrator_target and isinstance(narrator_target, dict) and narrator_target.get('type') == 'rvc_model':
                rvc_model = narrator_target
                print(f"📥 Using RVC Character Model: {rvc_model.get('model_name', 'Unknown')}")
            else:
                print("⚠️  Warning: narrator_target should be RVC Character Model for RVC conversion")
                print("🔄 Attempting conversion without specific model...")
            
            # Get RVC configuration from engine
            config = getattr(rvc_engine, 'config', {})
            
            # Generate cache key for this conversion
            cache_key = self._generate_rvc_cache_key(processed_source_audio, rvc_model, config)
            
            # Check for cached iterations
            cached_iterations = self._get_cached_rvc_iterations(cache_key, refinement_passes)
            
            # If we have the exact number of passes cached, return it immediately
            if refinement_passes in cached_iterations:
                print(f"💾 CACHE HIT: Using cached RVC conversion result for {refinement_passes} passes")
                return cached_iterations[refinement_passes]
            
            # Start from the highest cached iteration or from beginning
            start_iteration = 0
            current_audio = processed_source_audio
            
            # Find the highest cached iteration we can start from
            for i in range(refinement_passes, 0, -1):
                if i in cached_iterations:
                    print(f"💾 CACHE: Resuming RVC conversion from cached iteration {i}/{refinement_passes}")
                    current_audio = cached_iterations[i][0]  # Get the audio from cached result
                    start_iteration = i
                    break
            
            # Perform iterative RVC conversion
            for iteration in range(start_iteration, refinement_passes):
                iteration_num = iteration + 1
                print(f"🔄 RVC conversion pass {iteration_num}/{refinement_passes}...")
                
                # Convert audio tensor to format expected by RVC
                audio_input = self._convert_audio_for_rvc(current_audio)
                
                # Perform RVC conversion using the adapter with RVC model
                converted_audio_np, sample_rate = rvc_engine.convert_voice(
                    audio_input=audio_input,
                    rvc_model=rvc_model,  # Pass the RVC model from narrator_target
                    pitch_shift=config.get('pitch_shift', 0),
                    index_rate=config.get('index_rate', 0.75),
                    rms_mix_rate=config.get('rms_mix_rate', 0.25),
                    protect=config.get('protect', 0.25),
                    f0_method=config.get('f0_method', 'rmvpe'),
                    resample_sr=config.get('resample_sr', 0),
                    crepe_hop_length=160,
                    hubert_path=config.get('hubert_path'),  # Pass user's HuBERT model selection
                )
                
                # Convert back to ComfyUI audio format for next iteration
                converted_audio = self._convert_audio_from_rvc(converted_audio_np, sample_rate)
                current_audio = converted_audio
                
                # Cache this iteration result
                model_name = rvc_model.get('model_name', 'Unknown') if rvc_model else 'No Model'
                conversion_info = (
                    f"RVC Conversion: {model_name} model | "
                    f"Pitch: {config.get('pitch_shift', 0)} | "
                    f"Method: {config.get('f0_method', 'rmvpe')} | "
                    f"Index Rate: {config.get('index_rate', 0.75)} | "
                    f"Device: {config.get('device', 'auto')} | "
                    f"Pass: {iteration_num}/{refinement_passes}"
                )
                
                # Cache the result for this iteration
                self._cache_rvc_result(cache_key, iteration_num, (converted_audio, conversion_info))
            
            # Determine if we used cache
            cache_info = f"(resumed from cache at pass {start_iteration})" if start_iteration > 0 else "(no cache used)"
            
            # Final conversion info
            final_conversion_info = (
                f"RVC Conversion: {model_name} model | "
                f"Pitch: {config.get('pitch_shift', 0)} | "
                f"Method: {config.get('f0_method', 'rmvpe')} | "
                f"Index Rate: {config.get('index_rate', 0.75)} | "
                f"Device: {config.get('device', 'auto')} | "
                f"Refinement passes: {refinement_passes} {cache_info}"
            )
            
            print(f"✅ RVC voice conversion completed with {refinement_passes} refinement passes {cache_info}")
            return converted_audio, final_conversion_info
            
        except Exception as e:
            print(f"❌ RVC voice conversion failed: {e}")
            raise RuntimeError(f"RVC voice conversion failed: {e}")

    def _convert_audio_for_rvc(self, audio_dict):
        """Convert ComfyUI audio format to RVC-compatible format."""
        try:
            if not isinstance(audio_dict, dict) or "waveform" not in audio_dict:
                raise ValueError("Invalid audio format for RVC conversion")
            
            waveform = audio_dict["waveform"]
            sample_rate = audio_dict.get("sample_rate", 24000)
            
            # Convert tensor to numpy if needed
            if hasattr(waveform, 'numpy'):
                # Handle BFloat16 tensors which numpy can't directly convert (defensive programming)
                if hasattr(waveform, 'dtype') and waveform.dtype == torch.bfloat16:
                    audio_np = waveform.to(torch.float32).numpy()
                else:
                    audio_np = waveform.numpy()
            elif isinstance(waveform, torch.Tensor):
                # Handle BFloat16 tensors
                if waveform.dtype == torch.bfloat16:
                    audio_np = waveform.detach().cpu().to(torch.float32).numpy()
                else:
                    audio_np = waveform.detach().cpu().numpy()
            else:
                audio_np = waveform
            
            # Ensure mono audio - RVC expects 1D audio
            if audio_np.ndim > 1:
                if audio_np.shape[0] == 1:  # (1, samples)
                    audio_np = audio_np[0]
                elif audio_np.shape[1] == 1:  # (samples, 1)
                    audio_np = audio_np[:, 0]
                else:  # Multiple channels
                    audio_np = audio_np.mean(axis=0 if audio_np.shape[0] < audio_np.shape[1] else 1)
            
            return (audio_np, sample_rate)
            
        except Exception as e:
            raise ValueError(f"Failed to convert audio for RVC: {e}")

    def _convert_audio_from_rvc(self, audio_np, sample_rate):
        """Convert RVC output back to ComfyUI audio format."""
        try:
            # Ensure numpy array
            if not isinstance(audio_np, np.ndarray):
                audio_np = np.array(audio_np)
            
            # Ensure float32 in range [-1, 1]
            if audio_np.dtype != np.float32:
                if audio_np.dtype == np.int16:
                    audio_np = audio_np.astype(np.float32) / 32768.0
                else:
                    audio_np = audio_np.astype(np.float32)
            
            # Ensure audio is in proper range
            if np.max(np.abs(audio_np)) > 1.0:
                audio_np = audio_np / np.max(np.abs(audio_np))
            
            # Convert to tensor in ComfyUI format (batch, channels, samples)
            if audio_np.ndim == 1:
                # Mono: (samples,) -> (1, 1, samples)
                waveform = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
            else:
                # Multi-channel: ensure proper shape
                if audio_np.shape[0] > audio_np.shape[1]:
                    audio_np = audio_np.T
                waveform = torch.from_numpy(audio_np).unsqueeze(0)
            
            return {
                "waveform": waveform,
                "sample_rate": sample_rate
            }
            
        except Exception as e:
            raise ValueError(f"Failed to convert RVC output to ComfyUI format: {e}")

    def _create_proper_engine_node_instance(self, engine_data: Dict[str, Any]):
        """
        Create a proper engine VC node instance that has all the needed functionality.
        Uses caching to reuse instances and preserve model state across conversions.
        
        Args:
            engine_data: Engine configuration from TTS_engine input
            
        Returns:
            Proper engine VC node instance with all functionality
        """
        try:
            engine_type = engine_data.get("engine_type")
            config = engine_data.get("config", {})

            # Resolve device in config to prevent cache misses when switching between "auto" and actual device
            from utils.device import resolve_torch_device
            config_for_cache = dict(config)  # Make a copy to avoid modifying original
            if 'device' in config_for_cache:
                config_for_cache['device'] = resolve_torch_device(config_for_cache.get('device', 'auto'))

            # Create cache key based on engine type and stable config
            cache_key = f"{engine_type}_{hashlib.md5(str(sorted(config_for_cache.items())).encode()).hexdigest()[:8]}"
            
            # Check if we have a cached instance with the same configuration
            if cache_key in self._cached_engine_instances:
                cached_data = self._cached_engine_instances[cache_key]
                
                # Handle both old (direct instance) and new (timestamped dict) cache formats
                if isinstance(cached_data, dict) and 'instance' in cached_data:
                    # New timestamped format
                    cached_instance = cached_data['instance']
                    cache_timestamp = cached_data['timestamp']
                    
                    # Check if cache is still valid (not invalidated by model unloading)
                    from utils.models.comfyui_model_wrapper import is_engine_cache_valid
                    if is_engine_cache_valid(cache_timestamp):
                        print(f"🔄 Reusing cached {engine_type} VC engine instance (preserves model state)")
                        return cached_instance
                    else:
                        # Cache invalidated by model unloading, remove it
                        print(f"🗑️ Removing invalidated {engine_type} VC engine cache (models were unloaded)")
                        del self._cached_engine_instances[cache_key]
                else:
                    # Old format (direct instance) - assume invalid and remove
                    print(f"🗑️ Removing old-format {engine_type} VC engine cache (upgrading to timestamped format)")
                    del self._cached_engine_instances[cache_key]
            
            if engine_type == "chatterbox":
                # print(f"🔧 Creating new {engine_type} VC engine instance")
                
                # Import and create the original ChatterBox VC node using absolute import
                chatterbox_vc_path = os.path.join(nodes_dir, "chatterbox", "chatterbox_vc_node.py")
                chatterbox_vc_spec = importlib.util.spec_from_file_location("chatterbox_vc_module", chatterbox_vc_path)
                chatterbox_vc_module = importlib.util.module_from_spec(chatterbox_vc_spec)
                chatterbox_vc_spec.loader.exec_module(chatterbox_vc_module)
                
                ChatterboxVCNode = chatterbox_vc_module.ChatterboxVCNode
                engine_instance = ChatterboxVCNode()
                # Apply configuration
                for key, value in config.items():
                    if hasattr(engine_instance, key):
                        setattr(engine_instance, key, value)
                
                # Cache the instance with timestamp
                import time
                self._cached_engine_instances[cache_key] = {
                    'instance': engine_instance,
                    'timestamp': time.time()
                }
                return engine_instance
                
            elif engine_type == "chatterbox_official_23lang":
                # Import and create the ChatterBox Official 23-Lang VC processor
                chatterbox_23lang_vc_path = os.path.join(nodes_dir, "chatterbox_official_23lang", "chatterbox_official_23lang_vc_processor.py")
                chatterbox_23lang_vc_spec = importlib.util.spec_from_file_location("chatterbox_official_23lang_vc_module", chatterbox_23lang_vc_path)
                chatterbox_23lang_vc_module = importlib.util.module_from_spec(chatterbox_23lang_vc_spec)
                chatterbox_23lang_vc_spec.loader.exec_module(chatterbox_23lang_vc_module)
                
                ChatterboxOfficial23LangVCProcessor = chatterbox_23lang_vc_module.ChatterboxOfficial23LangVCProcessor
                engine_instance = ChatterboxOfficial23LangVCProcessor()
                # Apply configuration
                for key, value in config.items():
                    if hasattr(engine_instance, key):
                        setattr(engine_instance, key, value)
                
                # Cache the instance with timestamp
                import time
                self._cached_engine_instances[cache_key] = {
                    'instance': engine_instance,
                    'timestamp': time.time()
                }
                return engine_instance
                
            elif engine_type == "f5tts":
                # F5-TTS doesn't have voice conversion capability
                raise ValueError("F5-TTS engine does not support voice conversion. Use ChatterBox engine for voice conversion.")
                
            else:
                raise ValueError(f"Engine type '{engine_type}' does not support voice conversion. Currently supported: ChatterBox")
                
        except Exception as e:
            print(f"❌ Failed to create engine VC node instance: {e}")
            return None

    def _split_audio_into_chunks(self, audio: torch.Tensor, sample_rate: int,
                                 max_duration: int, method: str = "smart") -> List[torch.Tensor]:
        """
        Split audio into chunks using intelligent silence detection or fixed time intervals.

        Args:
            audio: Audio tensor [batch, channels, samples] or [channels, samples]
            sample_rate: Audio sample rate
            max_duration: Maximum chunk duration in seconds (0 = disabled)
            method: "smart" for silence-based or "fixed" for time-based splitting

        Returns:
            List of audio chunk tensors
        """
        # Normalize audio to [batch, channels, samples]
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)  # Add batch dimension
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        total_samples = audio.shape[-1]
        duration = total_samples / sample_rate

        # Chunking disabled (max_duration = 0) or audio is shorter than max duration
        if max_duration == 0:
            print(f"📊 Audio duration: {duration:.1f}s - chunking disabled, processing full audio")
            return [audio]

        if duration <= max_duration:
            print(f"📊 Audio duration: {duration:.1f}s - no chunking needed")
            return [audio]

        print(f"📊 Audio duration: {duration:.1f}s - splitting into chunks (max {max_duration}s each)")

        if method == "smart":
            return self._smart_chunk_audio(audio, sample_rate, max_duration)
        else:
            return self._fixed_chunk_audio(audio, sample_rate, max_duration)

    def _smart_chunk_audio(self, audio: torch.Tensor, sample_rate: int,
                          max_duration: int) -> List[torch.Tensor]:
        """
        Split audio at silence points near target chunk boundaries.

        Args:
            audio: Audio tensor [batch, channels, samples]
            sample_rate: Audio sample rate
            max_duration: Target chunk duration in seconds

        Returns:
            List of audio chunk tensors
        """
        # Detect silences
        silence_threshold = 0.01
        min_silence_duration = 0.1

        silent_regions = AudioProcessingUtils.detect_silence(
            audio, silence_threshold, min_silence_duration, sample_rate
        )

        print(f"🔍 Silence detection: Found {len(silent_regions)} silent regions")
        if silent_regions:
            print(f"   First silence: {silent_regions[0][0]:.2f}s - {silent_regions[0][1]:.2f}s")
            print(f"   Last silence: {silent_regions[-1][0]:.2f}s - {silent_regions[-1][1]:.2f}s")

        if not silent_regions:
            print("⚠️ No silences detected - falling back to fixed chunking")
            return self._fixed_chunk_audio(audio, sample_rate, max_duration)

        # Find optimal split points
        chunks = []
        current_pos = 0
        total_samples = audio.shape[-1]
        search_window = 5.0  # ±5 seconds search window

        while current_pos < total_samples:
            target_end = min(current_pos + int(max_duration * sample_rate), total_samples)

            # Find best silence within search window
            search_start = max(0, (target_end / sample_rate) - search_window)
            search_end = min(total_samples / sample_rate, (target_end / sample_rate) + search_window)

            best_silence = None
            best_silence_duration = 0

            for silence_start, silence_end in silent_regions:
                silence_mid = (silence_start + silence_end) / 2
                silence_duration = silence_end - silence_start

                # Check if silence midpoint is within search window AND after current position
                if search_start <= silence_mid <= search_end and silence_mid > (current_pos / sample_rate):
                    # Prefer longer silences (more natural pauses)
                    if silence_duration > best_silence_duration:
                        best_silence = silence_mid
                        best_silence_duration = silence_duration

            if best_silence:
                # Split at silence midpoint
                split_sample = int(best_silence * sample_rate)

                # Ensure we're making progress (split point must be after current position)
                if split_sample <= current_pos:
                    # Silence is behind us or at current position - use target instead
                    chunks.append(audio[:, :, current_pos:target_end])
                    current_pos = target_end
                    print(f"  ✂️ Fixed split at {target_end/sample_rate:.2f}s (silence too early)")
                else:
                    chunks.append(audio[:, :, current_pos:split_sample])
                    current_pos = split_sample
                    print(f"  ✂️ Smart split at {best_silence:.2f}s (silence: {best_silence_duration:.2f}s)")
            else:
                # No suitable silence found - use target position
                chunks.append(audio[:, :, current_pos:target_end])
                current_pos = target_end
                print(f"  ✂️ Fixed split at {target_end/sample_rate:.2f}s (no silence found)")

        print(f"📦 Created {len(chunks)} smart chunks")
        return chunks

    def _fixed_chunk_audio(self, audio: torch.Tensor, sample_rate: int,
                          max_duration: int) -> List[torch.Tensor]:
        """
        Split audio at fixed time intervals.

        Args:
            audio: Audio tensor [batch, channels, samples]
            sample_rate: Audio sample rate
            max_duration: Chunk duration in seconds

        Returns:
            List of audio chunk tensors
        """
        chunk_samples = int(max_duration * sample_rate)
        total_samples = audio.shape[-1]

        chunks = []
        for start_sample in range(0, total_samples, chunk_samples):
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunks.append(audio[:, :, start_sample:end_sample])

        print(f"📦 Created {len(chunks)} fixed-duration chunks")
        return chunks

    def _process_chunks_with_conversion(self, source_chunks: List[torch.Tensor],
                                       target_audio: Dict[str, Any],
                                       engine_instance: Any,
                                       engine_type: str,
                                       refinement_passes: int,
                                       config: Dict[str, Any],
                                       sample_rate: int) -> tuple[torch.Tensor, int]:
        """
        Process audio chunks through voice conversion and combine results.

        Args:
            source_chunks: List of source audio chunks
            target_audio: Target voice audio (full, not chunked)
            engine_instance: Engine VC node instance
            engine_type: Engine type string
            refinement_passes: Number of refinement passes
            config: Engine configuration
            sample_rate: Audio sample rate

        Returns:
            Tuple of (combined_audio_tensor, output_sample_rate)
        """
        from utils.audio.chunk_combiner import ChunkCombiner

        converted_chunks = []
        total_chunks = len(source_chunks)

        for i, chunk in enumerate(source_chunks, 1):
            print(f"🔄 Processing chunk {i}/{total_chunks}...")

            # Convert chunk to AUDIO dict format expected by engine nodes
            chunk_audio_dict = {
                "waveform": chunk,
                "sample_rate": sample_rate
            }

            # Call engine-specific conversion
            if engine_type == "chatterbox":
                language = config.get("language", "English")
                result = engine_instance.convert_voice(
                    source_audio=chunk_audio_dict,
                    target_audio=target_audio,
                    refinement_passes=refinement_passes,
                    device=config.get("device", "auto"),
                    language=language
                )
                converted_chunk_audio = result[0]

            elif engine_type == "chatterbox_official_23lang":
                language = config.get("language", "English")
                result = engine_instance.convert_voice(
                    source_audio=chunk_audio_dict,
                    target_audio=target_audio,
                    refinement_passes=refinement_passes,
                    device=config.get("device", "auto"),
                    language=language
                )
                converted_chunk_audio = result[0]

            else:
                raise ValueError(f"Unsupported engine type for chunking: {engine_type}")

            # Extract waveform tensor and sample rate from result
            if isinstance(converted_chunk_audio, dict) and "waveform" in converted_chunk_audio:
                converted_chunks.append(converted_chunk_audio["waveform"])
                # Get output sample rate from first chunk (all chunks should have same rate)
                if i == 1 and "sample_rate" in converted_chunk_audio:
                    output_sample_rate = converted_chunk_audio["sample_rate"]
            elif isinstance(converted_chunk_audio, torch.Tensor):
                converted_chunks.append(converted_chunk_audio)
            else:
                raise ValueError(f"Unexpected conversion result type: {type(converted_chunk_audio)}")

        # Use output sample rate if available, otherwise fallback to input sample rate
        if 'output_sample_rate' not in locals():
            output_sample_rate = sample_rate
            print(f"⚠️ Using input sample rate {sample_rate}Hz for combining (output rate unknown)")
        else:
            print(f"🔊 Output sample rate: {output_sample_rate}Hz")

        # Combine chunks using crossfade for smooth transitions
        print(f"🔗 Combining {total_chunks} converted chunks with crossfade...")
        combined_audio = ChunkCombiner.combine_chunks(
            converted_chunks,
            method="crossfade",
            crossfade_duration=0.05,  # 50ms crossfade for smooth transitions
            sample_rate=output_sample_rate
        )

        return combined_audio, output_sample_rate

    def convert_voice(self, TTS_engine: Dict[str, Any], source_audio: Dict[str, Any],
                     narrator_target: Dict[str, Any], refinement_passes: int,
                     max_chunk_duration: int = 30, chunk_method: str = "smart"):
        """
        Convert voice using the selected engine with intelligent chunking for long audio.
        This is a DELEGATION WRAPPER that preserves all original VC functionality.

        Args:
            TTS_engine: Engine configuration from engine nodes
            source_audio: Source audio to convert
            narrator_target: Target voice characteristics (renamed for consistency)
            refinement_passes: Number of conversion iterations
            max_chunk_duration: Maximum chunk duration in seconds (prevents OOM on long audio)
            chunk_method: Chunking method - "smart" (silence-based) or "fixed" (time-based)

        Returns:
            Tuple of (converted_audio, conversion_info)
        """
        # Apply Python 3.12 CUDNN fix before voice conversion to prevent VRAM spikes
        ensure_python312_cudnn_fix()

        try:
            # Check if this is an RVC_ENGINE (RVCEngineAdapter) or TTS_ENGINE (dict)
            if hasattr(TTS_engine, 'engine_type') and TTS_engine.engine_type == "rvc":
                # This is an RVC_ENGINE adapter
                return self._handle_rvc_conversion(TTS_engine, source_audio, narrator_target, refinement_passes)

            # Validate TTS_ENGINE input (traditional dict format)
            if not TTS_engine or not isinstance(TTS_engine, dict):
                raise ValueError("Invalid TTS_engine input - connect a TTS engine node or RVC Engine node")

            engine_type = TTS_engine.get("engine_type")
            config = TTS_engine.get("config", {})

            # FIX: Sanitize engine config - ComfyUI JSON serialization corrupts numeric types
            # When workflow JSON is saved/loaded, floats like 10.0 become integers 10
            # This particularly affects first run after reboot or when using Any Switch
            config = ConfigSanitizer.sanitize(config)
            
            if not engine_type:
                raise ValueError("TTS engine missing engine_type")
            
            print(f"🔄 Voice Changer: Starting {engine_type} voice conversion")
            
            # Validate engine supports voice conversion
            if engine_type not in ["chatterbox", "chatterbox_official_23lang", "rvc"]:
                raise ValueError(f"Engine '{engine_type}' does not support voice conversion. Currently supported engines: ChatterBox, ChatterBox Official 23-Lang, RVC")
            
            # Extract audio data from flexible inputs (support both AUDIO and NARRATOR_VOICE types)
            processed_source_audio = self._extract_audio_from_input(source_audio, "source_audio")
            processed_narrator_target = self._extract_audio_from_input(narrator_target, "narrator_target")

            # Create proper engine VC node instance to preserve ALL functionality
            engine_instance = self._create_proper_engine_node_instance(TTS_engine)
            if not engine_instance:
                raise RuntimeError("Failed to create engine VC node instance")

            # Get sample rate from source audio
            source_sample_rate = processed_source_audio.get("sample_rate", 22050)

            # Extract waveform tensor for chunking analysis
            source_waveform = processed_source_audio.get("waveform")
            if source_waveform is None:
                raise ValueError("Source audio missing waveform data")

            # Split source audio into chunks if needed
            source_chunks = self._split_audio_into_chunks(
                source_waveform,
                source_sample_rate,
                max_chunk_duration,
                chunk_method
            )

            # Prepare parameters for the original VC node's convert_voice method
            if engine_type == "chatterbox":
                # Extract language from engine config for multilingual VC support
                language = config.get("language", "English")
                print(f"🔄 Voice Changer: Using {language} language model for conversion")

                # Check if chunking is needed
                if len(source_chunks) > 1:
                    # Process with chunking
                    converted_waveform, output_sample_rate = self._process_chunks_with_conversion(
                        source_chunks,
                        processed_narrator_target,
                        engine_instance,
                        engine_type,
                        refinement_passes,
                        config,
                        source_sample_rate
                    )
                    # Wrap in AUDIO dict format with correct output sample rate
                    converted_audio = {
                        "waveform": converted_waveform,
                        "sample_rate": output_sample_rate
                    }
                else:
                    # No chunking needed - use original flow
                    result = engine_instance.convert_voice(
                        source_audio=processed_source_audio,
                        target_audio=processed_narrator_target,
                        refinement_passes=refinement_passes,
                        device=config.get("device", "auto"),
                        language=language
                    )
                    converted_audio = result[0]
                
                # Get detailed model information for debugging
                model_source = "unknown"
                model_repo = "unknown"
                if hasattr(engine_instance, 'model_manager') and hasattr(engine_instance.model_manager, 'get_model_source'):
                    model_source = engine_instance.model_manager.get_model_source("vc") or "local/bundled"
                
                # Get repository information for HuggingFace models
                if model_source == "huggingface":
                    try:
                        from engines.chatterbox.language_models import get_model_config
                        model_config = get_model_config(language)
                        if model_config:
                            model_repo = model_config.get("repo", "unknown")
                        else:
                            model_repo = "ResembleAI/chatterbox"  # Default English repo
                    except ImportError:
                        model_repo = "ResembleAI/chatterbox"  # Fallback
                
                chunking_info = ""
                if len(source_chunks) > 1:
                    chunking_info = f"Chunking: {len(source_chunks)} chunks ({chunk_method} mode, {max_chunk_duration}s max)\n"

                conversion_info = (
                    f"🔄 Voice Changer (Unified) - CHATTERBOX Engine:\n"
                    f"Language Model: {language}\n"
                    f"Model Source: {model_source}\n"
                    + (f"Repository: {model_repo}\n" if model_source == "huggingface" else "") +
                    chunking_info +
                    f"Refinement passes: {refinement_passes}\n"
                    f"Device: {config.get('device', 'auto')}\n"
                    f"Conversion completed successfully"
                )
                
            elif engine_type == "chatterbox_official_23lang":
                # Extract language from engine config for multilingual VC support
                language = config.get("language", "English")
                print(f"🔄 Voice Changer: Using {language} language model for ChatterBox Official 23-Lang conversion")

                # Check if chunking is needed
                if len(source_chunks) > 1:
                    # Process with chunking
                    converted_waveform, output_sample_rate = self._process_chunks_with_conversion(
                        source_chunks,
                        processed_narrator_target,
                        engine_instance,
                        engine_type,
                        refinement_passes,
                        config,
                        source_sample_rate
                    )
                    # Wrap in AUDIO dict format with correct output sample rate
                    converted_audio = {
                        "waveform": converted_waveform,
                        "sample_rate": output_sample_rate
                    }
                else:
                    # No chunking needed - use original flow
                    result = engine_instance.convert_voice(
                        source_audio=processed_source_audio,
                        target_audio=processed_narrator_target,
                        refinement_passes=refinement_passes,
                        device=config.get("device", "auto"),
                        language=language
                    )
                    converted_audio = result[0]
                
                # Get detailed model information for debugging
                model_source = "unknown"
                model_repo = "unknown"
                if hasattr(engine_instance, 'model_manager') and hasattr(engine_instance.model_manager, 'get_model_source'):
                    model_source = engine_instance.model_manager.get_model_source("vc") or "local/bundled"
                
                # Get repository information for HuggingFace models (if available)
                if model_source == "huggingface":
                    try:
                        # ChatterBox Official 23-Lang uses same language model configs
                        from engines.chatterbox.language_models import get_model_config
                        model_config = get_model_config(language)
                        if model_config:
                            model_repo = model_config.get("repo", "unknown")
                        else:
                            model_repo = "ResembleAI/chatterbox"  # Default English repo
                    except ImportError:
                        model_repo = "ResembleAI/chatterbox"  # Fallback
                
                chunking_info = ""
                if len(source_chunks) > 1:
                    chunking_info = f"Chunking: {len(source_chunks)} chunks ({chunk_method} mode, {max_chunk_duration}s max)\n"

                conversion_info = (
                    f"🔄 Voice Changer (Unified) - CHATTERBOX OFFICIAL 23-LANG Engine:\n"
                    f"Language Model: {language}\n"
                    f"Model Source: {model_source}\n"
                    + (f"Repository: {model_repo}\n" if model_source == "huggingface" else "") +
                    chunking_info +
                    f"Refinement passes: {refinement_passes}\n"
                    f"Device: {config.get('device', 'auto')}\n"
                    f"Conversion completed successfully"
                )
                
            else:
                # Future engines (RVC, etc.) will be handled here
                raise ValueError(f"Engine type '{engine_type}' voice conversion not yet implemented")
            
            print(f"✅ Voice Changer: {engine_type} conversion successful")
            return (converted_audio, conversion_info)
                
        except Exception as e:
            error_msg = f"❌ Voice conversion failed: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            # Return empty audio and error info
            empty_audio = AudioProcessingUtils.create_silence(1.0, 24000)
            empty_comfy = AudioProcessingUtils.format_for_comfyui(empty_audio, 24000)
            
            return (empty_comfy, error_msg)
    
    def _generate_rvc_cache_key(self, source_audio: Dict[str, Any], rvc_model: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate cache key for RVC voice conversion iterations"""
        # Create hash from source audio characteristics and RVC model
        # Convert BFloat16 to Float32 if needed before numpy conversion (defensive programming)
        waveform = source_audio["waveform"].cpu()
        if waveform.dtype == torch.bfloat16:
            waveform = waveform.to(torch.float32)
        source_hash = hashlib.md5(waveform.numpy().tobytes()).hexdigest()[:16]
        
        # Include RVC model information in cache key
        model_info = {
            'model_name': rvc_model.get('model_name', 'unknown') if rvc_model else 'no_model',
            'model_path': rvc_model.get('model_path', '') if rvc_model else '',
            'source_sr': source_audio["sample_rate"]
        }
        
        # Include ALL RVC engine and pitch options config for cache differentiation
        cache_data = {
            'source_hash': source_hash,
            'source_sr': source_audio["sample_rate"],
            # RVC Engine parameters
            'pitch_shift': config.get('pitch_shift', 0),
            'index_rate': config.get('index_rate', 0.75),
            'rms_mix_rate': config.get('rms_mix_rate', 0.25),
            'protect': config.get('protect', 0.25),
            'f0_method': config.get('f0_method', 'rmvpe'),
            'resample_sr': config.get('resample_sr', 0),
            'hubert_model': config.get('hubert_model', 'auto'),
            # RVC Pitch Extraction Options parameters
            'crepe_hop_length': config.get('crepe_hop_length', 160),
            'filter_radius': config.get('filter_radius', 3),
            'pitch_guidance': config.get('pitch_guidance', 1.0),
            'f0_autotune': config.get('f0_autotune', False),
            'model_info': str(sorted(model_info.items()))
        }
        
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_rvc_iterations(self, cache_key: str, max_iteration: int) -> Dict[int, Any]:
        """Get cached RVC iterations up to max_iteration"""
        if cache_key not in GLOBAL_RVC_ITERATION_CACHE:
            return {}
        
        cached_data = GLOBAL_RVC_ITERATION_CACHE[cache_key]
        return {i: cached_data[i] for i in cached_data if i <= max_iteration}
    
    def _cache_rvc_result(self, cache_key: str, iteration: int, result_tuple: tuple):
        """Cache a single RVC iteration result (limit to 5 iterations max)"""
        if cache_key not in GLOBAL_RVC_ITERATION_CACHE:
            GLOBAL_RVC_ITERATION_CACHE[cache_key] = {}
        
        # Only cache up to 5 iterations to prevent memory issues
        if iteration <= 5:
            GLOBAL_RVC_ITERATION_CACHE[cache_key][iteration] = result_tuple


# Register the node class
NODE_CLASS_MAPPINGS = {
    "UnifiedVoiceChangerNode": UnifiedVoiceChangerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnifiedVoiceChangerNode": "🔄 Voice Changer"
}