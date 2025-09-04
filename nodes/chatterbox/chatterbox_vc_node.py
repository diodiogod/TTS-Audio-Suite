"""
ChatterBox Voice Conversion Node - Migrated to use new foundation
Voice Conversion node using ChatterboxVC with improved architecture
"""

import torch
import tempfile
import os
import hashlib
from typing import Dict, Any

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

import torchaudio

# Global cache for VC iterations (max 5 to avoid memory issues)
GLOBAL_VC_ITERATION_CACHE = {}

class ChatterboxVCNode(BaseVCNode):
    """
    Voice Conversion node using ChatterboxVC - Voice Edition
    SUPPORTS BUNDLED CHATTERBOX
    """
    
    @classmethod
    def NAME(cls):
        return "🔄 ChatterBox Voice Conversion (diogod)"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_audio": ("AUDIO", {"tooltip": "The original voice audio you want to convert to sound like the target voice"}),
                "target_audio": ("AUDIO", {"tooltip": "The reference voice audio whose characteristics will be applied to the source audio"}),
                "refinement_passes": ("INT", {"default": 1, "min": 1, "max": 30, "step": 1, "tooltip": "Number of conversion iterations. Each pass refines the output to sound more like the target. Recommended: Max 5 passes - more can cause distortions. Each iteration is deterministic to reduce degradation."}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto", "tooltip": "Processing device: 'auto' selects best available, 'cuda' for GPU acceleration, 'cpu' for compatibility"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("converted_audio",)
    FUNCTION = "convert_voice"
    CATEGORY = "ChatterBox Voice"

    def __init__(self):
        super().__init__()

    def _generate_vc_cache_key(self, source_audio: Dict[str, Any], target_audio: Dict[str, Any], device: str) -> str:
        """Generate cache key for voice conversion iterations"""
        # Normalize audio inputs for VideoHelper compatibility
        source_audio = self._get_audio(source_audio, "source_audio")
        target_audio = self._get_audio(target_audio, "target_audio")
        
        # Create hash from source and target audio characteristics
        source_hash = hashlib.md5(source_audio["waveform"].cpu().numpy().tobytes()).hexdigest()[:16]
        target_hash = hashlib.md5(target_audio["waveform"].cpu().numpy().tobytes()).hexdigest()[:16]
        
        cache_data = {
            'source_hash': source_hash,
            'target_hash': target_hash,
            'source_sr': source_audio["sample_rate"],
            'target_sr': target_audio["sample_rate"],
            'device': device,
            'seed_base': 42  # Include seed base in cache key for deterministic results
        }
        
        cache_string = str(sorted(cache_data.items()))
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_iterations(self, cache_key: str, max_iteration: int) -> Dict[int, Dict[str, Any]]:
        """Get cached iterations up to max_iteration"""
        if cache_key not in GLOBAL_VC_ITERATION_CACHE:
            return {}
        
        cached_data = GLOBAL_VC_ITERATION_CACHE[cache_key]
        return {i: cached_data[i] for i in cached_data if i <= max_iteration}
    
    def _cache_iteration(self, cache_key: str, iteration: int, audio_result: Dict[str, Any]):
        """Cache a single iteration result (limit to 5 iterations max)"""
        if cache_key not in GLOBAL_VC_ITERATION_CACHE:
            GLOBAL_VC_ITERATION_CACHE[cache_key] = {}
        
        # Only cache up to 5 iterations to prevent memory issues
        if iteration <= 5:
            GLOBAL_VC_ITERATION_CACHE[cache_key][iteration] = audio_result

    def prepare_audio_files(self, source_audio: Dict[str, Any], target_audio: Dict[str, Any]) -> tuple[str, str]:
        """
        Prepare audio files for voice conversion by saving to temporary files.
        
        Args:
            source_audio: Source audio dictionary from ComfyUI (any format)
            target_audio: Target audio dictionary from ComfyUI (any format)
            
        Returns:
            Tuple of (source_path, target_path)
        """
        # Normalize audio inputs for VideoHelper compatibility
        source_audio = self._get_audio(source_audio, "source_audio")
        target_audio = self._get_audio(target_audio, "target_audio")
        
        # Save source audio to temporary file
        source_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        source_temp.close()
        
        source_waveform = source_audio["waveform"]
        if source_waveform.dim() == 3:
            source_waveform = source_waveform.squeeze(0)  # Remove batch dimension
        
        torchaudio.save(source_temp.name, source_waveform.cpu(), source_audio["sample_rate"])
        self._temp_files.append(source_temp.name)
        
        # Save target audio to temporary file
        target_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        target_temp.close()
        
        target_waveform = target_audio["waveform"]
        if target_waveform.dim() == 3:
            target_waveform = target_waveform.squeeze(0)  # Remove batch dimension
        
        torchaudio.save(target_temp.name, target_waveform.cpu(), target_audio["sample_rate"])
        self._temp_files.append(target_temp.name)
        
        return source_temp.name, target_temp.name

    def convert_voice(self, source_audio, target_audio, refinement_passes, device, language="English"):
        """
        Perform iterative voice conversion using the loaded model.
        
        Args:
            source_audio: Source audio from ComfyUI
            target_audio: Target voice audio from ComfyUI
            refinement_passes: Number of conversion iterations
            device: Target device
            language: Language model to use for conversion (English, German, Norwegian)
            
        Returns:
            Converted audio in ComfyUI format
        """
        def _process():
            # Load model with language support
            try:
                self.load_vc_model(device, language=language)
            except RuntimeError as e:
                error_str = str(e)
                # Handle unsupported language gracefully
                if "Voice conversion not supported" in error_str:
                    print(f"❌ Voice Conversion Error: {language} model does not support voice conversion")
                    print(f"   Please use a model with s3gen component (English, German, etc.)")
                    # Return silent audio to clearly indicate failure
                    print(f"🔇 Returning silent audio to indicate VC failure")
                    
                    # Create 1-second silent audio with same format as source
                    import torch
                    # Normalize source audio first to get proper format
                    normalized_source = self._get_audio(source_audio, "source_audio")
                    sample_rate = normalized_source["sample_rate"]
                    source_waveform = normalized_source["waveform"]
                    
                    
                    # Create 1-second silent audio tensor matching source format
                    if len(source_waveform.shape) >= 3:
                        # If has batch dimension, remove it for processing
                        source_waveform = source_waveform.squeeze(0)
                    
                    if len(source_waveform.shape) == 1:
                        # If 1D, create mono silent audio
                        silent_tensor = torch.zeros(sample_rate, dtype=torch.float32)
                    elif len(source_waveform.shape) == 2:
                        # If 2D, match the channel count but make it 1 second
                        num_channels = source_waveform.shape[0]
                        silent_tensor = torch.zeros(num_channels, sample_rate, dtype=torch.float32)
                    else:
                        # Fallback: create mono silent audio
                        silent_tensor = torch.zeros(sample_rate, dtype=torch.float32)
                    
                    # Use base class method to properly format for ComfyUI
                    silent_audio = self.format_audio_output(silent_tensor, sample_rate)
                    return (silent_audio,)
                else:
                    # Re-raise other types of RuntimeError
                    raise e
            
            # Generate cache key for this conversion
            cache_key = self._generate_vc_cache_key(source_audio, target_audio, device)
            
            # Check for cached iterations
            cached_iterations = self._get_cached_iterations(cache_key, refinement_passes)
            
            # If we have the exact number of passes cached, return it immediately
            if refinement_passes in cached_iterations:
                print(f"💾 CACHE HIT: Using cached voice conversion result for {refinement_passes} passes")
                return (cached_iterations[refinement_passes],)
            
            # Prepare target audio file (constant across all iterations)
            target_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            target_temp.close()
            
            target_waveform = target_audio["waveform"]
            if target_waveform.dim() == 3:
                target_waveform = target_waveform.squeeze(0)
            
            torchaudio.save(target_temp.name, target_waveform.cpu(), target_audio["sample_rate"])
            self._temp_files.append(target_temp.name)
            
            # Start from the highest cached iteration or from beginning
            start_iteration = 0
            current_audio = source_audio  # source_audio already normalized in prepare_audio_files()
            
            # Find the highest cached iteration we can start from
            for i in range(refinement_passes, 0, -1):
                if i in cached_iterations:
                    start_iteration = i
                    current_audio = cached_iterations[i]
                    print(f"💾 CACHE: Resuming from cached iteration {i}/{refinement_passes}")
                    break
            
            try:
                # Perform remaining voice conversion iterations
                for iteration in range(start_iteration, refinement_passes):
                    iteration_num = iteration + 1
                    
                    # Set deterministic seed for each iteration (base seed 42 + iteration)
                    # This helps reduce degradation by making each pass reproducible
                    iteration_seed = 42 + iteration_num
                    self.set_seed(iteration_seed)
                    
                    print(f"🔄 Voice conversion pass {iteration_num}/{refinement_passes}...")
                    
                    # Prepare current source audio file
                    source_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    source_temp.close()
                    
                    source_waveform = current_audio["waveform"]
                    if source_waveform.dim() == 3:
                        source_waveform = source_waveform.squeeze(0)
                    
                    torchaudio.save(source_temp.name, source_waveform.cpu(), current_audio["sample_rate"])
                    self._temp_files.append(source_temp.name)
                    
                    # Perform voice conversion
                    wav = self.vc_model.generate(
                        source_temp.name,
                        target_voice_path=target_temp.name
                    )
                    
                    # Update current_audio for next iteration
                    current_audio = self.format_audio_output(wav, self.vc_model.sr)
                    
                    # Cache this iteration result (only up to 5 iterations)
                    self._cache_iteration(cache_key, iteration_num, current_audio)
                
                cache_info = "with cache optimization" if start_iteration > 0 else "without cache"
                print(f"✅ Voice conversion completed with {refinement_passes} refinement passes ({cache_info})")
                return (current_audio,)
                
            finally:
                # Cleanup is handled automatically by the base class destructor
                # and the cleanup_temp_files method
                pass
        
        return self.process_with_error_handling(_process)