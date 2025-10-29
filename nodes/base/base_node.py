"""
Base Node - Common functionality for all ChatterBox Voice nodes
Provides shared methods and standardized interfaces
"""

import torch
import numpy as np
import tempfile
import os
from typing import Dict, Any, Optional, Tuple
import comfy.model_management as model_management

# Use absolute imports that work when loaded via importlib
import os
import sys

# Add project root directory to path for imports
# When loaded via importlib, __file__ might not resolve correctly
# So we'll use a more aggressive approach to find the project root
try:
    from utils.models.manager import model_manager
except ImportError:
    # Find project root by going up from this file location
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    nodes_dir = os.path.dirname(current_dir)  # nodes/
    project_root = os.path.dirname(nodes_dir)  # project root
    
    # Add to beginning of path and force reload of sys.modules
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Also try adding the utils directory directly
    utils_dir = os.path.join(project_root, 'utils')
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    
    # Clear any cached import failures
    if 'utils' in sys.modules:
        del sys.modules['utils']
    if 'utils.models' in sys.modules:
        del sys.modules['utils.models']
    if 'utils.models.manager' in sys.modules:
        del sys.modules['utils.models.manager']
    
    # Try import again
    from utils.models.manager import model_manager


class BaseChatterBoxNode:
    """
    Base class for all ChatterBox Voice nodes.
    Provides common functionality and standardized interfaces.
    """
    
    # Node metadata that can be overridden by subclasses
    CATEGORY = "ChatterBox Voice"
    FUNCTION = "process"
    
    def __init__(self):
        """Initialize base node with common properties."""
        self.model_manager = model_manager
        self.device = None
        self._temp_files = []  # Track temporary files for cleanup
    
    def __del__(self):
        """Cleanup temporary files when node is destroyed."""
        self.cleanup_temp_files()
    
    def cleanup_temp_files(self):
        """Clean up any temporary files created by the node."""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception:
                pass  # Ignore cleanup errors
        self._temp_files.clear()
    
    def resolve_device(self, device: str) -> str:
        """
        Resolve device string to actual device.
        
        Args:
            device: Device specification ('auto', 'cuda', 'cpu')
            
        Returns:
            Resolved device string
        """
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def set_seed(self, seed: int):
        """
        Set random seeds for reproducible generation.
        
        Args:
            seed: Random seed value (0 means no seed setting)
        """
        if seed != 0:
            # Clamp seed to valid NumPy range (0 to 2^32-1)
            clamped_seed = max(0, min(seed, 2**32 - 1))
            torch.manual_seed(clamped_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(clamped_seed)
            np.random.seed(clamped_seed)
    
    def check_interruption(self, operation_name: str = "Generation"):
        """
        Check if processing has been interrupted by ComfyUI.
        
        Args:
            operation_name: Name of the operation being performed
            
        Raises:
            InterruptedError: If processing has been interrupted
        """
        if model_management.interrupt_processing:
            raise InterruptedError(f"{operation_name} interrupted by user")
    
    def _get_audio(self, audio_input, input_name: str = "audio") -> Dict[str, Any]:
        """
        Simple audio input getter - handles all formats (AUDIO, Character Voices, VideoHelper).
        Use this instead of direct audio["waveform"] access for VideoHelper compatibility.
        
        Args:
            audio_input: Audio input in any supported format
            input_name: Name for error messages
            
        Returns:
            Standard AUDIO dict with 'waveform' and 'sample_rate' keys
        """
        return self.normalize_audio_input(audio_input, input_name)
    
    def normalize_audio_input(self, audio_input, input_name: str = "audio") -> Dict[str, Any]:
        """
        Universal audio input normalizer - handles all ComfyUI audio formats.
        Supports: AUDIO dict, Character Voices output, VideoHelper LazyAudioMap, etc.
        
        Args:
            audio_input: Audio input in any supported format
            input_name: Name for error messages
            
        Returns:
            Standard AUDIO dict with 'waveform' and 'sample_rate' keys
            
        Raises:
            ValueError: If audio format is not supported
        """
        # Import here to avoid circular imports
        try:
            from utils.audio.processing import AudioProcessingUtils
            return AudioProcessingUtils.normalize_audio_input(audio_input, input_name)
        except ImportError:
            # Fallback implementation if utils not available
            if audio_input is None:
                raise ValueError(f"{input_name} input is required")
            
            # Character Voices node output (NARRATOR_VOICE)
            if isinstance(audio_input, dict) and "audio" in audio_input:
                return self.normalize_audio_input(audio_input["audio"], input_name)
            
            # Standard AUDIO format or VideoHelper LazyAudioMap
            elif hasattr(audio_input, "__getitem__"):
                if "waveform" in audio_input and "sample_rate" in audio_input:
                    return {
                        "waveform": audio_input["waveform"], 
                        "sample_rate": audio_input["sample_rate"]
                    }
                else:
                    available_keys = list(audio_input.keys()) if hasattr(audio_input, "keys") else []
                    raise ValueError(f"Audio input missing required keys. Expected 'waveform' and 'sample_rate', found: {available_keys}")
            else:
                audio_type = type(audio_input).__name__
                raise ValueError(f"Unsupported audio format: {audio_type}")

    def handle_reference_audio(self, reference_audio: Optional[Dict[str, Any]], 
                             audio_prompt_path: str = "") -> Optional[str]:
        """
        Handle reference audio input, creating temporary files as needed.
        
        Args:
            reference_audio: Audio tensor from ComfyUI (any supported format)
            audio_prompt_path: Path to audio file on disk
            
        Returns:
            Path to audio file for use with ChatterBox models
        """
        audio_prompt = None
        
        if reference_audio is not None:
            # Normalize audio input to handle all formats (VideoHelper, Character Voices, etc.)
            normalized_audio = self.normalize_audio_input(reference_audio, "reference_audio")
            
            # Create temporary file for reference audio
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file.close()
            
            # Save waveform to temporary file
            waveform = normalized_audio["waveform"]
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)  # Remove batch dimension
            
            import torchaudio
            torchaudio.save(temp_file.name, waveform, normalized_audio["sample_rate"])
            
            audio_prompt = temp_file.name
            self._temp_files.append(temp_file.name)
            
        elif audio_prompt_path and os.path.exists(audio_prompt_path):
            audio_prompt = audio_prompt_path
        
        return audio_prompt
    
    def format_audio_output(self, audio_tensor: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
        """
        Format audio tensor for ComfyUI output.
        
        Args:
            audio_tensor: Audio tensor from model
            sample_rate: Sample rate of the audio
            
        Returns:
            Audio dictionary in ComfyUI format
        """
        # Ensure audio has batch dimension
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
        if audio_tensor.dim() == 2 and audio_tensor.shape[0] != 1:
            # If it's multi-channel, we might need to handle it differently
            # For now, assume it's already in [channels, samples] format
            pass
        
        # Add batch dimension for ComfyUI
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
        
        return {
            "waveform": audio_tensor,
            "sample_rate": sample_rate
        }
    
    def generate_info_string(self, **kwargs) -> str:
        """
        Generate information string about the processing.
        Can be overridden by subclasses to provide specific info.
        
        Args:
            **kwargs: Information to include in the string
            
        Returns:
            Formatted information string
        """
        info_parts = []
        for key, value in kwargs.items():
            if value is not None:
                info_parts.append(f"{key}: {value}")
        
        return ", ".join(info_parts)
    
    def validate_inputs(self, **inputs) -> Dict[str, Any]:
        """
        Validate and normalize inputs. Can be overridden by subclasses.
        
        Args:
            **inputs: Input parameters to validate
            
        Returns:
            Validated and normalized inputs
        """
        # Base implementation - just return inputs
        # Subclasses can override to add specific validation
        return inputs
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Args:
            model_type: Type of model ('tts' or 'vc')
            
        Returns:
            Dictionary with model information
        """
        return {
            "source": self.model_manager.get_model_source(model_type),
            "device": self.device,
            "available": self.model_manager.is_available[model_type],
        }
    
    @classmethod
    def get_input_types_base(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get base input types that are common across nodes.
        Subclasses can extend this.
        
        Returns:
            Base input types dictionary
        """
        return {
            "required": {
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "reference_audio": ("AUDIO", {"tooltip": "Optional reference audio for voice cloning"}),
                "audio_prompt_path": ("STRING", {"default": "", "tooltip": "Path to audio file on disk"}),
            }
        }
    
    def process_with_error_handling(self, process_func, *args, **kwargs):
        """
        Wrapper for processing functions that provides consistent error handling.
        
        Args:
            process_func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function
            
        Raises:
            Various exceptions with improved error messages
        """
        try:
            return process_func(*args, **kwargs)
        except InterruptedError:
            # Re-raise interruption errors as-is
            raise
        except ImportError as e:
            if "ChatterBox" in str(e):
                raise ImportError(
                    f"ChatterBox components not available. Please check installation or add bundled version. "
                    f"Original error: {e}"
                )
            else:
                raise
        except Exception as e:
            # Provide more context for other errors
            import traceback
            error_trace = traceback.format_exc()
            raise RuntimeError(
                f"Node processing failed: {type(e).__name__}: {e}\n"
                f"Traceback:\n{error_trace}"
            ) from e


class BaseTTSNode(BaseChatterBoxNode):
    """
    Base class specifically for TTS nodes with common TTS functionality.
    """
    
    def __init__(self):
        super().__init__()
        self.tts_model = None
    
    def load_tts_model(self, device: str = "auto", language: str = "English", force_reload: bool = False):
        """
        Load TTS model using the model manager.
        
        Args:
            device: Target device
            language: Language model to load
            force_reload: Force reload even if cached
            
        Returns:
            Loaded TTS model
        """
        device = self.resolve_device(device)
        self.device = device
        
        self.tts_model = self.model_manager.load_tts_model(device, language, force_reload)
        return self.tts_model
    
    def generate_tts_audio(self, text: str, audio_prompt: Optional[str] = None,
                          exaggeration: float = 0.5, temperature: float = 0.8,
                          cfg_weight: float = 0.5) -> torch.Tensor:
        """
        Generate TTS audio using the loaded model.
        
        Args:
            text: Text to synthesize
            audio_prompt: Optional audio prompt path
            exaggeration: Exaggeration parameter
            temperature: Temperature parameter
            cfg_weight: CFG weight parameter
            
        Returns:
            Generated audio tensor
        """
        if self.tts_model is None:
            raise RuntimeError("TTS model not loaded. Call load_tts_model() first.")

        # CRITICAL FIX: Reload model to correct device if it was offloaded
        # When models are cached, the model reference persists but may be on wrong device after "Clear VRAM"
        # Use the wrapper system to properly reload through ComfyUI model management
        if hasattr(self.tts_model, 'to') and hasattr(self.tts_model, 'device'):
            target_device = self.device if hasattr(self, 'device') else self.tts_model.device

            # Resolve "auto" to actual device
            if target_device == "auto":
                target_device = "cuda" if torch.cuda.is_available() else "cpu"

            # Check current device of model components (look for any PyTorch module with parameters)
            current_device = None
            for attr_name in ['t3', 's3gen', 've', 'ema_model']:  # Check common TTS model components
                if hasattr(self.tts_model, attr_name):
                    component = getattr(self.tts_model, attr_name)
                    if hasattr(component, 'parameters'):
                        try:
                            current_device = str(next(component.parameters()).device)
                            break
                        except StopIteration:
                            continue

            # Reload through unified interface if device mismatch
            if current_device and current_device != str(target_device):
                # print(f"🔄 Reloading TTS model from {current_device} to {target_device} via wrapper")

                # Find and call wrapper's model_load() instead of direct .to()
                try:
                    from utils.models.comfyui_model_wrapper.model_manager import tts_model_manager
                    for cache_key, wrapper in tts_model_manager._model_cache.items():
                        if hasattr(wrapper, 'model') and wrapper.model is self.tts_model:
                            wrapper.model_load(target_device)
                            # print(f"✅ Reloaded model via wrapper - ComfyUI management stays in sync")
                            break
                    else:
                        # Fallback: direct .to() if wrapper not found
                        print(f"⚠️ Wrapper not found, using direct .to() - 'Clear VRAM' may not work")
                        self.tts_model.to(target_device)
                except Exception as e:
                    # Fallback to direct .to()
                    print(f"⚠️ Wrapper reload failed ({e}), using direct .to()")
                    self.tts_model.to(target_device)

        # Use torch.no_grad() to ensure no gradients are tracked during inference
        with torch.no_grad():
            audio = self.tts_model.generate(
                text,
                audio_prompt_path=audio_prompt,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfg_weight
            )
        
        # Ensure tensor is completely detached from computation graph
        if hasattr(audio, 'detach'):
            audio = audio.detach()
        if hasattr(audio, 'clone'):
            audio = audio.clone()
        
        # Ensure audio is on CPU and proper dtype for ComfyUI compatibility
        if hasattr(audio, 'cpu'):
            audio = audio.cpu()
        
        # Ensure float32 dtype for ComfyUI video nodes
        if hasattr(audio, 'float'):
            audio = audio.float()  # Converts to float32
            
        return audio


class BaseVCNode(BaseChatterBoxNode):
    """
    Base class specifically for Voice Conversion nodes.
    """
    
    def __init__(self):
        super().__init__()
        self.vc_model = None
    
    def load_vc_model(self, device: str = "auto", force_reload: bool = False, language: str = "English"):
        """
        Load VC model using the model manager.
        
        Args:
            device: Target device
            force_reload: Force reload even if cached
            language: Language model to use for conversion (English, German, Norwegian)
            
        Returns:
            Loaded VC model
        """
        device = self.resolve_device(device)
        self.device = device
        
        self.vc_model = self.model_manager.load_vc_model(device, force_reload, language)
        return self.vc_model
    
    def convert_voice(self, source_path: str, target_path: str) -> torch.Tensor:
        """
        Perform voice conversion using the loaded model.
        
        Args:
            source_path: Path to source audio
            target_path: Path to target voice audio
            
        Returns:
            Converted audio tensor
        """
        if self.vc_model is None:
            raise RuntimeError("VC model not loaded. Call load_vc_model() first.")
        
        return self.vc_model.generate(source_path, target_voice_path=target_path)