"""
Import Manager - Centralized import handling with fallbacks for ChatterBox Voice
Handles complex import scenarios with bundled vs system packages
"""

import os
import sys
import warnings
import importlib.util
from typing import Optional, Dict, Any, Tuple, List


class ImportManager:
    """
    Manages complex import scenarios for ChatterBox Voice components.
    Handles bundled vs system imports with proper fallbacks.
    """
    
    def __init__(self, node_dir: Optional[str] = None, is_dev: bool = True):
        """
        Initialize ImportManager.
        
        Args:
            node_dir: Optional override for the node directory path
            is_dev: Whether to show debug messages
        """
        self.node_dir = node_dir or os.path.dirname(os.path.dirname(__file__))
        self.bundled_chatterbox_dir = os.path.join(self.node_dir, "chatterbox")
        self.is_dev = is_dev
        
        # Import status tracking
        self.import_status: Dict[str, Dict[str, Any]] = {
            "chatterbox_tts": {"available": False, "source": None, "error": None},
            "chatterbox_vc": {"available": False, "source": None, "error": None},
            "f5tts": {"available": False, "source": None, "error": None},
            "srt_parser": {"available": False, "source": None, "error": None},
            "audio_timing": {"available": False, "source": None, "error": None},
        }
        
        # Loaded modules cache
        self.loaded_modules: Dict[str, Any] = {}
    
    def _add_node_dir_to_path(self):
        """Add node directory to Python path for bundled imports."""
        if self.node_dir not in sys.path:
            sys.path.insert(0, self.node_dir)
    
    def _log_debug(self, message: str):
        """Log debug message if in dev mode."""
        if self.is_dev:
            try:
                print(message)
            except UnicodeEncodeError:
                # Fallback for Windows console encoding issues
                safe_message = message.encode('ascii', 'replace').decode('ascii')
                print(safe_message)
    
    def import_chatterbox_tts(self) -> Tuple[bool, Optional[Any], str]:
        """
        Import ChatterboxTTS with bundled first, then system fallback.
        
        Returns:
            Tuple of (success, module_class, source)
        """
        module_key = "chatterbox_tts"
        
        # Return cached if available
        if self.import_status[module_key]["available"]:
            return True, self.loaded_modules[module_key], self.import_status[module_key]["source"]
        
        # Try bundled first
        try:
            self._add_node_dir_to_path()
            from engines.chatterbox.tts import ChatterboxTTS
            
            self.loaded_modules[module_key] = ChatterboxTTS
            self.import_status[module_key] = {
                "available": True,
                "source": "bundled",
                "error": None
            }
            # Only log on error, not success
            pass
            return True, ChatterboxTTS, "bundled"
            
        except ImportError as bundled_error:
            # Try system installation
            try:
                from engines.chatterbox.tts import ChatterboxTTS
                
                self.loaded_modules[module_key] = ChatterboxTTS
                self.import_status[module_key] = {
                    "available": True,
                    "source": "system",
                    "error": None
                }
                self._log_debug("✅ ChatterboxTTS loaded from system package")
                return True, ChatterboxTTS, "system"
                
            except ImportError as system_error:
                self.import_status[module_key] = {
                    "available": False,
                    "source": None,
                    "error": f"Bundled: {bundled_error}, System: {system_error}"
                }
                self._log_debug(f"❌ ChatterboxTTS not available: {system_error}")
                return False, None, "none"
    
    def import_chatterbox_vc(self) -> Tuple[bool, Optional[Any], str]:
        """
        Import ChatterboxVC with bundled first, then system fallback.
        
        Returns:
            Tuple of (success, module_class, source)
        """
        module_key = "chatterbox_vc"
        
        # Return cached if available
        if self.import_status[module_key]["available"]:
            return True, self.loaded_modules[module_key], self.import_status[module_key]["source"]
        
        # Try bundled first
        try:
            self._add_node_dir_to_path()
            from engines.chatterbox.vc import ChatterboxVC
            
            self.loaded_modules[module_key] = ChatterboxVC
            self.import_status[module_key] = {
                "available": True,
                "source": "bundled",
                "error": None
            }
            # Only log on error, not success
            pass
            return True, ChatterboxVC, "bundled"
            
        except ImportError as bundled_error:
            # Try system installation
            try:
                from engines.chatterbox.vc import ChatterboxVC
                
                self.loaded_modules[module_key] = ChatterboxVC
                self.import_status[module_key] = {
                    "available": True,
                    "source": "system",
                    "error": None
                }
                self._log_debug("✅ ChatterboxVC loaded from system package")
                return True, ChatterboxVC, "system"
                
            except ImportError as system_error:
                self.import_status[module_key] = {
                    "available": False,
                    "source": None,
                    "error": f"Bundled: {bundled_error}, System: {system_error}"
                }
                self._log_debug(f"❌ ChatterboxVC not available: {system_error}")
                return False, None, "none"
    
    def import_f5tts(self) -> Tuple[bool, Any, str]:
        """
        Import F5-TTS with fallback handling.
        
        Returns:
            Tuple of (success, f5tts_module, source)
        """
        module_key = "f5tts"
        
        # Return cached if available
        if self.import_status[module_key]["available"]:
            return True, self.loaded_modules[module_key], self.import_status[module_key]["source"]
        
        try:
            # Try to import F5-TTS API
            from engines.f5_tts.api import F5TTS
            
            self.loaded_modules[module_key] = F5TTS
            self.import_status[module_key] = {
                "available": True,
                "source": "system",
                "error": None
            }
            # Only log on error, not success
            pass
            return True, F5TTS, "system"
            
        except ImportError as error:
            self.import_status[module_key] = {
                "available": False,
                "source": None,
                "error": str(error)
            }
            self._log_debug(f"❌ F5-TTS not available: {error}")
            return False, None, "none"
    
    def import_srt_modules(self) -> Tuple[bool, Dict[str, Any], str]:
        """
        Import SRT-related modules with multiple fallback strategies.
        
        Returns:
            Tuple of (success, modules_dict, source)
        """
        module_key = "srt_parser"
        
        # Return cached if available
        if self.import_status[module_key]["available"]:
            return True, self.loaded_modules[module_key], self.import_status[module_key]["source"]
        
        modules = {}
        
        # Strategy 1: Direct file loading from bundled directory
        srt_parser_path = os.path.join(self.bundled_chatterbox_dir, 'srt_parser.py')
        audio_timing_path = os.path.join(self.bundled_chatterbox_dir, 'audio_timing.py')
        
        if os.path.exists(srt_parser_path) and os.path.exists(audio_timing_path):
            try:
                # Load SRT parser module directly from file
                srt_parser_spec = importlib.util.spec_from_file_location("srt_parser", srt_parser_path)
                srt_parser_module = importlib.util.module_from_spec(srt_parser_spec)
                srt_parser_spec.loader.exec_module(srt_parser_module)
                
                # Extract SRT parser classes and functions
                modules.update({
                    "SRTParser": srt_parser_module.SRTParser,
                    "SRTSubtitle": srt_parser_module.SRTSubtitle,
                    "SRTParseError": srt_parser_module.SRTParseError,
                    "validate_srt_timing_compatibility": srt_parser_module.validate_srt_timing_compatibility,
                })
                
                # Try to load audio timing module
                try:
                    audio_timing_spec = importlib.util.spec_from_file_location("audio_timing", audio_timing_path)
                    audio_timing_module = importlib.util.module_from_spec(audio_timing_spec)
                    audio_timing_spec.loader.exec_module(audio_timing_module)
                    
                    # Extract audio timing classes and functions
                    modules.update({
                        "AudioTimingUtils": audio_timing_module.AudioTimingUtils,
                        "PhaseVocoderTimeStretcher": audio_timing_module.PhaseVocoderTimeStretcher,
                        "TimedAudioAssembler": audio_timing_module.TimedAudioAssembler,
                        "calculate_timing_adjustments": audio_timing_module.calculate_timing_adjustments,
                        "AudioTimingError": audio_timing_module.AudioTimingError,
                        "FFmpegTimeStretcher": getattr(audio_timing_module, "FFmpegTimeStretcher", None),
                    })
                    
                except Exception as timing_error:
                    self._log_debug(f"⚠️ Audio timing not fully available: {timing_error}")
                    # Add fallback implementations
                    modules.update(self._create_fallback_audio_timing())
                
                self.loaded_modules[module_key] = modules
                self.import_status[module_key] = {
                    "available": True,
                    "source": "direct_file",
                    "error": None
                }
                # SRT modules are bundled - no need to show success message
                return True, modules, "direct_file"
                
            except Exception as file_error:
                self._log_debug(f"❌ SRT modules failed to load (bundled files corrupted?): {file_error}")
        
        # Strategy 2: Try bundled package import
        if os.path.exists(self.bundled_chatterbox_dir):
            try:
                self._add_node_dir_to_path()
                from utils.timing.parser import SRTParser, SRTSubtitle, SRTParseError, validate_srt_timing_compatibility
                from engines.chatterbox.audio_timing import (
                    AudioTimingUtils, PhaseVocoderTimeStretcher, TimedAudioAssembler,
                    calculate_timing_adjustments, AudioTimingError
                )
                
                modules.update({
                    "SRTParser": SRTParser,
                    "SRTSubtitle": SRTSubtitle,
                    "SRTParseError": SRTParseError,
                    "validate_srt_timing_compatibility": validate_srt_timing_compatibility,
                    "AudioTimingUtils": AudioTimingUtils,
                    "PhaseVocoderTimeStretcher": PhaseVocoderTimeStretcher,
                    "TimedAudioAssembler": TimedAudioAssembler,
                    "calculate_timing_adjustments": calculate_timing_adjustments,
                    "AudioTimingError": AudioTimingError,
                })
                
                # Try to get FFmpegTimeStretcher
                try:
                    from engines.chatterbox.audio_timing import FFmpegTimeStretcher
                    modules["FFmpegTimeStretcher"] = FFmpegTimeStretcher
                except ImportError:
                    modules["FFmpegTimeStretcher"] = None
                
                self.loaded_modules[module_key] = modules
                self.import_status[module_key] = {
                    "available": True,
                    "source": "bundled_package",
                    "error": None
                }
                self._log_debug("✅ SRT modules loaded from bundled package")
                return True, modules, "bundled_package"
                
            except ImportError as bundled_error:
                self._log_debug(f"❌ Failed to load SRT from bundled package: {bundled_error}")
        
        # Strategy 3: Try system package import
        try:
            from utils.timing.parser import SRTParser, SRTSubtitle, SRTParseError, validate_srt_timing_compatibility
            from engines.chatterbox.audio_timing import (
                AudioTimingUtils, PhaseVocoderTimeStretcher, TimedAudioAssembler,
                calculate_timing_adjustments, AudioTimingError
            )
            
            modules.update({
                "SRTParser": SRTParser,
                "SRTSubtitle": SRTSubtitle,
                "SRTParseError": SRTParseError,
                "validate_srt_timing_compatibility": validate_srt_timing_compatibility,
                "AudioTimingUtils": AudioTimingUtils,
                "PhaseVocoderTimeStretcher": PhaseVocoderTimeStretcher,
                "TimedAudioAssembler": TimedAudioAssembler,
                "calculate_timing_adjustments": calculate_timing_adjustments,
                "AudioTimingError": AudioTimingError,
            })
            
            # Try to get FFmpegTimeStretcher
            try:
                from engines.chatterbox.audio_timing import FFmpegTimeStretcher
                modules["FFmpegTimeStretcher"] = FFmpegTimeStretcher
            except ImportError:
                modules["FFmpegTimeStretcher"] = None
            
            self.loaded_modules[module_key] = modules
            self.import_status[module_key] = {
                "available": True,
                "source": "system_package",
                "error": None
            }
            return True, modules, "system_package"
            
        except ImportError as system_error:
            self.import_status[module_key] = {
                "available": False,
                "source": None,
                "error": f"All import strategies failed. Last error: {system_error}"
            }
            self._log_debug(f"❌ SRT modules critically failed to load: {system_error}")
            return False, self._create_dummy_srt_modules(), "none"
    
    def _create_fallback_audio_timing(self) -> Dict[str, Any]:
        """Create minimal fallback implementations for audio timing."""
        import torch
        
        class AudioTimingUtils:
            @staticmethod
            def get_audio_duration(audio, sample_rate):
                if audio.dim() == 1:
                    return audio.size(0) / sample_rate
                elif audio.dim() == 2:
                    return audio.size(-1) / sample_rate
                else:
                    raise ValueError(f"Unsupported audio tensor dimensions: {audio.dim()}")
            
            @staticmethod
            def create_silence(duration_seconds, sample_rate, channels=1, device=None):
                num_samples = int(duration_seconds * sample_rate)
                if channels == 1:
                    return torch.zeros(num_samples, device=device)
                else:
                    return torch.zeros(channels, num_samples, device=device)
            
            @staticmethod
            def pad_audio_to_duration(audio, target_duration, sample_rate, pad_type="end"):
                current_duration = AudioTimingUtils.get_audio_duration(audio, sample_rate)
                if current_duration >= target_duration:
                    return audio
                
                pad_duration = target_duration - current_duration
                silence = AudioTimingUtils.create_silence(
                    pad_duration, sample_rate, 
                    channels=audio.shape[0] if audio.dim() == 2 else 1,
                    device=audio.device
                )
                
                if pad_type == "end":
                    return torch.cat([audio, silence], dim=-1)
                else:
                    return torch.cat([silence, audio], dim=-1)
            
            @staticmethod
            def seconds_to_samples(seconds, sample_rate):
                return int(seconds * sample_rate)
        
        class TimedAudioAssembler:
            def __init__(self, sample_rate):
                self.sample_rate = sample_rate
            
            def assemble_timed_audio(self, audio_segments, target_timings, fade_duration=0.01):
                return torch.cat(audio_segments, dim=-1)
        
        class AudioTimingError(Exception):
            pass
        
        def calculate_timing_adjustments(natural_durations, target_timings):
            adjustments = []
            for i, (natural_duration, (start_time, end_time)) in enumerate(zip(natural_durations, target_timings)):
                target_duration = end_time - start_time
                stretch_factor = target_duration / natural_duration if natural_duration > 0 else 1.0
                adjustments.append({
                    'segment_index': i,
                    'natural_duration': natural_duration,
                    'target_duration': target_duration,
                    'start_time': start_time,
                    'end_time': end_time,
                    'stretch_factor': stretch_factor,
                    'needs_stretching': abs(stretch_factor - 1.0) > 0.05,
                    'stretch_type': 'compress' if stretch_factor < 1.0 else 'expand' if stretch_factor > 1.0 else 'none'
                })
            return adjustments
        
        return {
            "AudioTimingUtils": AudioTimingUtils,
            "TimedAudioAssembler": TimedAudioAssembler,
            "AudioTimingError": AudioTimingError,
            "calculate_timing_adjustments": calculate_timing_adjustments,
            "PhaseVocoderTimeStretcher": None,  # Not available without librosa
            "FFmpegTimeStretcher": None,  # Not available without FFmpeg
        }
    
    def _create_dummy_srt_modules(self) -> Dict[str, Any]:
        """Create dummy SRT modules when imports fail."""
        
        class SRTParser:
            @staticmethod
            def parse_srt_content(content):
                raise ImportError("SRT support not available")
        
        class SRTSubtitle:
            def __init__(self, sequence=0, start_time=0.0, end_time=0.0, text=""):
                raise ImportError("SRT support not available - missing required modules")
        
        class SRTParseError(Exception):
            pass
        
        def validate_srt_timing_compatibility(*args, **kwargs):
            raise ImportError("SRT support not available - missing required modules")
        
        # Include fallback audio timing
        fallback_audio = self._create_fallback_audio_timing()
        
        return {
            "SRTParser": SRTParser,
            "SRTSubtitle": SRTSubtitle,
            "SRTParseError": SRTParseError,
            "validate_srt_timing_compatibility": validate_srt_timing_compatibility,
            **fallback_audio
        }
    
    def get_import_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current import status for all modules."""
        return self.import_status.copy()
    
    def get_availability_summary(self) -> Dict[str, bool]:
        """Get a summary of what's available."""
        return {
            "tts": self.import_status["chatterbox_tts"]["available"],
            "vc": self.import_status["chatterbox_vc"]["available"], 
            "srt": self.import_status["srt_parser"]["available"],
            "any_chatterbox": (
                self.import_status["chatterbox_tts"]["available"] or 
                self.import_status["chatterbox_vc"]["available"]
            ),
            "full_support": all(
                self.import_status[key]["available"] 
                for key in ["chatterbox_tts", "chatterbox_vc", "srt_parser"]
            )
        }


# Global import manager instance
import_manager = ImportManager()