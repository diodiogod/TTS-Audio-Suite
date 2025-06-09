# Version and constants
VERSION = "1.1.0"
IS_DEV = True  # Set to False for release builds
VERSION_DISPLAY = f"v{VERSION}" + (" (dev)" if IS_DEV else "")
SEPARATOR = "=" * 70

"""
ComfyUI Custom Nodes for ChatterboxTTS - Voice Edition
Enhanced with bundled ChatterBox support and improved chunking
SUPPORTS: Bundled ChatterBox (recommended) + System ChatterBox (fallback)
"""

import warnings
warnings.filterwarnings('ignore', message='.*PerthNet.*')
warnings.filterwarnings('ignore', message='.*LoRACompatibleLinear.*')
warnings.filterwarnings('ignore', message='.*requires authentication.*')

import torch
import torchaudio
import numpy as np
import folder_paths
import os
import tempfile
import re
from typing import List, Tuple, Optional
try:
    from chatterbox.audio_timing import FFmpegTimeStretcher, PhaseVocoderTimeStretcher
except ImportError:
    try:
        from .chatterbox.audio_timing import FFmpegTimeStretcher, PhaseVocoderTimeStretcher
    except ImportError:
        # Create dummy classes if import fails
        FFmpegTimeStretcher = type('FFmpegTimeStretcher', (object,), {})
        PhaseVocoderTimeStretcher = type('PhaseVocoderTimeStretcher', (object,), {})

# Define a global in-memory cache
GLOBAL_AUDIO_CACHE = {}

# Get the current node directory for bundled resources
NODE_DIR = os.path.dirname(__file__)
BUNDLED_CHATTERBOX_DIR = os.path.join(NODE_DIR, "chatterbox")
BUNDLED_MODELS_DIR = os.path.join(NODE_DIR, "models", "chatterbox")

# Smart import logic: Try bundled first, then system
CHATTERBOX_TTS_AVAILABLE = False
CHATTERBOX_VC_AVAILABLE = False
USING_BUNDLED_CHATTERBOX = False

# Try to import bundled ChatterBox first
try:
    # Add the node directory to Python path temporarily for bundled imports
    import sys
    if NODE_DIR not in sys.path:
        sys.path.insert(0, NODE_DIR)
    
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.vc import ChatterboxVC
    CHATTERBOX_TTS_AVAILABLE = True
    CHATTERBOX_VC_AVAILABLE = True
    USING_BUNDLED_CHATTERBOX = True
    if IS_DEV:
        print("✅ ChatterBox TTS package found!")
    
except ImportError as bundled_error:
    # Try system-installed ChatterBox as fallback
    try:
        from chatterbox.tts import ChatterboxTTS
        CHATTERBOX_TTS_AVAILABLE = True
    except ImportError as e:
        CHATTERBOX_TTS_AVAILABLE = False

    try:
        from chatterbox.vc import ChatterboxVC
        CHATTERBOX_VC_AVAILABLE = True
    except ImportError as e:
        CHATTERBOX_VC_AVAILABLE = False
    
    if CHATTERBOX_TTS_AVAILABLE and CHATTERBOX_VC_AVAILABLE:
        USING_BUNDLED_CHATTERBOX = False
        if IS_DEV:
            print("✅ ChatterBox TTS package found (system)!")
            
CHATTERBOX_AVAILABLE = CHATTERBOX_TTS_AVAILABLE and CHATTERBOX_VC_AVAILABLE

if not CHATTERBOX_AVAILABLE:
    # Create dummy classes so ComfyUI doesn't crash
    if not CHATTERBOX_TTS_AVAILABLE:
        class ChatterboxTTS:
            @classmethod
            def from_pretrained(cls, device):
                raise ImportError("ChatterboxTTS not available")
            
            @classmethod
            def from_local(cls, path, device):
                raise ImportError("ChatterboxTTS not available - install missing dependencies or add bundled version")
    
    if not CHATTERBOX_VC_AVAILABLE:
        class ChatterboxVC:
            @classmethod 
            def from_pretrained(cls, device):
                raise ImportError("ChatterboxVC not available")
                
            @classmethod
            def from_local(cls, path, device):
                raise ImportError("ChatterboxVC not available - install missing dependencies or add bundled version")


class ImprovedChatterBoxChunker:
    """Enhanced text chunker inspired by Orpheus TTS approach"""
    
    @staticmethod
    def split_into_chunks(text: str, max_chars: int = 400) -> List[str]:
        """
        Split text into chunks with better sentence boundary handling.
        Uses character-based limits like Orpheus TTS for more predictable chunk sizes.
        """
        if not text.strip():
            return []
            
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # If text is short enough, return as single chunk
        if len(text) <= max_chars:
            return [text]
        
        # Split into sentences using robust regex (same as Orpheus)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence exceeds limit and we have content, start new chunk
            if len(current_chunk) + len(sentence) + 1 > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            # If single sentence is too long, split it further
            elif len(sentence) > max_chars:
                # Add current chunk if not empty
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Split long sentence by commas (Orpheus approach)
                parts = re.split(r'(?<=,)\s+', sentence)
                sub_chunk = ""
                
                for part in parts:
                    if len(sub_chunk) + len(part) + 1 > max_chars:
                        if sub_chunk:
                            chunks.append(sub_chunk.strip())
                            sub_chunk = part
                        else:
                            # Even single part is too long - split arbitrarily
                            for i in range(0, len(part), max_chars):
                                chunk_part = part[i:i+max_chars].strip()
                                if chunk_part:
                                    chunks.append(chunk_part)
                    else:
                        sub_chunk = sub_chunk + ", " + part if sub_chunk else part
                
                # Set remaining as current chunk
                if sub_chunk:
                    current_chunk = sub_chunk
            else:
                # Normal sentence - add to current chunk
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    @staticmethod
    def add_silence_padding(audio: torch.Tensor, duration_ms: int = 50, sample_rate: int = 22050) -> torch.Tensor:
        """Add brief silence between chunks to improve naturalness"""
        silence_samples = int(duration_ms * sample_rate / 1000)
        
        # Create silence tensor with same shape as audio tensor
        if audio.dim() == 1:
            # 1D audio tensor [samples]
            silence = torch.zeros(silence_samples)
        elif audio.dim() == 2:
            # 2D audio tensor [channels, samples]
            silence = torch.zeros(audio.shape[0], silence_samples)
        else:
            # Fallback - just match the last dimension
            silence_shape = list(audio.shape)
            silence_shape[-1] = silence_samples
            silence = torch.zeros(*silence_shape)
        
        return torch.cat([audio, silence], dim=-1)


def find_chatterbox_models():
    """Find ChatterBox model files in order of priority"""
    model_paths = []
    
    # 1. Check for bundled models in node folder
    bundled_model_path = os.path.join(BUNDLED_MODELS_DIR, "s3gen.pt")
    if os.path.exists(bundled_model_path):
        model_paths.append(("bundled", BUNDLED_MODELS_DIR))
        return model_paths  # Return immediately if bundled models found
    
    # 2. Check ComfyUI models folder - first check the standard location
    comfyui_model_path_standard = os.path.join(folder_paths.models_dir, "chatterbox", "s3gen.pt")
    if os.path.exists(comfyui_model_path_standard):
        model_paths.append(("comfyui", os.path.dirname(comfyui_model_path_standard)))
        return model_paths
    
    # 3. Check legacy location (TTS/chatterbox) for backward compatibility
    comfyui_model_path_legacy = os.path.join(folder_paths.models_dir, "TTS", "chatterbox", "s3gen.pt")
    if os.path.exists(comfyui_model_path_legacy):
        model_paths.append(("comfyui", os.path.dirname(comfyui_model_path_legacy)))
        return model_paths
    
    # 3. HuggingFace download as fallback (only if no local models found)
    model_paths.append(("huggingface", None))
    
    return model_paths


class ChatterboxTTSNode:
    """
    Enhanced Text-to-Speech node using ChatterboxTTS - Voice Edition
    SUPPORTS BUNDLED CHATTERBOX + Enhanced Chunking
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello! This is the enhanced ChatterboxTTS with bundled support and improved chunking. It can handle very long texts by intelligently splitting them into smaller segments."
                }),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "exaggeration": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.25, 
                    "max": 2.0, 
                    "step": 0.05
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0.05, 
                    "max": 5.0, 
                    "step": 0.05
                }),
                "cfg_weight": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
                "audio_prompt_path": ("STRING", {"default": ""}),
                # ENHANCED CHUNKING CONTROLS - ALL OPTIONAL FOR BACKWARD COMPATIBILITY
                "enable_chunking": ("BOOLEAN", {"default": True}),
                "max_chars_per_chunk": ("INT", {"default": 400, "min": 100, "max": 1000, "step": 50}),
                "chunk_combination_method": (["auto", "concatenate", "silence_padding", "crossfade"], {"default": "auto"}),
                "silence_between_chunks_ms": ("INT", {"default": 100, "min": 0, "max": 500, "step": 25}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "generation_info")
    FUNCTION = "generate_speech"
    CATEGORY = "ChatterBox Voice"

    def __init__(self):
        self.model = None
        self.device = None
        self.chunker = ImprovedChatterBoxChunker()
        self.model_source = None  # Track where models are loaded from

    def load_model(self, device):
        if not CHATTERBOX_TTS_AVAILABLE:
            raise ImportError("ChatterboxTTS not available")
            
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.model is None or self.device != device:
            # Get available model paths in priority order
            model_paths = find_chatterbox_models()
            
            model_loaded = False
            for source, path in model_paths:
                try:
                    if source == "bundled" or source == "comfyui":
                        # Load from local files silently
                        self.model = ChatterboxTTS.from_local(path, device)
                        self.model_source = source
                        model_loaded = True
                        break
                    elif source == "huggingface":
                        # Only show download message when needed
                        if IS_DEV:
                            print("🌐 Downloading models from Hugging Face...")
                        self.model = ChatterboxTTS.from_pretrained(device)
                        self.model_source = "huggingface"
                        model_loaded = True
                        break
                except Exception as e:
                    if IS_DEV:
                        print(f"❌ Failed to load from {source}: {e}")
                    continue
            
            if not model_loaded:
                raise ImportError("Failed to load ChatterboxTTS from any source")
            
            self.device = device
            if IS_DEV:
                print(f"✅ ChatterboxTTS model loaded from {self.model_source}")

    def process_audio_chunk(self, chunk_text: str, audio_prompt: str, exaggeration: float, 
                           temperature: float, cfg_weight: float) -> torch.Tensor:
        """Process a single text chunk into audio"""
        return self.model.generate(
            chunk_text,
            audio_prompt_path=audio_prompt,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight
        )

    def add_crossfade(self, audio1: torch.Tensor, audio2: torch.Tensor, 
                     fade_duration: float = 0.1) -> torch.Tensor:
        """Add crossfade between two audio segments"""
        fade_samples = int(fade_duration * self.model.sr)
        
        if audio1.size(-1) < fade_samples or audio2.size(-1) < fade_samples:
            return torch.cat([audio1, audio2], dim=-1)
        
        fade_out = torch.linspace(1.0, 0.0, fade_samples)
        fade_in = torch.linspace(0.0, 1.0, fade_samples)
        
        audio1_end = audio1[..., -fade_samples:] * fade_out
        audio2_start = audio2[..., :fade_samples] * fade_in
        crossfaded = audio1_end + audio2_start
        
        return torch.cat([
            audio1[..., :-fade_samples],
            crossfaded,
            audio2[..., fade_samples:]
        ], dim=-1)

    def combine_audio_chunks(self, audio_segments: List[torch.Tensor], method: str, 
                           silence_ms: int, text_length: int) -> torch.Tensor:
        """Combine audio segments using specified method"""
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        # Auto-select best method based on text length
        if method == "auto":
            if text_length > 1000:  # Very long text
                method = "silence_padding"
            elif text_length > 500:  # Medium text
                method = "crossfade"
            else:  # Short text
                method = "concatenate"
            # Auto-selected method
        
        if method == "concatenate":
            # Simple concatenation
            return torch.cat(audio_segments, dim=-1)
        
        elif method == "silence_padding":
            # Add silence between chunks
            combined = audio_segments[0]
            for i in range(1, len(audio_segments)):
                combined = self.chunker.add_silence_padding(
                    combined, silence_ms, self.model.sr
                )
                combined = torch.cat([combined, audio_segments[i]], dim=-1)
            return combined
        
        elif method == "crossfade":
            # Use crossfade blending
            combined = audio_segments[0]
            for i in range(1, len(audio_segments)):
                combined = self.add_crossfade(combined, audio_segments[i])
            return combined
        
        else:
            # Fallback to concatenation
            return torch.cat(audio_segments, dim=-1)

    def generate_speech(self, text, device, exaggeration, temperature, cfg_weight, seed, 
                       reference_audio=None, audio_prompt_path="", 
                       enable_chunking=True, max_chars_per_chunk=400, 
                       chunk_combination_method="auto", silence_between_chunks_ms=100):
        
        self.load_model(device)
        
        # Handle None/empty values for backward compatibility
        if enable_chunking is None:
            enable_chunking = True
        if max_chars_per_chunk is None or max_chars_per_chunk < 100:
            max_chars_per_chunk = 400
        if chunk_combination_method is None or chunk_combination_method == "":
            chunk_combination_method = "auto"
        if silence_between_chunks_ms is None or silence_between_chunks_ms == "":
            silence_between_chunks_ms = 100
        
        # Set seed for reproducibility
        if seed != 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)

        # Handle reference audio input
        audio_prompt = None
        if reference_audio is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                waveform = reference_audio["waveform"]
                if waveform.dim() == 3:
                    waveform = waveform.squeeze(0)
                torchaudio.save(tmp_file.name, waveform, reference_audio["sample_rate"])
                audio_prompt = tmp_file.name
        elif audio_prompt_path and os.path.exists(audio_prompt_path):
            audio_prompt = audio_prompt_path

        # Determine if chunking is needed
        text_length = len(text)
        
        if not enable_chunking or text_length <= max_chars_per_chunk:
            # Process single chunk
            wav = self.process_audio_chunk(text, audio_prompt, exaggeration, temperature, cfg_weight)
            info = f"Generated {wav.size(-1) / self.model.sr:.1f}s audio from {text_length} characters (single chunk, {self.model_source} models)"
        else:
            # Split into chunks using improved chunker
            chunks = self.chunker.split_into_chunks(text, max_chars_per_chunk)
            # Process multiple chunks
            
            # Process each chunk
            audio_segments = []
            for i, chunk in enumerate(chunks):
                chunk_length = len(chunk)
                # Process chunk
                
                chunk_audio = self.process_audio_chunk(
                    chunk, audio_prompt, exaggeration, temperature, cfg_weight
                )
                audio_segments.append(chunk_audio)
            
            # Combine audio segments
            # Combine audio segments
            wav = self.combine_audio_chunks(
                audio_segments, chunk_combination_method, silence_between_chunks_ms, text_length
            )
            
            # Generate info
            total_duration = wav.size(-1) / self.model.sr
            avg_chunk_size = text_length // len(chunks)
            info = f"Generated {total_duration:.1f}s audio from {text_length} characters using {len(chunks)} chunks (avg {avg_chunk_size} chars/chunk, {self.model_source} models)"

        # Clean up temporary file
        if reference_audio is not None and audio_prompt:
            try:
                os.unlink(audio_prompt)
            except:
                pass

        # Return audio in ComfyUI format
        return (
            {
                "waveform": wav.unsqueeze(0),  # Add batch dimension
                "sample_rate": self.model.sr
            },
            info
        )

class ChatterboxVCNode:
    """
    Voice Conversion node using ChatterboxVC - Voice Edition
    SUPPORTS BUNDLED CHATTERBOX
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_audio": ("AUDIO",),
                "target_audio": ("AUDIO",),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("converted_audio",)
    FUNCTION = "convert_voice"
    CATEGORY = "ChatterBox Voice"

    def __init__(self):
        self.model = None
        self.device = None
        self.model_source = None

    def load_model(self, device):
        if not CHATTERBOX_VC_AVAILABLE:
            raise ImportError("ChatterboxVC not available")
            
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.model is None or self.device != device:
            # Get available model paths in priority order (same logic as TTS)
            model_paths = find_chatterbox_models()
            
            model_loaded = False
            for source, path in model_paths:
                try:
                    if source == "bundled" or source == "comfyui":
                        # Load from local files silently
                        self.model = ChatterboxVC.from_local(path, device)
                        self.model_source = source
                        model_loaded = True
                        break
                    elif source == "huggingface":
                        # Only show download message when needed
                        if IS_DEV:
                            print("🌐 Downloading voice conversion models from Hugging Face...")
                        self.model = ChatterboxVC.from_pretrained(device)
                        self.model_source = "huggingface"
                        model_loaded = True
                        break
                except Exception as e:
                    if IS_DEV:
                        print(f"❌ Failed to load VC from {source}: {e}")
                    continue
            
            if not model_loaded:
                raise ImportError("Failed to load ChatterboxVC from any source")
            
            self.device = device
            if IS_DEV:
                print(f"✅ ChatterboxVC model loaded from {self.model_source}")

    def convert_voice(self, source_audio, target_audio, device):
        self.load_model(device)

        # Save audio to temporary files
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as source_tmp:
            source_waveform = source_audio["waveform"]
            if source_waveform.dim() == 3:
                source_waveform = source_waveform.squeeze(0)  # Remove batch dimension if present
            torchaudio.save(source_tmp.name, source_waveform, source_audio["sample_rate"])
            source_path = source_tmp.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as target_tmp:
            target_waveform = target_audio["waveform"]
            if target_waveform.dim() == 3:
                target_waveform = target_waveform.squeeze(0)  # Remove batch dimension if present
            torchaudio.save(target_tmp.name, target_waveform, target_audio["sample_rate"])
            target_path = target_tmp.name

        try:
            # Perform voice conversion
            wav = self.model.generate(
                source_path,
                target_voice_path=target_path
            )

            # Clean up temporary files
            os.unlink(source_path)
            os.unlink(target_path)

            # Return audio in ComfyUI format
            return ({
                "waveform": wav.unsqueeze(0),  # Add batch dimension
                "sample_rate": self.model.sr
            },)

        except Exception as e:
            # Clean up on error
            try:
                os.unlink(source_path)
                os.unlink(target_path)
            except:
                pass
            raise e


# Initialize SRT support variables silently
SRT_SUPPORT_AVAILABLE = False
SRTParser = None
SRTSubtitle = None
SRTParseError = None
validate_srt_timing_compatibility = None
AudioTimingUtils = None
PhaseVocoderTimeStretcher = None
TimedAudioAssembler = None
calculate_timing_adjustments = None
AudioTimingError = None

# Import SRT support modules silently
import importlib.util

try:
    srt_imported = False
    
    # Check if SRT files exist in bundled chatterbox directory
    srt_parser_path = os.path.join(BUNDLED_CHATTERBOX_DIR, 'srt_parser.py')
    audio_timing_path = os.path.join(BUNDLED_CHATTERBOX_DIR, 'audio_timing.py')
    
    if os.path.exists(srt_parser_path) and os.path.exists(audio_timing_path):
        try:
            # Load SRT modules directly from files silently
            srt_parser_spec = importlib.util.spec_from_file_location("srt_parser", srt_parser_path)
            srt_parser_module = importlib.util.module_from_spec(srt_parser_spec)
            srt_parser_spec.loader.exec_module(srt_parser_module)
            
            # Extract SRT parser classes and functions
            SRTParser = srt_parser_module.SRTParser
            SRTSubtitle = srt_parser_module.SRTSubtitle
            SRTParseError = srt_parser_module.SRTParseError
            validate_srt_timing_compatibility = srt_parser_module.validate_srt_timing_compatibility
            
            # Try to load audio_timing module
            try:
                audio_timing_spec = importlib.util.spec_from_file_location("audio_timing", audio_timing_path)
                audio_timing_module = importlib.util.module_from_spec(audio_timing_spec)
                audio_timing_spec.loader.exec_module(audio_timing_module)
                
                # Extract audio timing classes and functions
                AudioTimingUtils = audio_timing_module.AudioTimingUtils
                PhaseVocoderTimeStretcher = audio_timing_module.PhaseVocoderTimeStretcher
                TimedAudioAssembler = audio_timing_module.TimedAudioAssembler
                calculate_timing_adjustments = audio_timing_module.calculate_timing_adjustments
                AudioTimingError = audio_timing_module.AudioTimingError
                
                srt_imported = True
                if IS_DEV:
                    print("✅ SRT TTS node available!")
            except Exception:
                if IS_DEV:
                    print("⚠️ Advanced audio timing not available - using fallback")
                
                # Create minimal fallback implementations for audio timing
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
                
                # Set fallback implementations
                PhaseVocoderTimeStretcher = None  # Not available without librosa
                
                srt_imported = True  # SRT parsing still works
            
        except Exception:
            if IS_DEV:
                print("❌ Failed to load SRT from files")
            pass
    
    # Try package imports if direct file import failed
    if not srt_imported:
        try:
            # Try bundled package import
            if os.path.exists(BUNDLED_CHATTERBOX_DIR):
                from chatterbox.srt_parser import SRTParser, SRTSubtitle, SRTParseError, validate_srt_timing_compatibility
                from chatterbox.audio_timing import (
                    AudioTimingUtils, PhaseVocoderTimeStretcher, TimedAudioAssembler,
                    calculate_timing_adjustments, AudioTimingError
                )
                srt_imported = True
                if IS_DEV:
                    print("✅ SRT loaded from bundled package")
        except ImportError:
            # Try system package import
            try:
                from chatterbox.srt_parser import SRTParser, SRTSubtitle, SRTParseError, validate_srt_timing_compatibility
                from chatterbox.audio_timing import (
                    AudioTimingUtils, PhaseVocoderTimeStretcher, TimedAudioAssembler,
                    calculate_timing_adjustments, AudioTimingError
                )
                srt_imported = True
                if IS_DEV:
                    print("✅ SRT loaded from system package")
            except ImportError:
                if IS_DEV:
                    print("❌ Failed to load SRT from packages")
                pass
    
    if srt_imported:
        SRT_SUPPORT_AVAILABLE = True
    else:
        raise ImportError("SRT import failed")
    
except Exception:
    SRT_SUPPORT_AVAILABLE = False
    if IS_DEV:
        print("❌ SRT support not available")
    
    # Create dummy classes for missing SRT components
    class SRTParser:
        @staticmethod
        def parse_srt_content(content):
            raise ImportError("SRT support not available")
    
    class SRTSubtitle:
        def __init__(self, sequence=0, start_time=0.0, end_time=0.0, text=""):
            raise ImportError("SRT support not available - missing required modules")
    
    class SRTParseError(Exception):
        pass
    
    class AudioTimingUtils:
        @staticmethod
        def get_audio_duration(*args, **kwargs):
            raise ImportError("SRT support not available - missing required modules")
        
        @staticmethod
        def create_silence(*args, **kwargs):
            raise ImportError("SRT support not available - missing required modules")
    
    class TimedAudioAssembler:
        def __init__(self, *args, **kwargs):
            raise ImportError("SRT support not available - missing required modules")
    
    class AudioTimingError(Exception):
        pass
    
    def validate_srt_timing_compatibility(*args, **kwargs):
        raise ImportError("SRT support not available - missing required modules")
    
    def calculate_timing_adjustments(*args, **kwargs):
        raise ImportError("SRT support not available - missing required modules")

class ChatterboxSRTTTSNode:
    """
    SRT Subtitle-aware Text-to-Speech node using ChatterboxTTS
    Generates timed audio that matches SRT subtitle timing
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "srt_content": ("STRING", {
                    "multiline": True,
                    "default": """1
00:00:01,000 --> 00:00:04,000
Hello! This is the first subtitle. I'll make it long on purpose.

2
00:00:04,50 --> 00:00:09,500
This is the second subtitle with precise timing.

3
00:00:10,000 --> 00:00:14,000
The audio will match these exact timings.""",
                    "tooltip": "The SRT subtitle content. Each entry defines a text segment and its precise start and end times."
                }),
                "device": (["auto", "cuda", "cpu"], {"default": "auto", "tooltip": "The device to run the TTS model on (auto, cuda, or cpu)."}),
                "exaggeration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.25,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Controls the expressiveness and emphasis of the generated speech. Higher values increase exaggeration."
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.05,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Controls the randomness and creativity of the generated speech. Higher values lead to more varied outputs."
                }),
                "cfg_weight": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Classifier-Free Guidance weight. Influences how strongly the model adheres to the input text."
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "tooltip": "Seed for reproducible speech generation. Set to 0 for random."}),
                "timing_mode": (["stretch_to_fit", "pad_with_silence", "smart_natural"], {
                    "default": "smart_natural",
                    "tooltip": "Determines how audio segments are aligned with SRT timings:\n🔹 stretch_to_fit: Stretches/compresses audio to exactly match SRT segment durations.\n🔹 pad_with_silence: Places natural audio at SRT start times, padding gaps with silence. May result in overlaps.\n🔹 smart_natural: Intelligently adjusts timings within 'timing_tolerance', prioritizing natural audio and shifting subsequent segments. Applies stretch/shrink within limits if needed."
                }),
            },
            "optional": {
                "reference_audio": ("AUDIO", {"tooltip": "Optional reference audio input from another ComfyUI node for voice cloning or style transfer. This is an alternative to 'audio_prompt_path'."}),
                "audio_prompt_path": ("STRING", {"default": "", "tooltip": "Path to an audio file on disk to use as a prompt for voice cloning or style transfer. This is an alternative to 'reference_audio'."}),
                "enable_audio_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, generated audio segments will be cached in memory to speed up subsequent runs with identical parameters."
                }),
                "fade_for_StretchToFit": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Duration (in seconds) for crossfading between audio segments in 'stretch_to_fit' mode."
                }),
                "max_stretch_ratio": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Maximum factor to slow down audio in 'smart_natural' mode. (e.g., 2.0x means audio can be twice as long). Recommend leaving at 1.0 for natural speech preservation and silence addition."
                }),
                "min_stretch_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Minimum factor to speed up audio in 'smart_natural' mode. (e.g., 0.5x means audio can be half as long). min=faster speech"
                }),
                "timing_tolerance": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Maximum allowed deviation (in seconds) for timing adjustments in 'smart_natural' mode. Higher values allow more flexibility."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio", "generation_info", "timing_report", "Adjusted_SRT", "warnings")
    FUNCTION = "generate_srt_speech"
    CATEGORY = "ChatterBox Voice"

    def __init__(self):
        self.model = None
        self.device = None
        self.model_source = None
        self.srt_parser = SRTParser() if SRT_SUPPORT_AVAILABLE else None
        # Audio segment cache for performance optimization
        self.cache_enabled = True


    def load_model(self, device):
        """Load ChatterboxTTS model (same as original node)"""
        if not CHATTERBOX_TTS_AVAILABLE:
            raise ImportError("ChatterboxTTS not available - check installation or add bundled version")
            
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.model is None or self.device != device:
            # Get available model paths in priority order
            model_paths = find_chatterbox_models()
            
            model_loaded = False
            for source, path in model_paths:
                try:
                    if source == "bundled" or source == "comfyui":
                        # Load from local files silently
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            self.model = ChatterboxTTS.from_local(path, device)
                            self.model_source = source
                            model_loaded = True
                            break
                    elif source == "huggingface":
                        # Only attempt HuggingFace download if it's in our paths
                        if source == "huggingface":
                            self.model = ChatterboxTTS.from_pretrained(device)
                            self.model_source = "huggingface"
                            model_loaded = True
                            break
                except Exception:
                    continue
            
            if not model_loaded:
                raise ImportError("ChatterboxTTS not available")
            
            self.device = device

    def _generate_segment_cache_key(self, subtitle_text, exaggeration, temperature, cfg_weight, seed,
                                   audio_prompt_component, model_source, device):
        """Generate cache key for a single audio segment based on generation parameters."""
        import hashlib
        
        # Create a hash of all parameters that affect TTS generation for this segment
        cache_data = {
            'text': subtitle_text,
            'exaggeration': exaggeration,
            'temperature': temperature,
            'cfg_weight': cfg_weight,
            'seed': seed,
            'audio_prompt_component': audio_prompt_component, # Use the consistent component
            'model_source': model_source,
            'device': device
        }
        # Convert to string and hash
        cache_string = str(sorted(cache_data.items()))
        cache_key = hashlib.md5(cache_string.encode()).hexdigest()
        return cache_key

    def _get_cached_segment_audio(self, segment_cache_key):
        """Retrieve cached audio for a single segment if available from in-memory cache."""
        if not self.cache_enabled or segment_cache_key not in GLOBAL_AUDIO_CACHE:
            return None
        
        cached_audio, natural_duration = GLOBAL_AUDIO_CACHE[segment_cache_key]
        # Cache hit - reuse silently
        return cached_audio, natural_duration

    def _cache_segment_audio(self, segment_cache_key, audio_tensor, natural_duration):
        """Cache generated audio for a single segment for future use in-memory."""
        if not self.cache_enabled:
            return
        
        GLOBAL_AUDIO_CACHE[segment_cache_key] = (audio_tensor.clone(), natural_duration) # Clone to avoid reference issues
        # Store in cache silently

    def generate_srt_speech(self, srt_content, device, exaggeration, temperature, cfg_weight, seed,
                            timing_mode, reference_audio=None, audio_prompt_path="",
                            max_stretch_ratio=2.0, min_stretch_ratio=0.5, fade_for_StretchToFit=0.01, enable_audio_cache=True, timing_tolerance=2.0):
        
        # Clear any previous warnings
        self._timing_warnings = []
        
        if not SRT_SUPPORT_AVAILABLE:
            raise ImportError("SRT support not available - missing required modules")
        
        self.load_model(device)
        
        # Update cache setting
        self.cache_enabled = enable_audio_cache
        # Initialize cache status tracking
        any_segment_cached = False
        
        # Set seed for reproducibility
        if seed != 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)

        # Handle reference audio input
        audio_prompt = None
        if reference_audio is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                waveform = reference_audio["waveform"]
                if waveform.dim() == 3:
                    waveform = waveform.squeeze(0)
                torchaudio.save(tmp_file.name, waveform, reference_audio["sample_rate"])
                audio_prompt = tmp_file.name
        elif audio_prompt_path and os.path.exists(audio_prompt_path):
            audio_prompt = audio_prompt_path

        try:
            # Parse SRT content silently
            subtitles = self.srt_parser.parse_srt_content(srt_content)
            
            # Validate timing compatibility
            warnings = validate_srt_timing_compatibility(subtitles, max_stretch_ratio, min_stretch_ratio)
            if warnings:
                # Add warnings to report
                if not hasattr(self, '_timing_warnings'):
                    self._timing_warnings = []
                self._timing_warnings.extend(warnings)
            
            # Determine the audio prompt component for cache key generation
            audio_prompt_component = ""
            if reference_audio is not None:
                import hashlib
                waveform_hash = hashlib.md5(reference_audio["waveform"].cpu().numpy().tobytes()).hexdigest()
                audio_prompt_component = f"ref_audio_{waveform_hash}_{reference_audio['sample_rate']}"
            elif audio_prompt_path:
                audio_prompt_component = audio_prompt_path

            # Generate audio segments
            audio_segments = []
            natural_durations = []
            
            for i, subtitle in enumerate(subtitles):
                
                # Generate segment-specific cache key
                segment_cache_key = self._generate_segment_cache_key(
                    subtitle.text, exaggeration, temperature, cfg_weight, seed,
                    audio_prompt_component, self.model_source, device
                )
                
                # Try to get cached audio for this segment
                cached_segment_data = self._get_cached_segment_audio(segment_cache_key)
                
                if cached_segment_data:
                    wav, natural_duration = cached_segment_data
                    # Cache hit - use silently
                    any_segment_cached = True
                else:
                    # Generate audio for this subtitle
                    wav = self.model.generate(
                        subtitle.text,
                        audio_prompt_path=audio_prompt,
                        exaggeration=exaggeration,
                        temperature=temperature,
                        cfg_weight=cfg_weight
                    )
                    natural_duration = AudioTimingUtils.get_audio_duration(wav, self.model.sr)
                    self._cache_segment_audio(segment_cache_key, wav, natural_duration)
                    # Generated new audio - cache silently
                
                audio_segments.append(wav)
                natural_durations.append(natural_duration)
            
            # Calculate basic adjustments for caching (this part remains the same)
            target_timings = [(sub.start_time, sub.end_time) for sub in subtitles]
            adjustments = calculate_timing_adjustments(natural_durations, target_timings)
            
            # Generate timing report (adjustments might be updated below for smart_natural)
            # This line needs to be after the smart_natural processing if adjustments are overridden
            # For now, we'll keep it here and ensure adjustments is correctly set before this call.
            # If timing_mode is smart_natural, adjustments will be overridden below.
            
            # Assemble final audio based on timing mode
            # Assemble audio using selected mode
            
            # Normalize all audio segments first to ensure consistent dimensions
            normalized_segments = []
            for i, audio in enumerate(audio_segments):
                if audio.dim() == 1:
                    normalized_segments.append(audio)
                elif audio.dim() == 2:
                    normalized_segments.append(audio)
                elif audio.dim() == 3 and audio.shape[0] == 1:
                    # Remove batch dimension if present
                    normalized = audio.squeeze(0)
                    # Track normalization adjustments
                    normalized_segments.append(normalized)
                else:
                    raise RuntimeError(f"Unsupported audio tensor shape for segment {i}: {audio.shape}")
            
            if timing_mode == "smart_natural":
                # Smart balanced timing: use natural audio but add minimal adjustments within tolerance
                # Process with smart natural mode
                final_audio, smart_adjustments = self._assemble_with_smart_timing(
                    normalized_segments, subtitles, self.model.sr, timing_tolerance,
                    max_stretch_ratio, min_stretch_ratio
                )
                total_duration = AudioTimingUtils.get_audio_duration(final_audio, self.model.sr)
                # Override adjustments for smart_natural mode
                adjustments = smart_adjustments
                
            elif timing_mode == "pad_with_silence":
                # Add silence to match timing without stretching
                # Process with silence padding mode
                final_audio = self._assemble_audio_with_overlaps(
                    normalized_segments, subtitles, self.model.sr
                )
                total_duration = AudioTimingUtils.get_audio_duration(final_audio, self.model.sr)
                
            else:  # stretch_to_fit
                # Use time stretching to match exact timing
                # Process with stretch to fit mode
                assembler = TimedAudioAssembler(self.model.sr)
                final_audio = assembler.assemble_timed_audio(
                    normalized_segments, target_timings, fade_duration=fade_for_StretchToFit
                )
                total_duration = AudioTimingUtils.get_audio_duration(final_audio, self.model.sr)
            
            # Generate timing report AFTER all adjustments are finalized
            timing_report = self._generate_timing_report(subtitles, adjustments, timing_mode)
            
            # Generate info with cache status and stretching method
            cache_status = "cached" if any_segment_cached else "generated"
            stretch_info = ""
            
            # Get stretching method info
            if timing_mode == "stretch_to_fit":
                current_stretcher = assembler.time_stretcher
            elif timing_mode == "smart_natural":
                # Use the stored stretcher type for smart_natural mode
                if hasattr(self, '_smart_natural_stretcher'):
                    if self._smart_natural_stretcher == "ffmpeg":
                        stretch_info = ", Stretching method: FFmpeg"
                    else:
                        stretch_info = ", Stretching method: Phase Vocoder"
                else:
                    stretch_info = ", Stretching method: Unknown"
            
            # For stretch_to_fit mode, examine the actual stretcher
            if timing_mode == "stretch_to_fit" and 'current_stretcher' in locals():
                if isinstance(current_stretcher, FFmpegTimeStretcher):
                    stretch_info = ", Stretching method: FFmpeg"
                elif isinstance(current_stretcher, PhaseVocoderTimeStretcher):
                    stretch_info = ", Stretching method: Phase Vocoder"
                else:
                    stretch_info = f", Stretching method: {current_stretcher.__class__.__name__}"
            
            info = (f"Generated {total_duration:.1f}s SRT-timed audio from {len(subtitles)} subtitles "
                   f"using {timing_mode} mode ({cache_status} segments, {self.model_source} models{stretch_info})")
            
            # Generation complete
            
            # Ensure final_audio is [channels, samples] for torchaudio.save
            if final_audio.dim() == 1:
                final_audio = final_audio.unsqueeze(0) # Convert [samples] to [1, samples]
            elif final_audio.dim() > 2:
                raise RuntimeError(f"Unexpected final_audio dimensions: {final_audio.dim()}D. Expected 1D or 2D.")

            # Generate the Adjusted_SRT string
            adjusted_srt_string = self._generate_adjusted_srt_string(subtitles, adjustments, timing_mode)

            return (
                {
                    "waveform": final_audio.unsqueeze(0),  # Add batch dimension for ComfyUI's format
                    "sample_rate": self.model.sr
                },
                info,
                timing_report,
                adjusted_srt_string,
                "\n".join(self._timing_warnings) if hasattr(self, '_timing_warnings') and self._timing_warnings else ""
            )
            
        except SRTParseError:
            raise ValueError("SRT parsing error")
        except AudioTimingError:
            raise ValueError("Audio timing error")
        except Exception:
            raise RuntimeError("SRT TTS generation failed")
        finally:
            # Clean up temporary file
            if reference_audio is not None and audio_prompt:
                try:
                    os.unlink(audio_prompt)
                except:
                    pass

    def _assemble_with_silence_padding(self, audio_segments: List[torch.Tensor],
                                     subtitles: List, sample_rate: int) -> torch.Tensor:
        """Assemble audio with silence padding to match SRT timing"""
        result_segments = []
        
        # Process audio segments with silence padding
        for i, (audio, subtitle) in enumerate(zip(audio_segments, subtitles)):
            
            # Normalize audio tensor to ensure consistent shape
            if audio.dim() == 1:
                # Keep 1D audio as-is
                normalized_audio = audio
            elif audio.dim() == 2:
                # Keep 2D audio as-is
                normalized_audio = audio
            elif audio.dim() == 3 and audio.shape[0] == 1:
                # Remove batch dimension if present
                normalized_audio = audio.squeeze(0)
                # Normalize audio dimensions
            else:
                raise RuntimeError(f"Unsupported audio tensor shape for segment {i}: {audio.shape}")
            
            # Add normalized audio segment
            result_segments.append(normalized_audio)
            
            # Add silence gap to next subtitle (if not last)
            if i < len(subtitles) - 1:
                next_subtitle = subtitles[i + 1]
                gap_duration = next_subtitle.start_time - subtitle.end_time
                
                if gap_duration > 0:
                    # Create silence gap
                    
                    # Calculate silence duration in samples
                    silence_samples = int(gap_duration * sample_rate)
                    
                    # Create silence tensor that exactly matches the normalized audio tensor shape
                    if normalized_audio.dim() == 1:
                        # 1D audio: create 1D silence [samples]
                        silence = torch.zeros(silence_samples, device=normalized_audio.device, dtype=normalized_audio.dtype)
                        # Create 1D silence
                    elif normalized_audio.dim() == 2:
                        # 2D audio: create 2D silence [channels, samples]
                        num_channels = normalized_audio.shape[0]
                        silence = torch.zeros(num_channels, silence_samples, device=normalized_audio.device, dtype=normalized_audio.dtype)
                        # Create 2D silence
                    else:
                        raise RuntimeError(f"Unsupported normalized audio tensor dimensions: {normalized_audio.dim()}")
                    
                    # Validate silence tensor
                    
                    # Final validation that silence matches audio dimensions
                    if silence.dim() != normalized_audio.dim():
                        raise RuntimeError(
                            f"Silence tensor dimension mismatch: audio is {normalized_audio.dim()}D {normalized_audio.shape}, "
                            f"but silence is {silence.dim()}D {silence.shape}"
                        )
                    
                    result_segments.append(silence)
        
        # Prepare segments for concatenation
        
        # Final validation: ensure all segments have the same number of dimensions
        if len(result_segments) > 1:
            reference_dim = result_segments[0].dim()
            reference_shape = result_segments[0].shape
            
            for i, segment in enumerate(result_segments):
                if segment.dim() != reference_dim:
                    raise RuntimeError(
                        f"TENSOR DIMENSION MISMATCH at segment {i}: "
                        f"Expected {reference_dim}D tensor like {reference_shape}, "
                        f"but got {segment.dim()}D tensor with shape {segment.shape}. "
                        f"This indicates a bug in tensor shape normalization."
                    )
        
        try:
            final_audio = torch.cat(result_segments, dim=-1)
            # Concatenate segments
            return final_audio
        except RuntimeError as e:
            # Concatenation failed
            raise RuntimeError("Failed to concatenate audio segments")

    def _generate_timing_report(self, subtitles: List,
                               adjustments: List[dict], timing_mode: str) -> str:
        """Generate detailed timing report"""
        report_lines = [
            f"SRT Timing Report ({timing_mode} mode)",
            "=" * 50,
            f"Total subtitles: {len(subtitles)}",
            f"Total duration: {subtitles[-1].end_time:.3f}s",
            "",
            "Per-subtitle analysis:"
        ]
        
        if timing_mode == "smart_natural":
            # For smart_natural mode, iterate directly over the detailed adjustments report
            for adj in adjustments:
                report_lines.append(
                    f"  {adj['sequence']:2d}. Original SRT: {adj['original_srt_start']:6.2f}-{adj['original_srt_end']:6.2f}s "
                    f"(Target: {adj['original_srt_duration']:.2f}s)"
                )
                report_lines.append(f"      Natural Audio: {adj['natural_audio_duration']:.3f}s")
                
                for action in adj['actions']:
                    report_lines.append(f"      - {action}")
                
                report_lines.append(
                    f"      Final Audio Duration: {adj['final_segment_duration']:.3f}s "
                    f"(Final SRT: {adj['final_srt_start']:6.2f}-{adj['final_srt_end']:6.2f}s)"
                )
                # Find the corresponding subtitle to get its text
                # This assumes subtitles are sorted by sequence or index
                original_subtitle_text = next((s.text for s in subtitles if s.sequence == adj['sequence']), "N/A")
                report_lines.append(f"      Text: {original_subtitle_text[:60]}{'...' if len(original_subtitle_text) > 60 else ''}")
        else:
            # For other modes, iterate using zip with original subtitles
            for i, (subtitle, adj) in enumerate(zip(subtitles, adjustments)):
                if timing_mode == "pad_with_silence":
                    # For pad_with_silence mode, show overlap/gap information
                    timing_info = ""
                    if adj['natural_duration'] > subtitle.duration:
                        # Audio is longer than SRT slot - will overlap
                        overlap = adj['natural_duration'] - subtitle.duration
                        timing_info = f" 🔁 [OVERLAP: +{overlap:.2f}s]"
                    elif i < len(subtitles) - 1:
                        # Check for silence gap to next subtitle
                        next_subtitle = subtitles[i + 1]
                        gap_duration = next_subtitle.start_time - subtitle.end_time
                        if gap_duration > 0:
                            timing_info = f" [+{gap_duration:.2f}s silence]"
                    
                    report_lines.append(
                        f"  {i+1:2d}. {subtitle.start_time:6.2f}-{subtitle.end_time:6.2f}s "
                        f"({subtitle.duration:.2f}s target, {adj['natural_duration']:.2f}s natural){timing_info}"
                    )
                else:
                    # For other modes (e.g., stretch_to_fit), show stretch information
                    stretch_info = ""
                    if adj['needs_stretching']:
                        stretch_info = f" [{adj['stretch_type']} {adj['stretch_factor']:.2f}x]"
                    
                    report_lines.append(
                        f"  {i+1:2d}. {subtitle.start_time:6.2f}-{subtitle.end_time:6.2f}s "
                        f"({subtitle.duration:.2f}s target, {adj['natural_duration']:.2f}s natural){stretch_info}"
                    )
                
                report_lines.append(f"      Text: {subtitle.text[:60]}{'...' if len(subtitle.text) > 60 else ''}")
        
        # Summary statistics
        if timing_mode == "pad_with_silence":
            total_gaps = 0
            total_gap_duration = 0
            total_overlaps = 0
            total_overlap_duration = 0
            
            for i, (subtitle, adj) in enumerate(zip(subtitles, adjustments)):
                if adj['natural_duration'] > subtitle.duration:
                    total_overlaps += 1
                    total_overlap_duration += adj['natural_duration'] - subtitle.duration
                elif i < len(subtitles) - 1:
                    gap_duration = subtitles[i + 1].start_time - subtitles[i].end_time
                    if gap_duration > 0:
                        total_gaps += 1
                        total_gap_duration += gap_duration
            
            summary_lines = [
                "",
                "Summary:",
                f"  Audio preserved at natural timing (no stretching)"
            ]
            
            if total_overlaps > 0:
                summary_lines.append(f"  Timing overlaps: {total_overlaps} segments, +{total_overlap_duration:.2f}s total overlap")
            
            if total_gaps > 0:
                summary_lines.append(f"  Silence gaps added: {total_gaps} gaps, {total_gap_duration:.2f}s total silence")
            
            if total_overlaps == 0 and total_gaps == 0:
                summary_lines.append(f"  Perfect timing match - no gaps or overlaps")
            
            report_lines.extend(summary_lines)
        elif timing_mode == "smart_natural":
            total_shifted = sum(1 for adj in adjustments if adj['next_segment_shifted_by'] > 0)
            total_stretched = sum(1 for adj in adjustments if abs(adj['stretch_factor_applied'] - 1.0) > 0.01)
            total_padded = sum(1 for adj in adjustments if adj['padding_added'] > 0)
            total_truncated = sum(1 for adj in adjustments if adj['truncated_by'] > 0)
            
            summary_lines = [
                "",
                "Summary (Smart Natural Mode):",
                f"  Segments with next segment shifted: {total_shifted}/{len(adjustments)}",
                f"  Segments with audio stretched/shrunk: {total_stretched}/{len(adjustments)}",
                f"  Segments with silence padded: {total_padded}/{len(adjustments)}",
                f"  Segments truncated: {total_truncated}/{len(adjustments)}",
            ]
            report_lines.extend(summary_lines)
        else:
            total_stretch_needed = sum(1 for adj in adjustments if adj['needs_stretching'])
            avg_stretch = np.mean([adj['stretch_factor'] for adj in adjustments])
            
            report_lines.extend([
                "",
                "Summary:",
                f"  Segments needing time adjustment: {total_stretch_needed}/{len(adjustments)}",
                f"  Average stretch factor: {avg_stretch:.2f}x",
            ])
        
        return "\n".join(report_lines)

    def _generate_adjusted_srt_string(self, subtitles: List, adjustments: List[dict], timing_mode: str) -> str:
        """
        Generates a multiline SRT string from the final adjusted timings.
        """
        srt_lines = []
        for i, adj in enumerate(adjustments):
            # Convert seconds to SRT time format: HH:MM:SS,ms
            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                seconds_int = int(seconds % 60)
                milliseconds = int((seconds - int(seconds)) * 1000)
                return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"

            # Determine start and end times based on timing_mode and available keys
            if timing_mode == "smart_natural":
                start_time_val = adj['final_srt_start']
                end_time_val = adj['final_srt_start'] + adj['final_segment_duration']
            elif timing_mode == "pad_with_silence":
                start_time_val = adj['start_time']
                end_time_val = adj['start_time'] + adj['natural_duration']
            else: # Default to stretch_to_fit or any other mode
                # Fallback to original SRT times if final adjusted times are not available
                start_time_val = adj.get('final_srt_start', adj.get('start_time'))
                end_time_val = adj.get('final_srt_end', adj.get('end_time'))
                
                # Ensure we have valid times, if not, use original subtitle times as a last resort
                if start_time_val is None or end_time_val is None:
                    # This should ideally not happen if adjustments are correctly populated
                    # but as a safeguard, use the original subtitle's times
                    original_subtitle = subtitles[i] # Assuming adjustments are in order
                    start_time_val = original_subtitle.start_time
                    end_time_val = original_subtitle.end_time
            # For stretch_to_fit, the original start_time and end_time are already the target times
            # and the audio is stretched to fit, so adj['end_time'] is correct.

            start_time_str = format_time(start_time_val)
            end_time_str = format_time(end_time_val)
            
            # Retrieve the original subtitle text using the segment_index or sequence
            # The 'subtitles' list is the original parsed list.
            # 'adj' contains 'segment_index' and 'sequence'.
            # For smart_natural, 'original_text' is already in adj.
            # For other modes, we need to look it up from the original 'subtitles' list.
            
            srt_text = adj.get('original_text')
            if srt_text is None: # If not smart_natural, get text from original subtitles list
                # Find the corresponding subtitle by index (assuming adjustments are in order)
                if i < len(subtitles):
                    srt_text = subtitles[i].text
                else:
                    srt_text = f"Subtitle {adj.get('sequence', i+1)}" # Fallback text

            srt_lines.append(str(adj.get('sequence', i+1))) # Use sequence if available, else index + 1
            srt_lines.append(f"{start_time_str} --> {end_time_str}")
            srt_lines.append(srt_text)
            srt_lines.append("") # Empty line separates entries
        
        return "\n".join(srt_lines)

    def _assemble_with_smart_timing(self, audio_segments: List[torch.Tensor],
                                   subtitles: List, sample_rate: int, tolerance: float,
                                   max_stretch_ratio: float, min_stretch_ratio: float) -> Tuple[torch.Tensor, List[dict]]:
        """
        Smart balanced timing: Adjusts SRT segment timings based on actual spoken duration
        and a user-defined timing_tolerance.
        
        Logic:
        1. Calculate actual spoken duration for each segment.
        2. Check Fit: Determine if spoken duration fits within allocated SRT timeframe.
        3. Adjust Next Segment Start (if needed): If spoken duration exceeds allocated time,
           attempt to shift the start time of the *next* SRT segment forward, within `timing_tolerance`.
        4. Stretch/Shrink (if still needed): If segment still doesn't fit after next segment adjustment,
           apply stretch/shrink factor to make it fit within the new interval, respecting `timing_tolerance`.
        5. Pad with Silence (last resort): If segment cannot be made to fit, pad remaining time with silence.
        
        Returns:
            Tuple[torch.Tensor, List[dict]]: The assembled audio and a list of dictionaries
            detailing the adjustments made for each segment.
        """
        processed_segments = []
        smart_adjustments_report = []
        
        # Create a mutable copy of subtitles to adjust start/end times
        mutable_subtitles = [SRTSubtitle(s.sequence, s.start_time, s.end_time, s.text) for s in subtitles]
        
        # Initialize stretcher for smart_natural mode
        try:
            # Try FFmpeg first
            print("Smart natural mode: Trying FFmpeg stretcher...")
            time_stretcher = FFmpegTimeStretcher()
            self._smart_natural_stretcher = "ffmpeg"
            print("Smart natural mode: Using FFmpeg stretcher")
        except AudioTimingError as e:
            # Fall back to Phase Vocoder
            print(f"Smart natural mode: FFmpeg initialization failed ({str(e)}), falling back to Phase Vocoder")
            time_stretcher = PhaseVocoderTimeStretcher()
            self._smart_natural_stretcher = "phase_vocoder"
            print("Smart natural mode: Using Phase Vocoder stretcher")
        
        # Process audio with smart natural timing
        
        for i, audio in enumerate(audio_segments):
            current_subtitle = mutable_subtitles[i]
            natural_duration = AudioTimingUtils.get_audio_duration(audio, sample_rate)
            
            original_srt_start = subtitles[i].start_time # Use original for reference
            original_srt_end = subtitles[i].end_time
            
            # Step 1: Check if natural duration fits within current SRT slot
            # This is the duration the SRT *originally* allocated for this segment
            initial_target_duration = original_srt_end - original_srt_start
            
            segment_report = {
                'segment_index': i,
                'sequence': current_subtitle.sequence,
                'original_srt_start': original_srt_start,
                'original_srt_end': original_srt_end,
                'original_srt_duration': initial_target_duration,
                'natural_audio_duration': natural_duration,
                'next_segment_shifted_by': 0.0,
                'stretch_factor_applied': 1.0,
                'padding_added': 0.0,
                'truncated_by': 0.0,
                'final_segment_duration': natural_duration, # Will be updated
                'final_srt_start': original_srt_start, # Will be updated
                'final_srt_end': original_srt_end, # Will be updated
                'original_text': subtitles[i].text, # Add original subtitle text
                'actions': []
            }
            
            # Process segment timing
            
            # Calculate how much extra time is needed for the natural audio
            time_needed_beyond_srt = natural_duration - initial_target_duration
            
            adjusted_current_segment_end = original_srt_end # This will be updated
            
            # Step 2 & 3: Adjust Next Segment Start (if needed)
            if time_needed_beyond_srt > 0: # Natural audio is longer than original SRT slot
                # Audio is longer than slot
                segment_report['actions'].append(f"Natural audio ({natural_duration:.3f}s) is longer than original SRT slot ({initial_target_duration:.3f}s) by {time_needed_beyond_srt:.3f}s.")
                
                if i + 1 < len(mutable_subtitles):
                    next_subtitle = mutable_subtitles[i+1]
                    original_next_srt_start = subtitles[i+1].start_time # Use original for reference
                    
                    # First, try to consume any existing gap to the next subtitle
                    existing_gap = original_next_srt_start - original_srt_end
                    if existing_gap > 0:
                        time_to_consume_from_gap = min(time_needed_beyond_srt, existing_gap)
                        time_needed_beyond_srt -= time_to_consume_from_gap
                        adjusted_current_segment_end += time_to_consume_from_gap
                        segment_report['actions'].append(f"Consumed {time_to_consume_from_gap:.3f}s from existing gap. Remaining excess: {time_needed_beyond_srt:.3f}s.")
                        # Gap consumed
                    
                    if time_needed_beyond_srt > 0: # Still need more time after consuming gap
                        next_natural_audio_duration = AudioTimingUtils.get_audio_duration(audio_segments[i+1], sample_rate)
                        
                        # Calculate "room" in the next segment: how much shorter its natural audio is than its SRT slot
                        next_segment_room = max(0.0, next_subtitle.duration - next_natural_audio_duration)
                        # Calculate available room
                        segment_report['actions'].append(f"Next segment (Seq {next_subtitle.sequence}) has {next_segment_room:.3f}s room.")

                        # How much can we shift the next subtitle without exceeding tolerance?
                        # This is the amount of time we can "borrow" from the next segment's start.
                        max_shift_allowed = min(tolerance, next_segment_room) # Only shift into its room, within tolerance
                        
                        # How much do we *want* to shift the next subtitle?
                        desired_shift = time_needed_beyond_srt
                        
                        actual_shift = min(desired_shift, max_shift_allowed)
                        
                        if actual_shift > 0:
                            # Shift the next subtitle's start and end times
                            next_subtitle.start_time += actual_shift
                            next_subtitle.end_time += actual_shift
                            adjusted_current_segment_end += actual_shift # Add to the already adjusted end
                            segment_report['next_segment_shifted_by'] = actual_shift
                            segment_report['actions'].append(f"Shifted next subtitle (Seq {next_subtitle.sequence}) by {actual_shift:.3f}s. New next SRT start: {next_subtitle.start_time:.3f}s.")
                            # Subtitle shifted
                        else:
                            segment_report['actions'].append("Cannot shift next subtitle within tolerance/available room.")
                            # Cannot shift subtitle
                    else: # No next subtitle or no excess after consuming gap
                        segment_report['actions'].append("No next subtitle to shift or excess consumed by gap.")
                        print("   No next subtitle to shift or excess consumed by gap.")
                else:
                    segment_report['actions'].append("No next subtitle to shift.")
                    print("   No next subtitle to shift.")
            
            # Step 4: Stretch/Shrink (if still needed)
            # The new target duration for the current segment is from its original start to its (potentially) adjusted end
            new_target_duration = adjusted_current_segment_end - original_srt_start
            
            # Calculate stretch factor needed to fit natural audio into the new target duration
            stretch_factor = new_target_duration / natural_duration if natural_duration > 0 else 1.0
            
            # Apply stretch factor limits based on max_stretch_ratio and min_stretch_ratio
            clamped_stretch_factor = max(min_stretch_ratio, min(max_stretch_ratio, stretch_factor))
            
            # Check if stretching is actually needed and if it's within acceptable limits
            if abs(clamped_stretch_factor - 1.0) > 0.01: # Apply stretch if deviation is more than 1%
                # Apply audio stretching
                segment_report['actions'].append(f"⏱️ Applying stretch/shrink: natural {natural_duration:.3f}s -> target {new_target_duration:.3f}s (factor: {clamped_stretch_factor:.3f}x).")
                segment_report['stretch_factor_applied'] = clamped_stretch_factor
                try:
                    stretched_audio = time_stretcher.time_stretch(audio, clamped_stretch_factor, sample_rate)
                    processed_audio = stretched_audio
                except Exception as e:
                    segment_report['actions'].append("Time stretching failed, using padding/truncation")
                    # Time stretching failed, use fallback
                    processed_audio = audio # Use original audio if stretching fails
            else:
                processed_audio = audio
                segment_report['actions'].append("No significant stretch/shrink needed.")
                # No stretching needed
            
            # Step 5: Pad with Silence (last resort) or Truncate
            final_processed_duration = AudioTimingUtils.get_audio_duration(processed_audio, sample_rate)
            
            if final_processed_duration < new_target_duration:
                padding_needed = new_target_duration - final_processed_duration
                if padding_needed > 0:
                    segment_report['padding_added'] = padding_needed
                    segment_report['actions'].append(f"Padding with {padding_needed:.3f}s silence to reach target duration.")
                    # Add silence padding
                    processed_audio = AudioTimingUtils.pad_audio_to_duration(processed_audio, new_target_duration, sample_rate, "end")
            elif final_processed_duration > new_target_duration:
                # Truncate if still too long
                truncated_by = final_processed_duration - new_target_duration
                segment_report['truncated_by'] = truncated_by
                segment_report['actions'].append(f"🚧 Truncating audio by {truncated_by:.3f}s.")
                # Truncate audio
                target_samples = AudioTimingUtils.seconds_to_samples(new_target_duration, sample_rate)
                processed_audio = processed_audio[..., :target_samples]
            
            processed_segments.append(processed_audio)
            
            # Update the current subtitle's end time to reflect the final processed duration
            # This is important for calculating the gap to the *next* segment in the final assembly
            current_subtitle.end_time = original_srt_start + AudioTimingUtils.get_audio_duration(processed_audio, sample_rate)
            
            segment_report['final_segment_duration'] = AudioTimingUtils.get_audio_duration(processed_audio, sample_rate)
            segment_report['final_srt_start'] = current_subtitle.start_time
            segment_report['final_srt_end'] = current_subtitle.end_time
            
            smart_adjustments_report.append(segment_report)
            
            # Add processed segment to report
            
        # Final assembly: concatenate all processed segments with silence in between
        # The mutable_subtitles now contain the potentially adjusted start/end times
        final_audio_parts = []
        current_output_time = 0.0
        
        for i, segment_audio in enumerate(processed_segments):
            current_subtitle = mutable_subtitles[i]
            
            # Add silence if there's a gap between current_output_time and current_subtitle's start_time
            if current_output_time < current_subtitle.start_time:
                gap_duration = current_subtitle.start_time - current_output_time
                # Add silence gap
                silence = AudioTimingUtils.create_silence(gap_duration, sample_rate,
                                                           channels=segment_audio.shape[0] if segment_audio.dim() == 2 else 1,
                                                           device=segment_audio.device)
                final_audio_parts.append(silence)
                current_output_time += gap_duration
            
            final_audio_parts.append(segment_audio)
            current_output_time += AudioTimingUtils.get_audio_duration(segment_audio, sample_rate)
            
            # Segment appended
            
        if not final_audio_parts:
            return torch.empty(0, device=audio_segments[0].device, dtype=audio_segments[0].dtype), smart_adjustments_report
            
        # Ensure all parts have the same number of dimensions before concatenating
        # This handles cases where some segments might be 1D and others 2D (e.g., mono vs stereo)
        # Assuming all audio segments are either 1D (mono) or 2D (multi-channel)
        target_dim = processed_segments[0].dim() if processed_segments else 1
        target_channels = processed_segments[0].shape[0] if target_dim == 2 else 1
        
        normalized_final_audio_parts = []
        for part in final_audio_parts:
            if part.dim() == target_dim:
                if target_dim == 2 and part.shape[0] != target_channels:
                    # Handle channel mismatch for 2D tensors (e.g., mono audio in stereo context)
                    if part.shape[0] == 1: # Mono audio, expand to target_channels
                        normalized_final_audio_parts.append(part.repeat(target_channels, 1))
                    else:
                        raise RuntimeError(f"Channel mismatch in final assembly: Expected {target_channels} channels, got {part.shape[0]}")
                else:
                    normalized_final_audio_parts.append(part)
            elif part.dim() == 1 and target_dim == 2: # Mono part, target is stereo
                normalized_final_audio_parts.append(part.unsqueeze(0).repeat(target_channels, 1))
            elif part.dim() == 2 and target_dim == 1: # Stereo part, target is mono (shouldn't happen if first segment is 1D)
                # This case implies a mix of mono and stereo, which is problematic.
                # For simplicity, if target is 1D, sum multi-channel to mono.
                normalized_final_audio_parts.append(torch.sum(part, dim=0))
            else:
                raise RuntimeError(f"Dimension mismatch in final assembly: Expected {target_dim}D, got {part.dim()}D")
        
        return torch.cat(normalized_final_audio_parts, dim=-1), smart_adjustments_report

    def _assemble_audio_with_overlaps(self, audio_segments: List[torch.Tensor],
                                     subtitles: List, sample_rate: int) -> torch.Tensor:
        """
        Assemble audio by placing segments at their SRT start times, allowing audible overlaps.
        Silence is implicitly added in gaps.
        """
        if not audio_segments:
            return torch.empty(0) # Return empty tensor if no segments

        # Determine output buffer properties from the first segment
        first_segment = audio_segments[0]
        num_channels = first_segment.shape[0] if first_segment.dim() == 2 else 1
        device = first_segment.device
        dtype = first_segment.dtype

        # Calculate total duration needed for the output buffer
        # This should be at least the end time of the last subtitle,
        # or the end time of the last audio segment if it extends beyond its subtitle.
        max_end_time = 0.0
        for i, (audio, subtitle) in enumerate(zip(audio_segments, subtitles)):
            segment_end_time = subtitle.start_time + (audio.size(-1) / sample_rate)
            max_end_time = max(max_end_time, segment_end_time)
        
        # Ensure the buffer is at least as long as the last subtitle's end time
        if subtitles:
            max_end_time = max(max_end_time, subtitles[-1].end_time)

        total_samples = int(max_end_time * sample_rate)
        

        # Initialize output buffer with zeros
        if num_channels == 1:
            output_audio = torch.zeros(total_samples, device=device, dtype=dtype)
        else:
            output_audio = torch.zeros(num_channels, total_samples, device=device, dtype=dtype)

        for i, (audio, subtitle) in enumerate(zip(audio_segments, subtitles)):
            # Process segment

            # Ensure normalized_audio matches the channel dimension of output_audio
            # The `audio` input to this method (`normalized_segments`) should already be 1D or 2D.
            normalized_audio = audio # Start with the input audio segment

            if num_channels == 1: # Output is mono (1D)
                if normalized_audio.dim() == 2:
                    # If segment is 2D (e.g., [1, samples] or [channels, samples]), squeeze to 1D
                    if normalized_audio.shape[0] == 1: # If it's [1, samples], just squeeze
                        normalized_audio = normalized_audio.squeeze(0)
                    else: # If it's multi-channel, sum to mono
                        normalized_audio = torch.sum(normalized_audio, dim=0)
                # If it's already 1D, no change needed
            else: # Output is stereo/multi-channel (2D)
                if normalized_audio.dim() == 1:
                    # If segment is mono (1D), expand to 2D and repeat channels
                    normalized_audio = normalized_audio.unsqueeze(0).repeat(num_channels, 1)
                elif normalized_audio.dim() == 2 and normalized_audio.shape[0] != num_channels:
                    # If segment is 2D but has wrong channel count, raise error
                    raise RuntimeError(f"Channel mismatch: output buffer has {num_channels} channels, but segment {i} has {normalized_audio.shape[0]} channels.")
            
            # Segment normalized


            start_sample = int(subtitle.start_time * sample_rate)
            end_sample_segment = start_sample + normalized_audio.size(-1)

            # Resize output_audio if current segment extends beyond current buffer size
            if end_sample_segment > output_audio.size(-1):
                new_total_samples = end_sample_segment
                if num_channels == 1:
                    new_output_audio = torch.zeros(new_total_samples, device=device, dtype=dtype)
                else:
                    new_output_audio = torch.zeros(num_channels, new_total_samples, device=device, dtype=dtype)
                
                # Copy existing audio to the new larger buffer
                current_len = output_audio.size(-1)
                if num_channels == 1:
                    new_output_audio[:current_len] = output_audio
                else:
                    new_output_audio[:, :current_len] = output_audio
                output_audio = new_output_audio
                # Buffer resized

            # Add (mix) the current audio segment into the output buffer
            # Ensure dimensions match for addition
            if output_audio.dim() == 1:
                output_audio[start_sample:end_sample_segment] += normalized_audio
            else:
                output_audio[:, start_sample:end_sample_segment] += normalized_audio
            
            # Segment placed

        # Assembly complete
        return output_audio



# Register nodes
NODE_CLASS_MAPPINGS = {
    "ChatterBoxVoiceTTS": ChatterboxTTSNode,
    "ChatterBoxVoiceVC": ChatterboxVCNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterBoxVoiceTTS": "🎤 ChatterBox Voice TTS",
    "ChatterBoxVoiceVC": "🔄 ChatterBox Voice Conversion",
}

# Add SRT node if available
if SRT_SUPPORT_AVAILABLE:
    NODE_CLASS_MAPPINGS["ChatterBoxSRTVoiceTTS"] = ChatterboxSRTTTSNode
    NODE_DISPLAY_NAME_MAPPINGS["ChatterBoxSRTVoiceTTS"] = "📺 ChatterBox SRT Voice TTS"

# Print startup banner
print(SEPARATOR)
print(f"🚀 ChatterBox Voice Extension {VERSION_DISPLAY}")

# Check for local models
model_paths = find_chatterbox_models()
first_source = model_paths[0][0] if model_paths else None
print(f"Using model source: {first_source}")

if first_source == "bundled":
    print("✓ Using bundled models")
elif first_source == "comfyui":
    print("✓ Using ComfyUI models")
elif first_source == "huggingface":
    print("⚠️ No local models found - will download from Hugging Face")
    print("💡 Tip: First generation will download models (~1GB)")
    print("   Models will be saved locally for future use")
else:
    print("⚠️ No local models found - will download from Hugging Face")
    print("💡 Tip: First generation will download models (~1GB)")
    print("   Models will be saved locally for future use")
print(SEPARATOR)

# Print final initialization with nodes list
print(f"🚀 ChatterBox Voice Extension {VERSION_DISPLAY} loaded with {len(NODE_DISPLAY_NAME_MAPPINGS)} nodes:")
for node in sorted(NODE_DISPLAY_NAME_MAPPINGS.values()):
    print(f"   • {node}")
print(SEPARATOR)
