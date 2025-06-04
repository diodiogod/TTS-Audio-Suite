"""
ComfyUI Custom Nodes for ChatterboxTTS - Voice Edition
Enhanced with bundled ChatterBox support and improved chunking
SUPPORTS: Bundled ChatterBox (recommended) + System ChatterBox (fallback)
"""

import torch
import torchaudio
import numpy as np
import folder_paths
import os
import tempfile
import re
from typing import List

# Get the current node directory for bundled resources
NODE_DIR = os.path.dirname(__file__)
BUNDLED_CHATTERBOX_DIR = os.path.join(NODE_DIR, "chatterbox")
BUNDLED_MODELS_DIR = os.path.join(NODE_DIR, "models", "chatterbox")

# Debug: Print what we're trying to import
print("🔍 Attempting to import ChatterBox modules...")
print(f"📁 Node directory: {NODE_DIR}")
print(f"📁 Looking for bundled ChatterBox at: {BUNDLED_CHATTERBOX_DIR}")
print(f"📁 Looking for bundled models at: {BUNDLED_MODELS_DIR}")

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
    print("✅ Using BUNDLED ChatterBox from node folder")
    CHATTERBOX_TTS_AVAILABLE = True
    CHATTERBOX_VC_AVAILABLE = True
    USING_BUNDLED_CHATTERBOX = True
    
except ImportError as bundled_error:
    print(f"📦 Bundled ChatterBox not found: {bundled_error}")
    
    # Try system-installed ChatterBox as fallback
    try:
        from chatterbox.tts import ChatterboxTTS
        print("✅ ChatterboxTTS imported from system installation")
        CHATTERBOX_TTS_AVAILABLE = True
    except ImportError as e:
        print(f"❌ System ChatterboxTTS import failed: {e}")
        CHATTERBOX_TTS_AVAILABLE = False

    try:
        from chatterbox.vc import ChatterboxVC
        print("✅ ChatterboxVC imported from system installation")
        CHATTERBOX_VC_AVAILABLE = True
    except ImportError as e:
        print(f"❌ System ChatterboxVC import failed: {e}")
        CHATTERBOX_VC_AVAILABLE = False
    
    if CHATTERBOX_TTS_AVAILABLE and CHATTERBOX_VC_AVAILABLE:
        print("✅ Using SYSTEM ChatterBox installation")
        USING_BUNDLED_CHATTERBOX = False

CHATTERBOX_AVAILABLE = CHATTERBOX_TTS_AVAILABLE and CHATTERBOX_VC_AVAILABLE

if not CHATTERBOX_AVAILABLE:
    print("💡 Creating dummy classes for missing ChatterBox components")
    print("🎯 To fix this:")
    print("   1. Install ChatterBox: pip install chatterbox-tts")
    print("   2. OR place ChatterBox code in the node folder for bundled approach")
    
    # Create dummy classes so ComfyUI doesn't crash
    if not CHATTERBOX_TTS_AVAILABLE:
        class ChatterboxTTS:
            @classmethod
            def from_pretrained(cls, device):
                raise ImportError("ChatterboxTTS not available - install missing dependencies or add bundled version")
            
            @classmethod
            def from_local(cls, path, device):
                raise ImportError("ChatterboxTTS not available - install missing dependencies or add bundled version")
    
    if not CHATTERBOX_VC_AVAILABLE:
        class ChatterboxVC:
            @classmethod 
            def from_pretrained(cls, device):
                raise ImportError("ChatterboxVC not available - install missing dependencies or add bundled version")
                
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
    """
    Smart model path detection with priority order:
    1. Bundled models in node folder (best for portability)
    2. ComfyUI models folder (standard location)
    3. Let ChatterBox download from HuggingFace (requires auth)
    """
    model_paths = []
    
    # 1. Check for bundled models in node folder
    if os.path.exists(BUNDLED_MODELS_DIR) and os.listdir(BUNDLED_MODELS_DIR):
        model_paths.append(("bundled", BUNDLED_MODELS_DIR))
        print(f"📦 Found bundled models at: {BUNDLED_MODELS_DIR}")
    
    # 2. Check ComfyUI models folder
    comfyui_model_path = os.path.join(folder_paths.models_dir, "TTS", "chatterbox")
    if os.path.exists(comfyui_model_path) and os.listdir(comfyui_model_path):
        model_paths.append(("comfyui", comfyui_model_path))
        print(f"📁 Found ComfyUI models at: {comfyui_model_path}")
    
    # 3. HuggingFace download as fallback
    model_paths.append(("huggingface", None))
    
    return model_paths


print("🔍 Defining ChatterboxTTSNode class with enhanced chunking...")

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
            raise ImportError("ChatterboxTTS not available - check installation or add bundled version")
            
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.model is None or self.device != device:
            print(f"Loading ChatterboxTTS model on {device}...")
            
            # Get available model paths in priority order
            model_paths = find_chatterbox_models()
            
            model_loaded = False
            for source, path in model_paths:
                try:
                    if source == "bundled":
                        print(f"📦 Loading from bundled models: {path}")
                        self.model = ChatterboxTTS.from_local(path, device)
                        self.model_source = "bundled"
                        model_loaded = True
                        break
                    elif source == "comfyui":
                        print(f"📁 Loading from ComfyUI models: {path}")
                        self.model = ChatterboxTTS.from_local(path, device)
                        self.model_source = "comfyui"
                        model_loaded = True
                        break
                    elif source == "huggingface":
                        print("🌐 Loading from Hugging Face (requires authentication)...")
                        self.model = ChatterboxTTS.from_pretrained(device)
                        self.model_source = "huggingface"
                        model_loaded = True
                        break
                except Exception as e:
                    print(f"❌ Failed to load from {source}: {e}")
                    continue
            
            if not model_loaded:
                raise ImportError("Failed to load ChatterboxTTS from any source")
            
            self.device = device
            print(f"✅ ChatterboxTTS model loaded from {self.model_source}!")

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
            print(f"🤖 Auto-selected combination method: {method}")
        
        if method == "concatenate":
            print("🔗 Using simple concatenation")
            return torch.cat(audio_segments, dim=-1)
        
        elif method == "silence_padding":
            print(f"🔗 Adding {silence_ms}ms silence between chunks")
            combined = audio_segments[0]
            for i in range(1, len(audio_segments)):
                combined = self.chunker.add_silence_padding(
                    combined, silence_ms, self.model.sr
                )
                combined = torch.cat([combined, audio_segments[i]], dim=-1)
            return combined
        
        elif method == "crossfade":
            print("🔗 Using crossfade blending")
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
            print(f"📝 Processing single chunk: {text_length} characters")
            wav = self.process_audio_chunk(text, audio_prompt, exaggeration, temperature, cfg_weight)
            info = f"Generated {wav.size(-1) / self.model.sr:.1f}s audio from {text_length} characters (single chunk, {self.model_source} models)"
        else:
            # Split into chunks using improved chunker
            chunks = self.chunker.split_into_chunks(text, max_chars_per_chunk)
            print(f"📝 Processing {len(chunks)} chunks from {text_length} characters")
            print(f"   Max chars per chunk: {max_chars_per_chunk}")
            print(f"   Combination method: {chunk_combination_method}")
            
            # Process each chunk
            audio_segments = []
            for i, chunk in enumerate(chunks):
                chunk_length = len(chunk)
                print(f"🎤 Chunk {i+1}/{len(chunks)}: {chunk_length} chars")
                print(f"   Preview: {chunk[:60]}{'...' if len(chunk) > 60 else ''}")
                
                chunk_audio = self.process_audio_chunk(
                    chunk, audio_prompt, exaggeration, temperature, cfg_weight
                )
                audio_segments.append(chunk_audio)
            
            # Combine audio segments
            print(f"🔗 Combining {len(audio_segments)} audio segments")
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

print("✅ ChatterboxTTSNode class defined")
print("🔍 Defining ChatterboxVCNode class...")

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
            raise ImportError("ChatterboxVC not available - check installation or add bundled version")
            
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.model is None or self.device != device:
            print(f"Loading ChatterboxVC model on {device}...")
            
            # Get available model paths in priority order (same logic as TTS)
            model_paths = find_chatterbox_models()
            
            model_loaded = False
            for source, path in model_paths:
                try:
                    if source == "bundled":
                        print(f"📦 Loading VC from bundled models: {path}")
                        self.model = ChatterboxVC.from_local(path, device)
                        self.model_source = "bundled"
                        model_loaded = True
                        break
                    elif source == "comfyui":
                        print(f"📁 Loading VC from ComfyUI models: {path}")
                        self.model = ChatterboxVC.from_local(path, device)
                        self.model_source = "comfyui"
                        model_loaded = True
                        break
                    elif source == "huggingface":
                        print("🌐 Loading VC from Hugging Face (requires authentication)...")
                        self.model = ChatterboxVC.from_pretrained(device)
                        self.model_source = "huggingface"
                        model_loaded = True
                        break
                except Exception as e:
                    print(f"❌ Failed to load VC from {source}: {e}")
                    continue
            
            if not model_loaded:
                raise ImportError("Failed to load ChatterboxVC from any source")
            
            self.device = device
            print(f"✅ ChatterboxVC model loaded from {self.model_source}!")

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

print("✅ ChatterboxVCNode class defined")

# Print setup summary
print("\n" + "="*60)
print("🎉 CHATTERBOX VOICE NODES LOADED SUCCESSFULLY!")
print("="*60)
if USING_BUNDLED_CHATTERBOX:
    print("📦 Using BUNDLED ChatterBox (self-contained)")
elif CHATTERBOX_AVAILABLE:
    print("🔧 Using SYSTEM ChatterBox installation")
else:
    print("❌ ChatterBox not available - install or bundle required")

print(f"📁 Node directory: {NODE_DIR}")
print(f"📁 Bundled ChatterBox: {os.path.exists(BUNDLED_CHATTERBOX_DIR)}")
print(f"📁 Bundled models: {os.path.exists(BUNDLED_MODELS_DIR)}")
print("="*60)
print()

# Node mappings for ComfyUI - UPDATED: Unique names to avoid conflicts
NODE_CLASS_MAPPINGS = {
    "ChatterBoxVoiceTTS": ChatterboxTTSNode,
    "ChatterBoxVoiceVC": ChatterboxVCNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterBoxVoiceTTS": "🎤 ChatterBox Voice TTS",
    "ChatterBoxVoiceVC": "🔄 ChatterBox Voice Conversion", 
}