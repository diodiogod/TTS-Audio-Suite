"""
Higgs Audio 2 Engine - Main TTS engine wrapper for ComfyUI integration
Provides high-quality text-to-speech with voice cloning capabilities
Based on boson_multimodal implementation by HiggsAudio team
"""

import torch
import torchaudio
import numpy as np
import os
import sys
import base64
import io
import json
import re
import time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# Add parent directory for imports
current_dir = os.path.dirname(__file__)
engines_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(engines_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import soundfile as sf
except ImportError:
    print("Warning: soundfile not installed. Install with: pip install soundfile")
    sf = None

# Import boson_multimodal modules
from .boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from .boson_multimodal.data_types import ChatMLSample, Message, AudioContent
from .boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample

# Import utilities
from utils.audio.processing import AudioProcessingUtils
from utils.audio.cache import CacheKeyGenerator, GLOBAL_AUDIO_CACHE, get_audio_cache
from utils.downloads.unified_downloader import unified_downloader
from utils.text.chunking import ImprovedChatterBoxChunker
from .higgs_audio_downloader import HiggsAudioDownloader
import folder_paths

# Global engine cache to avoid reloading models
_ENGINE_CACHE = {}

# Higgs Audio model configurations
HIGGS_AUDIO_MODELS = {
    "higgs-audio-v2-3B": {
        "generation_model": "bosonai/higgs-audio-v2-generation-3B-base",
        "tokenizer_model": "bosonai/higgs-audio-v2-tokenizer",
        "description": "Higgs Audio v2 3B parameter model"
    }
}


# Convert token-based limits to character-based for unified chunker
def tokens_to_chars(max_tokens: int) -> int:
    """Convert Higgs Audio token limit to character limit for unified chunker"""
    # Higgs Audio uses ~4 chars per token, but we use more conservative 3.5 for safety
    return int(max_tokens * 3.5)



class HiggsAudioEngine:
    """
    Main Higgs Audio 2 engine wrapper for ComfyUI
    Handles model loading, text generation, and voice cloning
    """
    
    def __init__(self):
        """Initialize Higgs Audio engine"""
        self.engine = None
        self.model_path = None
        self.tokenizer_path = None
        self.device = None
        # Use global shared cache (HiggsAudio generator already registered centrally)
        self.cache = get_audio_cache()
        self.downloader = HiggsAudioDownloader()
        
    
    
    def get_available_models(self) -> List[str]:
        """Get list of available Higgs Audio models"""
        return self.downloader.get_available_models()
    
    def initialize_engine(self, 
                         model_path: str = "bosonai/higgs-audio-v2-generation-3B-base",
                         tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer",
                         device: str = "auto") -> None:
        """
        Initialize or retrieve cached Higgs Audio engine
        
        Args:
            model_path: Path or HuggingFace model ID for generation model
            tokenizer_path: Path or HuggingFace model ID for audio tokenizer
            device: Device to use (auto, cuda, cpu)
        """
        global _ENGINE_CACHE
        
        # Auto-detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check cache
        cache_key = f"{model_path}_{tokenizer_path}_{device}"
        if cache_key in _ENGINE_CACHE:
            self.engine = _ENGINE_CACHE[cache_key]
            self.model_path = model_path
            self.tokenizer_path = tokenizer_path
            self.device = device
            print(f"💾 Using cached Higgs Audio engine")
            return
        
        print(f"🚀 Loading Higgs Audio 2 engine...")
        print(f"   Model: {model_path}")
        print(f"   Tokenizer: {tokenizer_path}")
        print(f"   Device: {device}")
        
        # Download models if needed
        model_path = self.downloader.download_model(model_path)
        tokenizer_path = self.downloader.download_tokenizer(tokenizer_path)
        
        try:
            # Create engine
            engine = HiggsAudioServeEngine(
                model_name_or_path=model_path,
                audio_tokenizer_name_or_path=tokenizer_path,
                device=device
            )
            
            # Cache engine
            _ENGINE_CACHE[cache_key] = engine
            self.engine = engine
            self.model_path = model_path
            self.tokenizer_path = tokenizer_path
            self.device = device
            
            print(f"✅ Higgs Audio 2 engine loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize Higgs Audio engine: {e}")
            raise
    
    def generate(self,
                text: str,
                reference_audio: Optional[Dict[str, Any]] = None,
                reference_text: str = "",
                audio_priority: str = "auto",
                system_prompt: str = "Generate audio following instruction.",
                max_new_tokens: int = 2048,
                temperature: float = 0.8,
                top_p: float = 0.6,
                top_k: int = 80,
                enable_chunking: bool = True,
                max_tokens_per_chunk: int = 225,
                silence_between_chunks_ms: int = 100,
                enable_cache: bool = True,
                character: str = "narrator",
                seed: int = -1) -> Tuple[Dict[str, Any], str]:
        """
        Generate audio from text using Higgs Audio 2
        
        Args:
            text: Input text to convert to speech
            voice_preset: Name of voice preset to use
            reference_audio: Reference audio for voice cloning
            reference_text: Text corresponding to reference audio
            audio_priority: Priority for audio source selection
            system_prompt: System prompt for generation
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            enable_chunking: Whether to chunk long text
            max_tokens_per_chunk: Maximum tokens per chunk
            silence_between_chunks_ms: Silence between chunks
            enable_cache: Whether to use caching
            character: Character name for generation
            seed: Random seed (-1 for random)
            
        Returns:
            Tuple of (audio dict, generation info string)
        """
        if not self.engine:
            raise RuntimeError("Engine not initialized. Call initialize_engine first.")
        
        start_time = time.time()
        
        # Check cache if enabled
        if enable_cache:
            cache_key = self.cache.generate_cache_key(
                engine_type='higgs_audio',
                text=text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                model_path=self.model_path,
                tokenizer_path=self.tokenizer_path,
                device=self.device,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                character=character
            )
            
            cached_result = self.cache.get_cached_audio(cache_key)
            if cached_result:
                audio_tensor, duration = cached_result
                print(f"💾 Using cached audio for Higgs Audio generation")
                return {"waveform": audio_tensor, "sample_rate": 24000}, f"Cached audio: {duration:.1f}s"
        
        # Process text for chunking if needed using unified chunker
        if enable_chunking:
            max_chars = tokens_to_chars(max_tokens_per_chunk)
            chunks = ImprovedChatterBoxChunker.split_into_chunks(text, max_chars)
            print(f"🧩 [HiggsAudio] Using unified chunker: {len(text)} chars -> {len(chunks)} chunks (max {max_chars} chars/chunk)")
        else:
            chunks = [text]
        
        print(f"🎤 Generating Higgs Audio for {len(chunks)} chunk(s)")
        
        # Generate audio for each chunk
        audio_segments = []
        total_tokens = 0
        
        for i, chunk in enumerate(chunks):
            chunk_tokens = len(chunk) // 4  # Approximate tokens for logging
            print(f"  Processing chunk {i+1}/{len(chunks)}: {len(chunk)} chars / ~{chunk_tokens} tokens")
            
            # Process single chunk
            chunk_audio, voice_info = self._process_single_chunk(
                chunk_text=chunk,
                reference_audio=reference_audio,
                reference_text=reference_text,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed if seed >= 0 else None
            )
            
            audio_segments.append(chunk_audio)
            total_tokens += chunk_tokens
        
        # Combine audio chunks
        if len(audio_segments) > 1:
            combined_audio = self._combine_audio_chunks(
                audio_segments, 
                combination_method="auto",
                silence_ms=silence_between_chunks_ms,
                crossfade_duration=0.1,
                text_length=len(text),
                original_text=text,
                text_chunks=chunks
            )
        else:
            combined_audio = audio_segments[0]
        
        # Calculate duration
        duration = combined_audio['waveform'].size(-1) / combined_audio['sample_rate']
        
        # Cache the result if enabled
        if enable_cache and 'cache_key' in locals():
            self.cache.cache_audio(cache_key, combined_audio['waveform'], duration)
        
        # Generate info string
        generation_time = time.time() - start_time
        info = (f"Generated {duration:.1f}s audio from {total_tokens} tokens "
                f"using {len(chunks)} chunk(s) in {generation_time:.1f}s")
        
        return combined_audio, info
    
    def generate_native_multispeaker(self,
                                   text: str,
                                   primary_reference_audio: Optional[Dict[str, Any]] = None,
                                   primary_reference_text: str = "",
                                   secondary_reference_audio: Optional[Dict[str, Any]] = None,
                                   secondary_reference_text: str = "",
                                   use_system_context: bool = True,
                                   system_prompt: str = "Generate audio following instruction.",
                                   max_new_tokens: int = 2048,
                                   temperature: float = 0.8,
                                   top_p: float = 0.6,
                                   top_k: int = 80,
                                   enable_cache: bool = True,
                                   character: str = "SPEAKER0",
                                   seed: int = -1) -> Tuple[Dict[str, Any], str]:
        """
        Generate audio using Higgs Audio 2's native multi-speaker capabilities
        
        Args:
            text: Input text with [SPEAKER0] and [SPEAKER1] tags
            primary_reference_audio: Reference audio for SPEAKER0
            primary_reference_text: Text corresponding to primary reference
            secondary_reference_audio: Reference audio for SPEAKER1  
            secondary_reference_text: Text corresponding to secondary reference
            use_system_context: If True, use system context mode; if False, use conversation mode
            system_prompt: System prompt for generation
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            enable_cache: Whether to use caching
            character: Character name for caching
            seed: Random seed (-1 for random)
            
        Returns:
            Tuple of (audio dict, generation info string)
        """
        if not self.engine:
            raise RuntimeError("Engine not initialized. Call initialize_engine first.")
        
        start_time = time.time()
        print(f"🎭 Native multi-speaker generation: {'System Context' if use_system_context else 'Conversation'} mode")
        
        # Build messages for ChatML format
        messages = []
        
        # Add system prompt
        if system_prompt.strip():
            messages.append(Message(role="system", content=system_prompt))
        
        # Add reference audios based on mode
        if use_system_context:
            # System context mode: Add both reference audios as system context
            if primary_reference_audio is not None:
                try:
                    primary_base64 = self._audio_to_base64(primary_reference_audio)
                    if primary_base64:
                        if primary_reference_text:
                            messages.append(Message(role="system", content=f"SPEAKER0 reference: {primary_reference_text}"))
                        audio_content = AudioContent(raw_audio=primary_base64, audio_url="")
                        messages.append(Message(role="system", content=[audio_content]))
                except Exception as e:
                    print(f"⚠️ Failed to encode primary reference audio: {e}")
            
            if secondary_reference_audio is not None:
                try:
                    secondary_base64 = self._audio_to_base64(secondary_reference_audio)
                    if secondary_base64:
                        if secondary_reference_text:
                            messages.append(Message(role="system", content=f"SPEAKER1 reference: {secondary_reference_text}"))
                        audio_content = AudioContent(raw_audio=secondary_base64, audio_url="")
                        messages.append(Message(role="system", content=[audio_content]))
                except Exception as e:
                    print(f"⚠️ Failed to encode secondary reference audio: {e}")
        else:
            # Conversation mode: Add reference audios as assistant messages
            if primary_reference_audio is not None:
                try:
                    primary_base64 = self._audio_to_base64(primary_reference_audio)
                    if primary_base64:
                        if primary_reference_text:
                            messages.append(Message(role="system", content=primary_reference_text))
                        audio_content = AudioContent(raw_audio=primary_base64, audio_url="")
                        messages.append(Message(role="assistant", content=[audio_content]))
                except Exception as e:
                    print(f"⚠️ Failed to encode primary reference audio: {e}")
            
            if secondary_reference_audio is not None:
                try:
                    secondary_base64 = self._audio_to_base64(secondary_reference_audio)
                    if secondary_base64:
                        if secondary_reference_text:
                            messages.append(Message(role="system", content=secondary_reference_text))
                        audio_content = AudioContent(raw_audio=secondary_base64, audio_url="")
                        messages.append(Message(role="assistant", content=[audio_content]))
                except Exception as e:
                    print(f"⚠️ Failed to encode secondary reference audio: {e}")
        
        # Add user text with SPEAKER tags
        messages.append(Message(role="user", content=text))
        
        # Create ChatML sample
        chat_sample = ChatMLSample(messages=messages)
        
        # Generate audio
        print(f"🗣️ Generating native multi-speaker audio...")
        
        try:
            output = self.engine.generate(
                chat_ml_sample=chat_sample,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else None,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                seed=seed
            )
            
            # Convert output to audio dict
            if hasattr(output, 'audio') and hasattr(output, 'sampling_rate'):
                audio_np = output.audio
                
                # Convert numpy to tensor
                if len(audio_np.shape) == 1:
                    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).float()
                elif len(audio_np.shape) == 2:
                    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).float()
                else:
                    audio_tensor = torch.from_numpy(audio_np).float()
                
                audio_result = {
                    "waveform": audio_tensor,
                    "sample_rate": output.sampling_rate
                }
                
                # Calculate duration and info
                duration = audio_tensor.size(-1) / output.sampling_rate
                generation_time = time.time() - start_time
                info = f"Native multi-speaker: {duration:.1f}s audio in {generation_time:.1f}s"
                
                print(f"  ✅ Generated {duration:.1f}s multi-speaker audio")
                return audio_result, info
            else:
                raise ValueError("Invalid audio output from Higgs Audio engine")
                
        except Exception as e:
            print(f"❌ Error during native multi-speaker generation: {e}")
            raise e
    
    def _process_single_chunk(self,
                             chunk_text: str,
                             reference_audio: Optional[Dict[str, Any]],
                             reference_text: str,
                             system_prompt: str,
                             max_new_tokens: int,
                             temperature: float,
                             top_p: float,
                             top_k: int,
                             seed: Optional[int] = None) -> Tuple[Dict[str, Any], str]:
        """
        Process a single text chunk to generate audio
        
        Returns:
            Tuple of (audio dict, voice info string)
        """
        # Build messages for ChatML format
        messages = []
        
        # Add system prompt
        if system_prompt.strip():
            messages.append(Message(role="system", content=system_prompt))
        
        # Determine voice source
        audio_for_cloning = None
        text_for_cloning = ""
        used_voice_info = "No voice cloning"
        
        # Check reference audio validity
        has_valid_reference = False
        if reference_audio is not None:
            try:
                if isinstance(reference_audio, dict) and "waveform" in reference_audio:
                    waveform = reference_audio["waveform"]
                    if hasattr(waveform, 'shape') and waveform.numel() > 0:
                        has_valid_reference = True
            except:
                pass
        
        # Use reference audio if available
        if has_valid_reference:
            audio_for_cloning = reference_audio
            text_for_cloning = reference_text.strip() or "Reference audio for voice cloning."
            used_voice_info = "Reference Audio Input"
            print(f"  Using reference audio for voice cloning")
        
        # Add voice cloning to messages if available
        if audio_for_cloning is not None:
            try:
                # Convert audio to base64
                audio_base64 = self._audio_to_base64(audio_for_cloning)
                if audio_base64:
                    # Add reference text as system message
                    if text_for_cloning:
                        messages.append(Message(role="system", content=text_for_cloning))
                    
                    # Add audio content as assistant message
                    audio_content = AudioContent(raw_audio=audio_base64, audio_url="")
                    messages.append(Message(role="assistant", content=[audio_content]))
                else:
                    used_voice_info = "Audio encoding failed - using basic TTS"
                    print("⚠️ Failed to encode reference audio")
            except Exception as e:
                print(f"❌ Error in audio processing: {e}")
                used_voice_info = f"Audio processing error: {str(e)}"
        
        # Add user text
        messages.append(Message(role="user", content=chunk_text))
        
        # Create ChatML sample
        chat_sample = ChatMLSample(messages=messages)
        
        # Generate audio
        print(f"🗣️ Generating audio...")
        
        try:
            output = self.engine.generate(
                chat_ml_sample=chat_sample,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k if top_k > 0 else None,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                seed=seed
            )
            
            # Convert output to audio dict
            if hasattr(output, 'audio') and hasattr(output, 'sampling_rate'):
                audio_np = output.audio
                
                # Convert numpy to tensor
                if len(audio_np.shape) == 1:
                    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0).float()
                elif len(audio_np.shape) == 2:
                    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).float()
                else:
                    audio_tensor = torch.from_numpy(audio_np).float()
                
                chunk_audio = {
                    "waveform": audio_tensor,
                    "sample_rate": output.sampling_rate
                }
                
                duration = audio_tensor.size(-1) / output.sampling_rate
                print(f"  ✅ Generated {duration:.1f}s audio")
                return chunk_audio, used_voice_info
            else:
                raise ValueError("Invalid audio output from Higgs Audio engine")
                
        except Exception as e:
            print(f"❌ Error during audio generation: {e}")
            raise e
    
    def _audio_to_base64(self, comfy_audio: Dict[str, Any]) -> str:
        """Convert ComfyUI audio format to base64 string"""
        if not sf:
            raise ImportError("soundfile is required for audio encoding")
        
        # Handle nested ComfyUI audio format
        if "waveform" in comfy_audio and isinstance(comfy_audio["waveform"], dict):
            # Nested format: {"waveform": {"waveform": tensor, "sample_rate": int}, ...}
            inner_audio = comfy_audio["waveform"]
            waveform = inner_audio["waveform"]
            sample_rate = inner_audio["sample_rate"]
        else:
            # Direct format: {"waveform": tensor, "sample_rate": int}
            waveform = comfy_audio["waveform"]
            sample_rate = comfy_audio["sample_rate"]
        
        # Ensure we have a tensor with dim() method
        if not hasattr(waveform, 'dim'):
            raise TypeError(f"Expected tensor with dim() method, got {type(waveform)}")
        
        # Handle tensor dimensions
        if waveform.dim() == 3:
            audio_np = waveform[0, 0].cpu().numpy()
        elif waveform.dim() == 2:
            audio_np = waveform[0].cpu().numpy()
        else:
            audio_np = waveform.cpu().numpy()
        
        # Write to buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, sample_rate, format='WAV')
        buffer.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        return audio_base64
    
    def _combine_audio_chunks(self, 
                             audio_segments: List[Dict[str, Any]], 
                             combination_method: str = "auto",
                             silence_ms: int = 100,
                             crossfade_duration: float = 0.1,
                             text_length: int = 0,
                             original_text: str = "",
                             text_chunks: List[str] = None) -> Dict[str, Any]:
        """Combine multiple audio chunks using modular combination utility"""
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        print(f"🔗 Combining {len(audio_segments)} chunks using '{combination_method}' method")
        
        # Extract waveforms and sample rate
        waveforms = [seg["waveform"] for seg in audio_segments]
        sample_rate = audio_segments[0]["sample_rate"]
        
        # Use modular chunk combiner
        from utils.audio.chunk_combiner import ChunkCombiner
        combined_waveform = ChunkCombiner.combine_chunks(
            audio_segments=waveforms,
            method=combination_method,
            silence_ms=silence_ms,
            crossfade_duration=crossfade_duration,
            sample_rate=sample_rate,
            text_length=text_length,
            original_text=original_text,
            text_chunks=text_chunks
        )
        
        print(f"  ✅ Combined waveform shape: {combined_waveform.shape}")
        return {"waveform": combined_waveform, "sample_rate": sample_rate}
    
    def cleanup(self):
        """Clean up resources"""
        global _ENGINE_CACHE
        
        if self.engine:
            # Clear from cache
            cache_key = f"{self.model_path}_{self.tokenizer_path}_{self.device}"
            if cache_key in _ENGINE_CACHE:
                del _ENGINE_CACHE[cache_key]
            
            self.engine = None
        
        # Run garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()