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
import logging

# boson_multimodal INFO logs now commented out directly in source
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

# Import boson_multimodal modules (imports register HiggsAudioConfig with transformers)
from .boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel  # Registers config first
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

# Import unified model interface for ComfyUI integration (replaces _ENGINE_CACHE)
from utils.models.unified_model_interface import load_tts_model

# Higgs Audio model configurations (matches downloader format)
HIGGS_AUDIO_MODELS = {
    "higgs-audio-v2-3B": {
        "generation_repo": "bosonai/higgs-audio-v2-generation-3B-base",
        "tokenizer_repo": "bosonai/higgs-audio-v2-tokenizer",
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
    
    # Class-level flag to prevent repetitive memory management warnings
    _memory_warning_shown = False
    
    def __init__(self):
        """Initialize Higgs Audio engine"""
        self.engine = None
        self.model_path = None
        self.tokenizer_path = None
        self.device = None
        # Use global shared cache (HiggsAudio generator already registered centrally)
        self.cache = get_audio_cache()
        self.downloader = HiggsAudioDownloader()

    def to(self, device):
        """
        Move all model components to the specified device.

        Critical for ComfyUI model management - ensures all components move together
        when models are detached to CPU and later reloaded to CUDA.

        Note: Higgs Audio uses CUDA graphs which are corrupted by CPU migration.
        After offloading to CPU, the model should be marked as invalid for reuse.
        """
        self.device = device

        # Move the underlying HiggsAudioServeEngine
        if hasattr(self.engine, 'to'):
            self.engine = self.engine.to(device)
        elif hasattr(self.engine, 'model') and hasattr(self.engine.model, 'to'):
            # Move the model component if engine doesn't have .to()
            self.engine.model = self.engine.model.to(device)

        return self

    def get_available_models(self) -> List[str]:
        """Get list of available Higgs Audio models"""
        return self.downloader.get_available_models()
    
    def initialize_engine(self, 
                         model_path: str = "bosonai/higgs-audio-v2-generation-3B-base",
                         tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer",
                         device: str = "auto",
                         enable_cuda_graphs: bool = True) -> None:
        """
        Initialize Higgs Audio engine using ComfyUI model management
        
        Args:
            model_path: Path or HuggingFace model ID for generation model
            tokenizer_path: Path or HuggingFace model ID for audio tokenizer
            device: Device to use (auto, cuda, cpu)
            enable_cuda_graphs: Whether to enable CUDA Graph optimization
        """        
        # Auto-detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Loading Higgs Audio 2 engine silently
        
        try:
            # Check if global cache invalidation occurred (models were unloaded)
            from utils.models.comfyui_model_wrapper import _global_cache_invalidation_flag
            should_force_reload = _global_cache_invalidation_flag > getattr(self, '_last_load_time', 0.0)
            if should_force_reload:
                print(f"🔄 Forcing model reload due to global cache invalidation")
                # Set flag to disable CUDA graphs for this reload cycle
                force_disable_cuda_graphs = True
            else:
                force_disable_cuda_graphs = False
            
            # Use unified model interface for ComfyUI integration
            engine = load_tts_model(
                engine_name="higgs_audio",
                model_name=model_path,
                device=device,
                model_path=self._get_smart_model_path(model_path),
                tokenizer_path=self._get_smart_tokenizer_path(tokenizer_path),
                enable_cuda_graphs=enable_cuda_graphs,  # Pass CUDA Graph setting
                force_reload=should_force_reload
            )
            
            # Update last load time
            import time
            self._last_load_time = time.time()
            
            self.engine = engine
            self.model_path = model_path
            self.tokenizer_path = tokenizer_path
            self.device = device
            
            # Disable CUDA graphs if this was loaded after cache invalidation (memory pressure)
            if force_disable_cuda_graphs and hasattr(self.engine, 'disable_cuda_graphs_for_memory_management'):
                self.engine.disable_cuda_graphs_for_memory_management()
            
            print(f"📦 Higgs Audio engine ready")
            
            # Show memory management warning only once per session
            if not HiggsAudioEngine._memory_warning_shown:
                print(f"⚠️  Memory Management Limitation: Higgs Audio uses CUDA graphs for performance")
                print(f"   CUDA graphs lock GPU memory and cannot be safely unloaded without crashes")
                print(f"   This model will stay in memory until ComfyUI restart")
                print(f"📝 For dynamic memory management, use ChatterBox or F5-TTS engines instead")
                HiggsAudioEngine._memory_warning_shown = True
            return
            
        except Exception as e:
            print(f"⚠️ Failed to load via unified interface: {e}")
            print(f"🔄 Falling back to direct loading...")
            # Fall back to direct loading if unified interface fails
            
            # Get smart paths that work with HiggsAudioServeEngine
            model_path_for_engine = self._get_smart_model_path(model_path)
            tokenizer_path_for_engine = self._get_smart_tokenizer_path(tokenizer_path)
            
            print(f"🔧 Using model: {model_path_for_engine}")
            print(f"🔧 Using tokenizer: {tokenizer_path_for_engine}")
            
            # Create engine with memory-efficient cache sizes for better VRAM compatibility
            # Reduced cache sizes to prevent OOM when other models are loaded
            engine = HiggsAudioServeEngine(
                model_name_or_path=model_path_for_engine,
                audio_tokenizer_name_or_path=tokenizer_path_for_engine,
                device=device,
                kv_cache_lengths=[1024, 2048, 4096]  # Smaller cache sizes to reduce VRAM usage
            )
            
            # Store engine (no manual caching - ComfyUI manages memory)
            self.engine = engine
            self.model_path = model_path
            self.tokenizer_path = tokenizer_path
            self.device = device
            
            print(f"📦 Higgs Audio engine ready")
            
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
                force_audio_gen: bool = False,
                ras_win_len: Optional[int] = 7,
                ras_max_num_repeat: int = 2,
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
                force_audio_gen=force_audio_gen,
                ras_win_len=ras_win_len,
                ras_max_num_repeat=ras_max_num_repeat,
                character=character,
                seed=seed  # Include seed in cache key so different seeds generate different outputs
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
                force_audio_gen=force_audio_gen,
                ras_win_len=ras_win_len,
                ras_max_num_repeat=ras_max_num_repeat,
                seed=seed if seed >= 0 else None
            )
            
            audio_segments.append(chunk_audio)
            total_tokens += chunk_tokens
        
        # Combine audio chunks with timing info
        from utils.audio.chunk_timing import ChunkTimingHelper
        combined_audio, chunk_info = ChunkTimingHelper.combine_audio_with_timing(
            audio_segments=audio_segments,
            combination_method="auto",
            silence_ms=silence_between_chunks_ms,
            crossfade_duration=0.1,
            sample_rate=24000,
            text_length=len(text),
            original_text=text,
            text_chunks=chunks,
            combiner_func=self._combine_audio_chunks
        )
        
        # Calculate duration
        duration = combined_audio['waveform'].size(-1) / combined_audio['sample_rate']
        
        # Cache the result if enabled
        if enable_cache and 'cache_key' in locals():
            self.cache.cache_audio(cache_key, combined_audio['waveform'], duration)
        
        # Generate enhanced info string with timing
        generation_time = time.time() - start_time
        base_info = (f"Generated {duration:.1f}s audio from {total_tokens} tokens "
                    f"using {len(chunks)} chunk(s) in {generation_time:.1f}s")
        
        enhanced_info = ChunkTimingHelper.enhance_generation_info(base_info, chunk_info)
        
        return combined_audio, enhanced_info
    
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
                                   force_audio_gen: bool = False,
                                   ras_win_len: Optional[int] = 7,
                                   ras_max_num_repeat: int = 2,
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
                force_audio_gen=force_audio_gen,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_max_num_repeat,
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
                             force_audio_gen: bool,
                             ras_win_len: Optional[int],
                             ras_max_num_repeat: int,
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
                force_audio_gen=force_audio_gen,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_max_num_repeat,
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
                # Duration info is included in the generation info returned to caller
                return chunk_audio, used_voice_info
            else:
                raise ValueError("Invalid audio output from Higgs Audio engine")
                
        except Exception as e:
            print(f"❌ Error during audio generation: {e}")
            raise e
    
    def _audio_to_base64(self, comfy_audio: Dict[str, Any]) -> str:
        """Convert ComfyUI audio format to base64 string, preserving device for later processing"""
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
        
        # Handle tensor dimensions and preserve device info
        if waveform.dim() == 3:
            audio_tensor = waveform[0, 0]
        elif waveform.dim() == 2:
            audio_tensor = waveform[0]
        else:
            audio_tensor = waveform
        
        # Convert to CPU for file operations (soundfile requires numpy)
        audio_np = audio_tensor.cpu().numpy()
        
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
                             text_chunks: List[str] = None,
                             return_info: bool = False,
                             sample_rate: int = None) -> Dict[str, Any]:
        """Combine multiple audio chunks using modular combination utility"""
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        print(f"🔗 Combining {len(audio_segments)} chunks using '{combination_method}' method")
        
        # Extract waveforms and sample rate
        waveforms = [seg["waveform"] for seg in audio_segments]
        sample_rate = audio_segments[0]["sample_rate"]
        
        # Use modular chunk combiner
        from utils.audio.chunk_combiner import ChunkCombiner
        result = ChunkCombiner.combine_chunks(
            audio_segments=waveforms,
            method=combination_method,
            silence_ms=silence_ms,
            crossfade_duration=crossfade_duration,
            sample_rate=sample_rate,
            text_length=text_length,
            original_text=original_text,
            text_chunks=text_chunks,
            return_info=return_info
        )
        
        if return_info:
            combined_waveform, chunk_info = result
            print(f"  ✅ Combined waveform shape: {combined_waveform.shape}")
            return {"waveform": combined_waveform, "sample_rate": sample_rate}, chunk_info
        else:
            combined_waveform = result
            print(f"  ✅ Combined waveform shape: {combined_waveform.shape}")
            return {"waveform": combined_waveform, "sample_rate": sample_rate}
    
    def _get_smart_model_path(self, model_path: str) -> str:
        """
        Get smart model path for HiggsAudioServeEngine:
        - Return verified local paths for complete local models
        - Download to TTS structure first, then return that path
        - Only fallback to repo IDs if download fails
        """
        # If already a valid local path, use it
        if os.path.exists(model_path) and os.path.isdir(model_path):
            if self._verify_model_completeness(model_path):
                return model_path
        
        # Check predefined models
        if model_path in HIGGS_AUDIO_MODELS:
            config = HIGGS_AUDIO_MODELS[model_path]
            repo_id = config["generation_repo"]
            
            # Check if we have complete local organized structure
            local_dir = os.path.join(self.downloader.base_path, model_path, "generation")
            if os.path.exists(local_dir) and self._verify_model_completeness(local_dir):
                # print(f"✅ Using verified local model: {local_dir}")
                return local_dir
            
            # Download to organized structure first using our downloader
            print(f"📥 Ensuring model is in TTS structure: {repo_id}")
            downloaded_path = self.downloader.download_model(repo_id)
            
            # If download succeeded and gave us a local path, use it
            if downloaded_path != repo_id and os.path.exists(downloaded_path):
                print(f"✅ Downloaded to TTS structure: {downloaded_path}")
                return downloaded_path
            
            # If download failed, raise error instead of fallback to repo ID
            raise RuntimeError(f"Failed to download model to TTS structure: {repo_id}")
        
        # For direct repo IDs, ensure download to organized structure
        if "/" in model_path:  # Looks like repo ID
            model_name = model_path.split('/')[-1]
            local_path = os.path.join(self.downloader.base_path, model_name)
            
            if os.path.exists(local_path) and self._verify_model_completeness(local_path):
                # print(f"✅ Using verified local model: {local_path}")
                return local_path
            
            # Download to organized structure first
            print(f"📥 Ensuring model is in TTS structure: {model_path}")
            downloaded_path = self.downloader.download_model(model_path)
            
            # If download succeeded and gave us a local path, use it
            if downloaded_path != model_path and os.path.exists(downloaded_path):
                print(f"✅ Downloaded to TTS structure: {downloaded_path}")
                return downloaded_path
            
            # If download failed, raise error instead of fallback to repo ID
            raise RuntimeError(f"Failed to download model to TTS structure: {model_path}")
        
        # Default: return as-is
        return model_path
    
    def _get_smart_tokenizer_path(self, tokenizer_path: str) -> str:
        """
        Get smart tokenizer path for HiggsAudioServeEngine following F5-TTS pattern:
        - Return verified local paths for complete local tokenizers
        - Return repo IDs otherwise (let HuggingFace handle cache/download)
        """
        # If already a valid local path, use it
        if os.path.exists(tokenizer_path) and os.path.isdir(tokenizer_path):
            if self._verify_tokenizer_completeness(tokenizer_path):
                return tokenizer_path
        
        # Check predefined models
        for model_name, config in HIGGS_AUDIO_MODELS.items():
            if config["tokenizer_repo"] == tokenizer_path:
                # Check if we have complete local organized structure
                local_dir = os.path.join(self.downloader.base_path, model_name, "tokenizer")
                if os.path.exists(local_dir) and self._verify_tokenizer_completeness(local_dir):
                    # print(f"✅ Using verified local tokenizer: {local_dir}")
                    return local_dir
                break
        
        # For direct repo IDs, check for local organized download
        if "/" in tokenizer_path:  # Looks like repo ID
            model_name = tokenizer_path.split('/')[-1]
            local_path = os.path.join(self.downloader.base_path, model_name)
            
            if os.path.exists(local_path) and self._verify_tokenizer_completeness(local_path):
                # print(f"✅ Using verified local tokenizer: {local_path}")
                return local_path
            
            # Download using unified downloader to follow TTS folder policy
            print(f"📥 Downloading tokenizer using unified downloader: {tokenizer_path}")
            downloaded_path = self.downloader.download_tokenizer(tokenizer_path)
            if downloaded_path:
                return downloaded_path
            else:
                raise RuntimeError(f"Failed to download tokenizer: {tokenizer_path}")

        # Default: try to download using unified downloader
        print(f"📥 Downloading tokenizer using unified downloader: {tokenizer_path}")
        downloaded_path = self.downloader.download_tokenizer(tokenizer_path)
        if downloaded_path:
            return downloaded_path
        else:
            raise RuntimeError(f"Failed to download tokenizer: {tokenizer_path}")
    
    def _verify_model_completeness(self, model_dir: str) -> bool:
        """Verify model directory has all essential files and they're not corrupted"""
        try:
            essential_files = ["config.json", "model.safetensors.index.json"]
            # Check files exist
            if not all(os.path.exists(os.path.join(model_dir, f)) for f in essential_files):
                return False
            
            # Quick corruption check - just verify config.json is valid JSON
            config_path = os.path.join(model_dir, "config.json")
            with open(config_path, 'r') as f:
                import json
                json.load(f)  # Will raise if corrupted
            
            return True
        except Exception:
            return False
    
    def _verify_tokenizer_completeness(self, tokenizer_dir: str) -> bool:
        """Verify tokenizer directory has all essential files"""
        try:
            essential_files = ["config.json", "model.pth"]
            return all(os.path.exists(os.path.join(tokenizer_dir, f)) for f in essential_files)
        except Exception:
            return False
    
    def cleanup(self):
        """Clean up resources (now handled by ComfyUI model management)"""
        if self.engine:
            # Models are now managed by ComfyUI's model management system
            # No manual cache cleanup needed - ComfyUI handles memory management
            self.engine = None
        
        # Run garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()