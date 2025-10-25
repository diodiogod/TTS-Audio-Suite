import asyncio
import base64
import torch
import numpy as np
from io import BytesIO
from dataclasses import dataclass
from typing import List, Optional, Union
from copy import deepcopy
from transformers import AutoTokenizer, AutoProcessor
from transformers.cache_utils import StaticCache, DynamicCache
from transformers.generation.streamers import BaseStreamer
from transformers.generation.stopping_criteria import StoppingCriteria
from dataclasses import asdict
from loguru import logger
import threading
import librosa
import torchaudio
from tqdm import tqdm


from ..dataset.chatml_dataset import ChatMLSample, ChatMLDatasetSample, prepare_chatml_sample
from ..model.higgs_audio import HiggsAudioModel
from ..model.higgs_audio.utils import revert_delay_pattern
from ..data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from ..audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer


@dataclass
class HiggsAudioStreamerDelta:
    """Represents a chunk of generated content, either text or audio tokens."""

    text: Optional[str] = None
    text_tokens: Optional[torch.Tensor] = None
    audio_tokens: Optional[torch.Tensor] = None
    finish_reason: Optional[str] = None


class AsyncHiggsAudioStreamer(BaseStreamer):
    """
    Async streamer that handles both text and audio token generation from Higgs-Audio model.
    Stores chunks in a queue to be consumed by downstream applications.

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenizer used to decode text tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt tokens in generation.
        timeout (`float`, *optional*):
            The timeout for the queue. If `None`, the queue will block indefinitely.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:
        ```python
        >>> from transformers import AutoTokenizer
        >>> from threading import Thread
        >>> import asyncio

        >>> tokenizer = AutoTokenizer.from_pretrained("path/to/higgs/tokenizer")
        >>> model = HiggsAudioModel.from_pretrained("path/to/higgs/model")
        >>> inputs = tokenizer(["Generate some text and audio:"], return_tensors="pt")

        >>> async def main():
        ...     streamer = AsyncHiggsAudioStreamer(tokenizer)
        ...     generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
        ...     thread = Thread(target=model.generate, kwargs=generation_kwargs)
        ...     thread.start()
        ...
        ...     async for delta in streamer:
        ...         if delta.text is not None:
        ...             print("Text:", delta.text)
        ...         if delta.audio_tokens is not None:
        ...             print("Audio tokens shape:", delta.audio_tokens.shape)
        >>> asyncio.run(main())
        ```
    """

    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        skip_prompt: bool = False,
        timeout: Optional[float] = None,
        audio_num_codebooks: int = 1,
        **decode_kwargs,
    ):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.timeout = timeout
        self.decode_kwargs = decode_kwargs
        self.audio_num_codebooks = audio_num_codebooks
        # Queue to store generated chunks
        self.queue = asyncio.Queue()
        self.stop_signal = None

        # Get running event loop
        self.loop = asyncio.get_running_loop()
        self.has_asyncio_timeout = hasattr(asyncio, "timeout")

        # State tracking
        self.next_tokens_are_prompt = True

    def put(self, value: torch.Tensor):
        """
        Receives tokens and processes them as either text or audio tokens.
        For text tokens, decodes and caches them until complete words are formed.
        For audio tokens, directly queues them.
        """
        if value.shape[0] > 1 and not self.next_tokens_are_prompt:
            # This is likely audio tokens (shape: [audio_num_codebooks])
            assert value.shape[0] == self.audio_num_codebooks, "Number of codebooks mismatch"
            delta = HiggsAudioStreamerDelta(audio_tokens=value)
            self.loop.call_soon_threadsafe(self.queue.put_nowait, delta)
            return

        # Skip prompt tokens if configured
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Process as text tokens
        if len(value.shape) > 1:
            value = value[0]

        text = self.tokenizer.decode(value, **self.decode_kwargs)
        delta = HiggsAudioStreamerDelta(text=text, text_tokens=value)
        self.loop.call_soon_threadsafe(self.queue.put_nowait, delta)

    def end(self):
        """Flushes any remaining text tokens and signals the end of generation."""
        self.next_tokens_are_prompt = True
        self.loop.call_soon_threadsafe(self.queue.put_nowait, self.stop_signal)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            if self.has_asyncio_timeout:
                async with asyncio.timeout(self.timeout):
                    value = await self.queue.get()
            else:
                value = await asyncio.wait_for(self.queue.get(), timeout=self.timeout)
        except asyncio.TimeoutError:
            raise TimeoutError()
        else:
            if value == self.stop_signal:
                raise StopAsyncIteration()
            else:
                return value


class TqdmProgressStreamer(BaseStreamer):
    """
    Progress bar streamer that shows token-by-token generation progress using tqdm.
    Displays progress like: "🗣️ Generating: 45%|████▌     | 234/512 tokens [00:12<00:15, 18.2it/s]"
    """
    
    def __init__(self, tokenizer: "AutoTokenizer", max_new_tokens: int, skip_prompt: bool = True):
        """
        Initialize the progress bar streamer.
        
        Args:
            tokenizer: The tokenizer used to decode tokens
            max_new_tokens: Maximum number of new tokens to generate
            skip_prompt: Whether to skip prompt tokens (default True)
        """
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.skip_prompt = skip_prompt
        self.generated_tokens = 0
        self.progress_bar = None
        self.next_tokens_are_prompt = True
        
    def put(self, value: torch.Tensor):
        """
        Process incoming tokens and update progress bar.
        
        Args:
            value: Tensor containing new token IDs
        """
        if self.skip_prompt and self.next_tokens_are_prompt:
            # Skip the first call which contains the prompt
            self.next_tokens_are_prompt = False
            return
        
        # Initialize progress bar on first non-prompt tokens
        if self.progress_bar is None:
            self.progress_bar = tqdm(
                total=self.max_new_tokens,
                desc="🗣️ Generating",
                unit="token",
                unit_scale=False,
                dynamic_ncols=True,
                leave=False  # Don't leave progress bar after completion
            )
        
        # Count new tokens generated
        if value.dim() > 1:
            new_tokens = value.shape[-1]
        else:
            new_tokens = 1
            
        self.generated_tokens += new_tokens
        
        # Update progress bar
        if self.progress_bar is not None:
            self.progress_bar.update(new_tokens)
    
    def end(self):
        """Clean up progress bar when generation ends."""
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None


class AsyncStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria that checks for stop signal from a threading event.

    Args:
        stop_signal (threading.Event): Event that will receive stop signals
    """

    def __init__(self, stop_signal: threading.Event):
        self.stop_signal = stop_signal

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.stop_signal.is_set():
            logger.info(f"Stop signal received. Can be caused by client disconnection.")
            return True
        return False


@dataclass
class HiggsAudioResponse:
    audio: Optional[np.ndarray] = None
    generated_audio_tokens: Optional[np.ndarray] = None
    sampling_rate: Optional[int] = None
    generated_text: str = ""
    generated_text_tokens: Optional[np.ndarray] = None
    usage: Optional[dict] = None


class HiggsAudioServeEngine:
    def __init__(
        self,
        model_name_or_path: str,
        audio_tokenizer_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: Union[torch.dtype, str] = "auto",
        kv_cache_lengths: List[int] = [2048, 8192, 16384],  # Larger cache sizes for newer transformers
        enable_cuda_graphs: bool = True,  # NEW: Control CUDA Graph optimization
    ):
        """
        Initialize the HiggsAudioServeEngine, a serving wrapper for the HiggsAudioModel.
        The model, tokenizer, and audio tokenizer will be downloaded from the Hugging Face Hub if they are not local.

        Args:
            model_name_or_path (str):
                The name or path of the model to load.
            audio_tokenizer_name_or_path (str):
                The name or path of the audio tokenizer to load.
            tokenizer_name_or_path (str):
                The name or path of the tokenizer to load.
            device (str):
                The device to use for the model.
            kv_cache_lengths (List[int]):
                The lengths of the KV caches to use for the model. Used for cuda graph capture when device is cuda.
            torch_dtype (Union[torch.dtype, str]):
                The dtype to use for the model.
            enable_cuda_graphs (bool):
                Whether to enable CUDA Graph optimization. If False, uses DynamicCache for memory safety.
        """
        self.device = device
        self.model_name_or_path = model_name_or_path
        self.audio_tokenizer_name_or_path = audio_tokenizer_name_or_path
        self.torch_dtype = torch_dtype
        
        # Store CUDA Graph setting IMMEDIATELY for cache creation
        self._cuda_graphs_enabled = enable_cuda_graphs
        self._cache_type = "StaticCache" if enable_cuda_graphs else "DynamicCache"
        
        print(f"🔧 HiggsAudioServeEngine: CUDA Graphs {'ENABLED' if enable_cuda_graphs else 'DISABLED'}")
        print(f"🔧 Cache Type: {self._cache_type}")

        # Initialize model and tokenizer
        # Load with attention implementation fix for newer transformers
        self.model = HiggsAudioModel.from_pretrained(
            model_name_or_path,
            dtype=torch_dtype,  # Changed from torch_dtype (deprecated in transformers 4.57.1+)
            attn_implementation="eager"  # Force eager attention for compatibility
        ).to(device)
        
        # Fix attention implementation for transformers compatibility
        self._fix_attention_implementation()
        
        # logger.info(f"Loaded model from {model_name_or_path}, dtype: {self.model.dtype}")

        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path
        # logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
        try:
            # First try auto-detection
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        except (KeyError, OSError) as e:
            # Fallback: explicitly use LlamaTokenizer for Higgs Audio models
            logger.info(f"Auto-detection failed ({e}), using LlamaTokenizer explicitly")
            from transformers import LlamaTokenizer
            self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name_or_path)

        # logger.info(f"Initializing Higgs Audio Tokenizer")
        self.audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_name_or_path, device=device)

        self.audio_num_codebooks = self.model.config.audio_num_codebooks
        self.audio_codebook_size = self.model.config.audio_codebook_size
        self.audio_tokenizer_tps = self.audio_tokenizer.tps
        self.samples_per_token = int(self.audio_tokenizer.sampling_rate // self.audio_tokenizer_tps)
        self.hamming_window_len = 2 * self.audio_num_codebooks * self.samples_per_token
        # Set the audio special tokens
        self.model.set_audio_special_tokens(self.tokenizer)

        # Prepare KV caches for different lengths
        cache_config = deepcopy(self.model.config.text_config)
        cache_config.num_hidden_layers = self.model.config.text_config.num_hidden_layers
        if self.model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self.model.config.audio_dual_ffn_layers)
            
        # Create cache type based on CUDA Graph setting
        if self._cuda_graphs_enabled:
            # Use StaticCache for CUDA Graph optimization (high performance)
            self.kv_caches = {
                length: StaticCache(
                    config=cache_config,
                    max_batch_size=1,
                    max_cache_len=length,
                    device=self.model.device,
                    dtype=self.model.dtype,
                )
                for length in sorted(kv_cache_lengths)
            }
            print(f"🔥 Created StaticCache buckets for CUDA Graph optimization")
        else:
            # Use DynamicCache for memory safety (no CUDA Graph capture)
            def safe_create_dynamic_cache():
                """Create DynamicCache with comprehensive compatibility handling"""
                try:
                    return DynamicCache()
                except (AttributeError, TypeError) as e:
                    # Fallback for transformers versions with property issues
                    logger.warning(f"DynamicCache() constructor failed: {e}, attempting manual construction...")
                    try:
                        cache = object.__new__(DynamicCache)
                        # Initialize all expected internal attributes
                        object.__setattr__(cache, '_key_cache', [])
                        object.__setattr__(cache, '_value_cache', [])
                        if hasattr(DynamicCache, '_seen_tokens'):
                            object.__setattr__(cache, '_seen_tokens', 0)
                        # Try to mark it as properly initialized
                        if hasattr(cache, '__dict__'):
                            cache.__dict__['_is_initialized'] = True
                        return cache
                    except Exception as fallback_error:
                        logger.error(f"Manual DynamicCache construction failed: {fallback_error}")
                        raise RuntimeError(f"Cannot create DynamicCache: {e}") from fallback_error

            self.kv_caches = {
                length: safe_create_dynamic_cache()
                for length in sorted(kv_cache_lengths)
            }
            print(f"🛡️ Created DynamicCache buckets for memory safety")

        if self.model.config.encode_whisper_embed:
            logger.info(f"Loading whisper processor")
            whisper_processor = AutoProcessor.from_pretrained(
                "openai/whisper-large-v3-turbo",
                trust_remote=True,
                device=self.device,
            )
        else:
            whisper_processor = None

        # Reuse collator to prepare inference samples
        self.collator = HiggsAudioSampleCollator(
            whisper_processor=whisper_processor,
            encode_whisper_embed=self.model.config.encode_whisper_embed,
            audio_in_token_id=self.model.config.audio_in_token_idx,
            audio_out_token_id=self.model.config.audio_out_token_idx,
            audio_stream_bos_id=self.model.config.audio_stream_bos_id,
            audio_stream_eos_id=self.model.config.audio_stream_eos_id,
            pad_token_id=self.model.config.pad_token_id,
            return_audio_in_tokens=False,
            use_delay_pattern=self.model.config.use_delay_pattern,
            audio_num_codebooks=self.model.config.audio_num_codebooks,
            round_to=1,
        )

        # Defer CUDA graph creation until first inference to prevent corruption during model loading
        self.device = device
        self.cuda_graphs_initialized = False
        self.enable_cuda_graphs = (device == "cuda")  # Normal behavior by default
        
        if device == "cuda":
            # logger.info("📝 CUDA graph capture deferred until first inference (prevents memory corruption)")
            pass
        else:
            # logger.info("CUDA graph capture skipped (not using CUDA device)")
            pass
    
    def _ensure_cuda_graphs(self):
        """Initialize CUDA graphs on first use if needed"""
        if self.device == "cuda" and self.enable_cuda_graphs and not self.cuda_graphs_initialized:
            try:
                # logger.info(f"🚀 Initializing CUDA graphs on first inference")
                self.model.capture_model(self.kv_caches.values())
                self.cuda_graphs_initialized = True
                # logger.info("✅ CUDA graph capture successful")
            except Exception as e:
                logger.warning(f"⚠️ CUDA graph capture failed: {e}")
                logger.info("Continuing without CUDA graphs (performance may be slower)")
                self.enable_cuda_graphs = False

    def _fix_attention_implementation(self):
        """Fix attention implementation for all LLaMA layers to avoid None errors"""
        # logger.info("Fixing attention implementation for transformers compatibility")
        
        # Fix the main config
        if hasattr(self.model.config, 'text_config'):
            if hasattr(self.model.config.text_config, '_attn_implementation'):
                if self.model.config.text_config._attn_implementation is None:
                    self.model.config.text_config._attn_implementation = "eager"
        
        # Fix all attention layers directly
        if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'model'):
            llm_model = self.model.language_model.model
            if hasattr(llm_model, 'layers'):
                for layer in llm_model.layers:
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'config'):
                        if hasattr(layer.self_attn.config, '_attn_implementation'):
                            if layer.self_attn.config._attn_implementation is None:
                                layer.self_attn.config._attn_implementation = "eager"

    def disable_cuda_graphs_permanently(self):
        """
        Permanently disable CUDA graphs for models that have been corrupted by CPU moves.
        This sacrifices performance but restores correct generation behavior.
        """
        try:
            logger.info("🚫 Permanently disabling CUDA graphs for resurrected model")
            
            # Disable CUDA graph functionality completely
            self.enable_cuda_graphs = False
            self.cuda_graphs_initialized = False
            
            # Clear any existing CUDA graphs that might be corrupted
            if hasattr(self.model, 'decode_graph_runners'):
                logger.info("🗑️ Clearing corrupted CUDA graph runners")
                for key in list(self.model.decode_graph_runners.keys()):
                    del self.model.decode_graph_runners[key]
                self.model.decode_graph_runners.clear()
            
            # Fix tensor tracking for the audio tokenizer components
            if hasattr(self.audio_tokenizer, 'semantic_model'):
                semantic_model = self.audio_tokenizer.semantic_model
                
                # Move to CPU and back to GPU to refresh tensor tracking
                original_device = next(semantic_model.parameters()).device
                semantic_model.cpu()
                semantic_model.to(original_device)
                semantic_model.eval()
                
                logger.info("✅ Fixed semantic model tensor tracking")
            
            # Fix encoder/decoder tensor tracking
            if hasattr(self.audio_tokenizer, 'encoder'):
                encoder = self.audio_tokenizer.encoder
                original_device = next(encoder.parameters()).device
                encoder.cpu()
                encoder.to(original_device)
                encoder.eval()
                
            if hasattr(self.audio_tokenizer, 'decoder_2'):
                decoder = self.audio_tokenizer.decoder_2
                original_device = next(decoder.parameters()).device
                decoder.cpu()
                decoder.to(original_device)
                decoder.eval()
            
            logger.info("✅ CUDA graphs disabled permanently - model should work correctly but slower")
            
        except Exception as e:
            logger.error(f"❌ Failed to disable CUDA graphs: {e}")
            logger.info("Continuing with existing state - generation may be corrupted")

    def disable_cuda_graphs_for_memory_management(self):
        """
        Disable CUDA graphs specifically for memory management scenarios.
        Called when cache invalidation indicates models will be unloaded/reloaded.
        """
        logger.info("🚫 Disabling CUDA graphs for memory management cycle")
        self.enable_cuda_graphs = False
        self.cuda_graphs_initialized = False

    def _prepare_inputs(self, chat_ml_sample: ChatMLSample, force_audio_gen: bool = False):
        input_tokens, _, audio_contents, _ = prepare_chatml_sample(
            chat_ml_sample,
            self.tokenizer,
        )

        postfix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if force_audio_gen:
            postfix += "<|audio_out_bos|>"
        postfix = self.tokenizer.encode(postfix, add_special_tokens=False)
        input_tokens.extend(postfix)

        # Configure the audio inputs
        audio_ids_l = []
        for audio_content in audio_contents:
            if audio_content.audio_url not in ["placeholder", ""]:
                # Use torchaudio instead of librosa for Python 3.13 compatibility
                raw_audio_tensor, sample_rate = torchaudio.load(audio_content.audio_url)
                # Resample if needed
                if sample_rate != self.audio_tokenizer.sampling_rate:
                    resampler = torchaudio.transforms.Resample(sample_rate, self.audio_tokenizer.sampling_rate)
                    raw_audio_tensor = resampler(raw_audio_tensor)
                # Convert to numpy and squeeze to 1D if needed
                raw_audio = raw_audio_tensor.squeeze().numpy()
            elif audio_content.raw_audio is not None:
                # Use torchaudio instead of librosa for Python 3.13 compatibility
                raw_audio_tensor, sample_rate = torchaudio.load(BytesIO(base64.b64decode(audio_content.raw_audio)))
                # Resample if needed
                if sample_rate != self.audio_tokenizer.sampling_rate:
                    resampler = torchaudio.transforms.Resample(sample_rate, self.audio_tokenizer.sampling_rate)
                    raw_audio_tensor = resampler(raw_audio_tensor)
                # Convert to numpy and squeeze to 1D if needed
                raw_audio = raw_audio_tensor.squeeze().numpy()
            else:
                raw_audio = None

            if raw_audio is not None:
                # Encode the audio (tokenizer requires numpy on CPU)
                audio_ids = self.audio_tokenizer.encode(raw_audio, self.audio_tokenizer.sampling_rate)
                # Move encoded audio_ids to the same device as the model instead of forcing CPU
                audio_ids_l.append(audio_ids.squeeze(0).to(self.device))

        if len(audio_ids_l) > 0:
            audio_ids_start = torch.tensor(
                np.cumsum(np.array([0] + [audio_ids.shape[1] for audio_ids in audio_ids_l])),
                dtype=torch.long,
                device=self.device,
            )[0:-1]
            audio_ids_concat = torch.cat(audio_ids_l, dim=1)
        else:
            audio_ids_start = None
            audio_ids_concat = None

        sample = ChatMLDatasetSample(
            input_ids=torch.LongTensor(input_tokens),
            label_ids=None,
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=None,
            audio_waveforms_start=None,
            audio_sample_rate=None,
            audio_speaker_indices=None,
        )
        data = self.collator([sample])
        inputs = asdict(data)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)

        return inputs

    def _prepare_kv_caches(self):
        # Get current model device (fresh lookup to avoid stale cache)
        model_device = next(self.model.parameters()).device
        
        # Check if we need to recreate caches due to device change or forced cleanup
        cache_device_mismatch = False
        force_cache_recreation = getattr(self, '_force_cache_recreation', False)
        
        if self.kv_caches and not force_cache_recreation:
            # Check if any cache is on wrong device
            for kv_cache in self.kv_caches.values():
                if hasattr(kv_cache, '__len__') and len(kv_cache) > 0:
                    try:
                        cache_tuple = kv_cache[0]  # Get first layer's cache
                        if cache_tuple is not None and len(cache_tuple) >= 2:
                            key_cache, _ = cache_tuple
                            cache_device = key_cache.device if key_cache is not None else model_device
                        else:
                            cache_device = model_device
                        if cache_device != model_device:
                            cache_device_mismatch = True
                            break
                    except (AttributeError, IndexError):
                        cache_device_mismatch = True
                        break
        
        # Recreate caches if device mismatch detected or forced recreation requested
        if cache_device_mismatch or force_cache_recreation:
            print(f"🔄 Recreating KV caches for device: {model_device}")
            # Get original cache lengths
            cache_lengths = list(self.kv_caches.keys()) if self.kv_caches else [1024, 2048, 4096]
            
            # Recreate caches with correct device
            from copy import deepcopy
            cache_config = deepcopy(self.model.config.text_config)
            cache_config.num_hidden_layers = self.model.config.text_config.num_hidden_layers
            if self.model.config.audio_dual_ffn_layers:
                cache_config.num_hidden_layers += len(self.model.config.audio_dual_ffn_layers)
                
            # Choose cache type based on CUDA Graph setting
            cuda_graphs_enabled = getattr(self, '_cuda_graphs_enabled', True)
            print(f"🔍 Debug: CUDA Graphs enabled = {cuda_graphs_enabled}")
            
            if cuda_graphs_enabled:
                # Use StaticCache for CUDA Graph optimization
                self.kv_caches = {
                    length: StaticCache(
                        config=cache_config,
                        max_batch_size=1,
                        max_cache_len=length,
                        device=model_device,
                        dtype=self.model.dtype,
                    )
                    for length in sorted(cache_lengths)
                }
                print(f"  🔥 Created StaticCache buckets for CUDA Graph optimization")
            else:
                # Use DynamicCache for memory safety (no pre-allocation)
                def safe_create_dynamic_cache():
                    """Create DynamicCache with comprehensive compatibility handling"""
                    try:
                        return DynamicCache()
                    except (AttributeError, TypeError) as e:
                        # Fallback for transformers versions with property issues
                        logger.warning(f"DynamicCache() constructor failed in _prepare_kv_caches: {e}, attempting manual construction...")
                        try:
                            cache = object.__new__(DynamicCache)
                            # Initialize all expected internal attributes
                            object.__setattr__(cache, '_key_cache', [])
                            object.__setattr__(cache, '_value_cache', [])
                            if hasattr(DynamicCache, '_seen_tokens'):
                                object.__setattr__(cache, '_seen_tokens', 0)
                            # Try to mark it as properly initialized
                            if hasattr(cache, '__dict__'):
                                cache.__dict__['_is_initialized'] = True
                            return cache
                        except Exception as fallback_error:
                            logger.error(f"Manual DynamicCache construction in _prepare_kv_caches failed: {fallback_error}")
                            raise RuntimeError(f"Cannot create DynamicCache: {e}") from fallback_error

                self.kv_caches = {
                    length: safe_create_dynamic_cache()
                    for length in sorted(cache_lengths)
                }
                print(f"  🛡️ Created DynamicCache buckets for memory safety")
            # Mark that we've created caches for this device to avoid recreation
            self._cache_device = model_device
            
            # Clear force recreation flag if it was set
            if force_cache_recreation:
                self._force_cache_recreation = False
                print(f"  ✅ Force cache recreation completed and flag cleared")
        else:
            # Store current device for future checks
            if not hasattr(self, '_cache_device'):
                self._cache_device = model_device
        
        # Reset all caches using explicit type checking to avoid ambiguity
        # (both StaticCache and DynamicCache have reset() method, so we use isinstance)
        for kv_cache in self.kv_caches.values():
            if isinstance(kv_cache, StaticCache):
                # StaticCache has built-in reset method
                try:
                    kv_cache.reset()
                except Exception as e:
                    print(f"  ⚠️ StaticCache reset error (may be expected): {type(e).__name__}")
            elif isinstance(kv_cache, DynamicCache):
                # DynamicCache needs manual clearing - use crop() API
                try:
                    kv_cache.crop(0)  # Clear all cached states
                    print(f"  🧹 Cleared DynamicCache state for fresh generation")
                except Exception as e:
                    print(f"  ⚠️ DynamicCache crop error: {type(e).__name__}: {e}")
            else:
                # Unknown cache type - log warning
                print(f"  ⚠️ Unknown cache type: {type(kv_cache).__name__}")

    def generate(
        self,
        chat_ml_sample: ChatMLSample,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        stop_strings: Optional[List[str]] = None,
        force_audio_gen: bool = False,
        ras_win_len: Optional[int] = 7,
        ras_win_max_num_repeat: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Generate audio from a chatml sample.
        Args:
            chat_ml_sample: A chatml sample.
            max_new_tokens: The maximum number of new tokens to generate.
            temperature: The temperature to use for the generation.
            top_p: The top p to use for the generation.
            stop_strings: A list of strings to stop the generation.
            force_audio_gen: Whether to force audio generation. This ensures the model generates audio tokens rather than text tokens.
            ras_win_len: The length of the RAS window. We use 7 by default. You can disable it by setting it to None or <=0.
            ras_win_max_num_repeat: The maximum number of times to repeat the RAS window.
        Returns:
            A dictionary with the following keys:
                audio: The generated audio.
                sampling_rate: The sampling rate of the generated audio.
        """
        # Initialize CUDA graphs on first inference (deferred from __init__)
        self._ensure_cuda_graphs()
        
        # Default stop strings
        if stop_strings is None:
            stop_strings = ["<|end_of_text|>", "<|eot_id|>"]
        if ras_win_len is not None and ras_win_len <= 0:
            ras_win_len = None

        with torch.no_grad():
            inputs = self._prepare_inputs(chat_ml_sample, force_audio_gen=force_audio_gen)
            prompt_token_ids = inputs["input_ids"][0].cpu().numpy()

            self._prepare_kv_caches()

            # Create progress bar streamer for token generation
            progress_streamer = TqdmProgressStreamer(
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                skip_prompt=True
            )

            # Minimal device enforcement - only move inputs if needed
            model_device = next(self.model.parameters()).device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.device != model_device:
                    inputs[k] = v.to(model_device)
            
            # Restore StaticCache for 50 it/s performance (with proper device handling)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stop_strings=stop_strings,
                tokenizer=self.tokenizer,
                do_sample=False if temperature == 0.0 else True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.kv_caches,  # Restore high-performance StaticCache
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                seed=seed,
                streamer=progress_streamer,  # Add progress bar streamer
            )
            
            # Clean up progress bar
            progress_streamer.end()

            if len(outputs[1]) > 0:
                wv_list = []
                for output_audio in outputs[1]:
                    vq_code = revert_delay_pattern(output_audio).clip(0, self.audio_codebook_size - 1)[:, 1:-1]
                    wv_numpy = self.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
                    wv_list.append(wv_numpy)
                wv_numpy = np.concatenate(wv_list)
            else:
                wv_numpy = None

            # We only support one request at a time now
            generated_text_tokens = outputs[0][0].cpu().numpy()[len(prompt_token_ids) :]
            generated_text = self.tokenizer.decode(generated_text_tokens)
            generated_audio_tokens = outputs[1][0].cpu().numpy()
            return HiggsAudioResponse(
                audio=wv_numpy,
                generated_audio_tokens=generated_audio_tokens,
                sampling_rate=self.audio_tokenizer.sampling_rate,
                generated_text=generated_text,
                generated_text_tokens=generated_text_tokens,
                usage={
                    "prompt_tokens": prompt_token_ids.shape[0],
                    "completion_tokens": generated_text_tokens.shape[0] + generated_audio_tokens.shape[1],
                    "total_tokens": (
                        prompt_token_ids.shape[0] + generated_text_tokens.shape[0] + generated_audio_tokens.shape[1]
                    ),
                    "cached_tokens": 0,
                },
            )
