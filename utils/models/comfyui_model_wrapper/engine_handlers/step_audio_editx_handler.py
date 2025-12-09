"""
Step Audio EditX engine handler with bitsandbytes int8/int4 support
"""

import torch
import gc
from typing import Optional, TYPE_CHECKING

from .base_handler import BaseEngineHandler

if TYPE_CHECKING:
    from ..base_wrapper import ComfyUIModelWrapper


class StepAudioEditXHandler(BaseEngineHandler):
    """
    Handler for Step Audio EditX engine with bitsandbytes quantization support.

    Bitsandbytes int8/int4 models can't use .to() for device movement.
    This handler detects quantized models and properly deletes them instead.
    """

    def partially_unload(self, wrapper: 'ComfyUIModelWrapper', device: str, memory_to_free: int) -> int:
        """
        Unload Step Audio EditX model, handling bitsandbytes int8/int4 models specially.
        """
        model = wrapper._model_ref() if wrapper._model_ref else None
        if model is None:
            return 0

        freed_memory = 0

        # Measure actual VRAM before unloading
        vram_before = 0
        if torch.cuda.is_available():
            try:
                vram_before = torch.cuda.memory_allocated()
            except Exception:
                pass

        try:
            model_info = f"{wrapper.model_info.model_type} model ({wrapper.model_info.engine})"

            # Check if this is a bitsandbytes quantized model
            # After unified_model_interface returns raw StepAudioTTS, check its LLM
            is_quantized = False

            # Check wrapper level (for legacy wrapped models)
            if hasattr(model, 'quantization_method') or hasattr(model, 'quantization'):
                is_quantized = True
                print(f"DEBUG: Found quantization attribute on model")

            # Check if model IS the TTS engine (raw StepAudioTTS from factory)
            if hasattr(model, 'llm') and hasattr(model.llm, 'modules'):
                try:
                    # Check for bitsandbytes quantized layers
                    quantized_modules = [m for m in model.llm.modules() if 'Int8' in str(type(m).__name__) or 'Int4' in str(type(m).__name__)]
                    if quantized_modules:
                        is_quantized = True
                        print(f"DEBUG: Found {len(quantized_modules)} quantized modules in model.llm")

                    # Also check for Linear8bitLt or Linear4bit classes from bitsandbytes
                    try:
                        import bitsandbytes as bnb
                        has_8bit = any(isinstance(m, bnb.nn.Linear8bitLt) for m in model.llm.modules())
                        has_4bit = any(hasattr(bnb.nn, 'Linear4bit') and isinstance(m, bnb.nn.Linear4bit) for m in model.llm.modules())
                        if has_8bit or has_4bit:
                            is_quantized = True
                            print(f"DEBUG: Found bitsandbytes quantization - 8bit={has_8bit}, 4bit={has_4bit}")
                    except ImportError:
                        pass
                except Exception as e:
                    print(f"DEBUG: Error checking llm modules: {e}")

            # Check inner TTS engine if it exists (for wrapped models)
            if hasattr(model, '_tts_engine') and model._tts_engine is not None:
                tts_engine = model._tts_engine
                if hasattr(tts_engine, 'llm') and hasattr(tts_engine.llm, 'modules'):
                    try:
                        is_quantized = is_quantized or any(
                            'Int8' in str(type(m).__name__) or 'Int4' in str(type(m).__name__)
                            for m in tts_engine.llm.modules()
                        )
                        if is_quantized:
                            print(f"DEBUG: Found quantized modules in _tts_engine.llm")
                    except Exception:
                        pass

            print(f"DEBUG: Quantization check for {model_info} - is_quantized={is_quantized}")

            if is_quantized:
                # Bitsandbytes quantized models can't use .to()
                # For int8/int4, we need to delete ALL references and force aggressive cleanup
                print(f"🔄 Unloading quantized {model_info} (bitsandbytes int8/int4)...")

                # Handle raw StepAudioTTS (has llm, cosy_model, audio_tokenizer)
                if hasattr(model, 'llm'):
                    if hasattr(model, 'llm') and model.llm is not None:
                        del model.llm
                        model.llm = None
                    if hasattr(model, 'cosy_model') and model.cosy_model is not None:
                        del model.cosy_model
                        model.cosy_model = None
                    if hasattr(model, 'audio_tokenizer') and model.audio_tokenizer is not None:
                        del model.audio_tokenizer
                        model.audio_tokenizer = None

                # Handle wrapped StepAudioEditXEngine (has _tts_engine, _tokenizer)
                if hasattr(model, '_tts_engine') and model._tts_engine is not None:
                    del model._tts_engine
                    model._tts_engine = None
                if hasattr(model, '_tokenizer') and model._tokenizer is not None:
                    del model._tokenizer
                    model._tokenizer = None

                # Delete the main reference
                del model
                wrapper._model_ref = None
                wrapper.current_device = device
                wrapper._is_loaded_on_gpu = False

                print(f"🗑️ Deleted quantized model from VRAM")

                # Measure actual VRAM freed after cleanup
                if torch.cuda.is_available():
                    try:
                        # Force cleanup first
                        gc.collect()
                        gc.collect()
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                        vram_after = torch.cuda.memory_allocated()
                        freed_memory = vram_before - vram_after
                        print(f"📊 VRAM: {vram_before // 1024 // 1024}MB → {vram_after // 1024 // 1024}MB (freed {freed_memory // 1024 // 1024}MB)")
                    except Exception as e:
                        freed_memory = wrapper._memory_size  # Fallback to estimate
                else:
                    freed_memory = wrapper._memory_size
            else:
                # Regular model - use standard .to() method
                if hasattr(model, 'to'):
                    try:
                        model.to(device)
                        freed_memory = wrapper._memory_size
                        wrapper.current_device = device
                        wrapper._is_loaded_on_gpu = False
                        print(f"🔄 Moved {model_info} to {device}, freed {freed_memory // 1024 // 1024}MB")
                    except Exception as e:
                        print(f"⚠️ Failed to move {model_info} to {device}: {e}")
                        freed_memory = wrapper._memory_size
                        wrapper.current_device = device
                        wrapper._is_loaded_on_gpu = False

        except Exception as e:
            print(f"⚠️ Error unloading {wrapper.model_info.engine} model: {e}")
            return 0

        # Force aggressive garbage collection and CUDA cache cleanup
        if freed_memory > 0:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception as e:
                    print(f"⚠️ CUDA synchronize warning (safe to ignore): {e}")

            # Multiple GC passes for bitsandbytes cleanup
            try:
                import gc
                gc.collect()
                gc.collect()
                gc.collect()
            except Exception as gc_error:
                print(f"⚠️ Garbage collection failed (safe to ignore): {gc_error}")

            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"⚠️ CUDA cache clear warning (safe to ignore): {e}")

        return freed_memory

    def model_unload(self, wrapper: 'ComfyUIModelWrapper', memory_to_free: Optional[int], unpatch_weights: bool) -> bool:
        """Full unload for Step Audio EditX"""
        if memory_to_free is not None and memory_to_free < wrapper.loaded_size():
            # Try partial unload first
            freed = self.partially_unload(wrapper, 'cpu', memory_to_free)
            success = freed >= memory_to_free
            print(f"{'✅' if success else '❌'} Partial unload: freed {freed // 1024 // 1024}MB (requested {memory_to_free // 1024 // 1024}MB)")
            return success

        # Full unload
        freed = self.partially_unload(wrapper, 'cpu', wrapper._memory_size)
        success = freed > 0
        print(f"{'✅' if success else '❌'} Full unload: freed {freed // 1024 // 1024}MB")
        return success
