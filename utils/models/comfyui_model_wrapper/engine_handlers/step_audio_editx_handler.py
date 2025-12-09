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

        try:
            model_info = f"{wrapper.model_info.model_type} model ({wrapper.model_info.engine})"

            # Check if this is a bitsandbytes quantized model
            # Step Audio EditX wraps the actual TTS engine, so check both wrapper and inner model
            is_quantized = False

            # Check wrapper level
            if hasattr(model, 'quantization_method') or hasattr(model, 'quantization'):
                is_quantized = True

            # Check inner TTS engine if it exists
            if hasattr(model, '_tts_engine') and model._tts_engine is not None:
                tts_engine = model._tts_engine
                if hasattr(tts_engine, 'modules'):
                    # Check if any module is quantized
                    try:
                        is_quantized = is_quantized or any(
                            'Int8' in str(type(m).__name__) or 'Int4' in str(type(m).__name__)
                            for m in tts_engine.modules()
                        )
                    except Exception:
                        pass

            print(f"DEBUG: Quantization check for {model_info} - is_quantized={is_quantized}")

            if is_quantized:
                # Bitsandbytes quantized models can't use .to()
                # For int8/int4, we need to delete ALL references and force aggressive cleanup
                print(f"🔄 Unloading quantized {model_info} (bitsandbytes int8/int4)...")

                # Delete inner TTS engine first if it exists
                if hasattr(model, '_tts_engine') and model._tts_engine is not None:
                    del model._tts_engine
                    model._tts_engine = None

                # Delete tokenizer if it exists
                if hasattr(model, '_tokenizer') and model._tokenizer is not None:
                    del model._tokenizer
                    model._tokenizer = None

                # Delete the wrapper
                freed_memory = wrapper._memory_size
                del model
                wrapper._model_ref = None
                wrapper.current_device = device
                wrapper._is_loaded_on_gpu = False

                # Mark as invalid for reuse - quantized models can't be moved back to GPU
                wrapper._is_valid_for_reuse = False
                print(f"🚫 Marked quantized model as invalid for reuse (must reload from scratch)")
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
