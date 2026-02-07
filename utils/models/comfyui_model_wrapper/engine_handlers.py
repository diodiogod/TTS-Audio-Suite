"""
Engine-specific handlers for model unloading and device management.

Different TTS engines have different architectures and device management requirements.
This module provides specialized handlers for each engine type.
"""

import torch
import gc


class BaseEngineHandler:
    """Base handler with default implementation for most engines"""

    def partially_unload(self, wrapper, device: str, memory_to_free: int) -> int:
        """
        Partially unload the model to free memory.

        Args:
            wrapper: ComfyUIModelWrapper instance
            device: Target device (usually 'cpu')
            memory_to_free: Amount of memory to free in bytes

        Returns:
            Amount of memory actually freed in bytes
        """
        if not wrapper._is_loaded_on_gpu:
            print(f"⚠️ Skipping unload: model already marked as not on GPU")
            return 0

        # Get the actual model
        model = wrapper._model_ref() if wrapper._model_ref else None
        if model is None:
            print(f"⚠️ Model reference is None, cannot unload")
            return 0

        # Move model to CPU
        try:
            if hasattr(model, 'to'):
                model.to(device)
                print(f"🔄 Moved {wrapper.model_info.model_type} model ({wrapper.model_info.engine}) to {device}, freed {wrapper._memory_size // 1024 // 1024}MB")

            # Update wrapper state
            wrapper.current_device = device
            wrapper._is_loaded_on_gpu = False

            # Force CUDA cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            return wrapper._memory_size
        except Exception as e:
            print(f"⚠️ Error moving model to {device}: {e}")
            return 0

    def model_unload(self, wrapper, memory_to_free=None, unpatch_weights=True) -> bool:
        """
        Fully unload the model from GPU memory.

        Args:
            wrapper: ComfyUIModelWrapper instance
            memory_to_free: Amount of memory to free (ignored for full unload)
            unpatch_weights: Whether to unpatch weights (TTS models don't use this)

        Returns:
            True if model was unloaded, False otherwise
        """
        print(f"🔄 TTS Model unload requested: {wrapper.model_info.engine} {wrapper.model_info.model_type}")

        # Use partially_unload to do the actual work
        freed = self.partially_unload(wrapper, 'cpu', wrapper._memory_size)
        return freed > 0


class StepAudioEditXHandler(BaseEngineHandler):
    """Specialized handler for Step Audio EditX with bitsandbytes int8 support"""

    def partially_unload(self, wrapper, device: str, memory_to_free: int) -> int:
        """
        Unload Step Audio EditX model, handling bitsandbytes int8 models specially.

        Bitsandbytes int8 models can't use .to() for device movement.
        We need to delete the model and let garbage collection handle it.
        """
        if not wrapper._is_loaded_on_gpu:
            print(f"⚠️ Skipping unload: model already marked as not on GPU")
            return 0

        # Get the actual model
        model = wrapper._model_ref() if wrapper._model_ref else None
        if model is None:
            print(f"⚠️ Model reference is None, cannot unload")
            return 0

        try:
            # For bitsandbytes int8 models, .to() doesn't work
            # Instead, delete the model and force garbage collection
            model_info = f"{wrapper.model_info.model_type} model ({wrapper.model_info.engine})"

            # Check if this is a bitsandbytes quantized model
            is_quantized = hasattr(model, 'quantization_method') or any(
                'Int8' in str(type(m).__name__) for m in model.modules() if hasattr(model, 'modules')
            )

            # print(f"DEBUG: Quantization check for {model_info} - is_quantized={is_quantized}, has_quantization_method={hasattr(model, 'quantization_method')}")

            if is_quantized:
                print(f"🔄 Unloading quantized {model_info} (bitsandbytes int8)...")
                # Delete model reference and let garbage collection handle it
                del model
                wrapper._model_ref = None
            else:
                # Regular model - use standard .to() method
                if hasattr(model, 'to'):
                    model.to(device)
                    print(f"🔄 Moved {model_info} to {device}")

            # Update wrapper state
            wrapper.current_device = device
            wrapper._is_loaded_on_gpu = False

            # Force CUDA cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return wrapper._memory_size
        except Exception as e:
            print(f"⚠️ Error unloading {wrapper.model_info.engine} model: {e}")
            return 0


class HiggsAudioHandler(BaseEngineHandler):
    """Specialized handler for Higgs Audio engine with CUDA graphs"""

    def model_unload(self, wrapper, memory_to_free=None, unpatch_weights=True) -> bool:
        """
        Higgs Audio uses CUDA graphs which are safely cleaned up during device migration.
        CUDA graphs will be recreated automatically on the next generation call.
        """
        result = super().model_unload(wrapper, memory_to_free, unpatch_weights)

        if result:
            # Mark model as invalid to force fresh CUDA graph initialization
            wrapper._is_valid_for_reuse = False
            print(f"🔄 Marked Higgs Audio model for cache refresh (CUDA graphs will be recreated)")

        return result


# Engine registry
_ENGINE_HANDLERS = {
    'chatterbox': BaseEngineHandler(),
    'chatterbox_official_23lang': BaseEngineHandler(),
    'f5tts': BaseEngineHandler(),
    'higgs_audio': HiggsAudioHandler(),
    'step_audio_editx': StepAudioEditXHandler(),
    'vibevoice': BaseEngineHandler(),
    'rvc': BaseEngineHandler(),
}


def get_engine_handler(engine_type: str):
    """
    Get the appropriate handler for an engine type.

    Args:
        engine_type: Engine identifier (e.g., 'chatterbox', 'higgs_audio')

    Returns:
        Engine handler instance
    """
    handler = _ENGINE_HANDLERS.get(engine_type)
    if handler is None:
        print(f"⚠️ No specialized handler for engine '{engine_type}', using default")
        return BaseEngineHandler()
    return handler
