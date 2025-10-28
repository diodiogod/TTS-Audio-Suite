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


class HiggsAudioHandler(BaseEngineHandler):
    """Specialized handler for Higgs Audio engine with CUDA graphs"""

    def model_unload(self, wrapper, memory_to_free=None, unpatch_weights=True) -> bool:
        """
        Higgs Audio uses CUDA graphs which get corrupted by CPU offloading.
        Mark model as invalid after unloading.
        """
        result = super().model_unload(wrapper, memory_to_free, unpatch_weights)

        if result:
            # Mark model as invalid to prevent reuse
            wrapper._is_valid_for_reuse = False
            print(f"🚫 Marked Higgs Audio model as invalid for reuse (CUDA graphs corrupted)")

        return result


# Engine registry
_ENGINE_HANDLERS = {
    'chatterbox': BaseEngineHandler(),
    'chatterbox_official_23lang': BaseEngineHandler(),
    'f5tts': BaseEngineHandler(),
    'higgs_audio': HiggsAudioHandler(),
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
