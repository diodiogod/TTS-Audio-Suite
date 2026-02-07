"""
Qwen3-TTS engine handler with CUDA graph cleanup support
"""

import torch
from typing import Optional, TYPE_CHECKING

from .generic_handler import GenericHandler

if TYPE_CHECKING:
    from ..base_wrapper import ComfyUIModelWrapper


class Qwen3TTSHandler(GenericHandler):
    """
    Handler for Qwen3-TTS engine with CUDA graph cleanup.

    Qwen3-TTS uses CUDA graphs for tokenizer optimization on streaming decode.
    On Windows, consecutive captures of CUDA graphs cause memory corruption:
    - First run: CUDA graph capture succeeds
    - Second run: Async allocator detects double-allocation at same address
    - Result: cudaMallocAsync INTERNAL ASSERT FAILED crash

    This handler clears CUDA graphs before unloading to allow safe reloading.
    """

    def partially_unload(self, wrapper: 'ComfyUIModelWrapper', device: str, memory_to_free: int) -> int:
        """
        Qwen3-TTS partial unload with CUDA graph cleanup.

        Clears CUDA graphs before moving to CPU to prevent memory corruption
        on consecutive model loads.
        """
        model = wrapper._model_ref() if wrapper._model_ref else None
        if model is None:
            return 0

        # Clear CUDA graphs before unloading
        try:
            self._clear_cuda_graphs(model, wrapper.model_info.engine)
        except Exception as e:
            print(f"⚠️ Failed to clear CUDA graphs: {e}")

        # Use standard CPU migration after clearing
        return super().partially_unload(wrapper, device, memory_to_free)

    def model_unload(self, wrapper: 'ComfyUIModelWrapper', memory_to_free: Optional[int], unpatch_weights: bool) -> bool:
        """
        Qwen3-TTS unload with CUDA graph cleanup.

        Clears CUDA graphs before full unload to prevent crashes.
        """
        model = wrapper._model_ref() if wrapper._model_ref else wrapper.model

        # Clear CUDA graphs before unloading
        try:
            self._clear_cuda_graphs(model, wrapper.model_info.engine)
        except Exception as e:
            print(f"⚠️ Failed to clear CUDA graphs: {e}")

        # Use standard unloading
        return super().model_unload(wrapper, memory_to_free, unpatch_weights)

    def _clear_cuda_graphs(self, model, engine: str):
        """Clear CUDA graphs from Qwen3-TTS decoder to prevent Windows allocation corruption"""
        try:
            # Debug: Print what type of model we received
            print(f"🔍 [DEBUG] Received model type: {type(model).__name__}")
            print(f"🔍 [DEBUG] Has _model: {hasattr(model, '_model')}")
            if hasattr(model, '_model'):
                print(f"🔍 [DEBUG] _model is None: {model._model is None}")
                if model._model is not None:
                    print(f"🔍 [DEBUG] _model type: {type(model._model).__name__}")
                    print(f"🔍 [DEBUG] _model has .model: {hasattr(model._model, 'model')}")

            # Navigate to decoder through various possible paths
            # wrapper.model = Qwen3TTSEngine (has ._model)
            # wrapper.model._model = Qwen3TTSModel (has .model)
            # wrapper.model._model.model = transformers model (has .speech_tokenizer)
            decoder = None

            # Path 1: Through Qwen3TTSEngine wrapper (most common)
            # model._model.model.speech_tokenizer.model.decoder
            if hasattr(model, '_model') and model._model is not None:
                if hasattr(model._model, 'model') and hasattr(model._model.model, 'speech_tokenizer'):
                    if hasattr(model._model.model.speech_tokenizer, 'model') and hasattr(model._model.model.speech_tokenizer.model, 'decoder'):
                        decoder = model._model.model.speech_tokenizer.model.decoder
                        print(f"🔍 Found Qwen3-TTS decoder at model._model.model.speech_tokenizer.model.decoder")

            # Path 2: Direct Qwen3TTSModel access (model.model.speech_tokenizer.model.decoder)
            if decoder is None and hasattr(model, 'model') and hasattr(model.model, 'speech_tokenizer'):
                if hasattr(model.model.speech_tokenizer, 'model') and hasattr(model.model.speech_tokenizer.model, 'decoder'):
                    decoder = model.model.speech_tokenizer.model.decoder
                    print(f"🔍 Found Qwen3-TTS decoder at model.model.speech_tokenizer.model.decoder")

            # Path 3: Direct transformers model access (model.speech_tokenizer.model.decoder)
            if decoder is None and hasattr(model, 'speech_tokenizer'):
                if hasattr(model.speech_tokenizer, 'model') and hasattr(model.speech_tokenizer.model, 'decoder'):
                    decoder = model.speech_tokenizer.model.decoder
                    print(f"🔍 Found Qwen3-TTS decoder at model.speech_tokenizer.model.decoder")

            if decoder is None:
                print(f"⚠️ Could not locate Qwen3-TTS decoder for CUDA graph cleanup")
                return

            # Check if decoder has CUDA graphs
            if hasattr(decoder, '_cuda_graph') and decoder._cuda_graph is not None:
                print(f"🧹 Clearing Qwen3-TTS CUDA graph to prevent allocation corruption...")

                # Synchronize before clearing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Try to release the graph pool using PyTorch's internal API
                try:
                    # Get the pool handle from the graph
                    if hasattr(decoder._cuda_graph, 'pool'):
                        pool = decoder._cuda_graph.pool()
                        print(f"🔧 Got graph pool handle")

                        # Try to release it using PyTorch's private API
                        if hasattr(torch._C, '_cuda_releasePool'):
                            torch._C._cuda_releasePool(pool)
                            print(f"✅ Released CUDA graph pool")
                except Exception as e:
                    print(f"⚠️ Could not release graph pool (will try standard cleanup): {e}")

                # Delete static tensors first (they hold references to captured allocations)
                # Must delete ALL references for pool to be freed
                if hasattr(decoder, '_static_input') and decoder._static_input is not None:
                    del decoder._static_input
                    decoder._static_input = None

                if hasattr(decoder, '_static_output') and decoder._static_output is not None:
                    del decoder._static_output
                    decoder._static_output = None

                # Now clear the graph itself
                graph = decoder._cuda_graph
                decoder._cuda_graph = None
                decoder._graph_window_size = None
                del graph  # Explicit deletion

                # Force CUDA synchronization and cache cleanup
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                # Force Python garbage collection
                import gc
                gc.collect()

                print(f"✅ Qwen3-TTS CUDA graph cleared successfully")
            else:
                print(f"📝 No CUDA graphs found in Qwen3-TTS decoder (already cleared or not captured)")

        except Exception as e:
            print(f"⚠️ Failed to clear Qwen3-TTS CUDA graphs: {e}")
