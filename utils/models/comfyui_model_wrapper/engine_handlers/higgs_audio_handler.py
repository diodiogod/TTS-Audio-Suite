"""
Higgs Audio engine handler with CUDA graph support
"""

import torch
from typing import Optional, TYPE_CHECKING

from .generic_handler import GenericHandler

if TYPE_CHECKING:
    from ..base_wrapper import ComfyUIModelWrapper


class HiggsAudioHandler(GenericHandler):
    """
    Handler for Higgs Audio engine with CUDA graph support.
    
    Higgs Audio uses CUDA graphs for optimization, which prevents safe unloading
    when enabled due to captured CUDA allocations.
    """
    
    def model_unload(self, wrapper: 'ComfyUIModelWrapper', memory_to_free: Optional[int], unpatch_weights: bool) -> bool:
        """
        Higgs Audio unload with CUDA graph cleanup.

        Safely clears CUDA graphs before unloading to prevent corruption.
        """
        # For stateless_tts (Higgs Audio wrapper), check if it's actually a Higgs Audio model
        is_higgs_audio = (wrapper.model_info.engine == "higgs_audio" or 
                         (wrapper.model_info.engine == "stateless_tts" and 
                          hasattr(wrapper.model, '_wrapped_engine')))
        
        if is_higgs_audio:
            # Check if this is a Higgs Audio model with CUDA Graphs enabled
            model = wrapper._model_ref() if wrapper._model_ref else wrapper.model

            # Clear CUDA graphs before unloading (using Qwen3-TTS cleanup pattern)
            try:
                self._clear_cuda_graphs(model, wrapper.model_info.engine)
            except Exception as e:
                print(f"⚠️ Failed to clear CUDA graphs: {e}")

        # Use standard unloading
        return super().model_unload(wrapper, memory_to_free, unpatch_weights)
    
    def partially_unload(self, wrapper: 'ComfyUIModelWrapper', device: str, memory_to_free: int) -> int:
        """
        Higgs Audio partial unload with CUDA graph cleanup.

        Safely clears CUDA graphs before device migration to prevent corruption.
        """
        model = wrapper._model_ref() if wrapper._model_ref else None
        if model is None:
            return 0

        # Clear CUDA graphs before moving to CPU (using Qwen3-TTS cleanup pattern)
        try:
            self._clear_cuda_graphs(model, wrapper.model_info.engine)
        except Exception as e:
            print(f"⚠️ Failed to clear CUDA graphs: {e}")

        # Use standard CPU migration after clearing
        return super().partially_unload(wrapper, device, memory_to_free)
    
    def _clear_cuda_graphs(self, model, engine: str):
        """Clear CUDA graphs if the model supports it (prevents corruption when moving to CPU)"""
        try:
            
            # The CUDA graphs are nested deeper in the Higgs Audio model structure
            # Try to find them through various paths
            cuda_model = None
            
            # Path 1: Direct access
            if hasattr(model, 'decode_graph_runners'):
                cuda_model = model

            # Path 2: Through engine attribute
            elif hasattr(model, 'engine') and hasattr(model.engine, 'model') and hasattr(model.engine.model, 'decode_graph_runners'):
                cuda_model = model.engine.model

            # Path 3: Through model attribute
            elif hasattr(model, 'model') and hasattr(model.model, 'decode_graph_runners'):
                cuda_model = model.model

            # Path 4: Search through all attributes recursively
            else:
                def find_cuda_model(obj, depth=0, max_depth=3):
                    if depth > max_depth:
                        return None
                    if hasattr(obj, 'decode_graph_runners'):
                        return obj
                    if hasattr(obj, '__dict__'):
                        for attr_name, attr_value in obj.__dict__.items():
                            if not attr_name.startswith('_') and attr_value is not None:
                                result = find_cuda_model(attr_value, depth + 1, max_depth)
                                if result:
                                    return result
                    return None

                cuda_model = find_cuda_model(model)
            
            if cuda_model:
                # Check for CUDA graphs and try to safely release them
                graph_count = sum(len(runners) for runners in cuda_model.decode_graph_runners.values())
                if graph_count > 0:
                    try:
                        # Synchronize before cleanup
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                        # DON'T manually clear decode_graph_runners - this breaks recreation
                        # DON'T set _force_cache_recreation - this also breaks things
                        # ONLY reset the cuda_graphs_initialized flag and let Higgs Audio recreate naturally

                        # Find the engine that has cuda_graphs_initialized flag
                        engine_with_caches = None
                        if hasattr(model, 'engine'):
                            engine_with_caches = model.engine
                        elif hasattr(model, '_wrapped_engine') and hasattr(model._wrapped_engine, 'engine'):
                            engine_with_caches = model._wrapped_engine.engine

                        if engine_with_caches and hasattr(engine_with_caches, 'cuda_graphs_initialized'):
                            # ONLY reset cuda_graphs_initialized - let natural recreation happen
                            engine_with_caches.cuda_graphs_initialized = False
                            print(f"🔧 Reset cuda_graphs_initialized flag - CUDA graphs will recreate naturally on next generation")

                        # Force CUDA synchronization and cache cleanup
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

                        # Force Python garbage collection
                        import gc
                        gc.collect()

                    except Exception as e:
                        print(f"⚠️ Failed to clear CUDA graphs: {e}")
                # else: No print needed for no graphs found
                    
        except Exception as e:
            print(f"⚠️ Failed to clear CUDA graphs: {e}")