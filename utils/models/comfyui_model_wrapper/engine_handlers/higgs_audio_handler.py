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
        Higgs Audio unload with CUDA graph checking.
        
        Refuses to unload if CUDA graphs are enabled to prevent crashes.
        """
        # For stateless_tts (Higgs Audio wrapper), check if it's actually a Higgs Audio model
        is_higgs_audio = (wrapper.model_info.engine == "higgs_audio" or 
                         (wrapper.model_info.engine == "stateless_tts" and 
                          hasattr(wrapper.model, '_wrapped_engine')))
        
        if is_higgs_audio:
            # Check if this is a Higgs Audio model with CUDA Graphs enabled
            cuda_graphs_enabled = getattr(wrapper.model, '_cuda_graphs_enabled', True)
            if cuda_graphs_enabled:
                print(f"⛔ CUDA Graph Mode: Unloading disabled to prevent crashes")
                print(f"   Model uses CUDA Graph optimization - cannot be safely unloaded")
                print(f"   To enable memory unloading, disable CUDA Graphs in engine settings")
                print(f"   Or restart ComfyUI to fully free memory")
                return False  # Refuse to unload
        
        # If CUDA graphs disabled, use standard unloading
        return super().model_unload(wrapper, memory_to_free, unpatch_weights)
    
    def partially_unload(self, wrapper: 'ComfyUIModelWrapper', device: str, memory_to_free: int) -> int:
        """
        Higgs Audio partial unload with CUDA graph clearing.
        
        Attempts to safely clear CUDA graphs before CPU migration.
        """
        model = wrapper._model_ref() if wrapper._model_ref else None
        if model is None:
            return 0
        
        try:
            # CRITICAL: Clear CUDA graphs before moving to CPU (prevents corruption)
            self._clear_cuda_graphs(model, wrapper.model_info.engine)
        except Exception as e:
            print(f"⚠️ Failed to clear CUDA graphs: {e}")
        
        # Use standard CPU migration after graph clearing
        return super().partially_unload(wrapper, device, memory_to_free)
    
    def _clear_cuda_graphs(self, model, engine: str):
        """Clear CUDA graphs if the model supports it (prevents corruption when moving to CPU)"""
        try:
            print(f"🔍 Checking for CUDA graphs in {engine} model...")
            print(f"🔍 Found Higgs Audio model, searching for decode_graph_runners...")
            
            # The CUDA graphs are nested deeper in the Higgs Audio model structure
            # Try to find them through various paths
            cuda_model = None
            
            # Path 1: Direct access
            if hasattr(model, 'decode_graph_runners'):
                cuda_model = model
                print(f"🔍 Found decode_graph_runners at top level")
            
            # Path 2: Through engine attribute
            elif hasattr(model, 'engine') and hasattr(model.engine, 'model') and hasattr(model.engine.model, 'decode_graph_runners'):
                cuda_model = model.engine.model
                print(f"🔍 Found decode_graph_runners in model.engine.model")
            
            # Path 3: Through model attribute
            elif hasattr(model, 'model') and hasattr(model.model, 'decode_graph_runners'):
                cuda_model = model.model
                print(f"🔍 Found decode_graph_runners in model.model")
            
            # Path 4: Search through all attributes recursively
            else:
                print(f"🔍 Searching recursively for decode_graph_runners...")
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
                                    print(f"🔍 Found decode_graph_runners in {attr_name} (depth {depth + 1})")
                                    return result
                    return None
                
                cuda_model = find_cuda_model(model)
            
            if cuda_model:
                # Check for CUDA graphs and try to safely release them
                graph_count = sum(len(runners) for runners in cuda_model.decode_graph_runners.values())
                if graph_count > 0:
                    print(f"🔍 Found {graph_count} CUDA graphs - attempting safe release")
                    try:
                        # Try to properly end/reset the CUDA graphs before clearing
                        # This should release the captured allocations properly
                        for key, runners in cuda_model.decode_graph_runners.items():
                            print(f"  🔧 Releasing {len(runners)} graphs for {key}")
                            for i, runner in enumerate(runners):
                                if hasattr(runner, 'graph') and runner.graph is not None:
                                    # Try to reset/end the graph properly
                                    try:
                                        # Reset the graph state
                                        if hasattr(runner.graph, 'reset'):
                                            runner.graph.reset()
                                        elif hasattr(runner, 'reset'):
                                            runner.reset()
                                        print(f"    ✅ Released graph {i+1}/{len(runners)}")
                                    except Exception as e:
                                        print(f"    ⚠️ Failed to reset graph {i+1}: {e}")
                            
                            # Now clear the runners
                            runners.clear()
                            
                        print(f"🧹 Attempted to release {graph_count} CUDA graphs safely")
                        
                        # Force CUDA synchronization to ensure graphs are properly released
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            print(f"🔄 CUDA synchronized after graph release")
                            
                    except Exception as e:
                        print(f"⚠️ Failed to release CUDA graphs: {e}, proceeding with standard unload")
                else:
                    print(f"📝 No CUDA graphs found in {engine} model")
            else:
                print(f"⚠️ Could not locate decode_graph_runners in {engine} model structure")
                    
        except Exception as e:
            print(f"⚠️ Failed to clear CUDA graphs: {e}")