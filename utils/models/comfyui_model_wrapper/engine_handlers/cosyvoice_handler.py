"""
CosyVoice engine handler with proper component-level device management
"""

import torch
import gc
from typing import Optional, TYPE_CHECKING

from .generic_handler import GenericHandler

if TYPE_CHECKING:
    from ..base_wrapper import ComfyUIModelWrapper


class CosyVoiceHandler(GenericHandler):
    """
    Handler for CosyVoice engine with proper device management.

    CosyVoice AutoModel doesn't have a .to() method, so we need to manually
    move all model components (LLM, Flow, HiFT, etc.) to CPU/GPU.
    """

    def partially_unload(self, wrapper: 'ComfyUIModelWrapper', device: str, memory_to_free: int) -> int:
        """
        CosyVoice partial unload with component-level device movement.

        Moves all CosyVoice model components (llm, flow, hift, campplus) to CPU.
        """
        if not wrapper._is_loaded_on_gpu:
            return 0

        model = wrapper._model_ref() if wrapper._model_ref else None
        if model is None:
            return 0

        freed_memory = 0

        try:
            # CosyVoice AutoModel has a 'model' attribute that contains the actual CosyVoice instance
            # The CosyVoice instance has: llm, flow, hift, campplus components
            if hasattr(model, 'model'):
                cosyvoice_instance = model.model

                # List of component names to move
                component_names = ['llm', 'flow', 'hift', 'campplus']
                components_moved = []

                for comp_name in component_names:
                    if hasattr(cosyvoice_instance, comp_name):
                        component = getattr(cosyvoice_instance, comp_name)
                        if component is not None and hasattr(component, 'to'):
                            try:
                                component.to(device)
                                freed_memory += self._estimate_model_memory(component)
                                components_moved.append(comp_name)
                            except Exception as e:
                                print(f"⚠️ Failed to move CosyVoice {comp_name} to {device}: {e}")

                if components_moved:
                    wrapper.current_device = device
                    wrapper._is_loaded_on_gpu = False
                    print(f"🔄 Moved CosyVoice components ({', '.join(components_moved)}) to {device}, freed ~{freed_memory // 1024 // 1024}MB")
                else:
                    print(f"⚠️ No CosyVoice components found to move")
            else:
                # Fallback: AutoModel might directly expose components
                component_names = ['llm', 'flow', 'hift', 'campplus']
                components_moved = []

                for comp_name in component_names:
                    if hasattr(model, comp_name):
                        component = getattr(model, comp_name)
                        if component is not None and hasattr(component, 'to'):
                            try:
                                component.to(device)
                                freed_memory += self._estimate_model_memory(component)
                                components_moved.append(comp_name)
                            except Exception as e:
                                print(f"⚠️ Failed to move CosyVoice {comp_name} to {device}: {e}")

                if components_moved:
                    wrapper.current_device = device
                    wrapper._is_loaded_on_gpu = False
                    print(f"🔄 Moved CosyVoice components ({', '.join(components_moved)}) to {device}, freed ~{freed_memory // 1024 // 1024}MB")

        except Exception as e:
            print(f"⚠️ Failed to partially unload CosyVoice model: {e}")

        # Force garbage collection after unloading
        if freed_memory > 0:
            # CRITICAL: Synchronize CUDA to ensure memory is actually freed
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"⚠️ CUDA cleanup warning (safe to ignore): {e}")

            try:
                gc.collect()
            except Exception as gc_error:
                print(f"⚠️ Garbage collection failed (safe to ignore): {gc_error}")

        return freed_memory

    def partially_load(self, wrapper: 'ComfyUIModelWrapper', device: str) -> bool:
        """
        CosyVoice partial load - move components from CPU to GPU.

        Mirrors partially_unload() by moving all CosyVoice components back to target device.
        """
        if wrapper._is_loaded_on_gpu:
            return True

        model = wrapper._model_ref() if wrapper._model_ref else None
        if model is None:
            return False

        try:
            # CosyVoice AutoModel has a 'model' attribute that contains the actual CosyVoice instance
            # The CosyVoice instance has: llm, flow, hift, campplus components
            if hasattr(model, 'model'):
                cosyvoice_instance = model.model

                # List of component names to move
                component_names = ['llm', 'flow', 'hift', 'campplus']
                components_moved = []

                for comp_name in component_names:
                    if hasattr(cosyvoice_instance, comp_name):
                        component = getattr(cosyvoice_instance, comp_name)
                        if component is not None and hasattr(component, 'to'):
                            try:
                                component.to(device)
                                # CRITICAL: Move all nested modules to ensure device consistency
                                # This fixes "Expected all tensors to be on the same device" errors
                                for module in component.modules():
                                    module.to(device)
                                components_moved.append(comp_name)
                            except Exception as e:
                                print(f"⚠️ Failed to move CosyVoice {comp_name} to {device}: {e}")

                if components_moved:
                    wrapper.current_device = device
                    wrapper._is_loaded_on_gpu = True
                    print(f"🔄 Moved CosyVoice components ({', '.join(components_moved)}) to {device}")
                    return True
                else:
                    print(f"⚠️ No CosyVoice components found to move")
                    return False
            else:
                # Fallback: AutoModel might directly expose components
                component_names = ['llm', 'flow', 'hift', 'campplus']
                components_moved = []

                for comp_name in component_names:
                    if hasattr(model, comp_name):
                        component = getattr(model, comp_name)
                        if component is not None and hasattr(component, 'to'):
                            try:
                                component.to(device)
                                # Move all nested modules
                                for module in component.modules():
                                    module.to(device)
                                components_moved.append(comp_name)
                            except Exception as e:
                                print(f"⚠️ Failed to move CosyVoice {comp_name} to {device}: {e}")

                if components_moved:
                    wrapper.current_device = device
                    wrapper._is_loaded_on_gpu = True
                    print(f"🔄 Moved CosyVoice components ({', '.join(components_moved)}) to {device}")
                    return True

        except Exception as e:
            print(f"⚠️ Failed to partially load CosyVoice model: {e}")
            return False

        return False

    def model_unload(self, wrapper: 'ComfyUIModelWrapper', memory_to_free: Optional[int], unpatch_weights: bool) -> bool:
        """
        CosyVoice full unload using CPU migration.
        """
        if memory_to_free is not None and memory_to_free < wrapper.loaded_size():
            # Try partial unload first
            freed = self.partially_unload(wrapper, 'cpu', memory_to_free)
            success = freed >= memory_to_free
            return success

        # Full unload - move to CPU
        freed = self.partially_unload(wrapper, 'cpu', wrapper._memory_size)
        return freed > 0
