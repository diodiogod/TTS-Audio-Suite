"""MOSS-TTS model lifecycle handling."""

import gc
from typing import TYPE_CHECKING

import torch

from .generic_handler import GenericHandler

if TYPE_CHECKING:
    from ..base_wrapper import ComfyUIModelWrapper


class MossTTSHandler(GenericHandler):
    """Keep normal offloading, with direct release for permanent removal."""

    @staticmethod
    def release(wrapper: "ComfyUIModelWrapper") -> int:
        model = wrapper._model_ref() if wrapper._model_ref else wrapper.model
        if model is None:
            return 0

        freed_memory = wrapper._memory_size
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # MOSS-TTS keeps both the language model and audio tokenizer under its
        # engine wrapper. Drop those references directly; moving an 8B model to
        # CPU during unload can exhaust system RAM and terminate ComfyUI.
        if hasattr(model, "_model"):
            model._model = None
        if hasattr(model, "_processor"):
            model._processor = None

        wrapper.model = None
        wrapper.model_info.model = None
        wrapper._model_ref = None
        wrapper.current_device = "cpu"
        wrapper._is_loaded_on_gpu = False
        wrapper._is_valid_for_reuse = False

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Released MOSS-TTS model directly (freed approximately {freed_memory // 1024 // 1024}MB)")
        return freed_memory
