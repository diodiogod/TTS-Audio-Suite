"""MOSS-SoundEffect v2 model lifecycle handling."""

import gc
from typing import TYPE_CHECKING

import torch

from .generic_handler import GenericHandler

if TYPE_CHECKING:
    from ..base_wrapper import ComfyUIModelWrapper


class MossSoundEffectV2Handler(GenericHandler):
    """Allow normal offload/reload, but directly destroy permanent removals."""

    @staticmethod
    def release(wrapper: "ComfyUIModelWrapper") -> int:
        model = wrapper._model_ref() if wrapper._model_ref else wrapper.model
        if model is None:
            return 0

        freed_memory = wrapper._memory_size
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        release = getattr(model, "release", None)
        if callable(release):
            release()

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

        print(
            "Released MOSS-SoundEffect v2 directly "
            f"(freed approximately {freed_memory // 1024 // 1024}MB)"
        )
        return freed_memory
