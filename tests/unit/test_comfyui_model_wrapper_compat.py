"""
Unit tests for ComfyUIModelWrapper compatibility with newer ComfyUI contracts.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

# Add custom node root to path BEFORE any project imports
custom_node_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(custom_node_root))

# Set up minimal environment to avoid ComfyUI imports
os.environ.setdefault("COMFYUI_TESTING", "1")

from utils.models.comfyui_model_wrapper.base_wrapper import ComfyUIModelWrapper, ModelInfo


class DummyModel:
    """Minimal wrapped model for ComfyUI wrapper tests."""


class TrackingModel:
    """Wrapped model that exposes ComfyUI memory hooks."""

    def __init__(self):
        self.calls = []

    def model_mmap_residency(self, free=False):
        self.calls.append(("mmap", free))
        if free:
            return 12, 34
        return 56, 78

    def pinned_memory_size(self):
        self.calls.append(("pinned", None))
        return 90

    def lowvram_patch_counter(self):
        self.calls.append(("lowvram", None))
        return 3


def build_wrapper(model=None, memory_size=4096, device="cpu"):
    wrapped_model = model or DummyModel()
    model_info = ModelInfo(
        model=wrapped_model,
        model_type="tts",
        engine="test_engine",
        device=device,
        memory_size=memory_size,
        load_device=device,
    )
    return ComfyUIModelWrapper(wrapped_model, model_info)


@pytest.mark.unit
class TestComfyUIModelWrapperCompatibility:
    def test_model_mmap_residency_falls_back_to_full_model_size(self):
        wrapper = build_wrapper(memory_size=12345)
        assert wrapper.model_mmap_residency() == (0, 12345)

    def test_model_mmap_residency_forwards_wrapped_model(self):
        wrapped = TrackingModel()
        wrapper = build_wrapper(model=wrapped)

        assert wrapper.model_mmap_residency() == (56, 78)
        assert wrapper.model_mmap_residency(free=True) == (12, 34)
        assert wrapped.calls[:2] == [("mmap", False), ("mmap", True)]

    def test_safe_defaults_for_pinned_and_lowvram_metrics(self):
        wrapper = build_wrapper(memory_size=2048)

        assert wrapper.pinned_memory_size() == 0
        assert wrapper.lowvram_patch_counter() == 0
        assert wrapper.get_ram_usage() == 2048

    def test_forwarded_metrics_use_wrapped_model_hooks(self):
        wrapped = TrackingModel()
        wrapper = build_wrapper(model=wrapped)

        assert wrapper.pinned_memory_size() == 90
        assert wrapper.lowvram_patch_counter() == 3
        assert ("pinned", None) in wrapped.calls
        assert ("lowvram", None) in wrapped.calls

    def test_current_loaded_device_returns_torch_device(self):
        wrapper = build_wrapper(device="cpu")
        assert wrapper.current_loaded_device() == torch.device("cpu")

    def test_model_patches_to_accepts_device_and_dtype_targets(self):
        wrapper = build_wrapper()

        wrapper.model_patches_to(torch.device("cpu"))
        wrapper.model_patches_to(torch.float16)

        assert wrapper.load_device == torch.device("cpu")
        assert wrapper.current_loaded_device() == torch.device("cpu")
        assert wrapper.model_dtype() == torch.float16
        assert wrapper.model_patches_models() == ()
