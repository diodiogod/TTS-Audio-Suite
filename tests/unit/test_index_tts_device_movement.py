"""
Unit tests for IndexTTS-2 / IndexTTS-2.5 engine device movement.

ComfyUI unloads the engine with model.to("cpu") and reloads it with model.to(device).
Both backends must move every component, including QwenEmotion and the plain tensors
that Module.to() skips, and IndexTTS-2 must keep its low-VRAM resting placement.
"""

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

custom_node_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(custom_node_root))
sys.path.insert(0, str(custom_node_root / "engines" / "index_tts"))

os.environ.setdefault("COMFYUI_TESTING", "1")

pytestmark = pytest.mark.unit

MODEL_NAMES = ["gpt", "semantic_model", "semantic_codec", "s2mel", "campplus_model", "bigvgan"]
CACHE_NAMES = [
    "cache_spk_cond", "cache_s2mel_style", "cache_s2mel_prompt", "cache_spk_audio_prompt",
    "cache_emo_cond", "cache_emo_audio_prompt", "cache_mel",
]


def _import_backend(name):
    try:
        return importlib.import_module(f"indextts.{name}")
    except Exception as exc:  # heavy optional deps (modelscope, omegaconf, ...)
        pytest.skip(f"indextts.{name} not importable here: {exc}")


class CachedTensorModule(nn.Module):
    """Mimics s2mel's gpt_fast Transformer: plain tensor attributes, not buffers."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.causal_mask = torch.ones(2, 2, dtype=torch.bool)


class FakeQwenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.hf_device_map = {"": "cuda:0"}


def _device_type(tensor):
    return tensor.device.type


def _build_engine(module, low_vram=False):
    engine = module.IndexTTS2.__new__(module.IndexTTS2)
    for name in MODEL_NAMES:
        setattr(engine, name, CachedTensorModule())
    qwen = module.QwenEmotion.__new__(module.QwenEmotion)
    qwen.model = FakeQwenModel()
    engine.qwen_emo = qwen
    engine.semantic_mean = torch.zeros(2)
    engine.semantic_std = torch.ones(2)
    engine.emo_matrix = (torch.zeros(2), torch.zeros(2))
    engine.spk_matrix = (torch.zeros(2), torch.zeros(2))
    for name in CACHE_NAMES:
        setattr(engine, name, torch.zeros(1))
    engine.low_vram = low_vram
    engine.device = "cuda:0"
    engine.load_device = "cpu" if low_vram else "cuda:0"
    return engine


def _all_tensors(engine):
    for name in MODEL_NAMES:
        module = getattr(engine, name)
        yield from module.parameters()
        yield module.causal_mask
    yield engine.semantic_mean
    yield engine.semantic_std
    yield from engine.emo_matrix
    yield from engine.spk_matrix


@pytest.mark.parametrize("backend", ["infer_v2", "infer_v2_5"])
def test_to_moves_every_component_and_clears_conditioning(backend, monkeypatch):
    module = _import_backend(backend)
    engine = _build_engine(module)
    moved_to = []
    monkeypatch.setattr(module.QwenEmotion, "to", lambda self, device: moved_to.append(device) or self)

    assert engine.to(torch.device("cpu")) is engine

    assert all(_device_type(t) == "cpu" for t in _all_tensors(engine))
    assert moved_to == ["cpu"]
    assert all(getattr(engine, name) is None for name in CACHE_NAMES)
    # inference code calls self.device.startswith(...)
    assert engine.device == "cpu"


def test_v2_low_vram_keeps_models_resting_on_cpu(monkeypatch):
    module = _import_backend("infer_v2")
    engine = _build_engine(module, low_vram=True)
    moved_to = []
    monkeypatch.setattr(module.QwenEmotion, "to", lambda self, device: moved_to.append(device) or self)
    moved_modules = []
    monkeypatch.setattr(module, "move_module", lambda m, device: moved_modules.append(device) or m)

    engine.to("cuda:0")

    assert set(moved_modules) == {"cpu"}
    assert engine.load_device == "cpu"
    assert engine.device == "cuda:0"
    # QwenEmotion is not part of the low-VRAM rotation, so it follows the target
    assert moved_to == ["cuda:0"]


@pytest.mark.parametrize("backend", ["infer_v2", "infer_v2_5"])
def test_qwen_emotion_moves_as_single_device(backend, monkeypatch):
    module = _import_backend(backend)
    qwen = module.QwenEmotion.__new__(module.QwenEmotion)
    qwen.model = FakeQwenModel()

    qwen.to("cpu")
    assert all(_device_type(p) == "cpu" for p in qwen.model.parameters())
    assert qwen.model.hf_device_map == {"": "cpu"}

    # A CUDA target re-selects the GPU the same way as at load time
    monkeypatch.setattr(module.QwenEmotion, "_select_device", staticmethod(lambda preferred=None: "cpu"))
    qwen.to("cuda:1")
    assert qwen.model.hf_device_map == {"": "cpu"}


def test_v2_5_qwen_emotion_loads_on_a_single_device(monkeypatch):
    module = _import_backend("infer_v2_5")
    calls = []
    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained",
                        lambda *a, **k: calls.append(k) or FakeQwenModel())
    monkeypatch.setattr(module.QwenEmotion, "_select_device", staticmethod(lambda preferred=None: "cuda:1"))

    module.QwenEmotion("unused", preferred_device="cuda:0")

    assert calls[0]["device_map"] == {"": "cuda:1"}


def test_move_module_moves_plain_tensor_attributes():
    from indextts.utils.module_device import move_module

    module = CachedTensorModule()
    module.causal_mask = module.causal_mask.to(torch.float16)
    meta = move_module(module, "meta")

    assert meta is module
    assert module.causal_mask.device.type == "meta"
    assert module.linear.weight.device.type == "meta"
