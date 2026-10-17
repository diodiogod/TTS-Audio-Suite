"""
Unit tests for IndexTTS-2 / IndexTTS-2.5 engine device movement.

ComfyUI unloads the engine with model.to("cpu") and reloads it with model.to(device).
Both backends must move every component, including QwenEmotion and the plain tensors
that Module.to() skips, and IndexTTS-2 must keep its low-VRAM resting placement.
"""

import importlib
import importlib.util
import gc
import os
import sys
import types
import weakref
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


@pytest.mark.parametrize("low_vram", [False, True])
def test_wrapper_respects_backend_placement_without_recursing(low_vram, monkeypatch):
    from utils.models.comfyui_model_wrapper.base_wrapper import ComfyUIModelWrapper, ModelInfo

    if not low_vram and not torch.cuda.is_available():
        pytest.skip("CUDA is required for ordinary GPU placement")
    module = _import_backend("infer_v2")
    engine = _build_engine(module, low_vram=low_vram)
    moves = []
    qwen_moves = []
    monkeypatch.setattr(module.QwenEmotion, "to", lambda self, device: qwen_moves.append(device) or self)
    monkeypatch.setattr(CachedTensorModule, "to", lambda self, device: moves.append(str(device)) or self)
    wrapper = ComfyUIModelWrapper(engine, ModelInfo(
        model=engine, model_type="tts", engine="index_tts", device="cpu",
        memory_size=4096, load_device="cuda:0",
    ))

    wrapper.model_load("cuda:0", _from_ensure_device=True)

    assert moves == ["cpu" if low_vram else "cuda:0"] * len(MODEL_NAMES)
    # The backend chooses Qwen's GPU; the wrapper must not move its child model again.
    assert qwen_moves == ["cuda:0"]
    assert all(p.device.type == "cpu" for p in engine.qwen_emo.model.parameters())
    assert wrapper._is_loaded_on_gpu
    assert wrapper.current_device == "cuda:0"


@pytest.fixture
def accel_module(monkeypatch):
    """Load the real lifecycle code without requiring FlashAttention or Triton."""
    package_name = "_tts_accel_lifecycle_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(custom_node_root / "engines/index_tts/indextts/accel")]
    attention = types.ModuleType(package_name + ".attention")
    attention.ForwardContext = object
    attention.get_forward_context = lambda: None
    attention.set_forward_context = lambda *args, **kwargs: None
    resets = []
    attention.reset_forward_context = lambda: resets.append(True)
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, attention.__name__, attention)
    name = package_name + ".accel_engine"
    spec = importlib.util.spec_from_file_location(name, Path(package.__path__[0]) / "accel_engine.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    # The relative import creates this module too; remove it after the test.
    cache_name = package_name + ".kv_manager"
    monkeypatch.setitem(sys.modules, cache_name, sys.modules.pop(cache_name))
    return module, resets


def test_accel_release_drops_graphs_model_cache_and_context(accel_module, monkeypatch):
    module, resets = accel_module
    accel = module.AccelInferenceEngine.__new__(module.AccelInferenceEngine)
    accel.model = nn.Linear(2, 2)
    accel.lm_head = nn.Linear(2, 2)
    accel.kv_manager = types.SimpleNamespace(kv_cache=torch.ones(4))
    accel.sampler = nn.Identity()
    accel.graphs = {1: torch.ones(2)}
    accel.graph_vars = {"input_ids": torch.ones(2)}
    accel.graph_pool = object()
    accel.graph_captured = True
    accel.current_sequences = [object()]
    references = [weakref.ref(value) for value in (
        accel.model, accel.lm_head, accel.kv_manager.kv_cache,
        accel.graphs[1], accel.graph_vars["input_ids"],
    )]
    synchronized = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: synchronized.append(str(device)))

    accel.release()
    accel.release()  # Repeated cleanup must also be safe.
    gc.collect()

    assert synchronized == ["cpu"]
    assert len(resets) == 2
    assert all(reference() is None for reference in references)
    assert not accel.graphs and not accel.graph_captured and not accel.current_sequences
    assert accel.graph_vars is None and accel.graph_pool is None


def test_gpt_movement_releases_acceleration_without_disabling_it():
    from indextts.gpt.model_v2 import UnifiedVoice

    voice = UnifiedVoice.__new__(UnifiedVoice)
    nn.Module.__init__(voice)
    voice.gpt = nn.Linear(2, 2)
    voice.use_accel = True
    releases = []
    voice.accel_engine = types.SimpleNamespace(release=lambda: releases.append(True))

    voice.to(dtype=torch.float64)

    assert releases == [True]
    assert voice.accel_engine is None and voice.use_accel
    assert next(voice.gpt.parameters()).dtype == torch.float64
    # Low-VRAM initialization must not allocate an accelerator on a different GPU.
    voice._init_accel_engine()
    assert voice.accel_engine is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the reload path")
def test_acceleration_is_rebuilt_for_generation_after_offload(monkeypatch):
    from indextts.gpt.model_v2 import UnifiedVoice
    from indextts.utils.module_device import move_module

    voice = UnifiedVoice.__new__(UnifiedVoice)
    nn.Module.__init__(voice)
    voice.gpt = nn.Linear(2, 2)
    voice.use_accel = True
    voice.accel_engine = None
    voice.spk_cond_mode = "campplus"
    voice.spk_emb_proj = nn.Identity()
    voice.max_mel_tokens = 8
    voice.stop_mel_token = 7
    voice.inference_model = types.SimpleNamespace(
        store_mel_emb=lambda value: None, embeddings=None, text_pos_embedding=None,
    )
    rebuilt = []
    released = []

    def rebuild():
        rebuilt.append(True)
        voice.accel_engine = types.SimpleNamespace(
            generate=lambda inputs, **kwargs: torch.cat([inputs, inputs.new_tensor([[3, 7]])], dim=1),
            release=lambda: released.append(True),
        )

    monkeypatch.setattr(voice, "_init_accel_engine", rebuild)
    monkeypatch.setattr(voice, "prepare_gpt_inputs", lambda cond, text, langs: (
        text.new_ones((1, 3)), cond, text.new_ones((1, 3)),
    ))
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    for _ in range(2):
        move_module(voice, device)
        tokens, _ = voice.inference_speech(
            torch.ones(1, 2, 1, device=device), torch.ones(1, 1, dtype=torch.long, device=device),
            emo_vec=torch.ones(1, 2, device=device), campplus_embedding=torch.ones(1, 2, device=device),
        )
        assert tokens.tolist() == [[3, 7]]
        assert tokens.device == device
        move_module(voice, "cpu")
        assert voice.accel_engine is None

    assert len(rebuilt) == len(released) == 2


@pytest.mark.parametrize("device_index", range(torch.cuda.device_count()))
def test_real_cuda_graphs_release_without_growing_on_reload(device_index, accel_module):
    module, _ = accel_module

    class GraphModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)
            self.config = types.SimpleNamespace(hidden_size=2)

        def forward(self, input_ids, return_dict=True):
            value = input_ids.float().unsqueeze(-1).expand(-1, -1, 2)
            return types.SimpleNamespace(last_hidden_state=self.linear(value))

    device = torch.device("cuda", device_index)
    snapshots = []
    with torch.cuda.device(device), torch.no_grad():
        for _ in range(2):
            accel = module.AccelInferenceEngine(
                GraphModel().to(device), nn.Linear(2, 2).to(device),
                num_layers=1, num_heads=1, head_dim=2, block_size=8, num_blocks=2,
            )
            accel._capture_cuda_graphs()
            with torch.inference_mode():
                accel.graph_vars["input_ids"][:1].fill_(3)
                accel.graphs[1].replay()
                torch.cuda.synchronize(device)
                torch.testing.assert_close(
                    accel.graph_vars["outputs"][:1],
                    accel.model(input_ids=accel.graph_vars["input_ids"][:1, None]).last_hidden_state[:, 0],
                )
            accel.release()
            gc.collect()
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            snapshots.append(torch.cuda.memory_allocated(device))

    # CUDA/cuBLAS workspaces can remain after the first capture. They must not
    # grow with a second render, and graph replay must run on the requested GPU.
    assert snapshots[1] == snapshots[0]
