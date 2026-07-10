import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
NODE_PATH = (
    REPO_ROOT
    / "nodes"
    / "engines"
    / "fish_audio_s2_engine_node.py"
)
SPEC = importlib.util.spec_from_file_location("fish_audio_s2_engine_node_test_module", NODE_PATH)
NODE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NODE_MODULE)


@pytest.mark.unit
def test_model_is_first_and_compile_defaults_enabled():
    inputs = NODE_MODULE.FishAudioS2EngineNode.INPUT_TYPES()

    assert list(inputs["required"])[0] == "model"
    assert "s2-pro" in inputs["required"]["model"][0]
    assert inputs["optional"]["compile"][1]["default"] is True

    node = NODE_MODULE.FishAudioS2EngineNode()
    (engine_config,) = node.create_engine_config()
    assert engine_config["config"]["compile"] is True


@pytest.mark.unit
def test_bnb_quantization_preserves_compile():
    node = NODE_MODULE.FishAudioS2EngineNode()

    (engine_config,) = node.create_engine_config(
        device="cuda",
        temperature=0.8,
        top_p=0.8,
        repetition_penalty=1.1,
        native_chunk_length=200,
        max_new_tokens=1024,
        context_length="8192",
        compile=True,
        model_variant="s2-pro",
        quantization="bnb_int8",
    )

    config = engine_config["config"]
    assert config["compile"] is True
    assert config["quantization"] == "bnb_int8"


@pytest.mark.unit
def test_fp8_variant_forces_quantization_none():
    node = NODE_MODULE.FishAudioS2EngineNode()

    (engine_config,) = node.create_engine_config(
        device="cuda",
        temperature=0.8,
        top_p=0.8,
        repetition_penalty=1.1,
        native_chunk_length=200,
        max_new_tokens=1024,
        context_length="8192",
        compile=False,
        model_variant="s2-pro-fp8",
        quantization="bnb_nf4",
        precision="float16",
    )

    config = engine_config["config"]
    assert config["model_variant"] == "s2-pro-fp8"
    assert config["quantization"] == "none"
    assert config["precision"] == "bfloat16"
