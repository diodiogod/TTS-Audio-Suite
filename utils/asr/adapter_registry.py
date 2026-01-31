"""
Adapter registry for ASR engines.
Keep mapping here so unified node doesn't know engine specifics.
"""

import importlib
from typing import Dict, Type


_ADAPTER_MAP: Dict[str, str] = {
    "qwen3_tts": "engines.adapters.asr_qwen3_adapter.Qwen3ASREngineAdapter",
    "qwen3": "engines.adapters.asr_qwen3_adapter.Qwen3ASREngineAdapter",
}


def get_asr_adapter_class(engine_type: str) -> Type:
    if not engine_type:
        raise ValueError("ASR engine_type is missing")
    if engine_type not in _ADAPTER_MAP:
        raise ValueError(f"ASR engine_type '{engine_type}' is not supported")

    module_path, class_name = _ADAPTER_MAP[engine_type].rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
