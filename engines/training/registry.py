"""
Lazy registry for engine-specific training backends.
"""

from __future__ import annotations

import importlib
from typing import Dict, Optional, Type

from engines.training.base_handler import BaseTrainingHandler, unpack_tts_engine


_HANDLERS: Dict[str, Type[BaseTrainingHandler]] = {}
_HANDLER_MODULES = {
    "rvc": "engines.rvc.training.handler",
}


def register_training_handler(engine_type: str, handler_cls: Type[BaseTrainingHandler]) -> None:
    _HANDLERS[engine_type] = handler_cls


def _ensure_handler_loaded(engine_type: str) -> None:
    if engine_type in _HANDLERS:
        return
    module_name = _HANDLER_MODULES.get(engine_type)
    if module_name:
        importlib.import_module(module_name)


def get_training_handler(engine_type: str) -> Optional[BaseTrainingHandler]:
    _ensure_handler_loaded(engine_type)
    handler_cls = _HANDLERS.get(engine_type)
    return handler_cls() if handler_cls else None


def get_training_handler_for_engine(tts_engine) -> Optional[BaseTrainingHandler]:
    engine_type, _ = unpack_tts_engine(tts_engine)
    return get_training_handler(engine_type)


__all__ = [
    "get_training_handler",
    "get_training_handler_for_engine",
    "register_training_handler",
]
