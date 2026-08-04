"""VoxCPM2 backend for the unified model-training node."""

from __future__ import annotations

from typing import Any, Dict

from engines.training.base_handler import BaseTrainingHandler
from engines.training.registry import register_training_handler

from .common import resolve_voxcpm2_settings


class VoxCPMTrainingHandler(BaseTrainingHandler):
    engine_type = "voxcpm"
    artifact_type = "lora_adapter"

    def _settings(self, tts_engine: Any) -> Dict[str, Any]:
        return resolve_voxcpm2_settings(self.ensure_engine_type(tts_engine))

    def build_default_training_config(self, tts_engine: Any) -> Dict[str, Any]:
        self._settings(tts_engine)
        return {
            "type": "training_config",
            "engine_type": "voxcpm",
            "training_mode": "lora_adapter",
            "max_train_steps": 1000,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "max_grad_norm": 1.0,
            "max_batch_tokens": 8192,
            "num_workers": 0,
            "cuda_device_index": -1,
            "save_steps": 500,
            "eval_steps": 500,
            "log_steps": 10,
            "lora_enable_lm": True,
            "lora_enable_dit": True,
            "lora_enable_proj": False,
            "lora_r": 32,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
        }

    def prepare_dataset(self, tts_engine: Any, **kwargs) -> Dict[str, Any]:
        from .dataset import prepare_voxcpm_dataset

        return prepare_voxcpm_dataset(self._settings(tts_engine), **kwargs)

    def train(
        self,
        tts_engine: Any,
        training_dataset: Dict[str, Any],
        training_config: Dict[str, Any],
        output_name: str = "",
        resume: bool = False,
        overwrite: bool = False,
        continue_from: Any = None,
        node_id: str = "",
    ) -> Dict[str, Any]:
        from .trainer import run_voxcpm_training_job

        return run_voxcpm_training_job(
            shared_settings=self._settings(tts_engine),
            dataset_info=training_dataset,
            training_config=training_config,
            output_name=output_name,
            resume=resume,
            overwrite=overwrite,
            continue_from=continue_from,
            node_id=node_id,
        )


register_training_handler("voxcpm", VoxCPMTrainingHandler)
