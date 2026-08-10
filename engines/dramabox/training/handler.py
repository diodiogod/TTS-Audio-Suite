"""DramaBox backend for the unified model-training node."""

from __future__ import annotations

from typing import Any, Dict

from engines.training.base_handler import BaseTrainingHandler
from engines.training.registry import register_training_handler


class DramaBoxTrainingHandler(BaseTrainingHandler):
    engine_type = "dramabox"
    artifact_type = "lora_adapter"

    def _shared_settings(self, tts_engine: Any) -> Dict[str, Any]:
        config = self.ensure_engine_type(tts_engine)
        return {
            "model_name": config.get("model_name", "DramaBox"),
            "device": str(config.get("device", "auto")),
            "precision": str(config.get("precision", "auto")),
        }

    def build_default_training_config(self, tts_engine: Any) -> Dict[str, Any]:
        self._shared_settings(tts_engine)
        return {
            "type": "training_config",
            "engine_type": "dramabox",
            "training_mode": "audio_lora",
            "base_model": "dev",
            "steps": 10000,
            "learning_rate": 1e-4,
            "lr_scheduler": "cosine",
            "warmup_steps": 500,
            "batch_size": 1,
            "grad_accum": 4,
            "max_grad_norm": 1.0,
            "save_every": 500,
            "log_every": 10,
            "seed": 42,
            "lora_rank": 128,
            "lora_alpha": 128,
            "lora_dropout": 0.1,
            "ref_ratio": 0.3,
            "max_ref_tokens": 200,
            "text_dropout": 0.4,
            "preprocess_batch_size": 8,
            "validation_config": "",
            "validation_gpu": "",
            "dry_run": False,
        }

    def prepare_dataset(self, tts_engine: Any, **kwargs) -> Dict[str, Any]:
        from .dataset import prepare_dramabox_dataset

        return prepare_dramabox_dataset(self._shared_settings(tts_engine), **kwargs)

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
        from .trainer import run_dramabox_training_job

        return run_dramabox_training_job(
            shared_settings=self._shared_settings(tts_engine),
            dataset_info=training_dataset,
            training_config=training_config,
            output_name=output_name,
            resume=resume,
            overwrite=overwrite,
            continue_from=continue_from,
            node_id=node_id,
        )


register_training_handler("dramabox", DramaBoxTrainingHandler)
