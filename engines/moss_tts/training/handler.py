"""
MOSS-TTS training backend for the unified model training node.
"""

from __future__ import annotations

from typing import Any, Dict

from engines.training.base_handler import BaseTrainingHandler
from engines.training.registry import register_training_handler
from engines.moss_tts.training.common import resolve_delay_training_variant


class MossTrainingHandler(BaseTrainingHandler):
    engine_type = "moss_tts"
    artifact_type = "lora_adapter"

    def _shared_settings(self, tts_engine: Any) -> Dict[str, Any]:
        config = self.ensure_engine_type(tts_engine)
        resolve_delay_training_variant(config)
        return {
            "model_variant": config.get("model_variant", "MOSS-TTS"),
            "codec_model": config.get("codec_model", "MOSS-Audio-Tokenizer"),
            "device": str(config.get("device", "auto")),
        }

    def build_default_training_config(self, tts_engine: Any) -> Dict[str, Any]:
        self._shared_settings(tts_engine)
        return {
            "type": "training_config",
            "engine_type": "moss_tts",
            "training_mode": "lora_adapter",
            "epochs": 3,
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 2e-6,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "max_train_steps": 30000,
            "max_grad_norm": 0.5,
            "num_workers": 0,
            "mixed_precision": "bf16",
            "gradient_checkpointing": True,
            "save_steps": 500,
            "eval_steps": 500,
            "log_steps": 10,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "trainable_lora_modules": "mlp",
        }

    def prepare_dataset(self, tts_engine: Any, **kwargs) -> Dict[str, Any]:
        from engines.moss_tts.training.dataset import prepare_moss_training_dataset

        return prepare_moss_training_dataset(self._shared_settings(tts_engine), **kwargs)

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
        from engines.moss_tts.training.trainer import run_moss_training_job

        return run_moss_training_job(
            shared_settings=self._shared_settings(tts_engine),
            dataset_info=training_dataset,
            training_config=training_config,
            output_name=output_name,
            resume=resume,
            overwrite=overwrite,
            continue_from=continue_from,
            node_id=node_id,
        )


register_training_handler("moss_tts", MossTrainingHandler)

