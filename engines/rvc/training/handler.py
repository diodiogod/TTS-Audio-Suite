"""
RVC training backend for the unified model training node.
"""

from __future__ import annotations

from typing import Any, Dict

from engines.training.base_handler import BaseTrainingHandler
from engines.training.registry import register_training_handler


class RVCTrainingHandler(BaseTrainingHandler):
    engine_type = "rvc"
    artifact_type = "voice_model"

    def _shared_settings(self, tts_engine: Any) -> Dict[str, Any]:
        config = self.ensure_engine_type(tts_engine)
        return {
            "hubert_model": config.get("hubert_model", "content-vec-best"),
            "hubert_path": config.get("hubert_path"),
            "f0_method": config.get("f0_method", "rmvpe"),
            "crepe_hop_length": int(config.get("crepe_hop_length", 160)),
            "device": str(config.get("device", "cpu")),
        }

    def build_default_training_config(self, tts_engine: Any) -> Dict[str, Any]:
        shared = self._shared_settings(tts_engine)
        return {
            "type": "training_config",
            "engine_type": "rvc",
            "training_mode": "voice_model",
            "gpu_ids": "",
            "epochs": 100,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "fp16_run": shared["device"].startswith("cuda"),
            "save_every_epoch": 5,
            "num_workers": 1,
            "cache_data_in_gpu": shared["device"].startswith("cuda"),
            "max_checkpoints": 1,
            "save_every_weights": False,
            "train_index": True,
            "save_best_model": True,
            "best_model_threshold": 30,
            "log_every_epoch": 1.0,
            "pretrained_generator": "auto",
            "pretrained_discriminator": "auto",
            "c_adv": 1.0,
            "c_mel": 45.0,
            "c_kl": 1.0,
            "c_fm": 2.0,
            "c_tefs": 0.0,
            "c_hd": 0.0,
            "c_tsi": 0.0,
            "c_gp": 0.0,
            "use_multiscale": False,
            "use_balancer": False,
            "use_pareto": False,
            "fast_mode": False,
        }

    def prepare_dataset(self, tts_engine: Any, **kwargs) -> Dict[str, Any]:
        from engines.rvc.training.dataset import prepare_rvc_training_dataset

        return prepare_rvc_training_dataset(self._shared_settings(tts_engine), **kwargs)

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
        from engines.rvc.training.trainer import run_rvc_training_job

        return run_rvc_training_job(
            shared_settings=self._shared_settings(tts_engine),
            dataset_info=training_dataset,
            training_config=training_config,
            output_name=output_name,
            resume=resume,
            overwrite=overwrite,
            continue_from=continue_from,
            node_id=node_id,
        )


register_training_handler("rvc", RVCTrainingHandler)
