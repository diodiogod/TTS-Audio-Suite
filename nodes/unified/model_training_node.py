"""
Unified model training router node.
"""

import os
import sys
import importlib.util

from engines.training.base_handler import unpack_tts_engine
from engines.training.registry import get_training_handler
from utils.models.engine_registry import engine_supports_training

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)
BaseTTSNode = base_module.BaseTTSNode


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")


class UnifiedModelTrainingNode(BaseTTSNode):
    @classmethod
    def NAME(cls):
        return "🎓 Model Training"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "TTS_engine": ("TTS_ENGINE", {
                    "tooltip": "Engine configuration used to route to the correct training backend. For RVC, only the shared training-relevant settings matter here, not inference-only knobs."
                }),
                "training_dataset": ("TRAINING_DATASET", {
                    "tooltip": "Prepared dataset payload from an engine-specific dataset prep node. Bad data quality will hurt you more than fancy training settings."
                }),
            },
            "optional": {
                "training_config": ("TRAINING_CONFIG", {
                    "tooltip": "Engine-specific training config. If omitted, backend defaults are used, but explicit config is better if you care about checkpoint cadence, resume behavior, or reproducibility."
                }),
                "continue_from": (any_typ, {
                    "tooltip": "Warm-start a new run from an existing finished model or artifacts. For RVC this is not exact resume; generator weights continue, but optimizer/discriminator state starts fresh."
                }),
                "output_name": ("STRING", {
                    "default": "",
                    "tooltip": "Optional exported model name override. Blank uses the dataset model name."
                }),
                "resume": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Exact resume from saved RVC training checkpoints for the latest compatible RVC job with the same output name, dataset, and sample rate. This only works if real G_*.pth and D_*.pth checkpoints exist."
                }),
                "overwrite": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reuse the resolved output name instead of making a unique suffix. Good for deliberate replacement, bad for archival. Do not use this with resume."
                }),
            },
            "hidden": {
                "node_id": ("STRING", {"default": "0"}),
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("TRAINING_ARTIFACTS", "STRING")
    RETURN_NAMES = ("training_artifacts", "training_info")
    FUNCTION = "train_model"
    CATEGORY = "TTS Audio Suite/🎓 Training"

    def train_model(
        self,
        TTS_engine,
        training_dataset,
        training_config=None,
        continue_from=None,
        output_name="",
        resume=False,
        overwrite=False,
        node_id="",
        unique_id="",
    ):
        engine_type, _ = unpack_tts_engine(TTS_engine)
        if not engine_supports_training(engine_type):
            raise ValueError(f"Engine '{engine_type}' does not support training")

        if not training_dataset or not isinstance(training_dataset, dict):
            raise ValueError("training_dataset must be a TRAINING_DATASET payload")
        dataset_engine = training_dataset.get("engine_type")
        if dataset_engine and dataset_engine != engine_type:
            raise ValueError(
                f"training_dataset engine_type '{dataset_engine}' does not match '{engine_type}'"
            )

        handler = get_training_handler(engine_type)
        if handler is None:
            raise RuntimeError(f"No training handler registered for engine '{engine_type}'")

        if training_config is None:
            training_config = handler.build_default_training_config(TTS_engine)
        elif not isinstance(training_config, dict):
            raise ValueError("training_config must be a TRAINING_CONFIG payload")

        config_engine = training_config.get("engine_type")
        if config_engine and config_engine != engine_type:
            raise ValueError(
                f"training_config engine_type '{config_engine}' does not match '{engine_type}'"
            )

        artifacts = handler.train(
            TTS_engine,
            training_dataset=training_dataset,
            training_config=training_config,
            output_name=output_name,
            resume=resume,
            overwrite=overwrite,
            continue_from=continue_from,
            node_id=node_id or unique_id,
        )
        info = artifacts.get("summary") or f"{engine_type} training complete"
        return artifacts, info


NODE_CLASS_MAPPINGS = {"UnifiedModelTrainingNode": UnifiedModelTrainingNode}
NODE_DISPLAY_NAME_MAPPINGS = {"UnifiedModelTrainingNode": "🎓 Model Training"}
