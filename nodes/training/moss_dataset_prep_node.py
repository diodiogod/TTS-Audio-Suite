"""
MOSS dataset preparation node for unified model training.
"""

import os
import sys
import importlib.util

from engines.training.registry import get_training_handler

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


class MossDatasetPrepNode(BaseTTSNode):
    @classmethod
    def NAME(cls):
        return "📦 MOSS Dataset Prep"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "TTS_engine": ("TTS_ENGINE", {
                    "tooltip": "Connect a MOSS engine here. The first training slice only supports Delay 8B LoRA training."
                }),
                "model_name": ("STRING", {
                    "default": "MyMossLoRA",
                    "tooltip": "Base name for the trained MOSS LoRA adapter folder."
                }),
                "dataset_source": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Path to the main MOSS manifest JSONL.\n"
                        "This is your training set manifest: one JSON row per clip.\n"
                        "In the normal workflow, connect the manifest path produced by MOSS Dataset Rows here."
                    )
                }),
            },
            "optional": {
                "validation_source": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Optional second manifest JSONL used only for validation.\n"
                        "\n"
                        "Use this if you already prepared a separate holdout set.\n"
                        "Example:\n"
                        "• dataset_source = your main training manifest\n"
                        "• validation_source = a smaller separate manifest reserved for evaluation\n"
                        "\n"
                        "If you leave this blank, the node will automatically split dataset_source into train + validation for you."
                    )
                }),
                "validation_split": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": (
                        "Only used when validation_source is blank.\n"
                        "Example: 0.05 means keep about 5% of dataset_source for validation and use the rest for training."
                    )
                }),
                "split_seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31 - 1,
                    "step": 1,
                    "tooltip": "Seed for the automatic train/validation split, so the same manifest gets split the same way every time."
                }),
                "prep_batch_size": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Batch size for audio-code extraction during dataset prep."
                }),
                "n_vq": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Optional codec codebook count override. 0 uses the model default."
                }),
                "encode_reference_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Pre-encode reference audio fields during dataset prep.\n"
                        "Leave this on if your manifest uses ref_audio/reference_audio.\n"
                        "If you are doing normal audio+transcript training with no reference clips, this setting does not matter much."
                    )
                }),
                "reuse_existing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reuse a matching prepared dataset cache instead of re-encoding audio codes every run."
                }),
            },
        }

    RETURN_TYPES = ("TRAINING_DATASET", "STRING")
    RETURN_NAMES = ("training_dataset", "dataset_info")
    FUNCTION = "prepare_dataset"
    CATEGORY = "TTS Audio Suite/🎓 Training"

    def prepare_dataset(self, TTS_engine, model_name, dataset_source, **kwargs):
        handler = get_training_handler("moss_tts")
        if handler is None:
            raise RuntimeError("MOSS training backend is not available")

        dataset = handler.prepare_dataset(
            TTS_engine,
            dataset_source=dataset_source,
            model_name=model_name,
            **kwargs,
        )
        info = (
            f"MOSS dataset ready: {dataset['model_name']} | "
            f"train={dataset['train_records']} | val={dataset['val_records']} | "
            f"sources={dataset.get('source_summary', 'unknown')}"
        )
        return dataset, info


NODE_CLASS_MAPPINGS = {"MossDatasetPrepNode": MossDatasetPrepNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MossDatasetPrepNode": "📦 MOSS Dataset Prep"}
