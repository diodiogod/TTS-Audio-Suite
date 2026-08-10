"""DramaBox dataset normalization and official-preprocessor node."""

import importlib.util
import os
import sys

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


class DramaBoxDatasetPrepNode(BaseTTSNode):
    @classmethod
    def NAME(cls):
        return "📦 DramaBox Dataset Prep"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "TTS_engine": ("TTS_ENGINE", {
                    "tooltip": "Connect a DramaBox engine. Its selected model supplies the official transformer, audio components, and Gemma paths.",
                }),
                "model_name": ("STRING", {
                    "default": "MyDramaBoxLoRA",
                    "tooltip": "Name used for the prepared dataset and eventual managed LoRA adapter.",
                }),
                "dataset_source": ("STRING", {
                    "default": "",
                    "tooltip": "JSONL/JSON manifest, TSV, gemini_synthetic index, or libriheavy index. Manifest rows should contain audio_filepath/audio_path and text/transcript.",
                }),
                "dataset_type": (["manifest", "tsv", "gemini_synthetic", "libriheavy"], {
                    "default": "manifest",
                    "tooltip": "Input format. The suite converts every format into the official ~-delimited speaker index used by the trainer.",
                }),
            },
            "optional": {
                "audio_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Base folder for relative audio paths. Blank resolves paths relative to the dataset file.",
                }),
                "min_duration": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.1,
                    "max": 60.0,
                    "step": 0.1,
                    "tooltip": "Minimum clip duration passed to the official preprocessor.",
                }),
                "max_duration": ("FLOAT", {
                    "default": 20.0,
                    "min": 0.5,
                    "max": 120.0,
                    "step": 0.5,
                    "tooltip": "Maximum clip duration passed to the official preprocessor.",
                }),
                "reuse_existing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reuse a matching normalized index and already-preprocessed cache when available.",
                }),
                "preprocess_now": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run the official Gemma/audio-VAE preprocessing now. Turn this off to prepare only the CPU-side index and let Model Training preprocess later.",
                }),
                "dry_run": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "CPU-safe index-only mode. No model download, Gemma load, or CUDA preprocessing is started.",
                }),
            },
        }

    RETURN_TYPES = ("TRAINING_DATASET", "STRING")
    RETURN_NAMES = ("training_dataset", "dataset_info")
    FUNCTION = "prepare_dataset"
    CATEGORY = "TTS Audio Suite/🎓 Training"

    def prepare_dataset(self, TTS_engine, model_name, dataset_source, dataset_type, **kwargs):
        handler = get_training_handler("dramabox")
        if handler is None:
            raise RuntimeError("DramaBox training backend is not available")
        dataset = handler.prepare_dataset(
            TTS_engine,
            dataset_source=dataset_source,
            model_name=model_name,
            dataset_type=dataset_type,
            **kwargs,
        )
        info = (
            f"DramaBox dataset ready: {dataset['model_name']} | "
            f"{dataset['train_records']} clips | "
            f"{len(dataset['speakers'])} speaker(s) | "
            f"preprocessed={dataset.get('preprocessed', False)}"
        )
        print(f"📦 {info}")
        return dataset, info


NODE_CLASS_MAPPINGS = {"DramaBoxDatasetPrepNode": DramaBoxDatasetPrepNode}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DramaBoxDatasetPrepNode": "📦 DramaBox Dataset Prep"
}
