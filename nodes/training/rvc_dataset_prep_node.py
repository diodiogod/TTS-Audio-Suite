"""
RVC dataset preparation node for unified model training.
"""

import os
import sys
import importlib.util
import re

from engines.training.registry import get_training_handler

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

OPT_AUDIO_INPUT_PATTERN = re.compile(r"^opt_audio(\d+)$")


class DynamicAudioOptionalInputs(dict):
    @staticmethod
    def _is_dynamic_audio_key(key):
        return isinstance(key, str) and OPT_AUDIO_INPUT_PATTERN.fullmatch(key) is not None

    def __contains__(self, key):
        return super().__contains__(key) or self._is_dynamic_audio_key(key)

    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        if self._is_dynamic_audio_key(key):
            return (
                "AUDIO",
                {
                    "forceInput": True,
                    "tooltip": "Optional training audio clip. Connect multiple AUDIO sources here and the node will collect them into one RVC dataset.",
                },
            )
        raise KeyError(key)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default


class RVCDatasetPrepNode(BaseTTSNode):
    @classmethod
    def NAME(cls):
        return "📦 RVC Dataset Prep"

    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = DynamicAudioOptionalInputs(
            {
                "cpu_workers": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "CPU workers for slicing and feature extraction. 1-4 is usually enough; more is not automatically better, especially on Windows."
                }),
                "chunk_seconds": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 15.0,
                    "step": 0.1,
                    "tooltip": "Target clip length after slicing. RVC usually likes short clips; 3.0s is a good default, and an effective range around 2-8s is normal."
                }),
                "overlap_seconds": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Small overlap helps avoid harsh cuts between slices. Too much overlap just duplicates data and slows prep."
                }),
                "max_volume": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Peak normalization ceiling for extracted clips. 0.95 is a safe default; avoid pushing everything to 1.0 unless you like clipping risk."
                }),
                "mute_ratio": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Extra silence augmentation ratio. Leave this at 0 for normal training; raise it only if the model is over-voicing silence or breathing badly."
                }),
                "reuse_existing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reuse the prepared dataset cache when settings match. Usually keep this on; rebuilding features every run is just wasted time."
                }),
                "opt_audio1": (
                    "AUDIO",
                    {
                        "forceInput": True,
                        "tooltip": "Optional training audio clip. Connect one or more AUDIO sources here for quick in-graph tests, or combine them with a dataset path.",
                    },
                ),
            }
        )
        return {
            "required": {
                "TTS_engine": ("TTS_ENGINE", {
                    "tooltip": "Connect the RVC engine here because dataset prep also needs HuBERT and pitch-extraction settings, not just the later training node."
                }),
                "model_name": ("STRING", {
                    "default": "MyVoice",
                    "tooltip": "Base name for the trained RVC voice. Changing only this should not force dataset re-prep when the actual data/settings stay the same."
                }),
                "dataset_source": ("STRING", {
                    "default": "",
                    "tooltip": "Optional folder, zip, or single audio file path. Relative paths resolve from ComfyUI input/ and input/datasets/. Clean single-speaker speech matters more than having one huge raw recording. Leave empty if you only use opt_audio inputs."
                }),
                "training_sample_rate": (["32k", "40k", "48k"], {
                    "default": "40k",
                    "tooltip": "Target training sample rate. 40k is the normal speech default and usually the right place to start."
                }),
            },
            "optional": optional_inputs,
        }

    RETURN_TYPES = ("TRAINING_DATASET", "STRING")
    RETURN_NAMES = ("training_dataset", "dataset_info")
    FUNCTION = "prepare_dataset"
    CATEGORY = "TTS Audio Suite/🎓 Training"

    def prepare_dataset(
        self,
        TTS_engine,
        model_name,
        dataset_source,
        training_sample_rate="40k",
        cpu_workers=1,
        chunk_seconds=3.0,
        overlap_seconds=0.3,
        max_volume=0.95,
        mute_ratio=0.0,
        reuse_existing=True,
        opt_audio1=None,
        **kwargs,
    ):
        handler = get_training_handler("rvc")
        if handler is None:
            raise RuntimeError("RVC training backend is not available")

        audio_inputs = self._collect_audio_inputs(opt_audio1=opt_audio1, **kwargs)
        dataset = handler.prepare_dataset(
            TTS_engine,
            dataset_source=dataset_source,
            model_name=model_name,
            sample_rate=training_sample_rate,
            cpu_workers=cpu_workers,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            max_volume=max_volume,
            mute_ratio=mute_ratio,
            reuse_existing=reuse_existing,
            audio_inputs=audio_inputs,
        )
        source_summary = dataset.get("source_summary", "unknown")
        info = (
            f"RVC dataset ready: {dataset['model_name']} | {dataset['sample_rate']} | "
            f"f0={dataset['f0_method']} | files={dataset.get('file_count', 0)} | sources={source_summary}"
        )
        return dataset, info

    def _collect_audio_inputs(self, opt_audio1=None, **kwargs):
        collected = []
        if opt_audio1 is not None:
            collected.append(("opt_audio1", opt_audio1))

        for key, value in kwargs.items():
            if value is None:
                continue
            match = OPT_AUDIO_INPUT_PATTERN.fullmatch(key)
            if match is not None:
                collected.append((key, value))

        collected.sort(key=lambda item: int(OPT_AUDIO_INPUT_PATTERN.fullmatch(item[0]).group(1)))
        return [self.normalize_audio_input(audio_input, input_name=name) for name, audio_input in collected]


NODE_CLASS_MAPPINGS = {"RVCDatasetPrepNode": RVCDatasetPrepNode}
NODE_DISPLAY_NAME_MAPPINGS = {"RVCDatasetPrepNode": "📦 RVC Dataset Prep"}
