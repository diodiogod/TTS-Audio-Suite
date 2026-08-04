"""Build and validate VoxCPM2 audio/text training datasets."""

import importlib.util
import json
import math
import os
import re
import sys

import folder_paths
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

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


_AUDIO_KEY = re.compile(r"^opt_audio(\d+)$")


class _DynamicAudioInputs(dict):
    def __contains__(self, key):
        return super().__contains__(key) or (
            isinstance(key, str) and _AUDIO_KEY.fullmatch(key) is not None
        )

    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        if isinstance(key, str) and _AUDIO_KEY.fullmatch(key):
            return ("AUDIO", {"forceInput": True})
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def _slugify(value):
    safe = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in str(value or "").strip()
    ).strip("_")
    return safe or "voxcpm2_dataset"


def _iter_clips(waveform):
    if waveform.ndim == 1:
        yield waveform
    elif waveform.ndim == 2:
        yield waveform
    elif waveform.ndim == 3:
        yield from waveform
    else:
        raise ValueError(f"Unsupported VoxCPM training audio shape: {tuple(waveform.shape)}")


def _stage_clip(audio, sample_rate, path):
    tensor = audio.detach().cpu().float()
    if tensor.ndim == 2:
        tensor = tensor.mean(dim=0)
    elif tensor.ndim != 1:
        raise ValueError(f"Expected one VoxCPM audio clip, got {tuple(tensor.shape)}")
    values = tensor.clamp(-1, 1).numpy()
    if int(sample_rate) != 16000:
        divisor = math.gcd(int(sample_rate), 16000)
        values = resample_poly(
            values,
            16000 // divisor,
            int(sample_rate) // divisor,
        )
    if len(values) < 800:
        raise ValueError("VoxCPM training clips must be at least 0.05 seconds")
    pcm16 = np.round(np.clip(values, -1, 1) * 32767).astype(np.int16)
    wavfile.write(path, 16000, pcm16)
    return len(pcm16) / 16000.0


class VoxCPMDatasetPrepNode(BaseTTSNode):
    @classmethod
    def NAME(cls):
        return "📦 VoxCPM2 Dataset Prep"

    @classmethod
    def INPUT_TYPES(cls):
        optional = _DynamicAudioInputs(
            {
                "dataset_source": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Existing official-format JSONL manifest. Each line needs "
                        '{"audio":"clip.wav","text":"transcript"}. Leave blank to use connected AUDIO inputs.'
                    ),
                }),
                "text_lines": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "When AUDIO inputs are connected, provide exactly one transcript "
                        "line per resulting clip, in input/batch order."
                    ),
                }),
                "validation_source": ("STRING", {
                    "default": "",
                    "tooltip": "Optional separate official-format validation JSONL.",
                }),
                "validation_split": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Automatic holdout ratio. Set 0 to train without validation.",
                }),
                "split_seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31 - 1,
                    "step": 1,
                }),
                "reuse_existing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reuse matching validated manifests.",
                }),
                "opt_audio1": ("AUDIO", {
                    "forceInput": True,
                    "tooltip": "Optional training audio. More AUDIO sockets are added dynamically.",
                }),
            }
        )
        return {
            "required": {
                "TTS_engine": ("TTS_ENGINE", {
                    "tooltip": "Connect a VoxCPM engine configured with VoxCPM2.",
                }),
                "model_name": ("STRING", {
                    "default": "MyVoxCPM2LoRA",
                    "tooltip": "Dataset and final adapter name.",
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("TRAINING_DATASET", "STRING")
    RETURN_NAMES = ("training_dataset", "dataset_info")
    FUNCTION = "prepare_dataset"
    CATEGORY = "TTS Audio Suite/🎓 Training"

    def _collect_audio(self, opt_audio1=None, **kwargs):
        values = []
        if opt_audio1 is not None:
            values.append(("opt_audio1", opt_audio1))
        for key, value in kwargs.items():
            if value is not None and _AUDIO_KEY.fullmatch(str(key)):
                values.append((key, value))
        values.sort(key=lambda item: int(_AUDIO_KEY.fullmatch(item[0]).group(1)))
        return values

    def _build_manifest(self, model_name, text_lines, audio_inputs):
        transcripts = str(text_lines or "").splitlines()
        normalized = []
        for input_name, value in audio_inputs:
            audio = self.normalize_audio_input(value, input_name=input_name)
            for clip in _iter_clips(audio["waveform"]):
                normalized.append((clip, int(audio["sample_rate"])))
        if len(transcripts) != len(normalized):
            raise ValueError(
                f"VoxCPM transcript count mismatch: {len(normalized)} audio clip(s), "
                f"{len(transcripts)} text line(s)."
            )
        if any(not line.strip() for line in transcripts):
            raise ValueError("VoxCPM training transcripts cannot be blank")

        root = os.path.join(
            folder_paths.get_input_directory(),
            "tts_audio_suite_training",
            "voxcpm",
            _slugify(model_name),
        )
        os.makedirs(root, exist_ok=True)
        records = []
        for index, ((clip, sample_rate), text) in enumerate(
            zip(normalized, transcripts), 1
        ):
            path = os.path.join(root, f"{index:05d}.wav")
            duration = _stage_clip(clip, sample_rate, path)
            records.append(
                {"audio": path, "text": text.strip(), "duration": duration}
            )
        manifest = os.path.join(root, "manifest.jsonl")
        with open(manifest, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return manifest

    def prepare_dataset(
        self,
        TTS_engine,
        model_name,
        dataset_source="",
        text_lines="",
        validation_source="",
        validation_split=0.05,
        split_seed=42,
        reuse_existing=True,
        opt_audio1=None,
        **kwargs,
    ):
        audio_inputs = self._collect_audio(opt_audio1=opt_audio1, **kwargs)
        source = str(dataset_source or "").strip()
        if audio_inputs:
            if source:
                raise ValueError(
                    "Use either dataset_source or connected AUDIO inputs, not both"
                )
            source = self._build_manifest(model_name, text_lines, audio_inputs)
        if not source:
            raise ValueError(
                "VoxCPM Dataset Prep needs dataset_source or at least one AUDIO input"
            )

        handler = get_training_handler("voxcpm")
        dataset = handler.prepare_dataset(
            TTS_engine,
            dataset_source=source,
            model_name=model_name,
            validation_source=validation_source,
            validation_split=validation_split,
            split_seed=split_seed,
            reuse_existing=reuse_existing,
        )
        info = (
            f"VoxCPM2 dataset ready: {dataset['model_name']} | "
            f"train={dataset['train_records']} | val={dataset['val_records']} | "
            "AudioVAE input=16 kHz"
        )
        return dataset, info


NODE_CLASS_MAPPINGS = {"VoxCPMDatasetPrepNode": VoxCPMDatasetPrepNode}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VoxCPMDatasetPrepNode": "📦 VoxCPM2 Dataset Prep"
}
