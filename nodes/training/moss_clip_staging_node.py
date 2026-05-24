"""
MOSS clip staging node for unified training workflows.
"""

import os
import re
import sys
import importlib.util
from typing import Dict, List

import folder_paths
import numpy as np
from scipy.io import wavfile

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
MIN_STAGED_CLIP_SECONDS = 0.05


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
                    "tooltip": "Optional training audio clip. Connect multiple AUDIO inputs here and the node will stage every resulting clip for later manifest row construction.",
                },
            )
        raise KeyError(key)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default


def _slugify(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value).strip())
    safe = safe.strip("_")
    return safe or "moss_dataset"


def _iter_audio_batches(waveform):
    if waveform.ndim == 1:
        yield waveform[None, :]
        return
    if waveform.ndim == 2:
        if waveform.shape[0] <= 8:
            yield waveform
            return
        for clip in waveform:
            yield clip[None, :]
        return
    if waveform.ndim != 3:
        raise ValueError(f"Unsupported audio tensor shape for MOSS clip staging: {tuple(waveform.shape)}")
    for clip in waveform:
        if clip.ndim == 1:
            yield clip[None, :]
        else:
            yield clip


def _write_audio_clip(audio_tensor, sample_rate: int, output_path: str):
    clip = audio_tensor.detach().cpu().float()
    if clip.ndim == 1:
        clip = clip.unsqueeze(0)
    if clip.ndim != 2:
        raise ValueError(f"Expected [channels, samples] audio clip, got shape {tuple(clip.shape)}")
    clip = clip.clamp(-1.0, 1.0)
    audio_np = clip.numpy()
    if audio_np.shape[0] <= 8:
        audio_np = audio_np.T
    wavfile.write(output_path, sample_rate, audio_np.astype(np.float32))


class MossClipStagingNode(BaseTTSNode):
    @classmethod
    def NAME(cls):
        return "🎞️ MOSS Clip Staging"

    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = DynamicAudioOptionalInputs(
            {
                "output_subdir": ("STRING", {
                    "default": "tts_audio_suite_training/moss_tts/staged_audio",
                    "tooltip": "Subdirectory inside ComfyUI input/ where staged MOSS training clips will be written."
                }),
                "overwrite": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Overwrite the existing staged clip folder if it already exists."
                }),
                "opt_audio1": (
                    "AUDIO",
                    {
                        "forceInput": True,
                        "tooltip": "Optional training audio clip. Connect one or more AUDIO sources here.",
                    },
                ),
            }
        )
        return {
            "required": {
                "dataset_name": ("STRING", {
                    "default": "MyMossDataset",
                    "tooltip": "Base name for the staged clip set."
                }),
            },
            "optional": optional_inputs,
        }

    RETURN_TYPES = ("MOSS_CLIP_DATASET", "STRING")
    RETURN_NAMES = ("clip_dataset", "dataset_info")
    FUNCTION = "stage_clips"
    CATEGORY = "TTS Audio Suite/🎓 Training"

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
        return [(name, self.normalize_audio_input(audio_input, input_name=name)) for name, audio_input in collected]

    def stage_clips(
        self,
        dataset_name: str,
        output_subdir: str = "",
        overwrite: bool = True,
        opt_audio1=None,
        **kwargs,
    ):
        audio_inputs = self._collect_audio_inputs(opt_audio1=opt_audio1, **kwargs)
        if not audio_inputs:
            raise ValueError("MOSS Clip Staging requires at least one connected AUDIO input")

        dataset_slug = _slugify(dataset_name)
        input_root = folder_paths.get_input_directory()
        subdir = str(output_subdir or "").strip().strip("/\\")
        output_root = os.path.join(input_root, subdir) if subdir else input_root
        dataset_dir = os.path.join(output_root, dataset_slug)

        if os.path.isdir(dataset_dir):
            if overwrite:
                import shutil
                shutil.rmtree(dataset_dir)
            else:
                raise FileExistsError(f"MOSS staged clip folder already exists: {dataset_dir}")
        os.makedirs(dataset_dir, exist_ok=True)

        clips: List[Dict[str, object]] = []
        clip_index = 0
        for input_name, normalized_audio in audio_inputs:
            sample_rate = int(normalized_audio["sample_rate"])
            waveform = normalized_audio["waveform"]
            source_clip_index = 0
            for clip in _iter_audio_batches(waveform):
                source_clip_index += 1
                duration_seconds = float(clip.shape[-1]) / float(sample_rate)
                if duration_seconds < MIN_STAGED_CLIP_SECONDS:
                    continue
                clip_index += 1
                clip_id = f"clip{clip_index:03d}"
                filename = f"{clip_index:05d}_{input_name}_{source_clip_index}.wav"
                clip_path = os.path.join(dataset_dir, filename)
                _write_audio_clip(clip, sample_rate, clip_path)
                clips.append({
                    "clip_id": clip_id,
                    "audio": clip_path,
                    "sample_rate": sample_rate,
                    "duration_seconds": duration_seconds,
                    "source_input": input_name,
                    "source_clip_index": source_clip_index,
                })

        if not clips:
            raise RuntimeError("MOSS Clip Staging produced no clips")

        dataset = {
            "type": "moss_clip_dataset",
            "dataset_name": dataset_name,
            "dataset_dir": dataset_dir,
            "clips": clips,
        }
        info = f"MOSS clip dataset ready: {dataset_name} | {len(clips)} clips"
        print(f"🎞️ {info}")
        return dataset, info


NODE_CLASS_MAPPINGS = {"MossClipStagingNode": MossClipStagingNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MossClipStagingNode": "🎞️ MOSS Clip Staging"}
