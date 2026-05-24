"""
MOSS manifest builder node for unified training workflows.
"""

import json
import os
import re
import sys
import importlib.util

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
                    "tooltip": "Optional training audio clip. Connect multiple AUDIO inputs here and the node will build one manifest row per resulting clip.",
                },
            )
        raise KeyError(key)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default


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
        raise ValueError(f"Unsupported audio tensor shape for MOSS manifest builder: {tuple(waveform.shape)}")
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


class MossManifestBuilderNode(BaseTTSNode):
    @classmethod
    def NAME(cls):
        return "📝 MOSS Manifest Builder"

    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = DynamicAudioOptionalInputs(
            {
                "output_subdir": ("STRING", {
                    "default": "tts_audio_suite_training/moss_tts/manifests",
                    "tooltip": "Subdirectory inside ComfyUI input/ where the manifest file and staged audio folder will be written."
                }),
                "overwrite": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Overwrite the existing manifest file if it already exists."
                }),
                "text_lines": ("STRING", {
                    "default": "Hello there, this is a training sample.\nHow are you today?",
                    "multiline": True,
                    "tooltip": (
                        "Audio mode only. One transcript line per resulting clip, in the same order as the connected AUDIO inputs.\n"
                        "If an AUDIO input contains multiple clips, each clip still needs its own line."
                    )
                }),
                "manifest_jsonl": ("STRING", {
                    "default": (
                        '{"audio":"./data/utt0001.wav","text":"Hello there, this is a training sample.","language":"en"}\n'
                        '{"audio":"./data/utt0002.wav","text":"How are you today?","ref_audio":"./data/ref.wav","language":"en"}'
                    ),
                    "multiline": True,
                    "tooltip": (
                        "Raw JSONL mode only. One JSON object per line in the official MOSS finetuning manifest style.\n"
                        "Use this if you want full manual control."
                    )
                }),
                "language": ("STRING", {
                    "default": "",
                    "tooltip": "Optional default language applied to every generated record in audio mode, or filled into raw JSONL records if they do not already define it."
                }),
                "instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional default instruction applied to every record unless a raw JSONL record already defines one."
                }),
                "quality": ("STRING", {
                    "default": "",
                    "tooltip": "Optional default quality field applied to every record."
                }),
                "sound_event": ("STRING", {
                    "default": "",
                    "tooltip": "Optional default sound_event field applied to every record."
                }),
                "ambient_sound": ("STRING", {
                    "default": "",
                    "tooltip": "Optional default ambient_sound field applied to every record."
                }),
                "duration_tokens": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Optional default MOSS tokens field applied to every record. 0 disables it."
                }),
                "opt_audio1": (
                    "AUDIO",
                    {
                        "forceInput": True,
                        "tooltip": "Optional training audio clip. If any opt_audio input is connected, the node switches to audio mode automatically and builds records from the audio clips instead of raw JSONL.",
                    },
                ),
            }
        )
        return {
            "required": {
                "mode": (["Audio Clips + Text Lines", "Raw JSONL"], {
                    "default": "Audio Clips + Text Lines",
                    "tooltip": "Choose whether to build the manifest from connected audio clips plus transcript lines, or from raw JSONL text."
                }),
                "manifest_name": ("STRING", {
                    "default": "moss_train.jsonl",
                    "tooltip": "Output manifest filename. .jsonl will be appended if missing."
                }),
            },
            "optional": optional_inputs,
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("manifest_path", "manifest_info")
    FUNCTION = "build_manifest"
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
        return [self.normalize_audio_input(audio_input, input_name=name) for name, audio_input in collected]

    @staticmethod
    def _apply_default_fields(record, *, language="", instruction="", quality="", sound_event="", ambient_sound="", duration_tokens=0):
        if language and not record.get("language"):
            record["language"] = language
        if instruction and not record.get("instruction"):
            record["instruction"] = instruction
        if quality and not record.get("quality"):
            record["quality"] = quality
        if sound_event and not record.get("sound_event"):
            record["sound_event"] = sound_event
        if ambient_sound and not record.get("ambient_sound"):
            record["ambient_sound"] = ambient_sound
        if int(duration_tokens or 0) > 0 and record.get("tokens") in (None, "", 0):
            record["tokens"] = int(duration_tokens)
        return record

    def _build_from_jsonl(
        self,
        raw_text: str,
        *,
        language="",
        instruction="",
        quality="",
        sound_event="",
        ambient_sound="",
        duration_tokens=0,
    ):
        if not raw_text:
            raise ValueError("manifest_jsonl is empty")

        records = []
        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except Exception as e:
                raise ValueError(f"MOSS manifest line {line_number} is not valid JSON: {e}") from e
            if not isinstance(record, dict):
                raise ValueError(f"MOSS manifest line {line_number} must be a JSON object")
            audio_path = record.get("audio")
            if not isinstance(audio_path, str) or not audio_path.strip():
                raise ValueError(f"MOSS manifest line {line_number} is missing a valid 'audio' field")
            self._apply_default_fields(
                record,
                language=language,
                instruction=instruction,
                quality=quality,
                sound_event=sound_event,
                ambient_sound=ambient_sound,
                duration_tokens=duration_tokens,
            )
            if not any(record.get(key) not in (None, "", []) for key in ("text", "instruction", "ambient_sound")):
                raise ValueError(
                    f"MOSS manifest line {line_number} must include at least one prompt field such as "
                    "'text', 'instruction', or 'ambient_sound'"
                )
            records.append(record)
        if not records:
            raise ValueError("MOSS manifest contains no valid records")
        return records

    def _build_from_audio(
        self,
        manifest_stem: str,
        output_dir: str,
        audio_inputs,
        text_lines: str,
        *,
        language="",
        instruction="",
        quality="",
        sound_event="",
        ambient_sound="",
        duration_tokens=0,
    ):
        if not audio_inputs:
            raise ValueError("Audio mode requires at least one connected opt_audio input")

        transcript_lines = [line.strip() for line in str(text_lines or "").splitlines() if line.strip()]
        if not transcript_lines:
            raise ValueError("Audio mode requires non-empty text_lines")

        audio_dir = os.path.join(output_dir, f"{manifest_stem}_assets")
        os.makedirs(audio_dir, exist_ok=True)

        staged_audio_paths = []
        for input_index, normalized_audio in enumerate(audio_inputs, start=1):
            sample_rate = int(normalized_audio["sample_rate"])
            waveform = normalized_audio["waveform"]
            clip_counter = 0
            for clip in _iter_audio_batches(waveform):
                clip_counter += 1
                output_name = f"{len(staged_audio_paths)+1:05d}_audio{input_index}_{clip_counter}.wav"
                output_path = os.path.join(audio_dir, output_name)
                _write_audio_clip(clip, sample_rate, output_path)
                staged_audio_paths.append(output_path)

        if len(transcript_lines) != len(staged_audio_paths):
            raise ValueError(
                f"Audio mode transcript mismatch: got {len(staged_audio_paths)} clip(s) but {len(transcript_lines)} non-empty text line(s). "
                "You need one text line per resulting clip."
            )

        records = []
        for audio_path, text in zip(staged_audio_paths, transcript_lines):
            record = {
                "audio": audio_path,
                "text": text,
            }
            self._apply_default_fields(
                record,
                language=language,
                instruction=instruction,
                quality=quality,
                sound_event=sound_event,
                ambient_sound=ambient_sound,
                duration_tokens=duration_tokens,
            )
            records.append(record)
        return records

    def build_manifest(
        self,
        mode: str,
        manifest_name: str,
        output_subdir: str = "",
        overwrite: bool = True,
        text_lines: str = "",
        manifest_jsonl: str = "",
        language: str = "",
        instruction: str = "",
        quality: str = "",
        sound_event: str = "",
        ambient_sound: str = "",
        duration_tokens: int = 0,
        opt_audio1=None,
        **kwargs,
    ):
        filename = str(manifest_name or "").strip() or "moss_train.jsonl"
        if not filename.endswith(".jsonl"):
            filename += ".jsonl"
        manifest_stem = filename[:-6] if filename.endswith(".jsonl") else filename

        input_root = folder_paths.get_input_directory()
        subdir = str(output_subdir or "").strip().strip("/\\")
        output_dir = os.path.join(input_root, subdir) if subdir else input_root
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, filename)

        if os.path.exists(manifest_path) and not overwrite:
            raise FileExistsError(f"MOSS manifest already exists: {manifest_path}")

        audio_inputs = self._collect_audio_inputs(opt_audio1=opt_audio1, **kwargs)
        if mode == "Audio Clips + Text Lines":
            records = self._build_from_audio(
                manifest_stem,
                output_dir,
                audio_inputs,
                text_lines,
                language=language.strip(),
                instruction=instruction.strip(),
                quality=quality.strip(),
                sound_event=sound_event.strip(),
                ambient_sound=ambient_sound.strip(),
                duration_tokens=duration_tokens,
            )
            mode_summary = "audio"
        else:
            records = self._build_from_jsonl(
                str(manifest_jsonl or "").strip(),
                language=language.strip(),
                instruction=instruction.strip(),
                quality=quality.strip(),
                sound_event=sound_event.strip(),
                ambient_sound=ambient_sound.strip(),
                duration_tokens=duration_tokens,
            )
            mode_summary = "raw-jsonl"

        with open(manifest_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        info = f"MOSS manifest ready: {os.path.basename(manifest_path)} | {len(records)} records | mode={mode_summary}"
        print(f"📝 {info}")
        return manifest_path, info


NODE_CLASS_MAPPINGS = {"MossManifestBuilderNode": MossManifestBuilderNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MossManifestBuilderNode": "📝 MOSS Manifest Builder"}
