"""
MOSS dataset rows node for turning staged clips into a training manifest.
"""

import json
import os
import sys
import importlib.util
from typing import Dict, List

import folder_paths

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


def _parse_required_lines(raw_text: str, expected_count: int, field_name: str) -> List[str]:
    lines = str(raw_text or "").splitlines()
    if len(lines) != expected_count:
        raise ValueError(
            f"MOSS {field_name} line count mismatch: expected {expected_count} line(s), got {len(lines)}. "
            "You need exactly one line per staged clip."
        )
    return [line.strip() for line in lines]


def _parse_optional_lines(raw_text: str, expected_count: int, field_name: str) -> List[str]:
    lines = str(raw_text or "").splitlines()
    if len(lines) > expected_count:
        raise ValueError(
            f"MOSS {field_name} line count mismatch: expected at most {expected_count} line(s), got {len(lines)}."
        )
    padded = lines + [""] * (expected_count - len(lines))
    return [line.strip() for line in padded]


def _apply_default(record: Dict[str, object], key: str, value):
    if value not in (None, "", [], 0):
        record[key] = value


class MossDatasetRowsNode(BaseTTSNode):
    @classmethod
    def NAME(cls):
        return "🧾 MOSS Dataset Rows"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_dataset": ("MOSS_CLIP_DATASET", {
                    "tooltip": "Staged clip dataset from MOSS Clip Staging."
                }),
                "manifest_name": ("STRING", {
                    "default": "moss_train.jsonl",
                    "tooltip": "Output manifest filename. .jsonl will be appended if missing."
                }),
                "text_lines": ("STRING", {
                    "default": "Hello there, this is a training sample.\nHow are you today?",
                    "multiline": True,
                    "tooltip": (
                        "Exactly one line per staged clip, in clip order.\n"
                        "Each line is the transcription/text for that clip's target audio.\n"
                        "Simple training: just give each clip its own transcript here.\n"
                        "Example: clip001 audio says 'Hello there' -> line 1 should be 'Hello there'."
                    )
                }),
            },
            "optional": {
                "reference_clip_lines": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Optional one line per staged clip.\n"
                        "Most single-speaker or fixed-voice LoRAs should leave this blank.\n"
                        "Use this only if a training row should also include reference audio for voice/style conditioning.\n"
                        "That means: text says what to say, target clip is the correct output, and reference clip says whose voice/style to follow.\n"
                        "Leave blank for normal audio+transcript training.\n"
                        "Each line may be blank, a single clip id like clip001, a 1-based clip number like 1, or a comma-separated list such as clip001,clip002.\n"
                        "Example: line 2 = clip001 means clip002 trains as: text from line 2 + reference voice from clip001 -> target audio clip002.\n"
                        "Single reference -> ref_audio. Multiple references -> reference_audio."
                    )
                }),
                "language_lines": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional one line per clip. Blank line uses default_language or no language field."
                }),
                "instruction_lines": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Optional one line per clip. Blank line uses default_instruction.\n"
                        "Use this for explicit labels such as 'speaking angrily', 'whispering', or 'calm narration'.\n"
                        "Do not use this for recording/presentation labels like 'telephone call quality'; those belong in quality_lines.\n"
                        "For many style-focused datasets, this is easier and clearer than using reference_clip_lines."
                    )
                }),
                "quality_lines": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Optional one line per clip. Blank line uses default_quality.\n"
                        "Use this for recording/presentation labels such as 'telephone call quality', 'studio recording', or similar audio-quality descriptors."
                    )
                }),
                "sound_event_lines": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional one line per clip. Blank line uses default_sound_event."
                }),
                "ambient_sound_lines": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional one line per clip. Blank line uses default_ambient_sound."
                }),
                "duration_tokens_lines": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional one line per clip. Blank line uses default_duration_tokens."
                }),
                "default_language": ("STRING", {
                    "default": "",
                    "tooltip": "Optional shared language fallback for rows whose language_lines entry is blank."
                }),
                "default_instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": (
                        "Optional shared instruction fallback for rows whose instruction_lines entry is blank.\n"
                        "Useful if most rows should carry the same style label, for example 'speaking angrily'."
                    )
                }),
                "default_quality": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Optional shared quality fallback.\n"
                        "Useful if most rows share the same recording/presentation quality label, for example 'telephone call quality'."
                    )
                }),
                "default_sound_event": ("STRING", {
                    "default": "",
                    "tooltip": "Optional shared sound_event fallback."
                }),
                "default_ambient_sound": ("STRING", {
                    "default": "",
                    "tooltip": "Optional shared ambient_sound fallback."
                }),
                "default_duration_tokens": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 8192,
                    "step": 1,
                    "tooltip": "Optional shared tokens fallback. 0 disables it."
                }),
                "output_subdir": ("STRING", {
                    "default": "tts_audio_suite_training/moss_tts/manifests",
                    "tooltip": "Subdirectory inside ComfyUI input/ where the generated manifest file will be written."
                }),
                "overwrite": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Overwrite the existing manifest file if it already exists."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("manifest_path", "manifest_info")
    FUNCTION = "build_rows"
    CATEGORY = "TTS Audio Suite/🎓 Training"

    def _resolve_reference_line(self, reference_line: str, clip_lookup: Dict[str, str]):
        raw = str(reference_line or "").strip()
        if not raw:
            return {}

        resolved_paths = []
        for token in [part.strip() for part in raw.split(",") if part.strip()]:
            key = token
            if key.isdigit():
                key = f"clip{int(key):03d}"
            path = clip_lookup.get(key)
            if not path:
                raise ValueError(
                    f"Unknown MOSS reference clip '{token}'. Use staged clip ids like clip001 "
                    "or 1-based numbers like 1,2,3."
                )
            resolved_paths.append(path)

        if not resolved_paths:
            return {}
        if len(resolved_paths) == 1:
            return {"ref_audio": resolved_paths[0]}
        return {"reference_audio": resolved_paths}

    def build_rows(
        self,
        clip_dataset,
        manifest_name: str,
        text_lines: str,
        reference_clip_lines: str = "",
        language_lines: str = "",
        instruction_lines: str = "",
        quality_lines: str = "",
        sound_event_lines: str = "",
        ambient_sound_lines: str = "",
        duration_tokens_lines: str = "",
        default_language: str = "",
        default_instruction: str = "",
        default_quality: str = "",
        default_sound_event: str = "",
        default_ambient_sound: str = "",
        default_duration_tokens: int = 0,
        output_subdir: str = "",
        overwrite: bool = True,
    ):
        if not isinstance(clip_dataset, dict) or clip_dataset.get("type") != "moss_clip_dataset":
            raise ValueError("clip_dataset must be a MOSS_CLIP_DATASET payload from MOSS Clip Staging")

        clips = clip_dataset.get("clips") or []
        if not clips:
            raise ValueError("clip_dataset contains no clips")

        clip_count = len(clips)
        text_values = _parse_required_lines(text_lines, clip_count, "text_lines")
        reference_values = _parse_optional_lines(reference_clip_lines, clip_count, "reference_clip_lines")
        language_values = _parse_optional_lines(language_lines, clip_count, "language_lines")
        instruction_values = _parse_optional_lines(instruction_lines, clip_count, "instruction_lines")
        quality_values = _parse_optional_lines(quality_lines, clip_count, "quality_lines")
        sound_event_values = _parse_optional_lines(sound_event_lines, clip_count, "sound_event_lines")
        ambient_sound_values = _parse_optional_lines(ambient_sound_lines, clip_count, "ambient_sound_lines")
        duration_tokens_values = _parse_optional_lines(duration_tokens_lines, clip_count, "duration_tokens_lines")

        clip_lookup = {str(clip["clip_id"]): str(clip["audio"]) for clip in clips}
        records = []
        for index, clip in enumerate(clips):
            record = {
                "audio": str(clip["audio"]),
            }
            if text_values[index]:
                record["text"] = text_values[index]

            reference_fields = self._resolve_reference_line(reference_values[index], clip_lookup)
            record.update(reference_fields)

            language_value = language_values[index] or str(default_language or "").strip()
            instruction_value = instruction_values[index] or str(default_instruction or "").strip()
            quality_value = quality_values[index] or str(default_quality or "").strip()
            sound_event_value = sound_event_values[index] or str(default_sound_event or "").strip()
            ambient_sound_value = ambient_sound_values[index] or str(default_ambient_sound or "").strip()
            duration_tokens_value = duration_tokens_values[index].strip()

            _apply_default(record, "language", language_value)
            _apply_default(record, "instruction", instruction_value)
            _apply_default(record, "quality", quality_value)
            _apply_default(record, "sound_event", sound_event_value)
            _apply_default(record, "ambient_sound", ambient_sound_value)

            if duration_tokens_value:
                try:
                    tokens = int(duration_tokens_value)
                except Exception as e:
                    raise ValueError(f"Invalid duration_tokens_lines value on row {index+1}: '{duration_tokens_value}'") from e
                if tokens > 0:
                    record["tokens"] = tokens
            elif int(default_duration_tokens or 0) > 0:
                record["tokens"] = int(default_duration_tokens)

            if not any(record.get(key) not in (None, "", []) for key in ("text", "instruction", "ambient_sound")):
                raise ValueError(
                    f"MOSS row {index+1} must include at least one prompt field such as text, instruction, or ambient_sound."
                )

            records.append(record)

        filename = str(manifest_name or "").strip() or "moss_train.jsonl"
        if not filename.endswith(".jsonl"):
            filename += ".jsonl"

        input_root = folder_paths.get_input_directory()
        subdir = str(output_subdir or "").strip().strip("/\\")
        output_dir = os.path.join(input_root, subdir) if subdir else input_root
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, filename)

        if os.path.exists(manifest_path) and not overwrite:
            raise FileExistsError(f"MOSS manifest already exists: {manifest_path}")

        with open(manifest_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        info = f"MOSS manifest rows ready: {os.path.basename(manifest_path)} | {len(records)} records"
        print(f"🧾 {info}")
        return manifest_path, info


NODE_CLASS_MAPPINGS = {"MossDatasetRowsNode": MossDatasetRowsNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MossDatasetRowsNode": "🧾 MOSS Dataset Rows"}
