"""Build a DramaBox training manifest from engine-neutral staged clips."""

import importlib.util
import json
import os
import sys
from typing import List

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


def _required_lines(raw_text: str, expected_count: int) -> List[str]:
    lines = str(raw_text or "").splitlines()
    if len(lines) != expected_count:
        raise ValueError(
            "DramaBox transcript line count mismatch: "
            f"expected {expected_count} line(s), got {len(lines)}. "
            "Enter exactly one transcript per staged clip."
        )
    return [line.strip() for line in lines]


def _optional_lines(raw_text: str, expected_count: int, field_name: str) -> List[str]:
    lines = str(raw_text or "").splitlines()
    if len(lines) > expected_count:
        raise ValueError(
            f"DramaBox {field_name} line count mismatch: expected at most "
            f"{expected_count} line(s), got {len(lines)}."
        )
    return [line.strip() for line in lines] + [""] * (expected_count - len(lines))


class DramaBoxDatasetRowsNode(BaseTTSNode):
    @classmethod
    def NAME(cls):
        return "🧾 DramaBox Dataset Rows"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_dataset": ("TRAINING_CLIP_DATASET", {
                    "tooltip": "Staged audio from Training Clip Staging.",
                }),
                "manifest_name": ("STRING", {
                    "default": "dramabox_train.jsonl",
                    "tooltip": "Output manifest filename. .jsonl is appended when missing.",
                }),
                "transcript_lines": ("STRING", {
                    "default": "Hello there, this is a training sample.\nThis is the second sample from the same speaker.",
                    "multiline": True,
                    "tooltip": "Exactly one line per staged clip, in clip order. Blank lines skip the corresponding clip.",
                }),
            },
            "optional": {
                "speaker_lines": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional speaker name per clip. Blank lines use default_speaker.",
                }),
                "language_lines": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional language code per clip. Blank lines use default_language.",
                }),
                "default_speaker": ("STRING", {
                    "default": "speaker_1",
                    "tooltip": "Speaker assigned when the corresponding speaker line is blank. Each DramaBox speaker needs at least two clips.",
                }),
                "default_language": ("STRING", {
                    "default": "en",
                    "tooltip": "Language code assigned when the corresponding language line is blank.",
                }),
                "output_subdir": ("STRING", {
                    "default": "tts_audio_suite_training/dramabox/manifests",
                    "tooltip": "Subdirectory inside ComfyUI input/ for the generated manifest.",
                }),
                "overwrite": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Overwrite an existing manifest with the same name.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("manifest_path", "manifest_info")
    FUNCTION = "build_rows"
    CATEGORY = "TTS Audio Suite/🎓 Training"

    def build_rows(
        self,
        clip_dataset,
        manifest_name: str,
        transcript_lines: str,
        speaker_lines: str = "",
        language_lines: str = "",
        default_speaker: str = "speaker_1",
        default_language: str = "en",
        output_subdir: str = "",
        overwrite: bool = True,
    ):
        if not isinstance(clip_dataset, dict) or clip_dataset.get("type") not in {
            "training_clip_dataset",
            "moss_clip_dataset",
        }:
            raise ValueError("clip_dataset must come from Training Clip Staging")

        clips = clip_dataset.get("clips") or []
        if not clips:
            raise ValueError("clip_dataset contains no clips")

        clip_count = len(clips)
        transcripts = _required_lines(transcript_lines, clip_count)
        speakers = _optional_lines(speaker_lines, clip_count, "speaker_lines")
        languages = _optional_lines(language_lines, clip_count, "language_lines")
        fallback_speaker = str(default_speaker or "").strip() or "speaker_1"
        fallback_language = str(default_language or "").strip() or "en"

        records = []
        speaker_counts = {}
        skipped_rows = 0
        for index, clip in enumerate(clips):
            if not transcripts[index]:
                skipped_rows += 1
                continue
            speaker = speakers[index] or fallback_speaker
            language = languages[index] or fallback_language
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            records.append({
                "audio_filepath": str(clip["audio"]),
                "text": transcripts[index],
                "speaker": speaker,
                "language": language,
                "duration": float(clip["duration_seconds"]),
                "sample_rate": int(clip["sample_rate"]),
                "samples": round(
                    float(clip["duration_seconds"]) * int(clip["sample_rate"])
                ),
            })

        if not records:
            raise RuntimeError(
                "DramaBox Dataset Rows produced no records. Add at least two "
                "non-empty transcripts for one speaker."
            )

        short_speakers = sorted(
            speaker for speaker, count in speaker_counts.items() if count < 2
        )
        if short_speakers:
            raise ValueError(
                "DramaBox needs at least two clips per speaker. Speakers with only "
                "one staged clip: " + ", ".join(short_speakers)
            )

        filename = str(manifest_name or "").strip() or "dramabox_train.jsonl"
        if not filename.lower().endswith(".jsonl"):
            filename += ".jsonl"
        input_root = folder_paths.get_input_directory()
        subdir = str(output_subdir or "").strip().strip("/\\")
        output_dir = os.path.join(input_root, subdir) if subdir else input_root
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, filename)
        if os.path.exists(manifest_path) and not overwrite:
            raise FileExistsError(f"DramaBox manifest already exists: {manifest_path}")

        with open(manifest_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        info = (
            f"DramaBox manifest ready: {os.path.basename(manifest_path)} | "
            f"{len(records)} clips | {len(speaker_counts)} speaker(s)"
        )
        if skipped_rows:
            info += f" | skipped {skipped_rows} blank transcript row(s)"
        print(f"🧾 {info}")
        return manifest_path, info


NODE_CLASS_MAPPINGS = {"DramaBoxDatasetRowsNode": DramaBoxDatasetRowsNode}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DramaBoxDatasetRowsNode": "🧾 DramaBox Dataset Rows"
}
