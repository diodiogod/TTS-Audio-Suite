"""
ZONOS3-Mini processor.

Handles character switching, language switching, pause tags, per-segment parameters and text chunking
for ZONOS3-Mini generation.
"""

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import comfy.model_management as model_management
import torch

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from engines.zonos3_mini.emphasis_spans import EmphasisParser, project_emphasis_spans
from engines.zonos3_mini.languages import DEFAULT_LANGUAGE, to_suite_language, to_zonos_language
from engines.zonos3_mini.steering_tags import apply_steering_tags
from utils.audio.chunk_timing import ChunkTimingHelper
from utils.text.character_parser import character_parser
from utils.text.chunking import ImprovedChatterBoxChunker
from utils.text.pause_processor import PauseTagProcessor
from utils.voice.character_logging import (
    format_resolved_character_block,
    resolved_character_label,
)
from utils.voice.reference import effective_voice_audio
from utils.text.segment_parameters import ParameterValidator, apply_segment_parameters
from utils.voice.discovery import get_available_characters, get_character_mapping, voice_discovery


class Zonos3MiniProcessor:
    """Internal processor for ZONOS3-Mini generation."""

    SAMPLE_RATE = 44100
    ENGINE_TYPE = "zonos3_mini"

    def __init__(self, adapter, engine_config: Dict[str, Any]):
        self.adapter = adapter
        self.config = engine_config.copy() if engine_config else {}
        self.chunker = ImprovedChatterBoxChunker()
        self._logged_settings = None

    def update_config(self, new_config: Dict[str, Any]):
        self.config = new_config.copy() if new_config else {}
        self.adapter.update_config(self.config)

    @staticmethod
    def _check_interrupt(
        item_index: Optional[int] = None,
        total_items: Optional[int] = None,
        character_name: Optional[str] = None,
    ) -> None:
        if not model_management.interrupt_processing:
            return

        if item_index is not None and total_items is not None:
            if character_name:
                raise InterruptedError(
                    f"ZONOS3-Mini text generation interrupted at item "
                    f"{item_index + 1}/{total_items}, character '{character_name}'"
                )
            raise InterruptedError(
                f"ZONOS3-Mini text generation interrupted at item {item_index + 1}/{total_items}"
            )

        raise InterruptedError("ZONOS3-Mini text generation interrupted by user")

    def _voice_log_note(self, voice_ref: Dict[str, Any], config: Dict[str, Any]) -> str:
        if isinstance(voice_ref, dict) and effective_voice_audio(voice_ref) is not None:
            return f" [cloned voice, {config.get('voice_mode', 'embedding')} mode]"
        return f" [preset voice: {config.get('voice_preset', 'Savannah')}]"

    def _log_settings(self) -> None:
        cfg = self.config
        controls = {k: v for k, v in (cfg.get("controls") or {}).items() if float(v) != 0.0}
        settings = (
            "⚙️ ZONOS3-Mini Settings: model={model}, device={device}/{dtype}, engine={engine}, decoder={decoder}, "
            "language={language}, "
            "voice_mode={voice_mode}, preset={preset}, LACI={laci}{controls}".format(
                model=cfg.get("model_variant", "ZONOS3-Mini"),
                device=cfg.get("device", "auto"),
                dtype=cfg.get("dtype", "auto"),
                engine=cfg.get("engine", "auto"),
                decoder=cfg.get("decoder", "keep2_plain"),
                language=to_zonos_language(cfg.get("language", DEFAULT_LANGUAGE)),
                voice_mode=cfg.get("voice_mode", "embedding"),
                preset=cfg.get("voice_preset", "Savannah"),
                laci="on" if cfg.get("laci") else "off",
                controls=f", controls={controls}" if controls else "",
            )
        )
        # once per run: an SRT run calls process_text for every subtitle with the same settings
        if settings != self._logged_settings:
            print(settings)
            self._logged_settings = settings

    def _setup_character_parser(self, text: str):
        resolved_language = to_suite_language(to_zonos_language(self.config.get("language", DEFAULT_LANGUAGE)))

        character_parser.language_resolver.default_language = resolved_language
        character_parser.default_language = resolved_language

        character_tags = re.findall(r"\[([^\]]+)\]", text or "")
        characters_from_tags = []
        for tag in character_tags:
            if not tag.startswith("pause:"):
                characters_from_tags.append(tag.split("|")[0].strip())

        available_chars = get_available_characters()
        character_aliases = voice_discovery.get_character_aliases()

        all_available = set()
        if available_chars:
            all_available.update(available_chars)
        for alias, target in character_aliases.items():
            all_available.add(alias.lower())
            all_available.add(target.lower())
        for char in characters_from_tags:
            all_available.add(char.lower())
        all_available.add("narrator")

        character_parser.set_available_characters(list(all_available))

        char_lang_defaults = voice_discovery.get_character_language_defaults()
        for char, lang in char_lang_defaults.items():
            character_parser.set_character_language_default(char, lang)

        character_parser.reset_session_cache()

    @staticmethod
    def _should_apply_segment_language(seg: Any, base_config: Dict[str, Any]) -> bool:
        segment_language = getattr(seg, "language", None)
        if not segment_language:
            return False

        if getattr(seg, "explicit_language", False):
            return True

        parser_default = str(character_parser.default_language or "en").strip().lower()
        return str(segment_language).strip().lower() != parser_default

    def _build_generation_items(
        self,
        text: str,
        voice_mapping: Dict[str, Any],
        seed: int,
        enable_chunking: bool,
        max_chars_per_chunk: int,
        show_text_logging: bool = True,
    ) -> List[Dict[str, Any]]:
        self._setup_character_parser(text)
        emphasis = EmphasisParser()
        marked_text = emphasis.prepare(text)

        narrator_info = voice_mapping.get("narrator", {})
        narrator_voice = narrator_info if isinstance(narrator_info, dict) else {"audio": narrator_info}

        base_config = self.config.copy()
        segment_objects = character_parser.parse_text_segments(marked_text, engine_type=self.ENGINE_TYPE)
        if not segment_objects:
            segment_objects = character_parser.parse_text_segments("narrator " + marked_text, engine_type=self.ENGINE_TYPE)

        characters = list({seg.character for seg in segment_objects if seg.character})
        character_mapping = get_character_mapping(characters, engine_type="audio_only")

        generation_items: List[Dict[str, Any]] = []

        for seg in segment_objects:
            self._check_interrupt()
            segment_text = (seg.text or "").strip()
            if not segment_text:
                continue

            segment_params = seg.parameters if seg.parameters else {}
            current_config = base_config.copy()
            current_seed = seed
            parameter_log = ""
            if segment_params:
                filtered_params = ParameterValidator.filter_parameters_for_engine(segment_params, self.ENGINE_TYPE)
                current_config = apply_steering_tags(
                    apply_segment_parameters(base_config, segment_params, self.ENGINE_TYPE)
                )
                if "seed" in current_config:
                    current_seed = int(current_config.get("seed", seed))
                parameter_log = ", ".join(f"{key}={value}" for key, value in filtered_params.items())

            if self._should_apply_segment_language(seg, base_config):
                current_config = current_config.copy()
                current_config["language"] = to_zonos_language(seg.language)
                print(f"  🌍 ZONOS3-Mini language switched to: {current_config['language']}")

            segment_language = to_zonos_language(current_config.get("language", DEFAULT_LANGUAGE))

            # character voice, else the narrator reference, else the preset voice (empty voice_ref)
            voice_ref = narrator_voice.copy()
            char_name = seg.character or "narrator"
            if char_name != "narrator" and char_name in voice_mapping:
                char_info = voice_mapping[char_name]
                if isinstance(char_info, dict):
                    voice_ref = char_info.copy()
            elif char_name != "narrator":
                audio_path, _ref_text = character_mapping.get(char_name, (None, None))
                if audio_path:
                    voice_ref = {"audio_path": audio_path}

            def _append_text_item(text_content: str):
                marked_text, spans = emphasis.strip_tags(text_content)
                text_content = marked_text.strip()
                if not text_content:
                    return
                if enable_chunking and len(text_content) > max_chars_per_chunk:
                    chunks = self.chunker.split_into_chunks(text_content, max_chars_per_chunk)
                    print(f"📝 ZONOS3-Mini: chunking {len(text_content)} characters into {len(chunks)} chunks")
                else:
                    chunks = [text_content]
                chunk_spans = project_emphasis_spans(marked_text, chunks, spans)
                for chunk, control_spans in zip(chunks, chunk_spans):
                    config = current_config.copy()
                    if control_spans:
                        config["control_spans"] = control_spans
                    generation_items.append(
                        {
                            "kind": "text",
                            "text": chunk,
                            "config": config,
                            "seed": current_seed,
                            "voice_ref": voice_ref.copy() if isinstance(voice_ref, dict) else voice_ref,
                            "character_name": char_name,
                            "language": segment_language,
                            "parameter_log": parameter_log,
                            "show_text_logging": show_text_logging,
                        }
                    )

            if PauseTagProcessor.has_pause_tags(segment_text):
                raw_pause_segments, _ = PauseTagProcessor.parse_pause_tags(segment_text)
                for frag_type, frag_content in raw_pause_segments:
                    if frag_type == "text":
                        _append_text_item(frag_content)
                    elif frag_type == "pause":
                        generation_items.append({"kind": "pause", "duration": float(frag_content)})
            else:
                _append_text_item(segment_text)

        emphasis.finish()
        return generation_items

    def process_text(
        self,
        text: str,
        voice_mapping: Dict[str, Any],
        seed: int,
        enable_chunking: bool = True,
        max_chars_per_chunk: int = 400,
        chunk_combination_method: str = "auto",
        silence_between_chunks_ms: int = 100,
        enable_audio_cache: bool = True,
        apply_edit_postprocessing: bool = True,
        show_text_logging: bool = True,
    ) -> List[Dict[str, Any]]:
        del chunk_combination_method, silence_between_chunks_ms  # applied by combine_audio_segments
        del apply_edit_postprocessing  # ZONOS3-Mini word emphasis is applied during inference
        self._log_settings()
        generation_items = self._build_generation_items(
            text=text,
            voice_mapping=voice_mapping,
            seed=seed,
            enable_chunking=bool(enable_chunking),
            max_chars_per_chunk=int(max_chars_per_chunk),
            show_text_logging=show_text_logging,
        )

        segment_records: List[Dict[str, Any]] = []
        try:
            for item_index, item in enumerate(generation_items):
                self._check_interrupt(
                    item_index=item_index,
                    total_items=len(generation_items),
                    character_name=item.get("character_name"),
                )
                if item["kind"] == "pause":
                    silence_duration = float(item["duration"])
                    silence = PauseTagProcessor.create_silence_segment(
                        silence_duration,
                        self.SAMPLE_RATE,
                        torch.device("cpu"),
                        torch.float32,
                    )
                    if silence.dim() == 1:
                        silence = silence.unsqueeze(0)
                    segment_records.append(
                        {
                            "waveform": silence,
                            "sample_rate": self.SAMPLE_RATE,
                            "text": f"[pause:{silence_duration:.3f}s]",
                        }
                    )
                    continue

                self.adapter.update_config(item["config"])
                display_name = resolved_character_label(item["character_name"], item["voice_ref"])
                print(
                    f"🎭 ZONOS3-Mini - Generating for '{display_name}' (Language: {item['language']})"
                    f"{self._voice_log_note(item['voice_ref'], item['config'])}:"
                )
                if item["parameter_log"]:
                    print(f"🎛️ ZONOS3-Mini params: {item['parameter_log']}")
                chunk_text = item["text"]
                if item["show_text_logging"]:
                    print(format_resolved_character_block(item["character_name"], chunk_text, item["voice_ref"]))

                emphasis_labels = []
                for span in item["config"].get("control_spans", ()):
                    phrase = chunk_text[span["start"]:span["end"]]
                    strength = span["controls"]["mood.arousal"]
                    emphasis_labels.append(f"{phrase!r} (strength {strength:g})")
                if emphasis_labels:
                    print(f"🎯 ZONOS3-Mini word emphasis: {', '.join(emphasis_labels)}")

                audio = self.adapter.generate_single(
                    text=chunk_text,
                    voice_ref=item["voice_ref"],
                    seed=item["seed"],
                    enable_audio_cache=enable_audio_cache,
                    character_name=item["character_name"],
                )
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                segment_records.append(
                    {
                        "waveform": audio.cpu(),
                        "sample_rate": self.SAMPLE_RATE,
                        "text": chunk_text,
                    }
                )
        finally:
            self.adapter.update_config(self.config)

        return segment_records

    def combine_audio_segments(
        self,
        segments: List[Dict[str, Any]],
        method: str = "auto",
        silence_ms: int = 100,
        original_text: str = "",
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        if not segments:
            empty = torch.zeros(0, dtype=torch.float32)
            if return_info:
                return empty, {}
            return empty

        audio_segments = [seg["waveform"] for seg in segments]
        text_chunks = [seg.get("text", "") for seg in segments]
        text_length = len(" ".join(text_chunks))

        combined_audio, chunk_info = ChunkTimingHelper.combine_audio_with_timing(
            audio_segments=audio_segments,
            combination_method=method,
            silence_ms=silence_ms,
            crossfade_duration=0.1,
            sample_rate=self.SAMPLE_RATE,
            text_length=text_length,
            original_text=original_text,
            text_chunks=text_chunks,
        )

        if return_info:
            return combined_audio, chunk_info
        return combined_audio
