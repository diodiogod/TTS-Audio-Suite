"""Text orchestration for the generic audio.cpp TTS engine."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch

from utils.audio.chunk_combiner import ChunkCombiner
from utils.text.character_parser import character_parser
from utils.text.pause_processor import PauseTagProcessor
from utils.text.segment_parameters import ParameterValidator, apply_segment_parameters
from utils.text.step_audio_editx_special_tags import get_edit_tags_for_segment
from utils.voice.discovery import get_available_characters, get_character_mapping, voice_discovery


class AudioCppProcessor:
    """Apply suite text features while accepting the runtime's response sample rate."""

    _RUNTIME_KEYS = (
        "connection_mode",
        "server_url",
        "external_server_url",
        "binary_path",
        "family",
        "package_id",
        "model_path",
        "model_id",
        "task",
        "backend",
        "device",
    )

    def __init__(self, adapter: Any, engine_config: Optional[Dict[str, Any]] = None):
        self.adapter = adapter
        self.config = dict(engine_config or {})
        self._sample_rate: Optional[int] = None

    @property
    def sample_rate(self) -> Optional[int]:
        return self._sample_rate

    def update_config(self, new_config: Optional[Dict[str, Any]]) -> None:
        new_value = dict(new_config or {})
        old_signature = tuple(self.config.get(key) for key in self._RUNTIME_KEYS)
        new_signature = tuple(new_value.get(key) for key in self._RUNTIME_KEYS)
        if old_signature != new_signature:
            self._sample_rate = None
        self.config = new_value
        self.adapter.update_config(new_value)

    def reset_sample_rate(self) -> None:
        """Begin a top-level generation without retaining an old server rate."""
        self._sample_rate = None

    @staticmethod
    def _check_interrupt() -> None:
        try:
            import comfy.model_management as model_management

            if getattr(model_management, "interrupt_processing", False) is True:
                raise InterruptedError("audio.cpp generation interrupted by user")
        except ImportError:
            return

    def _adopt_sample_rate(self, sample_rate: Any) -> int:
        try:
            value = int(sample_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"audio.cpp returned invalid sample rate: {sample_rate!r}") from exc
        if value <= 0:
            raise ValueError(f"audio.cpp returned invalid sample rate: {value}")
        if self._sample_rate is None:
            self._sample_rate = value
        elif self._sample_rate != value:
            raise RuntimeError(
                "audio.cpp returned inconsistent sample rates in one generation "
                f"({self._sample_rate} Hz then {value} Hz)"
            )
        return value

    def _setup_character_parser(self, text: str) -> None:
        language = str(self.config.get("language", "auto") or "auto").strip()
        fallback = "en" if language.lower() in {"", "auto", "none"} else language.lower()
        character_parser.language_resolver.default_language = fallback
        character_parser.default_language = fallback

        tagged = []
        for raw in re.findall(r"\[([^\]]+)\]", text or ""):
            name = raw.split("|", 1)[0].strip()
            if name and not name.lower().startswith(("pause:", "wait:", "stop:")):
                tagged.append(name)

        available = {str(item).lower() for item in (get_available_characters() or [])}
        for alias, target in voice_discovery.get_character_aliases().items():
            available.update((str(alias).lower(), str(target).lower()))
        available.update(name.lower() for name in tagged)
        available.add("narrator")
        character_parser.set_available_characters(sorted(available))
        for character, default_language in voice_discovery.get_character_language_defaults().items():
            character_parser.set_character_language_default(character, default_language)
        character_parser.reset_session_cache()

    @staticmethod
    def _should_apply_segment_language(segment: Any, base_config: Mapping[str, Any]) -> bool:
        language = str(getattr(segment, "language", "") or "").strip()
        if not language:
            return False
        if getattr(segment, "explicit_language", False):
            return True
        global_language = str(base_config.get("language", "auto") or "auto").strip().lower()
        parser_fallback = str(character_parser.default_language or "").strip().lower()
        return language.lower() != parser_fallback and language.lower() != global_language

    @staticmethod
    def _voice_for_character(
        character: str,
        voice_mapping: Mapping[str, Any],
        discovered: Mapping[str, Tuple[Optional[str], Optional[str]]],
    ) -> Dict[str, Any]:
        narrator = voice_mapping.get("narrator", {})
        voice = dict(narrator) if isinstance(narrator, Mapping) else {"audio": narrator}
        if character != "narrator" and character in voice_mapping:
            selected = voice_mapping[character]
            return dict(selected) if isinstance(selected, Mapping) else {"audio": selected}
        if character != "narrator":
            audio_path, reference_text = discovered.get(character, (None, None))
            if audio_path:
                return {"audio_path": audio_path, "reference_text": reference_text or ""}
        return voice

    @staticmethod
    def _chunks(text: str, enabled: bool, max_chars: int) -> List[str]:
        if not enabled:
            return [text]
        from utils.text.chunking import ImprovedChatterBoxChunker

        limit = ImprovedChatterBoxChunker.validate_chunking_params(max_chars)
        return ImprovedChatterBoxChunker.split_into_chunks(text, max_chars=limit)

    def get_character_order(self, text: str) -> List[str]:
        self._setup_character_parser(text)
        seen: List[str] = []
        for segment in character_parser.parse_text_segments(text, engine_type="audio_cpp"):
            character = segment.character or "narrator"
            if character not in seen:
                seen.append(character)
        return seen

    def process_text(
        self,
        text: str,
        voice_mapping: Optional[Dict[str, Any]],
        seed: int,
        enable_chunking: bool = True,
        max_chars_per_chunk: int = 400,
        chunk_combination_method: str = "auto",
        silence_between_chunks_ms: int = 100,
        enable_audio_cache: bool = True,
        apply_edit_postprocessing: bool = True,
        show_text_logging: bool = True,
        reset_sample_rate: bool = True,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        del chunk_combination_method, silence_between_chunks_ms
        if reset_sample_rate:
            self.reset_sample_rate()
        self._check_interrupt()
        voice_mapping = dict(voice_mapping or {})
        self._setup_character_parser(text)
        base_config = self.config.copy()
        segments = character_parser.parse_text_segments(text, engine_type="audio_cpp")
        if not segments and str(text or "").strip():
            segments = character_parser.parse_text_segments(
                f"[narrator]{text}", engine_type="audio_cpp"
            )

        characters = list({segment.character for segment in segments if segment.character})
        # GLM-TTS requires the transcript paired with its reference voice.
        # Other pinned families accept audio-only discovery and still receive a
        # transcript whenever one exists beside the character audio file.
        discovery_type = (
            "audio_and_text"
            if str(base_config.get("family", "")).lower() == "glm_tts"
            else "audio_only"
        )
        discovered = get_character_mapping(characters, engine_type=discovery_type)
        records: List[Dict[str, Any]] = []

        for segment in segments:
            self._check_interrupt()
            segment_text = str(segment.text or "").strip()
            if not segment_text:
                continue
            character = segment.character or "narrator"
            parameters = dict(segment.parameters or {})
            current_config = base_config
            current_seed = int(seed)
            if parameters:
                ParameterValidator.filter_parameters_for_engine(parameters, "audio_cpp")
                current_config = apply_segment_parameters(base_config, parameters, "audio_cpp")
                current_seed = int(current_config.get("seed", seed))
            if self._should_apply_segment_language(segment, base_config):
                current_config = current_config.copy()
                current_config["language"] = segment.language
            self.adapter.update_config(current_config)
            voice_ref = self._voice_for_character(character, voice_mapping, discovered)

            def generate_fragment(content: str, edit_tags: List[Any]) -> None:
                chunks = self._chunks(content, enable_chunking, max_chars_per_chunk)
                if show_text_logging:
                    print(f"🎭 audio.cpp - {character}: {content}")
                for chunk_index, chunk in enumerate(chunks):
                    self._check_interrupt()
                    waveform, response_rate = self.adapter.generate_single(
                        text=chunk,
                        voice_ref=voice_ref,
                        seed=current_seed + chunk_index,
                        enable_audio_cache=enable_audio_cache,
                        character_name=character,
                    )
                    sample_rate = self._adopt_sample_rate(response_rate)
                    waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
                    if waveform.dim() == 1:
                        waveform = waveform.unsqueeze(0)
                    if waveform.dim() != 2:
                        raise ValueError(
                            f"audio.cpp waveform must be [channels, samples], got {tuple(waveform.shape)}"
                        )
                    records.append(
                        {
                            "waveform": waveform,
                            "sample_rate": sample_rate,
                            "text": chunk,
                            "edit_tags": edit_tags if chunk_index == 0 else [],
                        }
                    )

            if PauseTagProcessor.has_pause_tags(segment_text):
                pause_parts, _ = PauseTagProcessor.parse_pause_tags(segment_text)
                for part_type, content in pause_parts:
                    if part_type == "text":
                        clean_text, edit_tags = get_edit_tags_for_segment(str(content))
                        if clean_text.strip():
                            generate_fragment(clean_text.strip(), edit_tags)
                    else:
                        records.append(
                            {
                                "pause_duration": float(content),
                                "text": f"[pause:{content}s]",
                                "edit_tags": [],
                            }
                        )
            else:
                clean_text, edit_tags = get_edit_tags_for_segment(segment_text)
                if clean_text.strip():
                    generate_fragment(clean_text.strip(), edit_tags)

        self.adapter.update_config(base_config)
        if any("pause_duration" in record for record in records):
            if self._sample_rate is None:
                raise ValueError("audio.cpp cannot render pauses before any response sample rate is known")
            for record in records:
                if "pause_duration" not in record:
                    continue
                record["waveform"] = PauseTagProcessor.create_silence_segment(
                    record.pop("pause_duration"), self._sample_rate, torch.device("cpu"), torch.float32
                )
                record["sample_rate"] = self._sample_rate

        if apply_edit_postprocessing and records and any(record.get("edit_tags") for record in records):
            from utils.audio.edit_post_processor import process_segments as apply_edits

            records = apply_edits(records, engine_config=base_config)
            for record in records:
                self._adopt_sample_rate(record.get("sample_rate"))
        return records

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
            return (empty, {}) if return_info else empty

        rates = {self._adopt_sample_rate(segment.get("sample_rate")) for segment in segments}
        if len(rates) != 1:
            raise RuntimeError(f"audio.cpp segments use inconsistent sample rates: {sorted(rates)}")
        sample_rate = rates.pop()
        waveforms = [segment["waveform"] for segment in segments]
        text_chunks = [str(segment.get("text", "")) for segment in segments]
        result = ChunkCombiner.combine_chunks(
            audio_segments=waveforms,
            method=method,
            silence_ms=int(silence_ms),
            crossfade_duration=0.1,
            sample_rate=sample_rate,
            text_length=len(" ".join(text_chunks)),
            original_text=original_text,
            text_chunks=text_chunks,
            return_info=return_info,
        )
        return result


# Compatibility with integration code that uses an all-caps acronym.
AudioCPPProcessor = AudioCppProcessor
