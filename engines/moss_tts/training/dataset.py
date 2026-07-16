"""
Dataset prep and teacher-forcing dataset for MOSS-TTS training.
"""

from __future__ import annotations

import os
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

from engines.moss_tts.impl.audio_tokenizer.modeling_moss_audio_tokenizer import MossAudioTokenizerModel
from engines.moss_tts.impl.delay.configuration_moss_tts import MossTTSDelayConfig
from engines.moss_tts.impl.delay.processing_moss_tts import MossTTSDelayProcessor
from engines.moss_tts.training.common import (
    dump_jsonl,
    fingerprint_paths,
    get_moss_training_root,
    load_jsonl,
    resolve_codec_path,
    resolve_delay_training_variant,
    resolve_model_path,
    split_train_val,
    slugify,
)


USER_MESSAGE_KEYS = ("text", "instruction", "tokens", "quality", "sound_event", "ambient_sound", "language")


@contextmanager
def _quiet_transformers_progress():
    transformers_logging.disable_progress_bar()
    try:
        yield
    finally:
        transformers_logging.enable_progress_bar()


def _normalize_audio_path_list(value: Any, field_name: str, allow_none: bool = False) -> Optional[List[Optional[str]]]:
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if allow_none:
            if not all(item is None or isinstance(item, str) for item in value):
                raise ValueError(f"`{field_name}` must be a string, null, or a list containing strings/nulls.")
        elif not all(isinstance(item, str) for item in value):
            raise ValueError(f"`{field_name}` must be a string or a list of strings.")
        return value
    raise TypeError(f"Unsupported `{field_name}` type: {type(value)}")


def _normalize_audio_codes(value: Any, field_name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim != 2:
        raise ValueError(f"`{field_name}` must have shape (T, n_vq), got {tuple(tensor.shape)}.")
    return tensor.cpu().contiguous()


def _normalize_audio_code_list(
    value: Any,
    field_name: str,
    allow_none: bool = False,
) -> Optional[List[Optional[torch.Tensor]]]:
    if value in (None, "", []):
        return None
    if torch.is_tensor(value):
        return [_normalize_audio_codes(value, field_name)]
    if isinstance(value, list):
        if not value:
            return None
        if allow_none and any(item is None for item in value):
            return [
                None if item is None else _normalize_audio_codes(item, f"{field_name}[{index}]")
                for index, item in enumerate(value)
            ]
        first_item = value[0]
        if torch.is_tensor(first_item):
            return [_normalize_audio_codes(item, f"{field_name}[{index}]") for index, item in enumerate(value)]
        if isinstance(first_item, list):
            if first_item and isinstance(first_item[0], list):
                return [_normalize_audio_codes(item, f"{field_name}[{index}]") for index, item in enumerate(value)]
            return [_normalize_audio_codes(value, field_name)]
    raise TypeError(f"Unsupported `{field_name}` type: {type(value)}")


def build_delay_training_processor(model_path: str):
    with _quiet_transformers_progress():
        model_config = MossTTSDelayConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, config=model_config)
    return MossTTSDelayProcessor(
        tokenizer=tokenizer,
        audio_tokenizer=None,
        model_config=model_config,
    )


def _build_encoding_processor(model_path: str, codec_path: str):
    with _quiet_transformers_progress():
        model_config = MossTTSDelayConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, config=model_config)
        audio_tokenizer = MossAudioTokenizerModel.from_pretrained(codec_path)
    return MossTTSDelayProcessor(
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer,
        model_config=model_config,
    )


def _collect_reference_paths(records: List[Dict[str, Any]]) -> List[str]:
    paths: List[str] = []
    for record in records:
        for field_name in ("ref_audio", "reference_audio"):
            values = _normalize_audio_path_list(record.get(field_name), field_name)
            if values is not None:
                paths.extend(values)
        reference = _normalize_audio_path_list(record.get("reference"), "reference", allow_none=True)
        if reference is not None:
            paths.extend([item for item in reference if item is not None])
    deduped: List[str] = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _attach_reference_audio_codes(records: List[Dict[str, Any]], path_to_codes: Dict[str, List[List[int]]]) -> None:
    for record in records:
        ref_audio = _normalize_audio_path_list(record.get("ref_audio"), "ref_audio")
        if ref_audio is not None:
            if len(ref_audio) != 1:
                raise ValueError("`ref_audio` only supports a single path.")
            record["ref_audio_codes"] = path_to_codes[ref_audio[0]]

        reference_audio = _normalize_audio_path_list(record.get("reference_audio"), "reference_audio")
        if reference_audio is not None:
            record["reference_audio_codes"] = [path_to_codes[path] for path in reference_audio]

        reference = _normalize_audio_path_list(record.get("reference"), "reference", allow_none=True)
        if reference is not None:
            record["reference_audio_codes"] = [
                None if path is None else path_to_codes[path]
                for path in reference
            ]


def _encode_manifest_records(
    raw_records: List[Dict[str, Any]],
    *,
    model_path: str,
    codec_path: str,
    batch_size: int,
    n_vq: Optional[int],
    encode_reference_audio: bool,
    device: str,
) -> List[Dict[str, Any]]:
    processor = _build_encoding_processor(model_path, codec_path)
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    target_audio_paths: List[str] = []
    for index, record in enumerate(raw_records):
        audio_path = record.get("audio")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError(f"Record {index} is missing a valid `audio` field.")
        target_audio_paths.append(audio_path)

    encoded_records = [dict(record) for record in raw_records]

    def _batch_encode(paths: List[str], desc: str) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for start in range(0, len(paths), max(1, int(batch_size))):
            batch_paths = paths[start : start + max(1, int(batch_size))]
            print(f"MOSS dataset prep: {desc} {min(start + len(batch_paths), len(paths))}/{len(paths)}")
            outputs.extend(processor.encode_audios_from_path(batch_paths, n_vq=n_vq))
        return outputs

    target_audio_codes = _batch_encode(target_audio_paths, "target audio")
    for record, codes in zip(encoded_records, target_audio_codes):
        record["audio_codes"] = codes.tolist()

    if encode_reference_audio:
        unique_reference_paths = _collect_reference_paths(encoded_records)
        if unique_reference_paths:
            reference_codes = _batch_encode(unique_reference_paths, "reference audio")
            reference_map = {
                path: codes.tolist()
                for path, codes in zip(unique_reference_paths, reference_codes)
            }
            _attach_reference_audio_codes(encoded_records, reference_map)

    return encoded_records


def prepare_moss_training_dataset(
    shared_settings: Dict[str, Any],
    *,
    dataset_source: str,
    model_name: str,
    validation_source: str = "",
    validation_split: float = 0.05,
    split_seed: int = 42,
    batch_size: int = 8,
    prep_batch_size: int = 0,
    n_vq: int = 0,
    encode_reference_audio: bool = True,
    reuse_existing: bool = True,
) -> Dict[str, Any]:
    # Node UI uses prep_batch_size; keep batch_size for compatibility with older callers.
    effective_batch_size = int(prep_batch_size) if int(prep_batch_size or 0) > 0 else int(batch_size)

    variant = resolve_delay_training_variant(shared_settings)

    train_manifest_path = os.path.abspath(dataset_source)
    if not os.path.isfile(train_manifest_path):
        raise FileNotFoundError(f"MOSS training manifest not found: {dataset_source}")
    val_manifest_path = os.path.abspath(validation_source) if str(validation_source or "").strip() else ""
    if val_manifest_path and not os.path.isfile(val_manifest_path):
        raise FileNotFoundError(f"MOSS validation manifest not found: {validation_source}")

    fingerprint_inputs = [train_manifest_path]
    if val_manifest_path:
        fingerprint_inputs.append(val_manifest_path)
    dataset_hash = fingerprint_paths(fingerprint_inputs)
    dataset_root = os.path.join(
        get_moss_training_root(),
        "datasets",
        f"{slugify(model_name)}_{slugify(variant)}_{dataset_hash[:10]}",
    )
    os.makedirs(dataset_root, exist_ok=True)

    raw_train_jsonl = os.path.join(dataset_root, "train_raw.jsonl")
    raw_val_jsonl = os.path.join(dataset_root, "val_raw.jsonl")
    prepared_train_jsonl = os.path.join(dataset_root, "train_prepared.jsonl")
    prepared_val_jsonl = os.path.join(dataset_root, "val_prepared.jsonl")

    if not (
        reuse_existing
        and os.path.isfile(prepared_train_jsonl)
        and os.path.isfile(prepared_val_jsonl)
    ):
        raw_train_records = load_jsonl(train_manifest_path)
        if not raw_train_records:
            raise ValueError("MOSS training manifest is empty")
        if val_manifest_path:
            raw_val_records = load_jsonl(val_manifest_path)
            if not raw_val_records:
                raise ValueError("MOSS validation manifest is empty")
        else:
            raw_train_records, raw_val_records = split_train_val(
                raw_train_records,
                validation_split=validation_split,
                split_seed=split_seed,
            )

        dump_jsonl(raw_train_records, raw_train_jsonl)
        dump_jsonl(raw_val_records, raw_val_jsonl)

        model_path = resolve_model_path(variant)
        codec_path = resolve_codec_path(shared_settings.get("codec_model", "MOSS-Audio-Tokenizer"))
        device = str(shared_settings.get("device", "cpu") or "cpu")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        prepared_train_records = _encode_manifest_records(
            raw_train_records,
            model_path=model_path,
            codec_path=codec_path,
            batch_size=effective_batch_size,
            n_vq=(int(n_vq) if int(n_vq or 0) > 0 else None),
            encode_reference_audio=encode_reference_audio,
            device=device,
        )
        prepared_val_records = _encode_manifest_records(
            raw_val_records,
            model_path=model_path,
            codec_path=codec_path,
            batch_size=effective_batch_size,
            n_vq=(int(n_vq) if int(n_vq or 0) > 0 else None),
            encode_reference_audio=encode_reference_audio,
            device=device,
        )
        dump_jsonl(prepared_train_records, prepared_train_jsonl)
        dump_jsonl(prepared_val_records, prepared_val_jsonl)

    prepared_train_records = load_jsonl(prepared_train_jsonl)
    prepared_val_records = load_jsonl(prepared_val_jsonl)
    return {
        "type": "training_dataset",
        "engine_type": "moss_tts",
        "training_mode": "lora_adapter",
        "model_variant": variant,
        "model_name": model_name,
        "dataset_dir": dataset_root,
        "raw_train_jsonl": raw_train_jsonl,
        "raw_val_jsonl": raw_val_jsonl,
        "prepared_train_jsonl": prepared_train_jsonl,
        "prepared_val_jsonl": prepared_val_jsonl,
        "train_records": len(prepared_train_records),
        "val_records": len(prepared_val_records),
        "source_summary": Path(train_manifest_path).name + (f" + {Path(val_manifest_path).name}" if val_manifest_path else ""),
    }


class MossTTSSFTDataset(Dataset):
    def __init__(
        self,
        records: Iterable[Dict[str, Any]],
        processor: MossTTSDelayProcessor,
        n_vq: Optional[int] = None,
    ) -> None:
        self.records = list(records)
        self.processor = processor
        self.n_vq = n_vq

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._pack_record(self.records[index])

    def _validate_code_list(
        self,
        codes_list: Optional[List[Optional[torch.Tensor]]],
        target_n_vq: int,
        field_name: str,
    ) -> Optional[List[Optional[torch.Tensor]]]:
        if codes_list is None:
            return None
        for codes in codes_list:
            if codes is None:
                continue
            if codes.shape[1] != target_n_vq:
                raise ValueError(
                    f"`{field_name}` n_vq={codes.shape[1]} does not match target n_vq={target_n_vq}."
                )
        return codes_list

    def _resolve_reference_codes(self, record: Dict[str, Any], target_n_vq: int) -> Optional[List[Optional[torch.Tensor]]]:
        for code_field in ("reference_audio_codes", "ref_audio_codes"):
            if record.get(code_field) is not None:
                return self._validate_code_list(
                    _normalize_audio_code_list(record[code_field], code_field, allow_none=(code_field == "reference_audio_codes")),
                    target_n_vq,
                    code_field,
                )
        return None

    def _pack_record(self, record: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "audio_codes" not in record:
            raise ValueError("Each record must contain `audio_codes`. Run MOSS Dataset Prep first.")

        target_codes = _normalize_audio_codes(record["audio_codes"], "audio_codes")
        target_n_vq = int(target_codes.shape[1])
        if self.n_vq is not None and target_n_vq != self.n_vq:
            raise ValueError(f"Expected n_vq={self.n_vq}, but got {target_n_vq}.")

        reference_codes = self._resolve_reference_codes(record, target_n_vq)
        user_kwargs: Dict[str, Any] = {
            "reference": reference_codes,
        }
        for key in USER_MESSAGE_KEYS:
            if record.get(key) is not None:
                user_kwargs[key] = record[key]

        user_message = self.processor.build_user_message(**user_kwargs)
        prompt = self.processor([[user_message]], mode="generation", n_vq=target_n_vq)
        assistant_message = self.processor.build_assistant_message(audio_codes_list=[target_codes])
        conversation = self.processor([[user_message, assistant_message]], mode="computing_loss", n_vq=target_n_vq)

        full_input_ids = conversation["input_ids"][0].cpu()
        prompt_length = int(prompt["input_ids"][0].shape[0])
        if prompt_length >= full_input_ids.shape[0]:
            raise ValueError("Prompt length must be shorter than the packed teacher-forcing sequence.")

        loss_mask = torch.zeros(full_input_ids.shape[0] - 1, dtype=torch.bool)
        loss_mask[prompt_length - 1 :] = True

        return {
            "input_ids": full_input_ids,
            "loss_mask": loss_mask,
        }

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [item["input_ids"] for item in batch]
        padded = self.processor._pad(input_ids_list)

        full_input_ids = padded["input_ids"].to(torch.long)
        full_attention_mask = padded["attention_mask"].bool()
        loss_masks = pad_sequence(
            [item["loss_mask"] for item in batch],
            batch_first=True,
            padding_value=False,
            padding_side="left",
        )

        labels = full_input_ids[:, 1:, :].clone()
        labels = labels.masked_fill(~loss_masks.unsqueeze(-1), -100)
        labels = labels.masked_fill(~full_attention_mask[:, 1:].unsqueeze(-1), -100)
        labels[:, :, 1:] = labels[:, :, 1:].masked_fill(
            labels[:, :, 1:] == self.processor.model_config.audio_pad_code,
            -100,
        )

        return {
            "input_ids": full_input_ids[:, :-1, :].contiguous(),
            "attention_mask": full_attention_mask[:, :-1].contiguous(),
            "labels": labels.contiguous(),
        }
