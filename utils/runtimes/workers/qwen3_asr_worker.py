from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
QWEN3_ASR_IMPL = PROJECT_ROOT / "engines" / "qwen3_asr" / "impl"
if str(QWEN3_ASR_IMPL) not in sys.path:
    sys.path.insert(0, str(QWEN3_ASR_IMPL))

from utils.runtimes.protocol import RuntimeJobResponse


def _emit(protocol_out, response: RuntimeJobResponse) -> None:
    protocol_out.write(json.dumps(response.to_dict(), ensure_ascii=True) + "\n")
    protocol_out.flush()


def _resolve_torch_dtype(precision: str):
    precision = (precision or "auto").lower()
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if precision in dtype_map:
        return dtype_map[precision]

    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def _sanitize_loader_settings(device: str, torch_dtype, attn_implementation: str):
    resolved_attn = "sdpa" if attn_implementation in ("auto", None, "") else attn_implementation
    if str(device) == "cpu":
        if torch_dtype == torch.float16:
            torch_dtype = torch.float32
        if resolved_attn == "flash_attention_2":
            resolved_attn = "sdpa"
    return torch_dtype, resolved_attn


def _load_audio_payload(payload: Dict[str, Any]) -> List[Any]:
    items = []
    for item in payload.get("items", []):
        kind = item.get("kind")
        if kind == "audio_path":
            items.append(item.get("audio_path"))
            continue
        if kind == "tensor_path":
            tensor_payload = torch.load(item["tensor_path"], map_location="cpu")
            waveform = tensor_payload["waveform"]
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.squeeze().cpu().numpy()
            items.append((waveform, int(tensor_payload.get("sample_rate", 16000))))
            continue
        raise RuntimeError(f"Unsupported isolated Qwen3-ASR audio payload kind: {kind}")
    return items


def _serialize_results(results) -> List[Dict[str, Any]]:
    payload = []
    for result in results:
        time_stamps = None
        raw_time_stamps = getattr(result, "time_stamps", None)
        if raw_time_stamps is not None:
            time_stamps = [
                {
                    "text": getattr(ts, "text", ""),
                    "start_time": float(getattr(ts, "start_time", 0.0)),
                    "end_time": float(getattr(ts, "end_time", 0.0)),
                }
                for ts in raw_time_stamps
            ]
        payload.append(
            {
                "language": getattr(result, "language", ""),
                "text": getattr(result, "text", ""),
                "time_stamps": time_stamps,
            }
        )
    return payload


def _serialize_alignment_results(results) -> List[List[Dict[str, Any]]]:
    payload = []
    for result in results:
        payload.append(
            [
                {
                    "text": getattr(item, "text", ""),
                    "start_time": float(getattr(item, "start_time", 0.0)),
                    "end_time": float(getattr(item, "end_time", 0.0)),
                }
                for item in result
            ]
        )
    return payload


def _build_streamer_factory():
    from utils.asr.progress_callback import ASRProgressStreamer

    return lambda max_new_tokens, _pb: ASRProgressStreamer(max_new_tokens or 256, None, label="ASR")


def main() -> int:
    protocol_out = sys.stdout
    sys.stdout = sys.stderr

    from engines.qwen3_tts.qwen3_asr_downloader import Qwen3ASRDownloader
    from engines.qwen3_asr.impl.qwen_asr.inference.qwen3_asr import Qwen3ASRModel
    from engines.qwen3_asr.impl.qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner

    engine = None

    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue

        request = None
        try:
            request = json.loads(stripped)
            action = request.get("action")
            payload = request.get("payload") or {}
            request_id = request.get("request_id")

            if action == "shutdown":
                _emit(protocol_out, RuntimeJobResponse(ok=True, result={"shutdown": True}, request_id=request_id))
                break

            if action == "ping":
                _emit(protocol_out, RuntimeJobResponse(ok=True, result={"pong": True}, request_id=request_id))
                continue

            if action == "initialize":
                downloader = Qwen3ASRDownloader()
                resolved_model_path = downloader.resolve_model_path(payload.get("model_path") or request.get("model_name"))
                device = request.get("device") or "auto"
                precision = payload.get("precision", "auto")
                attn_implementation = payload.get("attn_implementation", "sdpa")
                max_new_tokens = int(payload.get("max_new_tokens", 256))
                forced_aligner = payload.get("forced_aligner")
                forced_aligner_kwargs = payload.get("forced_aligner_kwargs")
                model_type = payload.get("model_type", "asr")

                torch_dtype = _resolve_torch_dtype(precision)
                torch_dtype, resolved_attn = _sanitize_loader_settings(device, torch_dtype, attn_implementation)

                if model_type == "aligner":
                    engine = Qwen3ForcedAligner.from_pretrained(
                        resolved_model_path,
                        dtype=torch_dtype,
                        device_map=device,
                        attn_implementation=resolved_attn,
                    )
                else:
                    loader_kwargs = {
                        "pretrained_model_name_or_path": resolved_model_path,
                        "dtype": torch_dtype,
                        "device_map": device,
                        "max_new_tokens": max_new_tokens,
                        "attn_implementation": resolved_attn,
                    }

                    if forced_aligner:
                        loader_kwargs["forced_aligner"] = downloader.resolve_model_path(forced_aligner)
                        loader_kwargs["forced_aligner_kwargs"] = forced_aligner_kwargs or {
                            "dtype": torch_dtype,
                            "device_map": device,
                            "attn_implementation": resolved_attn,
                        }

                    engine = Qwen3ASRModel.from_pretrained(**loader_kwargs)
                    engine._streamer_factory = _build_streamer_factory()
                    engine._progress_bar = None
                _emit(protocol_out, RuntimeJobResponse(ok=True, result={"model_name": request.get("model_name")}, request_id=request_id))
                continue

            if engine is None:
                raise RuntimeError("Qwen3-ASR worker received action before initialization")

            if action == "transcribe":
                results = engine.transcribe(
                    audio=_load_audio_payload(payload.get("audio") or {}),
                    context=payload.get("context", ""),
                    language=payload.get("language"),
                    return_time_stamps=bool(payload.get("return_time_stamps", False)),
                )
                _emit(
                    protocol_out,
                    RuntimeJobResponse(
                        ok=True,
                        result={"results": _serialize_results(results)},
                        request_id=request_id,
                    ),
                )
                continue

            if action == "align":
                results = engine.align(
                    audio=_load_audio_payload(payload.get("audio") or {}),
                    text=payload.get("text"),
                    language=payload.get("language"),
                )
                _emit(
                    protocol_out,
                    RuntimeJobResponse(
                        ok=True,
                        result={"results": _serialize_alignment_results(results)},
                        request_id=request_id,
                    ),
                )
                continue

            raise RuntimeError(f"Unsupported Qwen3-ASR worker action '{action}'")

        except Exception as exc:
            _emit(
                protocol_out,
                RuntimeJobResponse(
                    ok=False,
                    error=f"{exc}\n{traceback.format_exc()}",
                    request_id=(request.get("request_id") if isinstance(request, dict) else None),
                ),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
