from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.runtimes.protocol import RuntimeJobResponse


def _emit(protocol_out, response: RuntimeJobResponse) -> None:
    protocol_out.write(json.dumps(response.to_dict(), ensure_ascii=True) + "\n")
    protocol_out.flush()


def _load_audio_ref(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None

    kind = payload.get("kind")
    if kind == "audio_path":
        return payload.get("audio_path")

    if kind == "tensor_path":
        tensor_path = payload.get("tensor_path")
        if not tensor_path:
            raise RuntimeError("tensor_path payload missing file path")
        tensor_payload = torch.load(tensor_path, map_location="cpu")
        return {
            "waveform": tensor_payload["waveform"],
            "sample_rate": tensor_payload.get("sample_rate", 24000),
        }

    raise RuntimeError(f"Unsupported isolated Higgs audio payload kind: {kind}")


def _move_audio_result_to_cpu(audio_result: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in audio_result.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu()
        else:
            result[key] = value
    return result


def main() -> int:
    protocol_out = sys.stdout
    sys.stdout = sys.stderr

    from engines.higgs_audio.higgs_audio import HiggsAudioEngine

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
                engine = HiggsAudioEngine()
                engine.initialize_engine(
                    model_path=payload.get("model_path") or request.get("model_name") or "higgs-audio-v2-3B",
                    tokenizer_path=payload.get("tokenizer_path") or "bosonai/higgs-audio-v2-tokenizer",
                    device=request.get("device") or "auto",
                    enable_cuda_graphs=payload.get("enable_cuda_graphs", True),
                )
                _emit(
                    protocol_out,
                    RuntimeJobResponse(
                        ok=True,
                        result={"model_name": request.get("model_name")},
                        request_id=request_id,
                    ),
                )
                continue

            if engine is None:
                raise RuntimeError("Higgs Audio worker received action before initialization")

            if action == "generate_stateless":
                with torch.inference_mode():
                    audio_result, generation_info = engine.generate(
                        text=payload["text"],
                        reference_audio=_load_audio_ref(payload.get("reference_audio")),
                        reference_text=payload.get("reference_text", ""),
                        audio_priority=payload.get("audio_priority", "auto"),
                        system_prompt=payload.get("system_prompt", "Generate audio following instruction."),
                        max_new_tokens=payload.get("max_new_tokens", 2048),
                        temperature=payload.get("temperature", 0.8),
                        top_p=payload.get("top_p", 0.6),
                        top_k=payload.get("top_k", 80),
                        force_audio_gen=payload.get("force_audio_gen", True),
                        ras_win_len=payload.get("ras_win_len", 7),
                        ras_max_num_repeat=payload.get("ras_max_num_repeat", 2),
                        enable_chunking=payload.get("enable_chunking", True),
                        max_tokens_per_chunk=payload.get("max_tokens_per_chunk", 225),
                        silence_between_chunks_ms=payload.get("silence_between_chunks_ms", 100),
                        enable_cache=payload.get("enable_cache", True),
                        character=payload.get("character", "narrator"),
                        seed=payload.get("seed", -1),
                    )
            elif action == "generate_native_multispeaker_stateless":
                with torch.inference_mode():
                    audio_result, generation_info = engine.generate_native_multispeaker(
                        text=payload["text"],
                        primary_reference_audio=_load_audio_ref(payload.get("primary_reference_audio")),
                        primary_reference_text=payload.get("primary_reference_text", ""),
                        secondary_reference_audio=_load_audio_ref(payload.get("secondary_reference_audio")),
                        secondary_reference_text=payload.get("secondary_reference_text", ""),
                        use_system_context=payload.get("use_system_context", True),
                        system_prompt=payload.get("system_prompt", "Generate audio following instruction."),
                        max_new_tokens=payload.get("max_new_tokens", 2048),
                        temperature=payload.get("temperature", 0.8),
                        top_p=payload.get("top_p", 0.6),
                        top_k=payload.get("top_k", 80),
                        force_audio_gen=payload.get("force_audio_gen", True),
                        ras_win_len=payload.get("ras_win_len", 7),
                        ras_max_num_repeat=payload.get("ras_max_num_repeat", 2),
                        enable_cache=payload.get("enable_cache", True),
                        character=payload.get("character", "SPEAKER0"),
                        seed=payload.get("seed", -1),
                    )
            else:
                raise RuntimeError(f"Unsupported Higgs Audio worker action '{action}'")

            output_path = payload.get("output_path")
            if not output_path:
                raise RuntimeError(f"{action} payload missing output_path")
            torch.save(
                {
                    "audio_result": _move_audio_result_to_cpu(audio_result),
                    "generation_info": generation_info,
                },
                output_path,
            )
            _emit(
                protocol_out,
                RuntimeJobResponse(
                    ok=True,
                    result={"output_path": output_path},
                    request_id=request_id,
                ),
            )

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
