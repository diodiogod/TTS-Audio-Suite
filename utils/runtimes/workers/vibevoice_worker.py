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


def _load_voice_ref(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None

    kind = payload.get("kind")
    if kind == "audio_path":
        return {
            "audio_path": payload.get("audio_path"),
            "reference_text": payload.get("reference_text", ""),
            "character_name": payload.get("character_name", "narrator"),
        }

    if kind == "tensor_path":
        tensor_path = payload.get("tensor_path")
        if not tensor_path:
            raise RuntimeError("tensor_path payload missing file path")
        tensor_payload = torch.load(tensor_path, map_location="cpu")
        return {
            "waveform": tensor_payload["waveform"],
            "sample_rate": tensor_payload.get("sample_rate", 24000),
            "reference_text": payload.get("reference_text", ""),
            "character_name": payload.get("character_name", "narrator"),
        }

    raise RuntimeError(f"Unsupported isolated voice payload kind: {kind}")


def main() -> int:
    protocol_out = sys.stdout
    sys.stdout = sys.stderr

    from engines.vibevoice_engine import VibeVoiceEngine

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
                engine = VibeVoiceEngine()
                engine.initialize_engine(
                    model_name=request.get("model_name") or "vibevoice-1.5B",
                    device=request.get("device") or "auto",
                    attention_mode=payload.get("attention_mode", "auto"),
                    quantize_llm_4bit=payload.get("quantize_llm_4bit", False),
                )
                _emit(
                    protocol_out,
                    RuntimeJobResponse(
                        ok=True,
                        result={
                            "is_kugelaudio": bool(getattr(engine, "is_kugelaudio", False)),
                            "model_name": request.get("model_name"),
                        },
                        request_id=request_id,
                    ),
                )
                continue

            if engine is None:
                raise RuntimeError("VibeVoice worker received generation before initialization")

            if action == "generate_from_refs":
                voice_refs = [_load_voice_ref(item) for item in payload.get("voice_refs", [])]
                voice_samples = engine._prepare_voice_samples(voice_refs)
                result = engine.generate_speech(
                    text=payload["text"],
                    voice_samples=voice_samples,
                    cfg_scale=payload.get("cfg_scale", 1.3),
                    seed=payload.get("seed", 42),
                    use_sampling=payload.get("use_sampling", False),
                    temperature=payload.get("temperature", 0.95),
                    top_p=payload.get("top_p", 0.95),
                    inference_steps=payload.get("inference_steps", 20),
                    max_new_tokens=payload.get("max_new_tokens"),
                    enable_cache=payload.get("enable_cache", True),
                    character=payload.get("character", "narrator"),
                    stable_audio_component=payload.get("stable_audio_component", ""),
                    multi_speaker_mode=payload.get("multi_speaker_mode", "Custom Character Switching"),
                )
                output_path = payload.get("output_path")
                if not output_path:
                    raise RuntimeError("generate_from_refs payload missing output_path")
                torch.save(result, output_path)
                _emit(
                    protocol_out,
                    RuntimeJobResponse(
                        ok=True,
                        result={"output_path": output_path},
                        request_id=request_id,
                    ),
                )
                continue

            raise RuntimeError(f"Unsupported VibeVoice worker action '{action}'")

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
