"""Persistent Audio8 TTS worker for the shared Transformers 4 runtime."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
project_root_str = str(PROJECT_ROOT)
sys.path = [
    entry for entry in sys.path if str(Path(entry).resolve()) != project_root_str
]
sys.path.insert(0, project_root_str)

from utils.runtimes.protocol import RuntimeJobResponse  # noqa: E402


def _emit(protocol_out, response: RuntimeJobResponse) -> None:
    protocol_out.write(json.dumps(response.to_dict(), ensure_ascii=True) + "\n")
    protocol_out.flush()


def _load_reference_audio(
    payload: Optional[Dict[str, Any]],
) -> Any:
    if payload is None:
        return None
    kind = payload.get("kind")
    if kind == "audio_path":
        audio_path = payload.get("audio_path")
        if not audio_path:
            raise RuntimeError("Audio8 TTS audio_path payload is missing its file path")
        return audio_path
    if kind == "tensor_path":
        tensor_path = payload.get("tensor_path")
        if not tensor_path:
            raise RuntimeError(
                "Audio8 TTS tensor_path payload is missing its file path"
            )
        tensor_payload = torch.load(tensor_path, map_location="cpu")
        return {
            "array": tensor_payload["waveform"],
            "sampling_rate": int(tensor_payload["sample_rate"]),
        }
    raise RuntimeError(f"Unsupported Audio8 TTS reference payload kind: {kind}")


def main() -> int:
    protocol_out = sys.stdout
    sys.stdout = sys.stderr
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
                _emit(
                    protocol_out,
                    RuntimeJobResponse(
                        ok=True,
                        result={"shutdown": True},
                        request_id=request_id,
                    ),
                )
                break

            if action == "ping":
                _emit(
                    protocol_out,
                    RuntimeJobResponse(
                        ok=True,
                        result={"pong": True},
                        request_id=request_id,
                    ),
                )
                continue

            if action == "initialize":
                import transformers
                from packaging.version import Version

                transformers_version = Version(transformers.__version__)
                if not (Version("4.57.0") <= transformers_version < Version("5.0.0")):
                    raise RuntimeError(
                        "Audio8 TTS requires Transformers >=4.57.0,<5; "
                        f"worker loaded Transformers {transformers.__version__}"
                    )

                from engines.audio8_tts.audio8_tts_engine import (
                    Audio8TTSEngine,
                )

                engine = Audio8TTSEngine(
                    model_name=(request.get("model_name") or "Audio8-TTS-Preview-0.6b"),
                    model_dir=payload.get("model_path"),
                    device=request.get("device") or "auto",
                    dtype=payload.get("dtype", "auto"),
                )
                engine._ensure_model_loaded()
                _emit(
                    protocol_out,
                    RuntimeJobResponse(
                        ok=True,
                        result={
                            "model_name": engine.model_name,
                            "sample_rate": engine.SAMPLE_RATE,
                            "transformers_version": (transformers.__version__),
                        },
                        request_id=request_id,
                    ),
                )
                continue

            if engine is None:
                raise RuntimeError(
                    "Audio8 TTS worker received generation before initialization"
                )
            if action != "generate":
                raise RuntimeError(f"Unsupported Audio8 TTS worker action '{action}'")

            reference_audio = _load_reference_audio(payload.get("reference_audio"))
            audio = engine.generate(
                text=payload["text"],
                reference_audio=reference_audio,
                reference_text=payload.get("reference_text"),
                max_new_tokens=int(payload.get("max_new_tokens", 1024)),
                retry_max_new_tokens=int(payload.get("retry_max_new_tokens", 2000)),
                temperature=float(payload.get("temperature", 0.8)),
                top_p=float(payload.get("top_p", 0.95)),
                top_k=int(payload.get("top_k", 50)),
                do_sample=bool(payload.get("do_sample", True)),
                seed=int(payload.get("seed", 42)),
            )
            audio = torch.as_tensor(audio).detach().float().cpu()
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            if audio.ndim != 2 or audio.shape[0] != 1:
                raise RuntimeError(
                    f"Audio8 TTS generated invalid audio shape {tuple(audio.shape)}"
                )

            output_path = Path(payload["output_path"])
            torch.save(
                {
                    "audio": audio.contiguous(),
                    "sample_rate": engine.SAMPLE_RATE,
                },
                output_path,
            )
            _emit(
                protocol_out,
                RuntimeJobResponse(
                    ok=True,
                    result={
                        "output_path": str(output_path),
                        "sample_rate": engine.SAMPLE_RATE,
                    },
                    request_id=request_id,
                ),
            )
        except Exception as exc:
            request_id = (
                request.get("request_id") if isinstance(request, dict) else None
            )
            _emit(
                protocol_out,
                RuntimeJobResponse(
                    ok=False,
                    error=f"{exc}\n{traceback.format_exc()}",
                    request_id=request_id,
                ),
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
