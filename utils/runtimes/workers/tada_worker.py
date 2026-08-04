from __future__ import annotations

"""JSON-line worker for TADA's shared Transformers 4 runtime."""

import json
import os
import sys
import traceback
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENGINES_DIR = PROJECT_ROOT / "engines"

# ComfyUI may add the suite's engines directory directly to PYTHONPATH. In this
# worker that would make engines/tada shadow the official hume-tada package.
sys.path[:] = [
    path
    for path in sys.path
    if Path(path or os.getcwd()).resolve() != ENGINES_DIR.resolve()
]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from utils.runtimes.protocol import RuntimeJobResponse


def _emit(stream, response: RuntimeJobResponse) -> None:
    stream.write(json.dumps(response.to_dict(), ensure_ascii=True) + "\n")
    stream.flush()


def _load_reference(payload):
    if not isinstance(payload, dict):
        raise ValueError("TADA worker received an invalid reference-audio payload.")
    kind = payload.get("kind")
    if kind == "audio_path":
        return payload.get("audio_path")
    if kind == "tensor_path":
        tensor_path = payload.get("tensor_path")
        if not tensor_path:
            raise ValueError("TADA tensor reference is missing tensor_path.")
        saved = torch.load(tensor_path, map_location="cpu")
        return saved["waveform"], int(saved.get("sample_rate", 24000))
    raise ValueError(f"Unsupported TADA reference-audio payload kind: {kind}")


def main() -> int:
    protocol_out = sys.stdout
    sys.stdout = sys.stderr

    from engines.tada.tada_engine import TadaEngine

    engine = None
    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        request = None
        try:
            request = json.loads(stripped)
            request_id = request.get("request_id")
            action = request.get("action")
            payload = request.get("payload") or {}

            if action == "shutdown":
                if engine is not None:
                    engine.cleanup()
                    engine = None
                _emit(protocol_out, RuntimeJobResponse(ok=True, result={"shutdown": True}, request_id=request_id))
                break

            if action == "ping":
                _emit(protocol_out, RuntimeJobResponse(ok=True, result={"pong": True}, request_id=request_id))
                continue

            if action == "initialize":
                if request.get("runtime_profile") != "vibevoice_transformers4_shared":
                    raise RuntimeError("TADA can only initialize in vibevoice_transformers4_shared.")
                if engine is not None:
                    engine.cleanup()
                engine = TadaEngine(
                    model_name=request.get("model_name") or "TADA-1B",
                    model_path=payload["model_path"],
                    codec_path=payload["codec_path"],
                    tokenizer_path=payload["tokenizer_path"],
                    device=request.get("device") or "auto",
                    dtype=payload.get("dtype", "auto"),
                    attn_implementation=payload.get("attn_implementation", "sdpa"),
                    use_torch_compile=payload.get("use_torch_compile", False),
                    prompt_cache_size=payload.get("prompt_cache_size", 8),
                )
                _emit(
                    protocol_out,
                    RuntimeJobResponse(
                        ok=True,
                        result={"model_name": request.get("model_name"), "sample_rate": 24000},
                        request_id=request_id,
                    ),
                )
                continue

            if action == "cleanup":
                if engine is not None:
                    engine.cleanup()
                    engine = None
                _emit(protocol_out, RuntimeJobResponse(ok=True, result={"cleaned": True}, request_id=request_id))
                continue

            if engine is None:
                raise RuntimeError("TADA worker received a generation request before initialization.")

            if action == "generate":
                audio, sample_rate = engine.generate_speech(
                    text=payload["text"],
                    reference_audio=_load_reference(payload.get("reference_audio")),
                    reference_text=payload["reference_text"],
                    seed=payload.get("seed", 42),
                    language=payload.get("language", "English"),
                    acoustic_cfg_scale=payload.get("acoustic_cfg_scale", 1.6),
                    duration_cfg_scale=payload.get("duration_cfg_scale", 1.0),
                    cfg_schedule=payload.get("cfg_schedule", "cosine"),
                    time_schedule=payload.get("time_schedule", "logsnr"),
                    num_flow_matching_steps=payload.get("num_flow_matching_steps", 10),
                    noise_temperature=payload.get("noise_temperature", 0.9),
                    speed_up_factor=payload.get("speed_up_factor"),
                    num_transition_steps=payload.get("num_transition_steps", 5),
                    negative_condition_source=payload.get(
                        "negative_condition_source", "negative_step_output"
                    ),
                )
                output_path = payload["output_path"]
                torch.save(
                    {"audio": audio.detach().cpu().float(), "sample_rate": int(sample_rate)},
                    output_path,
                )
                _emit(
                    protocol_out,
                    RuntimeJobResponse(
                        ok=True,
                        result={"output_path": output_path, "sample_rate": int(sample_rate)},
                        request_id=request_id,
                    ),
                )
                continue

            raise RuntimeError(f"Unsupported TADA worker action '{action}'")

        except Exception as exc:
            request_id = request.get("request_id") if isinstance(request, dict) else None
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
