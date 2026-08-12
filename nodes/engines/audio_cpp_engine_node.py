"""ComfyUI configuration node for the generic audio.cpp backend."""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse


def _catalog_module():
    try:
        from utils.audio_cpp import catalog

        return catalog
    except ImportError:
        return None


def _fallback_specs() -> List[Dict[str, Any]]:
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "utils", "audio_cpp", "model_specs")
    )
    specs = []
    for path in glob.glob(os.path.join(root, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            if isinstance(value, dict) and value.get("family"):
                specs.append(value)
        except (OSError, json.JSONDecodeError):
            continue
    return specs


def _family_choices() -> List[str]:
    catalog = _catalog_module()
    if catalog is not None and callable(getattr(catalog, "family_choices", None)):
        choices = list(catalog.family_choices())
    else:
        choices = [spec["family"] for spec in _fallback_specs()]
    choices = sorted({str(choice) for choice in choices if str(choice).strip()})
    return choices or ["qwen3_tts"]


def _package_choices() -> List[str]:
    catalog = _catalog_module()
    if catalog is not None and callable(getattr(catalog, "package_choices", None)):
        choices = list(catalog.package_choices())
    else:
        choices = [
            package.get("id")
            for spec in _fallback_specs()
            for package in spec.get("packages", [])
            if isinstance(package, dict)
        ]
    return ["auto"] + sorted({str(choice) for choice in choices if choice})


def _recommended_package(family: str) -> str:
    catalog = _catalog_module()
    if catalog is not None and callable(getattr(catalog, "recommended_package", None)):
        value = catalog.recommended_package(family)
        if value:
            return str(value)
    for spec in _fallback_specs():
        if spec.get("family") != family:
            continue
        recommended = (spec.get("ui") or {}).get("recommended_package")
        if recommended:
            return str(recommended)
        for package in spec.get("packages", []):
            if package.get("default"):
                return str(package["id"])
    return "auto"


def _resolve_task(family: str, package_id: str, requested: str) -> str:
    requested = str(requested or "auto").lower()
    if requested in {"tts", "clon", "vdes"}:
        return requested
    catalog = _catalog_module()
    if catalog is not None and callable(getattr(catalog, "resolve_task", None)):
        return str(catalog.resolve_task(family, package_id, requested="auto")).lower()
    package_lower = package_id.lower()
    if "voicedesign" in package_lower or "voice_design" in package_lower:
        return "vdes"
    if family in {"chatterbox", "confucius4_tts"}:
        return "clon"
    return "tts"


def _validate_package(family: str, package_id: str) -> None:
    catalog = _catalog_module()
    getter = getattr(catalog, "get_package", None) if catalog is not None else None
    if not callable(getter) or package_id == "auto":
        return
    value = getter(package_id)
    if value is None:
        raise ValueError(f"Unknown audio.cpp package: {package_id}")
    package_family = value.get("family") if isinstance(value, Mapping) else getattr(value, "family", None)
    if package_family and str(package_family) != family:
        raise ValueError(f"audio.cpp package '{package_id}' does not belong to family '{family}'")


class AudioCppEngineNode:
    """Describe either a managed audio.cpp runtime or an existing installation."""

    @classmethod
    def NAME(cls):
        return "⚙️ audio.cpp Engine"

    @classmethod
    def INPUT_TYPES(cls):
        families = _family_choices()
        default_family = "qwen3_tts" if "qwen3_tts" in families else families[0]
        packages = _package_choices()
        return {
            "required": {
                "connection_mode": (
                    ["auto", "external_server", "existing_binary", "managed"],
                    {
                        "default": "auto",
                        "tooltip": "Auto prefers a supplied server or binary, then the suite-managed runtime.",
                    },
                ),
                "family": (families, {"default": default_family}),
                "package_id": (
                    packages,
                    {
                        "default": "auto",
                        "tooltip": "Auto selects the pinned recommended package for the chosen family.",
                    },
                ),
                "task": (
                    ["auto", "tts", "clon", "vdes"],
                    {
                        "default": "auto",
                        "tooltip": "Runtime task id. Auto follows the package; existing servers remain authoritative.",
                    },
                ),
                "backend": (["auto", "cuda", "cpu", "vulkan", "metal", "hip"], {"default": "auto"}),
                "device": ("INT", {"default": 0, "min": 0, "max": 31}),
                "threads": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 128,
                        "tooltip": "Native backend/OpenMP workers. Four matches the audio.cpp CLI default; tune for your CPU.",
                    },
                ),
                "language": ("STRING", {"default": "auto"}),
            },
            "optional": {
                "server_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Required only for external_server mode, for example http://127.0.0.1:8080.",
                    },
                ),
                "binary_path": ("STRING", {"default": ""}),
                "model_path": ("STRING", {"default": ""}),
                "model_id": ("STRING", {"default": ""}),
                "voice_id": ("STRING", {"default": ""}),
                "instruct": ("STRING", {"default": "", "multiline": True}),
                "temperature": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 5.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": -1, "min": -1, "max": 1000}),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 5.0, "step": 0.05},
                ),
                "max_tokens": ("INT", {"default": 0, "min": 0, "max": 131072}),
                "max_steps": ("INT", {"default": 0, "min": 0, "max": 4096}),
                "num_inference_steps": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": -1.0, "min": -1.0, "max": 100.0, "step": 0.05},
                ),
                "advanced_json": (
                    "STRING",
                    {
                        "default": "{}",
                        "multiline": True,
                        "tooltip": "Model-specific audio.cpp request options as a JSON object.",
                    },
                ),
                "auto_download_runtime": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Explicitly allow installing the pinned audio.cpp runtime into managed storage.",
                    },
                ),
                "auto_download_model": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Explicitly allow downloading the selected package into models/TTS/audio.cpp/models.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_config"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    def create_engine_config(
        self,
        connection_mode: str,
        family: str,
        package_id: str,
        task: str,
        backend: str,
        device: int,
        threads: int,
        language: str,
        server_url: str = "",
        binary_path: str = "",
        model_path: str = "",
        model_id: str = "",
        voice_id: str = "",
        instruct: str = "",
        temperature: float = -1.0,
        top_p: float = -1.0,
        top_k: int = -1,
        repetition_penalty: float = -1.0,
        max_tokens: int = 0,
        max_steps: int = 0,
        num_inference_steps: int = 0,
        guidance_scale: float = -1.0,
        advanced_json: str = "{}",
        auto_download_runtime: bool = False,
        auto_download_model: bool = False,
    ) -> tuple:
        mode = str(connection_mode).strip().lower()
        if mode not in {"auto", "external_server", "existing_binary", "managed"}:
            raise ValueError(f"Unsupported audio.cpp connection mode: {connection_mode}")
        family = str(family).strip()
        package_id = str(package_id or "auto").strip()
        if not family:
            raise ValueError("audio.cpp family is required")

        url = str(server_url or "").strip().rstrip("/")
        binary = os.path.abspath(os.path.expanduser(binary_path)) if binary_path.strip() else ""
        model = os.path.abspath(os.path.expanduser(model_path)) if model_path.strip() else ""
        if mode == "external_server":
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("audio.cpp external_server mode requires a valid HTTP(S) server_url")
        if mode == "existing_binary":
            if not binary:
                raise ValueError("audio.cpp existing_binary mode requires binary_path")
            if not os.path.isfile(binary):
                raise FileNotFoundError(f"audio.cpp binary not found: {binary}")

        try:
            advanced = json.loads(advanced_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid audio.cpp advanced JSON: {exc.msg}") from exc
        if not isinstance(advanced, Mapping):
            raise ValueError("audio.cpp advanced JSON must contain an object")

        uses_existing_server = mode == "external_server"
        if package_id == "auto" and not uses_existing_server:
            package_id = _recommended_package(family)
        if not uses_existing_server:
            _validate_package(family, package_id)
            resolved_task = _resolve_task(family, package_id, task)
        else:
            # The loaded model reported by /v1/models owns this decision.
            resolved_task = str(task or "auto").lower()

        config: Dict[str, Any] = {
            "engine_type": "audio_cpp",
            "connection_mode": mode,
            "family": family,
            "package_id": package_id,
            "requested_task": str(task or "auto").lower(),
            "task": resolved_task,
            "backend": str(backend).lower(),
            "device": int(device),
            "threads": int(threads),
            "language": str(language or "auto"),
            "server_url": url,
            "external_server_url": url,
            "binary_path": binary,
            "model_path": model,
            "model_id": str(model_id or "").strip(),
            "voice_id": str(voice_id or "").strip(),
            "instruct": str(instruct or "").strip(),
            "advanced_options": dict(advanced),
            "auto_download_runtime": bool(auto_download_runtime),
            "auto_download_model": bool(auto_download_model),
        }
        optional_values = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
            "guidance_scale": float(guidance_scale),
        }
        for key, value in optional_values.items():
            if value >= 0:
                config[key] = value
        for key, value in {
            "max_tokens": int(max_tokens),
            "max_steps": int(max_steps),
            "num_inference_steps": int(num_inference_steps),
        }.items():
            if value > 0:
                config[key] = value

        capabilities = ["tts", "voice_design"] if resolved_task == "vdes" else ["tts"]
        return ({"engine_type": "audio_cpp", "config": config, "capabilities": capabilities},)


NODE_CLASS_MAPPINGS = {"AudioCppEngineNode": AudioCppEngineNode}
NODE_DISPLAY_NAME_MAPPINGS = {"AudioCppEngineNode": "⚙️ audio.cpp Engine"}
