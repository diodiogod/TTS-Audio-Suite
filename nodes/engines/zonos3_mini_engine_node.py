"""
ZONOS3-Mini engine configuration node.
"""

import importlib.util
import os
import sys

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

from engines.zonos3_mini.languages import DEFAULT_LANGUAGE, LANGUAGES, to_zonos_language
from engines.zonos3_mini.voices import DEFAULT_PRESET, PRESETS
from engines.zonos3_mini.zonos3_mini_downloader import Zonos3MiniDownloader
from engines.zonos3_mini.steering_ui import validate_steering


class Zonos3MiniEngineNode(BaseTTSNode):
    """ZONOS3-Mini engine configuration node."""

    @classmethod
    def NAME(cls):
        return "⚙️ ZONOS3-Mini Engine"

    @classmethod
    def INPUT_TYPES(cls):
        try:
            models = Zonos3MiniDownloader().get_available_models()
        except Exception:
            models = ["ZONOS3-Mini"]
        return {
            "required": {
                "model_variant": (models, {
                    "default": "ZONOS3-Mini",
                    "tooltip": (
                        "Choose the ZONOS3-Mini model for speech generation and voice cloning.\n\n"
                        "First use downloads about 10.3 GB. Optional downloads:\n"
                        "• full_mr decoder: about 4 GB extra.\n"
                        "• Reference voice mode: about 0.8 GB extra.\n\n"
                        "Models are stored in models/TTS/zonos3_mini/."
                    )
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Where to run the model.\n\nAuto: use an NVIDIA GPU when available.\nCPU: much slower; requires embedding voice mode."
                }),
                "language": (list(LANGUAGES.keys()), {
                    "default": DEFAULT_LANGUAGE,
                    "tooltip": (
                        "Language to speak. Your voice clip can be in a different language.\n\n"
                        "Switch within the text with tags such as [fr:] or [de:Alice].\n"
                        "For Italian, use [it:] or [it:Name]; plain [it] is read as text."
                    )
                }),
                "voice_preset": (list(PRESETS.keys()), {
                    "default": DEFAULT_PRESET,
                    "tooltip": "Built-in voice to use when no narrator or character voice clip is supplied.\n\nA supplied voice clip takes priority over this preset."
                }),
                "voice_mode": (["embedding", "reference"], {
                    "default": "embedding",
                    "tooltip": (
                        "How the model uses your voice clip.\n\n"
                        "Embedding: extracts the voice's characteristics. Supports the Voice Characteristics node and CPU.\n"
                        "Reference: uses up to the first 60 seconds of the clip directly. Requires an NVIDIA GPU and an extra 0.8 GB download.\n\n"
                        "Neither mode needs a transcript. Try both to compare voice resemblance.\n"
                        "This is separate from the Fast/Reference backend choice."
                    )
                }),
                "temperature": ("FLOAT", {
                    "default": 0.75, "min": 0.1, "max": 2.5, "step": 0.01,
                    "tooltip": "Sampling temperature. Lower is steadier and more monotone; higher is more varied and expressive but less stable. Default 0.75."
                }),
                "top_k": ("INT", {
                    "default": 80, "min": 1, "max": 4096, "step": 1,
                    "tooltip": "Sample each audio token from the k most likely candidates. Lower is safer and flatter; higher allows more variety. Default 80."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.69, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Nucleus sampling: keep the most likely tokens whose probabilities add up to this value. Lower is more conservative. Default 0.69."
                }),
            },
            "optional": {
                "dtype": (["auto", "bfloat16", "float32"], {
                    "default": "auto",
                    "tooltip": "Number precision used by the model.\n\nAuto: bfloat16 on GPU, float32 on CPU.\nbfloat16 uses less GPU memory and is required for Fast."
                }),
                "decoder": (["keep2_plain", "full_mr"], {
                    "default": "keep2_plain",
                    "tooltip": "How generated audio is decoded.\n\nkeep2_plain: standard decoder.\nfull_mr: larger decoder for higher fidelity; adds about 4 GB of downloads and uses more GPU memory."
                }),
                "min_p": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Drop tokens whose probability is below this fraction of the most likely token's. 0 = off. Small values such as 0.05 remove unlikely outliers."
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 2.4, "min": 1.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Penalty on audio tokens that already appeared in the recent window. Prevents droning and stuck sounds; too high can make speech jumpy. 1 = off. Default 2.4."
                }),
                "repetition_window": ("INT", {
                    "default": 96, "min": 0, "max": 1024, "step": 1,
                    "tooltip": "How many recent frames (15 per second) the repetition penalty looks back over. Default 96 (6.4 s)."
                }),
                "repetition_codebooks": ("INT", {
                    "default": 6, "min": 0, "max": 10, "step": 1,
                    "tooltip": "How many of the 10 audio codebooks the repetition penalty applies to, counted from the first (coarsest). Default 6."
                }),
                "speaker_cfg_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "How strongly to reinforce the supplied voice.\n\n1: normal voice conditioning.\nAbove 1: stronger guidance toward the voice, at roughly half speed. Cannot be combined with LACI."
                }),
                "max_new_frames": ("INT", {
                    "default": 0, "min": 0, "max": 9000, "step": 1,
                    "tooltip": (
                        "Maximum audio length for each text chunk.\n\n"
                        "0: use the model's limit.\n"
                        "15 frames = 1 second; 9000 = 10 minutes.\n\n"
                        "Reaching the limit stops speech, even if text remains. Split longer scripts into smaller Unified chunks."
                    )
                }),
                "loudness_bucket": ("INT", {
                    "default": 10, "min": 0, "max": 10, "step": 1,
                    "tooltip": "How loudly the model speaks.\n\n0–9: quietest to loudest.\n10: let the model decide.\n\nChanges the spoken delivery, rather than adjusting volume afterwards."
                }),
                "fmax_bucket": ("INT", {
                    "default": 10, "min": 0, "max": 10, "step": 1,
                    "tooltip": "How much high-frequency detail the voice has.\n\n0–9: muffled to full-band sound.\n10: let the model decide."
                }),
                "speaking_rate_bucket": ("INT", {
                    "default": 30, "min": 0, "max": 30, "step": 1,
                    "tooltip": "How quickly the model speaks.\n\n0–29: slowest to fastest.\n30: let the model decide.\n\nChanges speech during generation; no audio stretching is applied."
                }),
                "laci": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Detect skipped or stalled passages and retry them automatically. Useful for long or difficult text; adds processing time.\n\nRequires speaker guidance and control guidance to both be 1."
                }),
                "short_text": (["auto", "off"], {
                    "default": "auto",
                    "tooltip": "Help the model handle very short text, such as one or two words.\n\nAuto: apply the model's short-text treatment; this may enable LACI.\nOff: generate the text without that treatment."
                }),
                "text_normalization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Convert numbers, dates and currency into spoken words.\n\nFor example, '$5.32' becomes 'five dollars thirty two cents'.\nTurn off when the text is already written exactly as it should be spoken."
                }),
                "silence_prefill_frames": ("INT", {
                    "default": 3, "min": 0, "max": 30, "step": 1,
                    "tooltip": "A short silent lead-in used internally to start speech. It is removed from the output.\n\n15 frames = 1 second. Default: 3 frames."
                }),
                "decode_context": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use the end of the voice clip to help smooth the start of generated speech.\n\nApplies only to reference voice mode."
                }),
                "headroom_db": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 24.0, "step": 0.5,
                    "tooltip": "Output level reduction in dB applied to the decoded audio to leave peak headroom. 0 = decoder's full level."
                }),
                "fade_out_ms": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1000.0, "step": 10.0,
                    "tooltip": "Fade-out applied to the end of each generated chunk, in milliseconds. 0 = off."
                }),
                "engine": (["auto", "fast", "reference"], {
                    "default": "auto",
                    "tooltip": (
                        "Choose how the model generates speech.\n\n"
                        "Auto: use Fast when available, otherwise Reference.\n"
                        "Fast: quicker generation.\n"
                        "Reference: slower; also works on CPU and in float32.\n\n"
                        "Both support the model's full context, for takes of up to 10 minutes.\n"
                        "The backends can sound different, even with the same seed.\n\n"
                        "Fast setup: NVIDIA GPU (cuda:0), bfloat16 and Triton. Start ComfyUI with --disable-cuda-malloc."
                    )
                }),
                "voice_characteristics": ("ZONOS3_VOICE_CHARACTERISTICS", {
                    "tooltip": "Connect 🌈 ZONOS3-Mini Voice Characteristics (gender, age, pitch, texture, recording character, accents). Requires voice_mode 'embedding'."
                }),
                "delivery": ("ZONOS3_DELIVERY", {
                    "tooltip": "Connect 🌈 ZONOS3-Mini Emotion & Delivery (delivery, mood, emotions and effect strength over time). Works with both voice modes; extra control guidance needs embedding mode."
                }),
            },
        }

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_config"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    def create_engine_config(
        self,
        model_variant: str,
        device: str,
        language: str,
        voice_preset: str,
        voice_mode: str,
        temperature: float,
        top_k: int,
        top_p: float,
        dtype: str = "auto",
        engine: str = "auto",
        decoder: str = "keep2_plain",
        min_p: float = 0.0,
        repetition_penalty: float = 2.4,
        repetition_window: int = 96,
        repetition_codebooks: int = 6,
        speaker_cfg_scale: float = 1.0,
        max_new_frames: int = 0,
        loudness_bucket: int = 10,
        fmax_bucket: int = 10,
        speaking_rate_bucket: int = 30,
        laci: bool = False,
        short_text: str = "auto",
        text_normalization: bool = True,
        silence_prefill_frames: int = 3,
        decode_context: bool = False,
        headroom_db: float = 3.0,
        fade_out_ms: float = 0.0,
        voice_characteristics=None,
        delivery=None,
    ) -> tuple:
        for expected, steering in (("voice_characteristics", voice_characteristics), ("delivery", delivery)):
            if isinstance(steering, dict):
                kind = steering.get("kind")
                if kind != expected:
                    raise ValueError(f"ZONOS3-Mini: connect {kind} to the {kind} socket, not {expected}.")
        controls = {}
        for steering in (voice_characteristics, delivery):
            if isinstance(steering, dict):
                for direction, dose in (steering.get("controls") or {}).items():
                    controls[direction] = float(dose)

        config = {
            "engine_type": "zonos3_mini",
            "model_variant": model_variant,
            "device": device,
            "dtype": dtype,
            "engine": engine,
            "decoder": decoder,
            "language": language,
            "voice_preset": voice_preset,
            "voice_mode": voice_mode,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "min_p": float(min_p),
            "repetition_penalty": float(repetition_penalty),
            "repetition_window": int(repetition_window),
            "repetition_codebooks": int(repetition_codebooks),
            "speaker_cfg_scale": float(speaker_cfg_scale),
            "controls_cfg_scale": 1.0,
            "max_new_frames": int(max_new_frames),
            "loudness_bucket": int(loudness_bucket),
            "fmax_bucket": int(fmax_bucket),
            "speaking_rate_bucket": int(speaking_rate_bucket),
            "laci": bool(laci),
            "short_text": short_text,
            "text_normalization": bool(text_normalization),
            "silence_prefill_frames": int(silence_prefill_frames),
            "decode_context": bool(decode_context),
            "headroom_db": float(headroom_db),
            "fade_out_ms": float(fade_out_ms),
            "controls": controls,
            "steer_hold": -1.0,
            "steer_onset_seconds": -1.0,
            "steer_ramp_seconds": -1.0,
        }
        if isinstance(delivery, dict):
            for key in ("steer_hold", "steer_onset_seconds", "steer_ramp_seconds", "controls_cfg_scale"):
                if key in delivery:
                    config[key] = float(delivery[key])

        validate_steering(voice_mode, controls, config["controls_cfg_scale"], bool(laci), float(speaker_cfg_scale))

        print(f"⚙️ ZONOS3-Mini: Configured on {device}")
        print(
            f"   Model: {model_variant} | Language: {language} ({to_zonos_language(language)}) | "
            f"Dtype: {dtype} | Engine: {engine} | Decoder: {decoder}"
        )
        print(
            f"   Voice: mode={voice_mode}, preset={voice_preset} (used without reference audio) | "
            f"LACI: {'on' if laci else 'off'}"
        )
        if controls:
            print(f"   Steering: {controls}")

        engine_data = {
            "engine_type": "zonos3_mini",
            "config": config,
            "adapter_class": "Zonos3MiniEngineAdapter",
            "capabilities": ["tts"],
        }
        return (engine_data,)
