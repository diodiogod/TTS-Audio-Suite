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
from engines.zonos3_mini.steering_ui import control_catalog


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
                    "tooltip": "Zyphra ZONOS3-Mini (Apache-2.0 weights), 44.1 kHz multilingual TTS with voice cloning. First use downloads about 10.3 GB into ComfyUI/models/TTS/zonos3_mini/ZONOS3-Mini/; the full_mr decoder adds 4 GB and voice_mode 'reference' adds 0.8 GB, each fetched only when first used. Uses an adapted version of Zyphra's fast CUDA-graph engine or their plain-PyTorch reference implementation (see 'engine'). 'local:<folder>' entries are complete model folders you placed under models/TTS/zonos3_mini/."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run ZONOS3-Mini on. Auto uses CUDA when available. CPU works with voice_mode 'embedding' but is far slower than real time."
                }),
                "language": (list(LANGUAGES.keys()), {
                    "default": DEFAULT_LANGUAGE,
                    "tooltip": "Language of the text. Selects text normalization (numbers, dates, currency) and the alignment heads used by LACI; the voice is independent of it, so cross-lingual cloning is simply a reference voice in one language plus a different language here. Inline language tags such as [fr:] or [de:Alice] switch per segment; for Italian use [it:] or [it:Name] because a bare [it] is treated as text."
                }),
                "voice_preset": (list(PRESETS.keys()), {
                    "default": DEFAULT_PRESET,
                    "tooltip": "Built-in ZONOS3 voice, used only when no narrator or character reference audio is connected. Root voices: AmericanFemale, AmericanMale, BritishFemale. English: Cillian (Irish), Imogen/Rupert (British), Jett (energetic), Morgana (witch), Sakura (anime), Savannah/Wyatt (American), Thaddeus (announcer), Willow (ASMR). Others: Gaspard (French), Liesel/Wolfgang (German), Ananya/Arjun (Hindi), Ginevra/Lorenzo (Italian), Minjun/Seoyeon (Korean), Haoran/Meilin (Mandarin), Leonor/Thiago (Portuguese), Rodrigo/Ximena (Spanish), Chaitanya/Sravya (Telugu)."
                }),
                "voice_mode": (["embedding", "reference"], {
                    "default": "embedding",
                    "tooltip": "How a voice is given to the model. embedding: the clip is reduced to a 192-number speaker embedding; works on CPU and is required for Voice Characteristics. reference: the clip (up to its first 60 s) is encoded to codec codes and used as an audio prefix; closest cloning of timbre and recording character, CUDA only, downloads an extra 0.8 GB encoder. No reference transcript is needed in either mode."
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
                    "tooltip": "Model precision. Auto uses bfloat16 on CUDA (less VRAM, faster) and float32 on CPU. float32 on CUDA is mainly for diagnosis."
                }),
                "decoder": (["keep2_plain", "full_mr"], {
                    "default": "keep2_plain",
                    "tooltip": "Audio decoder head. keep2_plain is the model default (0.5 GB). full_mr is the larger multi-resolution head (extra 4 GB download, more VRAM) for higher decoding fidelity."
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
                    "tooltip": "Classifier-free guidance toward the reference voice. 1 = off. Above 1 pushes harder toward the speaker but runs the model twice per frame (about half speed) and cannot be combined with LACI."
                }),
                "max_new_frames": ("INT", {
                    "default": 0, "min": 0, "max": 9000, "step": 1,
                    "tooltip": "Upper limit on generated audio per text chunk: 15 frames = 1 second; 9000 frames = 10 minutes. 0 uses the selected backend's remaining context. Fast reserves 4096 positions shared by text, reference audio and generated frames, so its audio limit is below 4 minutes 33 seconds per chunk. Reference retains the model's 32768-position context for longer takes. A larger value cannot override the backend's context limit. Reaching either limit stops generation; unread text is not automatically continued. Use smaller Unified chunks for longer scripts."
                }),
                "loudness_bucket": ("INT", {
                    "default": 10, "min": 0, "max": 10, "step": 1,
                    "tooltip": "Native loudness conditioning, 0 (quietest) to 9 (loudest). 10 = unknown, lets the model decide. This is a model input, not post-processing gain."
                }),
                "fmax_bucket": ("INT", {
                    "default": 10, "min": 0, "max": 10, "step": 1,
                    "tooltip": "Native audio-bandwidth conditioning, 0 (narrow, muffled) to 9 (full band). 10 = unknown, lets the model decide."
                }),
                "speaking_rate_bucket": ("INT", {
                    "default": 30, "min": 0, "max": 30, "step": 1,
                    "tooltip": "Native speaking-rate conditioning, 0 (slowest) to 29 (fastest). 30 = unknown, lets the model decide. Changes how the model speaks rather than time-stretching audio."
                }),
                "laci": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "LACI watches the model's attention over the text while it speaks, detects skipped or stalled passages and regenerates that stretch (up to 8 retries). Helps long or difficult text; costs a little speed. Needs speaker_cfg_scale and controls_cfg_scale at 1."
                }),
                "short_text": (["auto", "off"], {
                    "default": "auto",
                    "tooltip": "Handling of very short inputs (one or two words), which the model otherwise tends to garble. auto applies Zyphra's short-text treatment, which may enable LACI for that text. off sends the text as is."
                }),
                "text_normalization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Spell out numbers, dates, currency and abbreviations in the selected language before synthesis ('$5.32' becomes 'five dollars thirty two cents'). Disable when the text is already written as it should be spoken."
                }),
                "silence_prefill_frames": ("INT", {
                    "default": 3, "min": 0, "max": 30, "step": 1,
                    "tooltip": "Silent frames placed before generation starts (15 frames = 1 second), giving the model a clean onset. They are not part of the output audio. Default 3."
                }),
                "decode_context": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Reference voice mode only: decode the generated audio with the end of the reference clip as context, which can smooth the first instants of speech. Ignored in embedding mode."
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
                    "tooltip": "Choose how the model runs. Fast uses an adapted version of Zyphra's CUDA-graph backend for faster generation. It needs CUDA (cuda:0), bfloat16, Triton (triton-windows on Windows) and the native PyTorch GPU allocator. Start ComfyUI with --disable-cuda-malloc for Fast; cudaMallocAsync is incompatible with its CUDA graphs, so Auto uses Reference with that allocator. To reduce VRAM reserved for its GPU cache, this integration limits Fast to 4096 context positions shared by text, reference audio and generated speech: less than 4 minutes 33 seconds of audio per chunk, with the actual limit reduced by the inputs. Use smaller Unified chunks or Reference for longer individual takes. Reference preserves the model's 32768-position context and advertised up-to-10-minute capability; it also works on CPU and in float32. Auto prefers Fast, including its smaller context, and reports when initialization falls back to Reference. The same seed can produce different audio on each backend."
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

        if voice_mode != "embedding":
            embedding_ids = {effect["id"] for effect in control_catalog()["effects"] if effect["embedding_only"]}
            if any(dose != 0 and direction in embedding_ids for direction, dose in controls.items()):
                raise ValueError("ZONOS3-Mini Voice Characteristics requires embedding voice mode. Select embedding or disconnect Voice Characteristics before generating in reference mode.")

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

        if config["controls_cfg_scale"] != 1.0:
            if voice_mode != "embedding":
                raise ValueError("ZONOS3-Mini control guidance requires embedding voice mode. Use normal guidance (1) in Emotion & Delivery for reference mode.")
            if laci or float(speaker_cfg_scale) != 1.0:
                raise ValueError("ZONOS3-Mini control guidance cannot be combined with LACI or speaker guidance. Use normal control guidance (1).")
            hidden_ids = {effect["id"] for effect in control_catalog()["effects"] if not effect["embedding_only"]}
            if not any(dose != 0 and direction in hidden_ids for direction, dose in controls.items()):
                raise ValueError("ZONOS3-Mini control guidance requires an active delivery effect. Use normal guidance (1) while effects are off.")

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
