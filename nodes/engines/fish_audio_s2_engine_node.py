"""Fish Audio S2 Pro engine configuration node."""

import importlib.util
import json
import os
import re
import sys

current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

base_spec = importlib.util.spec_from_file_location("base_node_module", os.path.join(nodes_dir, "base", "base_node.py"))
base_module = importlib.util.module_from_spec(base_spec)
sys.modules.setdefault("base_node_module", base_module)
base_spec.loader.exec_module(base_module)

SPEAKER_INPUT_PATTERN = re.compile(r"^speaker([2-9]\d*)$")


class DynamicSpeakerInputs(dict):
    @staticmethod
    def _is_speaker_key(key):
        return isinstance(key, str) and SPEAKER_INPUT_PATTERN.fullmatch(key) is not None

    def __contains__(self, key):
        return super().__contains__(key) or self._is_speaker_key(key)

    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        if self._is_speaker_key(key):
            return ("*", {
                "forceInput": True,
                "tooltip": "Positional Fish speaker reference. Connect a Character Voices output with exact reference transcript.",
            })
        raise KeyError(key)

    def get(self, key, default=None):
        return self[key] if key in self else default


class FishAudioS2EngineNode(base_module.BaseTTSNode):
    @classmethod
    def NAME(cls):
        return "⚙️ Fish Audio S2 Pro Engine"

    @classmethod
    def _get_model_options(cls):
        """List local checkpoints first, followed by downloadable variants."""
        downloadable = ["s2-pro", "s2-pro-fp8"]
        try:
            from engines.fish_audio_s2.downloader import FishAudioS2Downloader

            local = [
                f"local:{os.path.basename(os.path.normpath(path))}"
                for path in FishAudioS2Downloader.local_model_paths()
            ]
        except Exception:
            local = []
        return local + [option for option in downloadable if option not in local]

    @classmethod
    def INPUT_TYPES(cls):
        optional = DynamicSpeakerInputs({
            "normalize": ("BOOLEAN", {"default": True, "tooltip": "Official English/Chinese number normalization."}),
            "cache_reference": ("BOOLEAN", {"default": True, "tooltip": "Cache encoded reference codes inside the Fish runtime."}),
            "precision": (["bfloat16", "float16"], {"default": "bfloat16"}),
            "compile": ("BOOLEAN", {"default": True, "tooltip": "Enable the official torch.compile path (recommended). The first generation is slower, but later runs are much faster; testing showed about 6 it/s to 41 it/s, roughly a 6.8x speedup. This can also be tested with BNB quantization, although bitsandbytes compatibility may vary by version."}),
            "quantization": (["none", "bnb_int8", "bnb_nf4"], {
                "default": "none",
                "tooltip": (
                    "Optional on-the-fly quantization for the official s2-pro checkpoint.\n\n"
                    "• none: Full checkpoint loading\n"
                    "• bnb_int8: BitsAndBytes INT8 load-time quantization\n"
                    "• bnb_nf4: BitsAndBytes NF4 load-time quantization\n\n"
                    "BNB quantization reuses the official checkpoint and requires CUDA plus bitsandbytes. "
                    "It is ignored for the separate s2-pro-fp8 checkpoint."
                ),
            }),
            "multi_speaker_mode": (["Native Multi-Speaker", "Custom Character Switching"], {
                "default": "Native Multi-Speaker",
                "tooltip": (
                    "Native Multi-Speaker sends all character turns in one Fish dialogue request. "
                    "This preserves native multi-turn context and long-form behavior.\n\n"
                    "Custom Character Switching generates every parsed character segment independently "
                    "as local speaker 0, using only that character's reference. This can reduce speaker "
                    "leakage but loses cross-turn context and requires more generation calls."
                ),
            }),
            "language_prompting": (["Auto Inline Tag", "Off"], {
                "default": "Auto Inline Tag",
                "tooltip": (
                    "Fish has no native language parameter, so the suite can optionally prepend a natural-language "
                    "inline instruction for resolved segment languages.\n\n"
                    "• Auto Inline Tag: alias/default languages like de or fr become Fish prompt tags such as <German>\n"
                    "• Off: keep Fish language fully text-only and ignore alias/default language switching"
                ),
            }),
            "speaker2": ("*", {
                "forceInput": True,
                "tooltip": "Optional second speaker reference. Speaker 1 is always the Unified node narrator/opt_narrator input. Additional speaker inputs appear automatically.",
            }),
        })
        return {"required": {
            "model": (cls._get_model_options(), {
                "default": "s2-pro",
                "tooltip": "Fish Audio S2 model selection. Options prefixed with local: are already present in configured ComfyUI TTS model folders; unprefixed options are downloadable checkpoints.",
            }),
            "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
            "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
            "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.9, "max": 2.0, "step": 0.05}),
            "native_chunk_length": ("INT", {"default": 200, "min": 100, "max": 1000, "step": 25,
                "tooltip": "Official UTF-8 byte limit for grouping native speaker turns."}),
            "max_new_tokens": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64}),
            "context_length": (["4096", "8192", "16384", "32768"], {
                "default": "8192",
                "tooltip": "Native context and KV-cache size. Larger values support longer dialogue but use substantially more VRAM.",
            }),
        }, "optional": optional}

    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_config"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"

    @classmethod
    def IS_CHANGED(cls, model="s2-pro", device="auto", temperature=0.8, top_p=0.8, repetition_penalty=1.1,
                   native_chunk_length=200, max_new_tokens=1024, context_length="8192",
                   normalize=True, cache_reference=True, precision="bfloat16",
                   compile=True, model_variant=None, quantization="none",
                   multi_speaker_mode="Native Multi-Speaker", language_prompting="Auto Inline Tag",
                   speaker2=None, **kwargs):
        model_selection = model_variant if model_variant is not None else model
        speaker_keys = sorted(
            key for key in kwargs.keys()
            if SPEAKER_INPUT_PATTERN.fullmatch(key) is not None
        )
        fingerprint = {
            "device": device,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
            "native_chunk_length": int(native_chunk_length),
            "max_new_tokens": int(max_new_tokens),
            "context_length": str(context_length),
            "normalize": bool(normalize),
            "cache_reference": bool(cache_reference),
            "precision": precision,
            "compile": bool(compile),
            "model": model_selection,
            "quantization": quantization,
            "multi_speaker_mode": multi_speaker_mode,
            "language_prompting": language_prompting,
            "speaker2_connected": speaker2 is not None,
            "dynamic_speakers_connected": {
                key: kwargs.get(key) is not None for key in speaker_keys
            },
        }
        return json.dumps(fingerprint, sort_keys=True)

    def create_engine_config(self, model="s2-pro", device="auto", temperature=0.8, top_p=0.8, repetition_penalty=1.1,
                             native_chunk_length=200, max_new_tokens=1024, context_length="8192",
                             normalize=True, cache_reference=True, precision="bfloat16",
                             compile=True, model_variant=None, quantization="none",
                             multi_speaker_mode="Native Multi-Speaker", language_prompting="Auto Inline Tag",
                             speaker2=None, **kwargs):
        model_selection = model_variant if model_variant is not None else model
        runtime_variant = model_selection
        if str(model_selection).startswith("local:"):
            try:
                from engines.fish_audio_s2.downloader import FishAudioS2Downloader
                local_path = FishAudioS2Downloader.resolve_local_model_path(model_selection)
                runtime_variant = FishAudioS2Downloader.resolve_model_variant(model_selection, local_path)
            except Exception:
                runtime_variant = "s2-pro"
        speakers = []
        if speaker2 is not None:
            speakers.append((2, speaker2))
        for key, value in kwargs.items():
            match = SPEAKER_INPUT_PATTERN.fullmatch(key)
            if match and value is not None:
                speakers.append((int(match.group(1)), value))
        speakers = [value for _, value in sorted(speakers, key=lambda item: item[0])]
        if runtime_variant != "s2-pro":
            precision = "bfloat16"
            quantization = "none"
        print(
            "   Settings: "
            f"model={model_selection}, quantization={quantization}, "
            f"multi_speaker_mode={multi_speaker_mode}, language_prompting={language_prompting}, "
            f"temperature={float(temperature)}, "
            f"top_p={float(top_p)}, repetition_penalty={float(repetition_penalty)}, "
            f"native_chunk_length={int(native_chunk_length)}, max_new_tokens={int(max_new_tokens)}, "
            f"context_length={int(context_length)}, precision={precision}, "
            f"normalize={bool(normalize)}, cache_reference={bool(cache_reference)}, compile={bool(compile)}"
        )
        config = {
            "engine_type": "fish_audio_s2", "model_variant": model_selection, "device": device,
            "temperature": float(temperature), "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
            "native_chunk_length": int(native_chunk_length), "max_new_tokens": int(max_new_tokens),
            "context_length": int(context_length),
            "normalize": bool(normalize), "cache_reference": bool(cache_reference),
            "precision": precision, "compile": bool(compile),
            "quantization": quantization,
            "multi_speaker_mode": multi_speaker_mode,
            "language_prompting": language_prompting,
            "speaker_references": speakers, "runtime_mode": "main_subprocess",
        }
        return ({"engine_type": "fish_audio_s2", "config": config, "capabilities": ["tts"]},)
