"""
RVC training config node for unified model training.
"""

import os
import sys
import importlib.util

import folder_paths

from engines.rvc.impl.rvc_downloader import PRETRAINED_MODELS_D, PRETRAINED_MODELS_G
from utils.models.extra_paths import get_all_tts_model_paths

# Add project root directory to path for imports
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


def _list_pretrained_candidates(kind: str):
    discovered = ["auto", ""]
    bundled = PRETRAINED_MODELS_G if kind == "G" else PRETRAINED_MODELS_D
    discovered.extend(sorted(set(bundled)))

    search_dirs = [
        os.path.join(search_root, "pretrained_v2")
        for search_root in get_all_tts_model_paths("TTS")
    ]
    search_dirs.extend([
        os.path.join(folder_paths.models_dir, "TTS", "pretrained_v2"),
        os.path.join(folder_paths.models_dir, "pretrained_v2"),
    ])

    deduped_dirs = []
    seen_dirs = set()
    for search_dir in search_dirs:
        normalized_dir = os.path.normpath(search_dir)
        if normalized_dir in seen_dirs:
            continue
        seen_dirs.add(normalized_dir)
        deduped_dirs.append(search_dir)

    for search_dir in deduped_dirs:
        if not os.path.isdir(search_dir):
            continue
        for filename in sorted(os.listdir(search_dir)):
            if not filename.endswith(".pth"):
                continue
            if kind == "G" and "G" not in filename:
                continue
            if kind == "D" and "D" not in filename:
                continue
            discovered.append(f"local:{filename}")

    deduped = []
    seen = set()
    for value in discovered:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _with_tooltip(config: dict, tooltip: str):
    merged = dict(config)
    merged["tooltip"] = tooltip
    return merged


class RVCTrainingConfigNode(BaseTTSNode):
    @classmethod
    def NAME(cls):
        return "🎛️ RVC Training Config"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "epochs": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 20000,
                    "step": 1,
                    "tooltip": "Actual RVC epoch count. Around 20 is a smoke test, 50-100 is a first real pass, and 100+ is where cleaner datasets usually become worth judging. Loss helps spot trends, but listening tests still decide quality."
                }),
                "batch_size": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Use the largest stable batch that fits VRAM. Bigger is usually faster and a bit smoother; lower it first if you hit OOM or instability."
                }),
                "learning_rate": ("FLOAT", {
                    "default": 1e-4,
                    "min": 1e-8,
                    "max": 1.0,
                    "step": 1e-8,
                    "tooltip": "1e-4 is the normal safe default. Lower it if training gets spiky or noisy; raising it is one of the fastest ways to make adversarial training worse."
                }),
                "gpu_ids": ("STRING", {
                    "default": "",
                    "tooltip": "CUDA device ids in the odd RVC format, for example '0' or '0-1'. Leave blank unless you deliberately want multi-GPU behavior."
                }),
                "num_workers": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16,
                    "step": 1,
                    "tooltip": "PyTorch dataloader workers. Keep this at 0 on Windows; worker processes there can respawn ComfyUI internals and break training."
                }),
            },
            "optional": {
                "fp16_run": ("BOOLEAN", _with_tooltip(
                    {"default": True},
                    "Keep this on for normal CUDA training. Turn it off only if you see NaNs, weird instability, or hardware/driver-specific fp16 issues."
                )),
                "save_every_epoch": ("INT", _with_tooltip(
                    {"default": 5, "min": 0, "max": 1000},
                    "Checkpoint cadence. Save a resumable training checkpoint every N epochs. 5 is a good default, 1 is safest, and 0 disables periodic resume checkpoints entirely."
                )),
                "cache_data_in_gpu": ("BOOLEAN", _with_tooltip(
                    {"default": True},
                    "Cache batches in VRAM for speed. Usually worth it if you have room; turn it off if VRAM pressure becomes the real bottleneck."
                )),
                "max_checkpoints": ("INT", _with_tooltip(
                    {"default": 1, "min": 1, "max": 999, "step": 1},
                    "Checkpoint retention policy. Keep the newest N checkpoint pairs and delete older ones. 1 means keep only the latest resumable checkpoint, 3 means keep the latest three, and so on."
                )),
                "save_every_weights": ("BOOLEAN", _with_tooltip(
                    {"default": False},
                    "Export extra standalone weight files at each save interval. Mostly useful for inspection or experiments; normal training does not need this."
                )),
                "train_index": ("BOOLEAN", _with_tooltip(
                    {"default": True},
                    "Build the matched FAISS index after training. Usually keep this on; normal RVC inference often sounds worse without the proper index."
                )),
                "save_best_model": ("BOOLEAN", _with_tooltip(
                    {"default": True},
                    "Track generator loss and save the lowest-loss model as an extra inference checkpoint candidate. Useful for listening tests, but lowest loss is not guaranteed to be the best sounding model."
                )),
                "best_model_threshold": ("INT", _with_tooltip(
                    {"default": 30, "min": 1, "max": 100},
                    "Initial gate for first 'best model' capture. Leave this alone unless you are debugging why low-loss checkpoint saving triggers too early or too often."
                )),
                "log_every_epoch": ("FLOAT", _with_tooltip(
                    {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1},
                    "Logging cadence within an epoch. This affects dashboard/log granularity, not training quality. 1.0 is about once per epoch, 0.5 is about twice."
                )),
                "pretrained_generator": (_list_pretrained_candidates("G"), _with_tooltip(
                    {"default": "auto"},
                    "Generator initialization checkpoint. Keep this on auto unless you know exactly why you want a different base; training from scratch on a small dataset is usually a bad idea."
                )),
                "pretrained_discriminator": (_list_pretrained_candidates("D"), _with_tooltip(
                    {"default": "auto"},
                    "Discriminator initialization checkpoint. Auto is the sane default and should stay matched to the generator/sample-rate setup."
                )),
                "c_adv": ("FLOAT", _with_tooltip(
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1},
                    "Adversarial loss weight. Advanced knob. Do not touch this unless you are intentionally rebalancing training and understand the side effects."
                )),
                "c_mel": ("FLOAT", _with_tooltip(
                    {"default": 45.0, "min": 0.0, "max": 100.0, "step": 0.1},
                    "Mel reconstruction weight. One of the main terms keeping speech recognizable. Usually leave it at default unless you are doing real loss tuning."
                )),
                "c_kl": ("FLOAT", _with_tooltip(
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1},
                    "KL regularization weight. Changing this can absolutely destabilize training, so leave it alone unless you know the tradeoff you want."
                )),
                "c_fm": ("FLOAT", _with_tooltip(
                    {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1},
                    "Feature-matching weight. Usually helps texture/stability. Leave it at default unless you are deliberately rebalancing losses."
                )),
                "c_tefs": ("FLOAT", _with_tooltip(
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1},
                    "Experimental TEFS auxiliary loss. Keep this at 0 unless you are explicitly experimenting with that feature."
                )),
                "c_hd": ("FLOAT", _with_tooltip(
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1},
                    "Experimental harmonic-detail auxiliary loss. Keep this at 0 for normal training."
                )),
                "c_tsi": ("FLOAT", _with_tooltip(
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1},
                    "Experimental TSI auxiliary loss. Keep this at 0 unless you are intentionally testing it."
                )),
                "c_gp": ("FLOAT", _with_tooltip(
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1},
                    "Gradient-penalty weight. This is mostly a stabilization/debug knob, not a normal training control. Leave it at 0 unless adversarial loss is misbehaving."
                )),
                "use_multiscale": ("BOOLEAN", _with_tooltip(
                    {"default": False},
                    "Use experimental multiscale mel loss. Heavier and not needed for a normal first pass."
                )),
                "use_balancer": ("BOOLEAN", _with_tooltip(
                    {"default": False},
                    "Enable dynamic loss balancing. Advanced only. If you do not already know why you want this, you do not want this."
                )),
                "use_pareto": ("BOOLEAN", _with_tooltip(
                    {"default": False},
                    "Pareto-style balancing on top of the dynamic balancer. Ignore this unless you are already deliberately using the balancer."
                )),
                "fast_mode": ("BOOLEAN", _with_tooltip(
                    {"default": False},
                    "Speed tweak for the advanced balancer path. It does nothing useful unless you already enabled those advanced balancing options."
                )),
            },
        }

    RETURN_TYPES = ("TRAINING_CONFIG", "STRING")
    RETURN_NAMES = ("training_config", "config_info")
    FUNCTION = "create_config"
    CATEGORY = "TTS Audio Suite/🎓 Training"

    def create_config(self, **kwargs):
        config = {
            "type": "training_config",
            "engine_type": "rvc",
            "training_mode": "voice_model",
            **kwargs,
        }
        info = (
            f"RVC training config: {config['epochs']} epochs | batch {config['batch_size']} | "
            f"lr {config['learning_rate']}"
        )
        return config, info


NODE_CLASS_MAPPINGS = {"RVCTrainingConfigNode": RVCTrainingConfigNode}
NODE_DISPLAY_NAME_MAPPINGS = {"RVCTrainingConfigNode": "🎛️ RVC Training Config"}
