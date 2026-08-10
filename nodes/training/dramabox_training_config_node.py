"""DramaBox IC-LoRA training configuration node."""

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


class DramaBoxTrainingConfigNode(BaseTTSNode):
    @classmethod
    def NAME(cls):
        return "🎛️ DramaBox Training Config"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "training_mode": (["Audio LoRA (IC-LoRA)"], {
                    "default": "Audio LoRA (IC-LoRA)",
                    "tooltip": "Official DramaBox audio-branch IC-LoRA training mode.",
                }),
                "base_model": (["dev", "distilled"], {
                    "default": "dev",
                    "tooltip": "Official timestep schedule. dev is the normal DramaBox fine-tuning choice; distilled is experimental.",
                }),
                "steps": ("INT", {
                    "default": 10000,
                    "min": 1,
                    "max": 1000000,
                    "step": 100,
                    "tooltip": "Optimizer steps. The upstream example uses 10,000; listen to saved checkpoints instead of assuming the final step is best.",
                }),
                "learning_rate": ("FLOAT", {
                    "default": 1e-4,
                    "min": 1e-8,
                    "max": 1.0,
                    "step": 1e-6,
                    "tooltip": "LoRA learning rate. The official example uses 1e-4 for a fresh adapter.",
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Per-device batch size. Keep this at 1 unless the dataset and GPU have room.",
                }),
                "grad_accum": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Gradient accumulation steps. This increases effective batch size without loading more samples at once.",
                }),
                "lora_rank": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 512,
                    "step": 1,
                    "tooltip": "LoRA rank. The official DramaBox example uses 128.",
                }),
                "lora_alpha": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 1024,
                    "step": 1,
                    "tooltip": "LoRA alpha. Keeping alpha equal to rank gives a 1.0 adapter scale.",
                }),
                "lora_dropout": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "LoRA dropout. The official small-dataset example uses 0.1.",
                }),
            },
            "optional": {
                "lr_scheduler": (["cosine", "linear", "constant"], {
                    "default": "cosine",
                    "tooltip": "Learning-rate schedule passed to the official trainer.",
                }),
                "warmup_steps": ("INT", {
                    "default": 500,
                    "min": 0,
                    "max": 100000,
                    "step": 10,
                    "tooltip": "Warmup steps before the selected schedule. The official example uses 500.",
                }),
                "max_grad_norm": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Gradient clipping threshold.",
                }),
                "ref_ratio": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Fraction of a training target used as the appended voice-reference tail.",
                }),
                "max_ref_tokens": ("INT", {
                    "default": 200,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Maximum reference tokens after audio patchification.",
                }),
                "text_dropout": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Probability of dropping text conditioning so the adapter learns to use the reference voice path.",
                }),
                "save_every": ("INT", {
                    "default": 500,
                    "min": 1,
                    "max": 100000,
                    "step": 10,
                    "tooltip": "Checkpoint cadence. The official trainer requires a value of at least 1.",
                }),
                "log_every": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Human-readable console update cadence. The training panel receives quieter per-step updates.",
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31 - 1,
                    "step": 1,
                    "tooltip": "Training random seed.",
                }),
                "preprocess_batch_size": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1,
                    "tooltip": "Audio/text preprocessing batch size. Lower this if preprocessing runs out of memory.",
                }),
                "validation_config": ("STRING", {
                    "default": "",
                    "tooltip": "Optional path to the official val_config YAML. Validation launches another full inference process at each save step and requires a separate GPU.",
                }),
                "validation_gpu": ("STRING", {
                    "default": "",
                    "tooltip": "Physical CUDA device index reserved for validation, for example 1. Required when validation_config is set and must differ from the training GPU.",
                }),
                "dry_run": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "CPU-safe preflight only: writes the normalized official config and command without loading DramaBox weights or starting CUDA training.",
                }),
            },
        }

    RETURN_TYPES = ("TRAINING_CONFIG", "STRING")
    RETURN_NAMES = ("training_config", "config_info")
    FUNCTION = "create_config"
    CATEGORY = "TTS Audio Suite/🎓 Training"

    def create_config(self, **kwargs):
        kwargs["training_mode"] = "audio_lora"
        config = {
            "type": "training_config",
            "engine_type": "dramabox",
            **kwargs,
        }
        info = (
            f"DramaBox audio LoRA config: {config['base_model']} | "
            f"{config['steps']} steps | batch {config['batch_size']} | "
            f"rank {config['lora_rank']} | lr {config['learning_rate']}"
        )
        return config, info


NODE_CLASS_MAPPINGS = {"DramaBoxTrainingConfigNode": DramaBoxTrainingConfigNode}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DramaBoxTrainingConfigNode": "🎛️ DramaBox Training Config"
}
