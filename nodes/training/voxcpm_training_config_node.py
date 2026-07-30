"""Official VoxCPM2 LoRA training settings."""

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


class VoxCPMTrainingConfigNode(BaseTTSNode):
    @classmethod
    def NAME(cls):
        return "🎛️ VoxCPM2 LoRA Config"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_train_steps": ("INT", {
                    "default": 1000, "min": 1, "max": 1000000, "step": 100,
                    "tooltip": "Official trainer iteration count. Use 1-5 first for a VRAM smoke test.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Per-device batch size. Keep 1 until measured on your GPU.",
                }),
                "gradient_accumulation_steps": ("INT", {
                    "default": 8, "min": 1, "max": 256, "step": 1,
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.0001, "min": 0.00000001, "max": 1.0, "step": 0.00000001,
                    "tooltip": "Official VoxCPM LoRA default is 1e-4.",
                }),
                "lora_r": ("INT", {
                    "default": 32, "min": 1, "max": 512, "step": 1,
                }),
                "lora_alpha": ("INT", {
                    "default": 32, "min": 1, "max": 1024, "step": 1,
                }),
            },
            "optional": {
                "weight_decay": ("FLOAT", {
                    "default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001,
                }),
                "warmup_steps": ("INT", {
                    "default": 100, "min": 0, "max": 100000, "step": 1,
                }),
                "max_grad_norm": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1,
                }),
                "max_batch_tokens": ("INT", {
                    "default": 8192, "min": 0, "max": 131072, "step": 256,
                    "tooltip": "Filters individual samples that would exceed this packed-token budget. 0 disables filtering.",
                }),
                "num_workers": ("INT", {
                    "default": 0, "min": 0, "max": 16, "step": 1,
                    "tooltip": "0 is the safest Windows setting.",
                }),
                "cuda_device_index": ("INT", {
                    "default": -1, "min": -1, "max": 31, "step": 1,
                    "tooltip": (
                        "-1 selects the CUDA GPU with the most free VRAM. "
                        "Use 0, 1, etc. to force a specific CUDA device."
                    ),
                }),
                "save_steps": ("INT", {
                    "default": 500, "min": 0, "max": 100000, "step": 10,
                }),
                "eval_steps": ("INT", {
                    "default": 500, "min": 0, "max": 100000, "step": 10,
                    "tooltip": "0 disables validation passes.",
                }),
                "log_steps": ("INT", {
                    "default": 10, "min": 1, "max": 10000, "step": 1,
                }),
                "lora_dropout": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                }),
                "lora_enable_lm": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Official default: adapt the language-model blocks.",
                }),
                "lora_enable_dit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Official default: adapt the diffusion-transformer blocks.",
                }),
                "lora_enable_proj": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Official default is off. Enable only for deliberate experiments.",
                }),
            },
        }

    RETURN_TYPES = ("TRAINING_CONFIG", "STRING")
    RETURN_NAMES = ("training_config", "config_info")
    FUNCTION = "create_config"
    CATEGORY = "TTS Audio Suite/🎓 Training"

    def create_config(self, **kwargs):
        config = {
            "type": "training_config",
            "engine_type": "voxcpm",
            "training_mode": "lora_adapter",
            **kwargs,
        }
        if not (
            config.get("lora_enable_lm")
            or config.get("lora_enable_dit")
            or config.get("lora_enable_proj")
        ):
            raise ValueError("At least one VoxCPM LoRA module group must be enabled")
        info = (
            f"VoxCPM2 LoRA: steps={config['max_train_steps']} | "
            f"batch={config['batch_size']}x{config['gradient_accumulation_steps']} | "
            f"r={config['lora_r']}, alpha={config['lora_alpha']}"
        )
        return config, info


NODE_CLASS_MAPPINGS = {"VoxCPMTrainingConfigNode": VoxCPMTrainingConfigNode}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VoxCPMTrainingConfigNode": "🎛️ VoxCPM2 LoRA Config"
}
