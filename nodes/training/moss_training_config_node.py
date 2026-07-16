"""
MOSS training config node for unified model training.
"""

import os
import sys
import importlib.util

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


class MossTrainingConfigNode(BaseTTSNode):
    @classmethod
    def NAME(cls):
        return "🎛️ MOSS Training Config"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "training_mode": (["LoRA Adapter (Delay 8B)"], {
                    "default": "LoRA Adapter (Delay 8B)",
                    "tooltip": "Trains LoRA adapters for the selected MOSS Delay 8B v1.0 or v1.5 base model."
                }),
                "epochs": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Epoch count used when max_train_steps is 0."
                }),
                "max_train_steps": ("INT", {
                    "default": 30000,
                    "min": 0,
                    "max": 1000000,
                    "step": 100,
                    "tooltip": "Hard training-step cap. Set 0 to use epochs instead."
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "tooltip": "Per-device batch size. Keep it low on 8B models."
                }),
                "gradient_accumulation_steps": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Micro-batch accumulation. This is how you fake a larger effective batch without immediately OOMing."
                }),
                "learning_rate": ("FLOAT", {
                    "default": 2e-6,
                    "min": 1e-8,
                    "max": 1.0,
                    "step": 1e-8,
                    "tooltip": "LoRA learning rate. The upstream Norwegian example used 2e-6."
                }),
            },
            "optional": {
                "weight_decay": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 1e-4,
                    "tooltip": "Standard AdamW weight decay."
                }),
                "warmup_steps": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 100000,
                    "step": 1,
                    "tooltip": "Warmup step count before cosine decay."
                }),
                "max_grad_norm": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Gradient clipping. The upstream Norwegian example used 0.5."
                }),
                "num_workers": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Dataloader workers. 0 is the safest default."
                }),
                "mixed_precision": (["bf16", "fp16", "no"], {
                    "default": "bf16",
                    "tooltip": "Accelerate mixed precision mode."
                }),
                "gradient_checkpointing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep this on for 8B training unless you deliberately want higher VRAM usage."
                }),
                "base_quantization": (["none", "4bit_nf4"], {
                    "default": "none",
                    "tooltip": "Base-model VRAM strategy. 4bit_nf4 enables QLoRA-style loading of the frozen 8B base model with bitsandbytes. This is the main knob if 24 GB VRAM is still not enough."
                }),
                "bnb_4bit_compute_dtype": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Compute dtype used by 4-bit quantized layers. auto follows the mixed-precision choice. fp32 is safer but costs more VRAM."
                }),
                "bnb_4bit_use_double_quant": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable nested quantization for 4-bit base loading. Usually worth keeping on."
                }),
                "save_steps": ("INT", {
                    "default": 500,
                    "min": 0,
                    "max": 100000,
                    "step": 10,
                    "tooltip": "Checkpoint save cadence. 0 disables intermediate checkpoints."
                }),
                "eval_steps": ("INT", {
                    "default": 500,
                    "min": 0,
                    "max": 100000,
                    "step": 10,
                    "tooltip": "Validation cadence. 0 disables periodic validation."
                }),
                "log_steps": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Console/progress update cadence."
                }),
                "lora_r": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 512,
                    "step": 1,
                    "tooltip": "LoRA rank."
                }),
                "lora_alpha": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 1024,
                    "step": 1,
                    "tooltip": "LoRA alpha."
                }),
                "lora_dropout": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "LoRA dropout."
                }),
                "trainable_lora_modules": (["mlp", "mlp_plus_o", "all"], {
                    "default": "mlp",
                    "tooltip": "Which Delay backbone modules receive LoRA adapters. mlp matches the released Norwegian example best."
                }),
            },
        }

    RETURN_TYPES = ("TRAINING_CONFIG", "STRING")
    RETURN_NAMES = ("training_config", "config_info")
    FUNCTION = "create_config"
    CATEGORY = "TTS Audio Suite/🎓 Training"

    def create_config(self, **kwargs):
        kwargs["training_mode"] = "lora_adapter"
        config = {
            "type": "training_config",
            "engine_type": "moss_tts",
            **kwargs,
        }
        info = (
            f"MOSS training config: LoRA Delay 8B | steps {config['max_train_steps'] or 'epoch-based'} | "
            f"batch {config['batch_size']} | lr {config['learning_rate']} | "
            f"base {config.get('base_quantization', 'none')}"
        )
        return config, info


NODE_CLASS_MAPPINGS = {"MossTrainingConfigNode": MossTrainingConfigNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MossTrainingConfigNode": "🎛️ MOSS Training Config"}
