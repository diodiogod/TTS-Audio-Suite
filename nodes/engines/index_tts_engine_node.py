"""
IndexTTS-2 Engine Configuration Node

Provides comprehensive configuration interface for IndexTTS-2 TTS engine with all
official parameters exposed for experimentation and fine-tuning.
"""

import os
import sys
import importlib.util
from typing import Dict, Any, List

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load base_node module directly
base_node_path = os.path.join(nodes_dir, "base", "base_node.py")
base_spec = importlib.util.spec_from_file_location("base_node_module", base_node_path)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["base_node_module"] = base_module
base_spec.loader.exec_module(base_module)

# Import the base class
BaseTTSNode = base_module.BaseTTSNode

import folder_paths
from utils.models.extra_paths import get_all_tts_model_paths, find_model_in_paths

# AnyType for flexible input types (accepts any data type)
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_typ = AnyType("*")


class IndexTTSEngineNode(BaseTTSNode):
    """
    IndexTTS-2 TTS Engine configuration node.
    Provides IndexTTS-2 parameters and creates engine adapter for unified nodes.
    """
    
    @classmethod
    def NAME(cls):
        return "⚙️ IndexTTS-2 Engine"
    
    @classmethod
    def INPUT_TYPES(cls):        
        # Get available model paths
        model_paths = cls._get_model_paths()
        
        return {
            "required": {
                # Model Configuration
                "model_path": (model_paths, {
                    "default": model_paths[0] if model_paths else "IndexTTS-2",
                    "tooltip": "IndexTTS-2 model selection:\n• local:ModelName: Use locally installed model (respects extra_model_paths.yaml)\n• ModelName: Auto-download model if not found locally\n• Downloads respect extra_model_paths.yaml configuration"
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run IndexTTS-2 model on. 'auto' selects best available."
                }),
                
                # IndexTTS-2 Unique Features
                "emotion_alpha": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Emotion intensity control (0.0-2.0). Affects emotion control from connected emotion nodes. 1.0=full emotion, 0.5=50% blend, 0.0=neutral."
                }),
                "use_random": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable random sampling for more varied generation. Can improve diversity."
                }),
                
                # Text Processing
                "max_text_tokens_per_segment": ("INT", {
                    "default": 120, "min": 50, "max": 300, "step": 10,
                    "tooltip": "Maximum text tokens per segment. Longer segments may cause quality issues."
                }),
                "interval_silence": ("INT", {
                    "default": 200, "min": 0, "max": 1000, "step": 50,
                    "tooltip": "Silence duration between segments in milliseconds."
                }),
                
                # Generation Parameters
                "temperature": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1,
                    "tooltip": "Controls randomness. Higher values = more creative, lower = more consistent."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Nucleus sampling threshold. Controls probability mass of tokens to consider."
                }),
                "top_k": ("INT", {
                    "default": 30, "min": 1, "max": 100, "step": 5,
                    "tooltip": "Top-k sampling parameter. Lower values = more focused generation."
                }),
                "do_sample": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable sampling for generation. Disable for deterministic output."
                }),
                
                # Advanced Generation
                "length_penalty": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Length penalty for beam search. Positive values favor longer sequences."
                }),
                "num_beams": ("INT", {
                    "default": 3, "min": 1, "max": 10, "step": 1,
                    "tooltip": "Number of beams for beam search. Higher values = better quality but slower."
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 10.0, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "Penalty for repeated tokens. Higher values reduce repetition."
                }),
                "max_mel_tokens": ("INT", {
                    "default": 1500, "min": 500, "max": 3000, "step": 100,
                    "tooltip": "Maximum mel-spectrogram tokens to generate. Controls output length limit."
                }),
                
                # Model Options
                "use_fp16": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use FP16 for faster inference. Disable if you encounter numerical issues."
                }),
                "use_deepspeed": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use DeepSpeed optimization. Requires DeepSpeed installation."
                }),
            },
            "optional": {
                # Unified Emotion Control - Using multitype input for better connection suggestions
                "emotion_control": (any_typ, {
                    "tooltip": """• 🌈 Emotion Vectors - Manual emotion control sliders
• 🎭 Character Voices (opt_narrator) - Audio-based emotion reference
• 🌈 Text Emotion - AI-analyzed emotion from text
• Direct AUDIO - Any audio input for emotion reference
Character emotion tags [Alice:emotion_ref] will override this for specific characters."""
                }),
                
                # CUDA Kernel Option
                "use_cuda_kernel": (["auto", "true", "false"], {
                    "default": "auto",
                    "tooltip": "Use BigVGAN CUDA kernels for faster vocoding. Auto-detects availability."
                }),
            }
        }
    
    RETURN_TYPES = ("TTS_ENGINE",)
    RETURN_NAMES = ("TTS_engine",)
    FUNCTION = "create_engine_adapter"
    CATEGORY = "TTS Audio Suite/⚙️ Engines"
    
    @classmethod
    def _get_model_paths(cls) -> List[str]:
        """Get available IndexTTS-2 model paths following F5TTS pattern."""
        paths = ["IndexTTS-2"]  # Auto-download option (just model name)

        try:
            # Check all configured TTS model paths
            all_tts_paths = get_all_tts_model_paths('TTS')

            for base_path in all_tts_paths:
                # Check direct path (models/TTS/IndexTTS-2)
                index_direct = os.path.join(base_path, "IndexTTS-2")
                if os.path.exists(os.path.join(index_direct, "config.yaml")):
                    local_model = "local:IndexTTS-2"
                    if local_model not in paths:
                        paths.insert(0, local_model)  # Insert at beginning

                # Check organized path (models/TTS/IndexTTS/IndexTTS-2)
                index_organized = os.path.join(base_path, "IndexTTS")
                if os.path.exists(index_organized):
                    for item in os.listdir(index_organized):
                        model_dir = os.path.join(index_organized, item)
                        if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "config.yaml")):
                            local_model = f"local:{item}"
                            if local_model not in paths:
                                paths.insert(-1, local_model)  # Insert before auto-download
        except Exception:
            # Fallback to original behavior if extra_paths fails
            base_dir = os.path.join(folder_paths.models_dir, "TTS", "IndexTTS")
            if os.path.exists(base_dir):
                for item in os.listdir(base_dir):
                    model_dir = os.path.join(base_dir, item)
                    if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "config.yaml")):
                        local_model = f"local:{item}"
                        if local_model not in paths:
                            paths.insert(-1, local_model)  # Insert before auto-download

        return paths
    
    def create_engine_adapter(
        self,
        model_path: str,
        device: str,
        emotion_alpha: float,
        use_random: bool,
        max_text_tokens_per_segment: int,
        interval_silence: int,
        temperature: float,
        top_p: float,
        top_k: int,
        do_sample: bool,
        length_penalty: float,
        num_beams: int,
        repetition_penalty: float,
        max_mel_tokens: int,
        use_fp16: bool,
        use_deepspeed: bool,
        use_cuda_kernel: str = "auto",
        emotion_control = None,
    ):
        """
        Create IndexTTS-2 engine adapter with configuration.
        
        Returns:
            Tuple containing IndexTTS-2 engine configuration data
        """
        try:
            # Process unified emotion control
            emotion_audio = None
            emotion_vector = None
            use_emotion_text = False
            emotion_text = None
            is_dynamic_template = False

            if emotion_control:
                if isinstance(emotion_control, dict):
                    # Check the type of emotion control
                    emotion_type = emotion_control.get("type")

                    if emotion_type == "emotion_vectors":
                        # Emotion vectors from options node
                        emotion_vectors = emotion_control.get("emotion_vectors", {})
                        emotions = [
                            emotion_vectors.get("happy", 0.0),
                            emotion_vectors.get("angry", 0.0),
                            emotion_vectors.get("sad", 0.0),
                            emotion_vectors.get("afraid", 0.0),
                            emotion_vectors.get("disgusted", 0.0),
                            emotion_vectors.get("melancholic", 0.0),
                            emotion_vectors.get("surprised", 0.0),
                            emotion_vectors.get("calm", 0.0)
                        ]
                        if any(e > 0.0 for e in emotions):
                            emotion_vector = emotions

                    elif emotion_type == "qwen_emotion":
                        # QwenEmotion text analysis
                        use_emotion_text = emotion_control.get("use_emotion_text", False)
                        emotion_text = emotion_control.get("emotion_text", "")
                        is_dynamic_template = emotion_control.get("is_dynamic_template", False)

                    elif "waveform" in emotion_control or "audio" in emotion_control:
                        # Direct audio input (NARRATOR_VOICE from Character Voices or AUDIO)
                        emotion_audio = emotion_control

                elif hasattr(emotion_control, 'get') and ("waveform" in emotion_control or "audio" in emotion_control):
                    # Direct audio input
                    emotion_audio = emotion_control
            
            # Parse CUDA kernel option
            cuda_kernel_option = None
            if use_cuda_kernel == "true":
                cuda_kernel_option = True
            elif use_cuda_kernel == "false":
                cuda_kernel_option = False
            # "auto" stays as None for auto-detection
            
            # Create configuration dictionary
            config = {
                "model_path": model_path,
                "device": device,
                "emotion_audio": emotion_audio,  # Will be None if not connected, audio dict if connected
                "emotion_alpha": emotion_alpha,
                "use_emotion_text": use_emotion_text,
                "emotion_text": emotion_text if emotion_text and emotion_text.strip() else None,
                "use_random": use_random,
                "max_text_tokens_per_segment": max_text_tokens_per_segment,
                "interval_silence": interval_silence,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": do_sample,
                "length_penalty": length_penalty,
                "num_beams": num_beams,
                "repetition_penalty": repetition_penalty,
                "max_mel_tokens": max_mel_tokens,
                "use_fp16": use_fp16,
                "use_deepspeed": use_deepspeed,
                "emotion_vector": emotion_vector,
                "use_cuda_kernel": cuda_kernel_option,
                "is_dynamic_template": is_dynamic_template,
                "engine_type": "index_tts"
            }
            
            print(f"⚙️ IndexTTS-2: Configured on {device}")
            print(f"   Model: {model_path}")
            emotion_desc = f"alpha={emotion_alpha}, use_text={use_emotion_text}"
            if is_dynamic_template:
                emotion_desc += " (dynamic template)"
            print(f"   Emotion: {emotion_desc}")
            print(f"   Generation: temp={temperature}, top_p={top_p}, top_k={top_k}")
            print(f"   Chunking: max_tokens={max_text_tokens_per_segment}, silence={interval_silence}ms")
            
            # Return engine data for consumption by unified TTS nodes
            engine_data = {
                "engine_type": "index_tts",
                "config": config,
                "adapter_class": "IndexTTSAdapter"
            }
            
            return (engine_data,)
            
        except Exception as e:
            print(f"❌ IndexTTS-2 Engine error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return error config
            error_config = {
                "engine_type": "index_tts",
                "config": {
                    "model_path": model_path,
                    "device": "cpu",  # Fallback to CPU
                    "emotion_alpha": 1.0,
                    "temperature": 0.8,
                    "error": str(e)
                },
                "adapter_class": "IndexTTSAdapter"
            }
            return (error_config,)


# Register the node
NODE_CLASS_MAPPINGS = {
    "IndexTTS Engine": IndexTTSEngineNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTS Engine": "IndexTTS-2 Engine"
}