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
                    "default": model_paths[0] if model_paths else "auto-download",
                    "tooltip": "IndexTTS-2 model path. 'auto-download' will download the model automatically."
                }),
                "device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "Device to run IndexTTS-2 model on. 'auto' selects best available."
                }),
                
                # IndexTTS-2 Unique Features
                "emotion_alpha": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Emotion intensity control (0.0-2.0). Affects both emotion_audio references and emotion vectors (happy, sad, etc.). 1.0=full emotion, 0.5=50% blend, 0.0=neutral."
                }),
                "use_emotion_text": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Extract emotions from text using QwenEmotion model instead of audio references."
                }),
                "emotion_text": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Custom emotion description text (e.g., 'happy and excited'). Only used if use_emotion_text is enabled."
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
                # IndexTTS-2 Emotion Disentanglement
                "emotion_audio": ("AUDIO", {
                    "tooltip": "Reference audio to extract emotion from (e.g., angry speech, sad voice). IndexTTS-2 will copy the emotional style from this audio while keeping the main speaker's voice identity. Connect Character Voices node for voice+text or direct audio input."
                }),
                
                # Emotion Vectors from separate options node
                "emotion_vectors": ("INDEXTS_EMOTION_VECTORS", {
                    "tooltip": "Connect IndexTTS-2 Emotion Vectors Options node for advanced emotion control with 8 different emotion types (happy, angry, sad, etc.)"
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
        """Get available IndexTTS-2 model paths."""
        paths = ["auto-download"]
        
        # Check for existing models
        base_dir = os.path.join(folder_paths.models_dir, "TTS", "IndexTTS")
        if os.path.exists(base_dir):
            for item in os.listdir(base_dir):
                model_dir = os.path.join(base_dir, item)
                if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "config.yaml")):
                    paths.append(model_dir)
        
        return paths
    
    def create_engine_adapter(
        self,
        model_path: str,
        device: str,
        emotion_alpha: float,
        use_emotion_text: bool,
        emotion_text: str,
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
        emotion_audio = None,
        emotion_vectors = None,
    ):
        """
        Create IndexTTS-2 engine adapter with configuration.
        
        Returns:
            Tuple containing IndexTTS-2 engine configuration data
        """
        try:
            # Create emotion vector from emotion_vectors input or defaults
            emotion_vector = None
            if emotion_vectors:
                # Use emotion vectors from options node - match official IndexTTS-2 order
                emotions = [
                    emotion_vectors.get("happy", 0.0),
                    emotion_vectors.get("angry", 0.0), 
                    emotion_vectors.get("sad", 0.0),
                    emotion_vectors.get("afraid", 0.0),  # Official IndexTTS-2 uses "afraid"
                    emotion_vectors.get("disgusted", 0.0),
                    emotion_vectors.get("melancholic", 0.0),
                    emotion_vectors.get("surprised", 0.0),
                    emotion_vectors.get("calm", 0.0)
                ]
                if any(e > 0.0 for e in emotions):
                    emotion_vector = emotions
            
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
                "emotion_text": emotion_text if emotion_text.strip() else None,
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
                "engine_type": "index_tts"
            }
            
            print(f"⚙️ IndexTTS-2: Configured on {device}")
            print(f"   Model: {model_path}")
            print(f"   Emotion: alpha={emotion_alpha}, use_text={use_emotion_text}")
            print(f"   Generation: temp={temperature}, top_p={top_p}, top_k={top_k}")
            print(f"   Chunking: max_tokens={max_text_tokens_per_segment}, silence={interval_silence}ms")
            print(f"🐛 emotion_audio in engine config: {type(emotion_audio)} | exists: {emotion_audio is not None}")
            
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