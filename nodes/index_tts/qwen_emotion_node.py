"""
QwenEmotion Text Analysis Node for IndexTTS-2

Provides text-based emotion analysis using the QwenEmotion model for dynamic emotion control.
"""

import os
import sys
from typing import List

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import folder_paths


class QwenEmotionNode:
    """
    QwenEmotion Text Analysis node for IndexTTS-2.
    Analyzes text to extract emotion vectors automatically.
    """

    @classmethod
    def NAME(cls):
        return "🌈 IndexTTS-2 Text Emotion"

    @classmethod
    def INPUT_TYPES(cls):
        # Get available QwenEmotion models
        models = cls._get_available_qwen_models()

        return {
            "required": {
                "qwen_model": (models, {
                    "default": models[0] if models else "qwen0.6bemo4-merge",
                    "tooltip": "QwenEmotion model for text emotion analysis. Use 'local:' prefix for local models or select from available downloadable models."
                }),
                "emotion_text": ("STRING", {
                    "default": "Happy character speaking: {seg}",
                    "multiline": True,
                    "tooltip": "Text describing the desired emotion. Use {seg} placeholder for dynamic per-segment analysis (e.g., 'Angry man shouting: {seg}', 'Calm narrator: {seg}'). Without {seg}, applies same emotion to all segments."
                }),
            }
        }

    RETURN_TYPES = ("EMOTION_CONTROL",)
    RETURN_NAMES = ("emotion_control",)
    FUNCTION = "create_emotion_control"
    CATEGORY = "TTS Audio Suite/Engines/IndexTTS-2"

    @classmethod
    def _get_available_qwen_models(cls) -> List[str]:
        """Get available QwenEmotion models (both downloadable and local)"""
        # Available downloadable QwenEmotion models (part of IndexTTS-2)
        models = ["qwen0.6bemo4-merge"]

        # Check for local QwenEmotion models in IndexTTS directories
        base_dirs = [
            os.path.join(folder_paths.models_dir, "TTS", "IndexTTS"),
            os.path.join(folder_paths.models_dir, "IndexTTS"),  # Legacy
        ]

        for base_dir in base_dirs:
            if os.path.exists(base_dir):
                for item in os.listdir(base_dir):
                    index_model_dir = os.path.join(base_dir, item)
                    if os.path.isdir(index_model_dir):
                        # Check for complete qwen subdirectories (must have required files)
                        for subitem in os.listdir(index_model_dir):
                            if subitem.startswith("qwen") and os.path.isdir(os.path.join(index_model_dir, subitem)):
                                qwen_dir = os.path.join(index_model_dir, subitem)
                                # Verify it's a complete QwenEmotion model
                                required_files = ["config.json", "model.safetensors", "tokenizer.json"]
                                if all(os.path.exists(os.path.join(qwen_dir, f)) for f in required_files):
                                    local_model = f"local:{subitem}"
                                    if local_model not in models:
                                        models.append(local_model)

        return models

    def create_emotion_control(self, qwen_model: str, emotion_text: str):
        """
        Create emotion control data for QwenEmotion text analysis.

        Args:
            qwen_model: QwenEmotion model to use
            emotion_text: Text describing desired emotion

        Returns:
            Emotion control dict for IndexTTS-2 adapter
        """
        # Check if using dynamic per-segment template
        is_dynamic = "{seg}" in emotion_text

        emotion_control = {
            "type": "qwen_emotion",
            "use_emotion_text": True,
            "emotion_text": emotion_text,
            "qwen_model": qwen_model,
            "is_dynamic_template": is_dynamic
        }

        if is_dynamic:
            print(f"🌈 IndexTTS-2 Text Emotion: Dynamic per-segment analysis with model '{qwen_model}' - template: '{emotion_text}'")
        else:
            print(f"🌈 IndexTTS-2 Text Emotion: Static emotion analysis with model '{qwen_model}' for '{emotion_text}'")

        return (emotion_control,)


# Export for ComfyUI
NODE_CLASS_MAPPINGS = {
    "QwenEmotionNode": QwenEmotionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenEmotionNode": "🌈 IndexTTS-2 Text Emotion"
}