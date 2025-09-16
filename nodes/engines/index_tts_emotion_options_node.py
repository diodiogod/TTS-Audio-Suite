"""
IndexTTS-2 Emotion Vectors Options Node

Provides emotion vector sliders for IndexTTS-2 engine to reduce clutter in main engine node.
Returns emotion vector configuration that can be optionally connected to IndexTTS-2 Engine.
"""

import os
import sys

# Add project root directory to path for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)  # nodes/
project_root = os.path.dirname(nodes_dir)  # project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class IndexTTSEmotionOptionsNode:
    """
    IndexTTS-2 Emotion Vectors Options node.
    Provides emotion vector sliders for fine-grained emotion control.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                # Manual Emotion Vector (8 emotions) - Using sliders like RVC
                "Happy": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.2, "step": 0.01,
                    "display": "slider",
                    "tooltip": "Happy emotion intensity (0.0-1.2). Higher values make speech sound more joyful and upbeat."
                }),
                "Angry": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.2, "step": 0.01,
                    "display": "slider",
                    "tooltip": "Angry emotion intensity (0.0-1.2). Higher values make speech sound more aggressive and harsh."
                }),
                "Sad": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.2, "step": 0.01,
                    "display": "slider",
                    "tooltip": "Sad emotion intensity (0.0-1.2). Higher values make speech sound more melancholic and downcast."
                }),
                "Surprised": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.2, "step": 0.01,
                    "display": "slider",
                    "tooltip": "Surprised emotion intensity (0.0-1.2). Higher values make speech sound more shocked and amazed."
                }),
                "Afraid": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.2, "step": 0.01,
                    "display": "slider",
                    "tooltip": "Afraid emotion intensity (0.0-1.2). Higher values make speech sound more scared and anxious."
                }),
                "Disgusted": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.2, "step": 0.01,
                    "display": "slider",
                    "tooltip": "Disgusted emotion intensity (0.0-1.2). Higher values make speech sound more repulsed and revolted."
                }),
                "Calm": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.2, "step": 0.01,
                    "display": "slider",
                    "tooltip": "Calm emotion intensity (0.0-1.2). Higher values make speech sound more peaceful and relaxed."
                }),
                "Melancholic": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.2, "step": 0.01,
                    "display": "slider",
                    "tooltip": "Melancholic emotion intensity (0.0-1.2). Higher values make speech sound more thoughtfully sad and wistful."
                }),
            }
        }
    
    RETURN_TYPES = ("EMOTION_CONTROL",)
    RETURN_NAMES = ("emotion_control",)
    FUNCTION = "create_emotion_vectors"
    CATEGORY = "TTS Audio Suite/Engines/IndexTTS-2"
    DESCRIPTION = "Configure emotion vector intensities for IndexTTS-2. Connect to IndexTTS-2 Engine node for advanced emotion control using 8 different emotion types."

    @classmethod 
    def NAME(cls):
        return "🌈 IndexTTS-2 Emotion Vectors"
    
    def create_emotion_vectors(self, Happy=0.0, Angry=0.0, Sad=0.0, 
                             Surprised=0.0, Afraid=0.0, Disgusted=0.0,
                             Calm=0.0, Melancholic=0.0):
        """
        Create emotion vectors configuration.
        
        Returns:
            Emotion vectors configuration dictionary
        """
        emotion_vectors = {
            "happy": Happy,
            "angry": Angry, 
            "sad": Sad,
            "surprised": Surprised,
            "afraid": Afraid,
            "disgusted": Disgusted,
            "calm": Calm,
            "melancholic": Melancholic
        }
        
        # Only include non-zero emotions for cleaner output
        active_emotions = {k: v for k, v in emotion_vectors.items() if v > 0.0}
        
        if active_emotions:
            print(f"🎭 IndexTTS-2 Emotion Vectors: {active_emotions}")
        else:
            print(f"🎭 IndexTTS-2 Emotion Vectors: All neutral (0.0)")

        # Create unified emotion control format
        emotion_control = {
            "type": "emotion_vectors",
            "emotion_vectors": emotion_vectors
        }

        return (emotion_control,)


# Register the node
NODE_CLASS_MAPPINGS = {
    "IndexTTSEmotionOptionsNode": IndexTTSEmotionOptionsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTSEmotionOptionsNode": "🌈 IndexTTS-2 Emotion Vectors"
}