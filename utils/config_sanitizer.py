"""
Engine Config Sanitizer - Fixes ComfyUI JSON serialization type corruption

When workflow JSONs are saved/loaded, numeric types get corrupted:
- Floats like 10.0 become integers 10
- Device strings become floats like 0.6

This sanitizer fixes these issues for all TTS engine configs.
Particularly important for first run after reboot or when using Any Switch node.
"""

from typing import Dict, Any


class ConfigSanitizer:
    """Sanitizes TTS engine configs to fix JSON serialization corruption."""

    # Parameters that should be floats
    FLOAT_PARAMS = {
        'repetition_penalty', 'temperature', 'top_p', 'emotion_alpha',
        'length_penalty', 'speed', 'target_rms', 'cross_fade_duration',
        'cfg_weight', 'exaggeration', 'cfg_strength', 'min_p', 'fade_for_StretchToFit',
        'max_stretch_ratio', 'min_stretch_ratio', 'timing_tolerance'
    }

    # Parameters that should be strings
    STRING_PARAMS = {
        'device', 'language', 'model', 'model_version', 'model_path'
    }

    # Parameters that should be integers
    INT_PARAMS = {
        'top_k', 'num_beams', 'max_mel_tokens', 'interval_silence',
        'max_text_tokens_per_segment', 'seed', 'refinement_passes'
    }

    @classmethod
    def sanitize(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize engine config to fix JSON serialization type corruption.

        Args:
            config: Engine configuration dictionary

        Returns:
            Sanitized configuration dictionary with corrected types
        """
        if not config:
            return config

        for key, value in config.items():
            # Skip None values
            if value is None:
                continue

            # Convert integers to floats for float parameters
            if key in cls.FLOAT_PARAMS:
                if isinstance(value, int):
                    config[key] = float(value)

            # Convert floats/ints to strings for string parameters
            elif key in cls.STRING_PARAMS:
                if isinstance(value, (int, float)):
                    # Fix corrupted device values (0.6, 0.7 instead of strings)
                    if key == 'device':
                        if value == 0:
                            config[key] = "cuda:0"
                        else:
                            config[key] = "auto"
                    else:
                        config[key] = str(value)

            # Convert floats to integers for int parameters
            elif key in cls.INT_PARAMS:
                if isinstance(value, float):
                    config[key] = int(value)

        return config

    @classmethod
    def sanitize_and_log(cls, config: Dict[str, Any], node_name: str = "TTS Engine") -> Dict[str, Any]:
        """
        Sanitize config and log any corrections made.

        Args:
            config: Engine configuration dictionary
            node_name: Name of the node for logging context

        Returns:
            Sanitized configuration dictionary
        """
        if not config:
            return config

        corrections_made = []

        for key, value in config.items():
            if value is None:
                continue

            original_value = value

            # Convert integers to floats for float parameters
            if key in cls.FLOAT_PARAMS and isinstance(value, int):
                config[key] = float(value)
                corrections_made.append(f"'{key}': {original_value} (int) → {config[key]} (float)")

            # Convert floats/ints to strings for string parameters
            elif key in cls.STRING_PARAMS and isinstance(value, (int, float)):
                if key == 'device' and value == 0:
                    config[key] = "cuda:0"
                    corrections_made.append(f"'{key}': {original_value} → 'cuda:0' (corrupted device)")
                elif key == 'device':
                    config[key] = "auto"
                    corrections_made.append(f"'{key}': {original_value} → 'auto' (corrupted device)")
                else:
                    config[key] = str(value)
                    corrections_made.append(f"'{key}': {original_value} → '{config[key]}' (str)")

            # Convert floats to integers for int parameters
            elif key in cls.INT_PARAMS and isinstance(value, float):
                config[key] = int(value)
                corrections_made.append(f"'{key}': {original_value} (float) → {config[key]} (int)")

        if corrections_made:
            print(f"⚠️ {node_name} config corrections (JSON serialization fix):")
            for correction in corrections_made:
                print(f"   {correction}")

        return config
