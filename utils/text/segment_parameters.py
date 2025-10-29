"""
Segment-level Parameter System for Per-Segment TTS Parameter Control

Allows parameters like seed, temperature, cfg, etc. to be set inline in text/SRT
using pipe-separated syntax: [Alice|seed:42|temperature:0.5]

Parameters override node defaults for a single segment, then revert after.
"""

from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


# Parameter aliases: user_input -> canonical_name
PARAMETER_ALIASES = {
    'seed': 'seed',
    'temp': 'temperature',
    'temperature': 'temperature',
    'cfg': 'cfg',
    'cfg_weight': 'cfg',  # cfg_weight alias -> cfg (more universal)
    'cfgweight': 'cfg',
    'exaggeration': 'exaggeration',
    'exag': 'exaggeration',
    'speed': 'speed',
    'top_p': 'top_p',
    'top_k': 'top_k',
    'topk': 'top_k',
    'topp': 'top_p',
    'inference_steps': 'inference_steps',
    'steps': 'inference_steps',
    'emotion_alpha': 'emotion_alpha',
    # NOTE: 'emotion' intentionally NOT aliased to avoid confusion with emotion reference syntax [Alice:Bob]
}

# Parameter compatibility matrix: canonical_name -> set of engine_types
PARAMETER_ENGINES = {
    'seed': {
        'chatterbox', 'chatterbox_official_23lang', 'f5tts', 'higgs_audio',
        'vibevoice', 'index_tts'
    },
    'temperature': {
        'chatterbox', 'chatterbox_official_23lang', 'f5tts', 'higgs_audio',
        'vibevoice', 'index_tts'
    },
    'cfg': {
        'f5tts', 'vibevoice', 'index_tts', 'chatterbox', 'chatterbox_official_23lang'
    },
    'exaggeration': {
        'chatterbox', 'chatterbox_official_23lang'
    },
    'speed': {
        'f5tts'
    },
    'top_p': {
        'higgs_audio', 'vibevoice', 'index_tts'
    },
    'top_k': {
        'higgs_audio', 'index_tts'
    },
    'inference_steps': {
        'vibevoice'
    },
    'emotion_alpha': {
        'index_tts'
    }
}

# Parameter type validation: canonical_name -> (type, min, max, description)
PARAMETER_VALIDATION = {
    'seed': (int, 0, 2**32 - 1, "Random seed for reproducible generation"),
    'temperature': (float, 0.1, 2.0, "Randomness/creativity control (lower=more deterministic)"),
    'cfg': (float, 0.0, 20.0, "Classifier-free guidance strength"),
    'exaggeration': (float, 0.0, 2.0, "Voice emotion exaggeration (ChatterBox only)"),
    'speed': (float, 0.5, 2.0, "Speech speed multiplier (F5-TTS only)"),
    'top_p': (float, 0.0, 1.0, "Nucleus sampling probability"),
    'top_k': (int, 1, 100, "Top-k sampling"),
    'inference_steps': (int, 1, 100, "Number of inference steps"),
    'emotion_alpha': (float, 0.0, 1.0, "Emotion strength (IndexTTS-2 only)")
}

# Mapping of canonical parameter names to node config keys (handles engine-specific naming)
# When applying to config, use these keys
PARAMETER_NODE_KEYS = {
    'seed': 'seed',
    'temperature': 'temperature',
    'cfg': {'default': 'cfg_weight', 'f5tts': 'cfg_strength'},  # Engine-specific mapping
    'exaggeration': 'exaggeration',
    'speed': 'speed',
    'top_p': 'top_p',
    'top_k': 'top_k',
    'inference_steps': 'inference_steps',
    'emotion_alpha': 'emotion_alpha'
}


class ParameterValidator:
    """Validates and filters parameters based on engine type and value constraints."""

    @staticmethod
    def normalize_parameter_name(param_name: str) -> Optional[str]:
        """
        Normalize parameter name (lowercase) and resolve aliases.

        Args:
            param_name: Raw parameter name from user input

        Returns:
            Canonical parameter name, or None if unknown
        """
        normalized = param_name.strip().lower()
        return PARAMETER_ALIASES.get(normalized)

    @staticmethod
    def is_parameter_supported(param_name: str, engine_type: str) -> bool:
        """
        Check if a parameter is supported by the given engine.

        Args:
            param_name: Canonical parameter name (already normalized)
            engine_type: Engine type (e.g., 'chatterbox', 'f5tts')

        Returns:
            True if parameter is supported, False otherwise
        """
        if param_name not in PARAMETER_ENGINES:
            return False
        return engine_type in PARAMETER_ENGINES[param_name]

    @staticmethod
    def validate_parameter_value(param_name: str, value: Any) -> Tuple[bool, Optional[str], Any]:
        """
        Validate and convert parameter value.

        Args:
            param_name: Canonical parameter name (already normalized)
            value: Raw value from segment tag

        Returns:
            Tuple of (is_valid, error_message, converted_value)
        """
        if param_name not in PARAMETER_VALIDATION:
            return False, f"Unknown parameter: {param_name}", value

        expected_type, min_val, max_val, description = PARAMETER_VALIDATION[param_name]

        try:
            # Try to convert to expected type
            if expected_type == int:
                converted = int(float(value))  # Handle "42.0" -> 42
            elif expected_type == float:
                converted = float(value)
            else:
                converted = value

            # Check bounds
            if converted < min_val or converted > max_val:
                return False, f"{param_name} must be between {min_val} and {max_val}, got {converted}", value

            return True, None, converted

        except (ValueError, TypeError) as e:
            return False, f"{param_name} must be {expected_type.__name__}, got {type(value).__name__}", value

    @staticmethod
    def filter_parameters_for_engine(
        parameters: Dict[str, Any],
        engine_type: str,
        warn: bool = True
    ) -> Dict[str, Any]:
        """
        Filter parameters to only those supported by the given engine.
        Optionally warns about unsupported parameters.

        Args:
            parameters: Dictionary of parameters
            engine_type: Engine type
            warn: Whether to log warnings for unsupported parameters

        Returns:
            Filtered dictionary with only supported parameters
        """
        filtered = {}

        for param_name, value in parameters.items():
            if ParameterValidator.is_parameter_supported(param_name, engine_type):
                filtered[param_name] = value
            elif warn:
                logger.warning(
                    f"⚠️ Parameter '{param_name}' not supported by {engine_type} engine, ignoring"
                )

        return filtered


class SegmentParameterParser:
    """Parses and extracts parameters from pipe-separated tags like [Alice|seed:42|temperature:0.5]"""

    @staticmethod
    def parse_tag_segments(tag_content: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Parse a character tag into segments (character/language info) and parameters.

        Examples:
            "[Alice]" -> (["Alice"], {})
            "[Alice|seed:42]" -> (["Alice"], {"seed": 42})
            "[fr:Alice|seed:42]" -> (["fr:Alice"], {"seed": 42})
            "[seed:42|fr:Alice]" -> (["fr:Alice"], {"seed": 42})  # Order-independent
            "[Alice|SEED:42|TEMPERATURE:0.5]" -> (["Alice"], {"seed": 42, "temperature": 0.5})  # Case-insensitive

        Args:
            tag_content: Content inside brackets (e.g., "Alice|seed:42|temperature:0.5")

        Returns:
            Tuple of (tag_segments, parameters_dict)
            - tag_segments: List of non-parameter parts (language:character format)
            - parameters_dict: Dictionary of extracted parameters (canonical names, lowercase)
        """
        if not tag_content or not isinstance(tag_content, str):
            return [], {}

        # Split by pipe character
        parts = tag_content.split('|')
        tag_segments = []
        parameters = {}

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Check if this part is a parameter (contains :)
            if ':' in part:
                # Could be parameter or language:character
                # Parameters are like "seed:42", "temperature:0.5"
                # Language:character is like "fr:Alice", "en:Bob"
                key, value = part.split(':', 1)
                key_stripped = key.strip()
                value = value.strip()

                # Try to normalize parameter name (handles aliases and case-insensitivity)
                canonical_name = ParameterValidator.normalize_parameter_name(key_stripped)

                if canonical_name:
                    # It's a parameter - validate and add
                    is_valid, error_msg, converted = ParameterValidator.validate_parameter_value(canonical_name, value)
                    if is_valid:
                        parameters[canonical_name] = converted
                    else:
                        logger.warning(f"⚠️ Invalid parameter: {error_msg}")
                else:
                    # It's likely language:character, keep as segment
                    tag_segments.append(part)
            else:
                # No colon, could be just character name
                tag_segments.append(part)

        return tag_segments, parameters

    @staticmethod
    def reconstruct_character_tag(tag_segments: List[str]) -> str:
        """
        Reconstruct a character tag from segments (without parameters).

        Args:
            tag_segments: List of non-parameter parts

        Returns:
            Reconstructed tag content (e.g., "Alice" or "fr:Alice")
        """
        if not tag_segments:
            return ""

        # Join all non-parameter segments with |
        # But typically there's only one segment (the character/language part)
        # Multiple segments shouldn't happen with proper parsing
        return tag_segments[0] if len(tag_segments) == 1 else "|".join(tag_segments)


class SegmentParameterCollector:
    """Collects parameters from a text segment's character tags."""

    @staticmethod
    def extract_parameters_from_text(text: str) -> Tuple[str, Dict[str, Any], str]:
        """
        Extract parameters from a text segment's character tags.

        Example:
            "[Alice|seed:42] Hi there"
            -> ("Hi there", {"seed": 42}, "Alice")

        Args:
            text: Text segment potentially with character tags

        Returns:
            Tuple of (cleaned_text, parameters, character_name)
            - cleaned_text: Text with tags removed
            - parameters: Extracted parameters
            - character_name: Extracted character name (or empty string if none)
        """
        import re

        # Match all character tags like [content]
        # Using same pattern as character_parser but we'll extract parameters
        tag_pattern = re.compile(r'\[(?!(?:pause|wait|stop):)(?!(?:it|IT|italian|Italian)\])([^\]]+)\]')

        parameters = {}
        character_name = ""

        def replace_tag(match):
            nonlocal parameters, character_name

            tag_content = match.group(1)
            tag_segments, tag_params = SegmentParameterParser.parse_tag_segments(tag_content)

            # Extract parameters
            parameters.update(tag_params)

            # Reconstruct character tag (without parameters)
            if tag_segments:
                character_name = tag_segments[0]  # Usually just one segment

            # Return empty string to remove the tag from text
            return ""

        # Replace all tags and collect parameters
        cleaned_text = tag_pattern.sub(replace_tag, text).strip()

        return cleaned_text, parameters, character_name


def parse_segment_text(text: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
    """
    Convenience function to parse segment text with parameters.

    Args:
        text: Text segment potentially with [Character|param:value] tags

    Returns:
        Tuple of (cleaned_text, parameters, character_name)
    """
    return SegmentParameterCollector.extract_parameters_from_text(text)


def filter_parameters(parameters: Dict[str, Any], engine_type: str) -> Dict[str, Any]:
    """
    Convenience function to filter parameters for an engine.

    Args:
        parameters: Dictionary of parameters
        engine_type: Engine type

    Returns:
        Filtered parameters supported by the engine
    """
    return ParameterValidator.filter_parameters_for_engine(parameters, engine_type)


def apply_segment_parameters(
    base_config: Dict[str, Any],
    segment_params: Dict[str, Any],
    engine_type: str
) -> Dict[str, Any]:
    """
    Apply segment-level parameters to override base config for a single segment.

    Args:
        base_config: Base node configuration
        segment_params: Segment-level parameters to apply
        engine_type: Engine type

    Returns:
        New config dict with segment parameters applied (doesn't modify original)
    """
    # Create a copy to avoid modifying original
    config = base_config.copy()

    # Filter parameters for this engine
    filtered_params = ParameterValidator.filter_parameters_for_engine(segment_params, engine_type)

    # Apply each parameter
    for param_name, value in filtered_params.items():
        # Map parameter name to actual config key (handles engine-specific naming)
        param_mapping = PARAMETER_NODE_KEYS.get(param_name, param_name)

        # Check if mapping is engine-specific (dict) or generic (string)
        if isinstance(param_mapping, dict):
            # Engine-specific mapping: use engine-specific key if available, else use 'default'
            config_key = param_mapping.get(engine_type, param_mapping.get('default', param_name))
        else:
            # Generic mapping: use the same key for all engines
            config_key = param_mapping

        config[config_key] = value

    return config
