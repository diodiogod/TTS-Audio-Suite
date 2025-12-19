"""
Step Audio EditX Audio Editor Node

Specialized audio editing node for Step Audio EditX unique capabilities:
- Emotion editing (14 emotions)
- Style editing (32 styles)
- Speed control (4 levels)
- Paralinguistic effects (10 types)
- Denoising and VAD

This is NOT voice conversion - it's specialized audio manipulation that modifies
the characteristics of existing audio while preserving the content.

For paralinguistic mode: Include <tags> directly in audio_text where you want sounds inserted.
Example: "Hello <Laughter> how are you?" - the transcript is auto-derived as "Hello how are you?"
"""

import os
import sys
import torch
import hashlib
import tempfile
import torchaudio
import folder_paths
from typing import Dict, Any, Optional

# Add project root for imports
current_dir = os.path.dirname(__file__)
nodes_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(nodes_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.audio.processing import AudioProcessingUtils
from utils.text.step_audio_editx_special_tags import (
    convert_step_audio_editx_tags,
    strip_paralinguistic_tags,
    has_step_audio_editx_tags,
    get_paralinguistic_options_for_ui,
)

# Global cache for iteration results - allows user to try 1, then 3, then 2 without re-running
GLOBAL_EDITX_ITERATION_CACHE: Dict[str, Dict[int, Any]] = {}


class StepAudioEditXAudioEditorNode:
    """
    Step Audio EditX Audio Editor - Specialized audio editing with emotion, style,
    speed, paralinguistic effects, and audio cleanup capabilities.

    This node allows editing existing audio to modify its emotional expression,
    speaking style, speed, or to add non-verbal sounds while preserving content.
    """

    # Edit type options
    EDIT_TYPES = ["emotion", "style", "speed", "paralinguistic", "denoise", "vad"]

    # Emotion options (14 total)
    EMOTION_OPTIONS = [
        "none",  # No emotion edit
        "happy", "sad", "angry", "excited", "calm", "fearful", "surprised", "disgusted",
        "confusion", "empathy", "embarrass", "depressed", "coldness", "admiration",
        "remove"  # Remove emotion
    ]

    # Style options (32 total + none + remove)
    STYLE_OPTIONS = [
        "none",  # No style edit
        "whisper", "serious", "child", "older", "girl", "pure", "sister", "sweet",
        "exaggerated", "ethereal", "generous", "recite", "act_coy", "warm", "shy",
        "comfort", "authority", "chat", "radio", "soulful", "gentle", "story", "vivid",
        "program", "news", "advertising", "roar", "murmur", "shout", "deeply", "loudly",
        "arrogant", "friendly",
        "remove"  # Remove style
    ]

    # Speed options
    SPEED_OPTIONS = ["none", "faster", "slower", "more faster", "more slower"]

    @classmethod
    def INPUT_TYPES(cls):
        # Build paralinguistic tags hint for tooltip - show ALL tags
        available_tags = get_paralinguistic_options_for_ui()
        tags_hint = ", ".join([f"<{t}>" for t in available_tags])

        return {
            "required": {
                "input_audio": ("AUDIO", {
                    "tooltip": "Input audio to edit (0.5-30 seconds limit)"
                }),
                "audio_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": f"Transcript of the input audio.\n\nFor paralinguistic mode: Include tags where you want sounds inserted.\nExample: 'Hello <Laughter> how are you?'\n\nAvailable tags: {tags_hint}"
                }),
                "edit_type": (cls.EDIT_TYPES, {
                    "default": "emotion",
                    "tooltip": "Type of edit to apply:\n- emotion: Change emotional expression\n- style: Change speaking style\n- speed: Adjust speaking speed\n- paralinguistic: Add non-verbal sounds (use <tags> in audio_text)\n- denoise: Remove background noise\n- vad: Remove silent portions"
                }),
            },
            "optional": {
                "emotion": (cls.EMOTION_OPTIONS, {
                    "default": "none",
                    "tooltip": "Emotion to apply (only used when edit_type='emotion')"
                }),
                "style": (cls.STYLE_OPTIONS, {
                    "default": "none",
                    "tooltip": "Speaking style to apply (only used when edit_type='style')"
                }),
                "speed": (cls.SPEED_OPTIONS, {
                    "default": "none",
                    "tooltip": "Speed adjustment (only used when edit_type='speed')"
                }),
                "n_edit_iterations": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "tooltip": "Number of editing iterations. Higher values = stronger effect but may reduce quality. 1-2 recommended for subtle changes, 3-5 for dramatic changes.\n\nIterations are cached - try 3, then 2, then 1 without re-running!"
                }),
                "tts_engine": ("TTS_ENGINE", {
                    "tooltip": "Step Audio EditX engine configuration. If not provided, will create default engine."
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("edited_audio", "edit_info")
    FUNCTION = "edit_audio"
    CATEGORY = "TTS Audio Suite/🎨 Step Audio EditX"

    def __init__(self):
        self._engine = None
        self._cached_settings = None  # Track settings used to create cached engine

    def _get_or_create_engine(self, tts_engine_data=None, inline_tag_precision=None, inline_tag_device=None, pre_loaded_engine=None):
        """
        Get existing engine from config or create default one.

        Args:
            tts_engine_data: Engine data from connected Step Audio EditX Engine node
            inline_tag_precision: Precision override for inline edit tags (auto, fp32, fp16, bf16, int8, int4)
            inline_tag_device: Device override for inline edit tags (auto, cuda, cpu, xpu)
            pre_loaded_engine: Pre-loaded engine from TTS processor (for EditPostProcessor reuse)
        """
        # If pre-loaded engine provided directly, return it as-is (EditPostProcessor path)
        if pre_loaded_engine is not None:
            print(f"✅ Using pre-loaded Step Audio EditX engine (reusing from TTS generation)")
            # The pre_loaded_engine is already wrapped by the factory, no need to wrap again
            return pre_loaded_engine
        else:
            if tts_engine_data is not None and "pre_loaded_engine" in tts_engine_data:
                print(f"⚠️ DEBUG: tts_engine_data has pre_loaded_engine={tts_engine_data.get('pre_loaded_engine')}, but pre_loaded_engine param is None")

        if tts_engine_data is not None:
            # Engine provided via TTS_ENGINE from Step Audio EditX Engine node
            engine_type = tts_engine_data.get("engine_type", "")
            if engine_type != "step_audio_editx":
                raise ValueError(
                    f"Wrong engine type: '{engine_type}'\n"
                    f"This node only works with Step Audio EditX.\n"
                    f"Either:\n"
                    f"  • Connect '⚙️ Step Audio EditX Engine' node\n"
                    f"  • Or leave tts_engine disconnected (uses default settings)"
                )

            # Get config and load engine via unified interface (reuses if already loaded)
            config = tts_engine_data.get("config", {})

            from utils.models.unified_model_interface import unified_model_interface, ModelLoadConfig
            from utils.device import resolve_torch_device

            # Build config for unified interface
            model_config = ModelLoadConfig(
                engine_name="step_audio_editx",
                model_type="tts",  # Use "tts" to match adapter's cache key
                model_name=config.get("model_path", "Step-Audio-EditX"),
                device=resolve_torch_device(config.get("device", "auto")),
                additional_params={
                    "torch_dtype": config.get("torch_dtype", "bfloat16"),
                    "quantization": config.get("quantization")
                }
            )

            # Load via unified interface (will reuse if already loaded)
            engine = unified_model_interface.load_model(model_config)

            # Store generation params for potential use
            engine._temperature = config.get("temperature", 0.7)
            engine._do_sample = config.get("do_sample", True)
            engine._max_new_tokens = config.get("max_new_tokens", 8192)

            return engine

        # Create default engine if not provided - use unified interface
        # Also reload if engine was deleted (e.g., by memory management for quantized models)
        # OR if inline settings have changed

        # Check if settings changed
        current_settings = (inline_tag_precision, inline_tag_device)
        settings_changed = (self._cached_settings is not None and
                           self._cached_settings != current_settings)

        if settings_changed:
            print(f"⚙️ Inline tag settings changed: {self._cached_settings} → {current_settings}, reloading engine...")
            self._engine = None
            self._cached_settings = None

        # Check if engine was deleted (MUST be after settings change check, before early return)
        engine_was_deleted = (self._engine is not None and
                             hasattr(self._engine, '_tts_engine') and
                             self._engine._tts_engine is None)

        if engine_was_deleted:
            print("⚠️ Step Audio EditX engine was deleted, reloading...")
            self._engine = None
        elif self._engine is not None:
            # Using cached engine - ensure it's on the correct device
            from utils.device import resolve_torch_device
            device_setting = inline_tag_device if inline_tag_device else "auto"
            target_device = resolve_torch_device(device_setting)

            # Check if model components need to be moved to GPU
            needs_device_move = False
            if hasattr(self._engine, 'llm') and self._engine.llm is not None:
                try:
                    current_device = next(self._engine.llm.parameters()).device
                    if str(current_device) != str(target_device):
                        needs_device_move = True
                except (StopIteration, AttributeError):
                    pass

            if needs_device_move:
                print(f"🔄 Moving cached Step Audio EditX engine from {current_device} to {target_device}...")
                import torch
                # Move LLM
                if hasattr(self._engine, 'llm') and self._engine.llm is not None:
                    self._engine.llm = self._engine.llm.to(target_device)
                # Move CosyVoice
                if hasattr(self._engine, 'cosy_model') and self._engine.cosy_model is not None:
                    if hasattr(self._engine.cosy_model, 'cosy_impl') and self._engine.cosy_model.cosy_impl is not None:
                        self._engine.cosy_model.cosy_impl = self._engine.cosy_model.cosy_impl.to(target_device)
                        self._engine.cosy_model.device = torch.device(target_device)
                print(f"✅ Engine moved to {target_device}")
            else:
                print(f"♻️ Using cached Step Audio EditX engine (precision={inline_tag_precision}, device={inline_tag_device})")

            return self._engine

        if self._engine is None:
            from utils.models.unified_model_interface import unified_model_interface, ModelLoadConfig
            from utils.device import resolve_torch_device

            # Use inline tag settings if provided, otherwise use defaults
            device_setting = inline_tag_device if inline_tag_device else "auto"
            precision_setting = inline_tag_precision if inline_tag_precision else "auto"

            # Convert precision setting to torch_dtype and quantization
            torch_dtype_val = "bfloat16"  # default
            quantization_val = None

            if precision_setting in ["fp32", "fp16", "bf16"]:
                # Map short names to full names
                dtype_map = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
                torch_dtype_val = dtype_map.get(precision_setting, "bfloat16")
                quantization_val = None
            elif precision_setting in ["int8", "int4"]:
                # Quantization - torch_dtype doesn't matter (will be quantized)
                torch_dtype_val = "bfloat16"
                quantization_val = precision_setting
            elif precision_setting == "auto":
                # Auto defaults to bf16
                torch_dtype_val = "bfloat16"
                quantization_val = None

            # Always print precision/device to verify settings are being used
            print(f"🔄 Loading Step Audio EditX engine (precision={precision_setting}, device={device_setting}, quantization={quantization_val})...")

            # Resolve model path to actual filesystem path, respecting extra_model_paths.yaml
            import folder_paths
            import os
            from utils.models.extra_paths import find_model_in_paths

            # Try to find Step Audio EditX model in any configured TTS path
            model_path = find_model_in_paths("Step-Audio-EditX", model_type='TTS', subdirs=['step_audio_editx'])

            # Fallback to default ComfyUI path if not found in extra paths
            if not model_path:
                model_path = os.path.join(folder_paths.models_dir, "TTS", "step_audio_editx", "Step-Audio-EditX")

            config = ModelLoadConfig(
                engine_name="step_audio_editx",
                model_type="tts",  # Use "tts" to match adapter's cache key
                model_name="Step-Audio-EditX",
                model_path=model_path,
                device=resolve_torch_device(device_setting),
                additional_params={"torch_dtype": torch_dtype_val, "quantization": quantization_val}
            )

            # Load engine via unified interface (already wrapped with edit_single method)
            engine = unified_model_interface.load_model(config)

            self._engine = engine
            # Cache the settings used to create this engine
            self._cached_settings = current_settings

        return self._engine

    def _validate_audio_duration(self, audio_tensor, sample_rate):
        """Validate audio duration is within limits (0.5-30 seconds)."""
        duration = audio_tensor.shape[-1] / sample_rate
        if duration < 0.5:
            raise ValueError(f"Input audio too short: {duration:.2f}s (minimum: 0.5s)")
        if duration > 30.0:
            raise ValueError(
                f"Input audio too long: {duration:.2f}s (maximum: 30s)\n\n"
                f"This is a model architecture limitation of Step Audio EditX.\n"
                f"For longer audio, split it manually at natural pauses and edit each segment separately."
            )
        return duration

    def _get_edit_info_for_type(self, edit_type, emotion, style, speed):
        """Get the appropriate edit_info value based on edit_type."""
        if edit_type == "emotion":
            if emotion == "none":
                return None
            return emotion
        elif edit_type == "style":
            if style == "none":
                return None
            return style
        elif edit_type == "speed":
            if speed == "none":
                return None
            return speed
        else:
            # paralinguistic, denoise, vad don't use edit_info from dropdowns
            return None

    def _generate_cache_key(self, audio_tensor, audio_text, edit_type, edit_info, target_text):
        """Generate a unique cache key for iteration caching."""
        # Use audio content hash + edit parameters
        audio_bytes = audio_tensor.cpu().numpy().tobytes()
        audio_hash = hashlib.md5(audio_bytes).hexdigest()[:16]

        key_parts = [
            audio_hash,
            audio_text,
            edit_type,
            str(edit_info),
            str(target_text)
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached_iterations(self, cache_key: str, max_iteration: int) -> Dict[int, Any]:
        """Get cached iterations up to max_iteration."""
        if cache_key not in GLOBAL_EDITX_ITERATION_CACHE:
            return {}
        cached_data = GLOBAL_EDITX_ITERATION_CACHE[cache_key]
        return {i: cached_data[i] for i in cached_data if i <= max_iteration}

    def _cache_iteration_result(self, cache_key: str, iteration: int, result: torch.Tensor):
        """Cache a single iteration result (limit to 5 iterations max)."""
        if cache_key not in GLOBAL_EDITX_ITERATION_CACHE:
            GLOBAL_EDITX_ITERATION_CACHE[cache_key] = {}

        # Only cache up to 5 iterations to prevent memory issues
        if iteration <= 5:
            GLOBAL_EDITX_ITERATION_CACHE[cache_key][iteration] = result.clone()

    def edit_audio(
        self,
        input_audio,
        audio_text,
        edit_type,
        emotion="none",
        style="none",
        speed="none",
        n_edit_iterations=1,
        tts_engine=None,
        suppress_progress=False,
        inline_tag_precision=None,
        inline_tag_device=None
    ):
        """
        Edit audio with specified modification.

        Args:
            input_audio: ComfyUI audio dict with waveform and sample_rate
            audio_text: Transcript of input audio. For paralinguistic mode, include <tags>.
            edit_type: Type of edit (emotion, style, speed, paralinguistic, denoise, vad)
            emotion: Emotion to apply
            style: Style to apply
            speed: Speed adjustment
            n_edit_iterations: Number of editing iterations (1-5)
            tts_engine: Optional TTS_ENGINE configuration from Step Audio EditX Engine node
            suppress_progress: If True, suppress progress messages (useful when called from edit_post_processor)

        Returns:
            Tuple of (edited_audio_dict, edit_info_string)
        """
        # Validate audio_text is provided (except for denoise/vad)
        if edit_type not in ["denoise", "vad"] and not audio_text.strip():
            raise ValueError(
                "audio_text is REQUIRED for this edit type. "
                "Please provide the transcript of the input audio."
            )

        # Extract audio from ComfyUI format
        if isinstance(input_audio, dict) and 'waveform' in input_audio:
            audio_tensor = input_audio['waveform']
            sample_rate = input_audio.get('sample_rate', 24000)
        else:
            raise ValueError("Invalid audio format. Expected ComfyUI audio dict with 'waveform' key.")

        # Handle 3D tensor [batch, channels, samples] -> [channels, samples]
        if audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(0)

        # Validate duration
        duration = self._validate_audio_duration(audio_tensor, sample_rate)
        # Compact header for edit post-processor (detailed info shown by post-processor)
        # print(f"🎨 Step Audio EditX: Editing {duration:.2f}s audio with '{edit_type}' mode")

        # Get engine (pass inline tag settings if provided)
        # Extract pre_loaded_engine if tts_engine is a dict with 'pre_loaded_engine'
        pre_loaded = tts_engine.get('pre_loaded_engine') if isinstance(tts_engine, dict) else None
        step_audio_engine = self._get_or_create_engine(tts_engine, inline_tag_precision, inline_tag_device, pre_loaded)

        # Process text based on edit_type
        # Normalize newlines to spaces - the model instruction format doesn't handle newlines well
        clean_audio_text = ' '.join(audio_text.strip().split())
        target_text_with_tags = None
        edit_info = None

        if edit_type == "paralinguistic":
            # Check if text contains paralinguistic tags
            if not has_step_audio_editx_tags(audio_text):
                raise ValueError(
                    "Paralinguistic mode requires <tags> in audio_text.\n"
                    "Example: 'Hello <Laughter> how are you?'\n"
                    "Available: <Laughter>, <Sigh>, <Breathing>, <Uhm>, etc."
                )

            # Derive clean transcript by stripping tags (already normalized above)
            clean_audio_text = ' '.join(strip_paralinguistic_tags(audio_text).split())

            # Convert <tags> to [tags] for engine, normalize whitespace
            target_text_with_tags = ' '.join(convert_step_audio_editx_tags(audio_text).split())

            print("   " + "="*60)
            print(f"   {audio_text}")
            print("   " + "="*60)

        else:
            # For other edit types, strip paralinguistic tags from transcript
            # (they're only meaningful in paralinguistic mode)
            clean_audio_text = ' '.join(strip_paralinguistic_tags(clean_audio_text).split())

            # Get edit_info from dropdowns
            edit_info = self._get_edit_info_for_type(edit_type, emotion, style, speed)

            # Validate edit_info is set for types that need it
            if edit_type in ["emotion", "style", "speed"] and edit_info is None:
                raise ValueError(f"Please select a {edit_type} option (not 'none')")

        # Generate cache key for iteration caching
        cache_key = self._generate_cache_key(
            audio_tensor, clean_audio_text, edit_type, edit_info, target_text_with_tags
        )

        # Check for cached iterations
        cached_iterations = self._get_cached_iterations(cache_key, n_edit_iterations)

        # If we have the exact number of passes cached, return immediately
        if n_edit_iterations in cached_iterations:
            print(f"💾 CACHE HIT: Using cached edit result for {n_edit_iterations} iterations")
            edited_tensor = cached_iterations[n_edit_iterations]

            # Ensure 3D for ComfyUI
            if edited_tensor.dim() == 1:
                edited_tensor = edited_tensor.unsqueeze(0).unsqueeze(0)
            elif edited_tensor.dim() == 2:
                edited_tensor = edited_tensor.unsqueeze(0)

            output_sample_rate = step_audio_engine.get_sample_rate()
            output_duration = edited_tensor.shape[-1] / output_sample_rate

            edit_info_str = (
                f"Step Audio EditX Edit Complete (CACHED):\n"
                f"Duration: {duration:.2f}s -> {output_duration:.2f}s\n"
                f"Edit type: {edit_type}\n"
                f"Iterations: {n_edit_iterations} (from cache)\n"
                f"Sample rate: {output_sample_rate}Hz"
            )

            return (
                {"waveform": edited_tensor, "sample_rate": output_sample_rate},
                edit_info_str
            )

        # Find highest cached iteration to resume from
        start_iteration = 0
        current_tensor = audio_tensor
        current_sample_rate = sample_rate  # Track current sample rate

        for i in range(n_edit_iterations, 0, -1):
            if i in cached_iterations:
                print(f"💾 CACHE: Resuming from cached iteration {i}/{n_edit_iterations}")
                current_tensor = cached_iterations[i]
                start_iteration = i
                current_sample_rate = step_audio_engine.get_sample_rate()  # Cached tensors are at engine sample rate
                break

        # Save initial/resumed audio to temp file for engine
        comfyui_temp_dir = folder_paths.get_temp_directory()

        # Get generation params from engine config or use defaults
        max_new_tokens = getattr(step_audio_engine, '_max_new_tokens', 8192)
        temperature = getattr(step_audio_engine, '_temperature', 0.7)
        do_sample = getattr(step_audio_engine, '_do_sample', True)

        try:
            # Perform remaining iterations
            for iteration in range(start_iteration, n_edit_iterations):
                iteration_num = iteration + 1
                if not suppress_progress:
                    print(f"🔄 Edit iteration {iteration_num}/{n_edit_iterations}...")

                # Create progress bar for this iteration
                progress_bar = None
                try:
                    import comfy.utils
                    progress_bar = comfy.utils.ProgressBar(max_new_tokens)
                except (ImportError, AttributeError):
                    pass

                # Save current tensor to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=comfyui_temp_dir) as tmp_file:
                    temp_audio_path = tmp_file.name
                    save_tensor = current_tensor
                    if save_tensor.dim() == 1:
                        save_tensor = save_tensor.unsqueeze(0)
                    elif save_tensor.dim() == 3:
                        save_tensor = save_tensor.squeeze(0)

                    # CRITICAL: Resample to engine's sample rate (24000 Hz) if needed
                    # Input audio might be 44100 Hz, 48000 Hz, etc.
                    # Saving with wrong sample rate causes pitch shift
                    # Step Audio EditX always uses 24000 Hz
                    target_sample_rate = 24000
                    if current_sample_rate != target_sample_rate:
                        print(f"   Resampling audio from {current_sample_rate}Hz to {target_sample_rate}Hz")
                        # Use torchaudio for resampling
                        resampler = torchaudio.transforms.Resample(
                            orig_freq=current_sample_rate,
                            new_freq=target_sample_rate
                        )
                        resampled_tensor = resampler(save_tensor.cpu())
                        torchaudio.save(temp_audio_path, resampled_tensor.cpu(), target_sample_rate)
                    else:
                        torchaudio.save(temp_audio_path, save_tensor.cpu(), current_sample_rate)

                # Perform single edit with progress tracking
                current_tensor = step_audio_engine.edit_single(
                    input_audio_path=temp_audio_path,
                    audio_text=clean_audio_text,
                    edit_type=edit_type,
                    edit_info=edit_info,
                    text=target_text_with_tags,
                    progress_bar=progress_bar,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample
                )

                # Update sample rate - engine always returns 24000 Hz audio
                current_sample_rate = step_audio_engine.get_sample_rate()

                # Cache this iteration
                self._cache_iteration_result(cache_key, iteration_num, current_tensor)

                # Cleanup temp file
                if os.path.exists(temp_audio_path):
                    try:
                        os.unlink(temp_audio_path)
                    except Exception:
                        pass

            edited_tensor = current_tensor

            # Get sample rate from engine
            output_sample_rate = step_audio_engine.get_sample_rate()

            # Ensure output is 3D [batch, channels, samples] for ComfyUI
            if edited_tensor.dim() == 1:
                edited_tensor = edited_tensor.unsqueeze(0).unsqueeze(0)
            elif edited_tensor.dim() == 2:
                edited_tensor = edited_tensor.unsqueeze(0)

            # Calculate output duration
            output_duration = edited_tensor.shape[-1] / output_sample_rate

            # Build edit info string
            edit_details = f"Edit type: {edit_type}"
            if edit_info:
                edit_details += f"\nValue: {edit_info}"
            if target_text_with_tags:
                edit_details += f"\nTarget: {target_text_with_tags}"

            cache_info = f"(resumed from iteration {start_iteration})" if start_iteration > 0 else ""

            edit_info_str = (
                f"Step Audio EditX Edit Complete:\n"
                f"Duration: {duration:.2f}s -> {output_duration:.2f}s\n"
                f"Transcript: {clean_audio_text}\n"
                f"{edit_details}\n"
                f"Iterations: {n_edit_iterations} {cache_info}\n"
                f"Sample rate: {output_sample_rate}Hz"
            )

            # Return in ComfyUI format
            return (
                {"waveform": edited_tensor, "sample_rate": output_sample_rate},
                edit_info_str
            )

        except Exception as e:
            # Cleanup on error
            raise


# Node class mapping for ComfyUI registration
__all__ = ["StepAudioEditXAudioEditorNode"]
