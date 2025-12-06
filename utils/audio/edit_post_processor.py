"""
Step Audio EditX Inline Edit Post-Processor

Applies Step Audio EditX edits (emotion, style, speed, paralinguistic) to TTS-generated
audio segments that contain inline edit tags.

This post-processor works with ANY TTS engine - the edit tags are stripped before TTS
generation, then edits are applied as a post-processing step.

Usage:
    # After TTS generation returns segments
    segments = EditPostProcessor.process_segments(segments, engine_config)

Architecture:
    1. Scans segments for edit_tags metadata
    2. Lazy-loads Step Audio EditX engine only if edits are needed
    3. Applies edits in sequence (emotion → style → speed → paralinguistic)
    4. Returns modified segments for assembly
"""

import torch
import tempfile
import os
from typing import Dict, List, Any, Optional
import comfy.model_management as model_management

# Lazy imports to avoid circular dependencies
_step_audio_engine = None
_step_audio_loaded = False


def _get_step_audio_engine():
    """Lazy load Step Audio EditX engine."""
    global _step_audio_engine, _step_audio_loaded

    if _step_audio_loaded:
        return _step_audio_engine

    try:
        from engines.step_audio_editx.step_audio_editx import StepAudioEditXEngine
        from utils.device import resolve_torch_device

        device = resolve_torch_device("auto")
        print(f"🎨 EditPostProcessor: Loading Step Audio EditX engine on {device}...")

        _step_audio_engine = StepAudioEditXEngine()
        _step_audio_engine.load_model(
            model_path="Step-Audio-EditX",
            device=device,
            torch_dtype="bfloat16",
            quantization=None
        )
        _step_audio_loaded = True
        print(f"✅ EditPostProcessor: Step Audio EditX engine loaded")
        return _step_audio_engine

    except Exception as e:
        print(f"❌ EditPostProcessor: Failed to load Step Audio EditX engine: {e}")
        _step_audio_loaded = True  # Mark as "attempted" to avoid retry loops
        return None


def _validate_audio_duration(audio_tensor: torch.Tensor, sample_rate: int) -> bool:
    """
    Validate audio duration is within Step Audio EditX limits (0.5-30s).

    Args:
        audio_tensor: Audio waveform tensor
        sample_rate: Sample rate in Hz

    Returns:
        True if valid, False otherwise
    """
    # Get samples count from last dimension
    if audio_tensor.dim() == 1:
        samples = audio_tensor.shape[0]
    elif audio_tensor.dim() == 2:
        samples = audio_tensor.shape[-1]
    elif audio_tensor.dim() == 3:
        samples = audio_tensor.shape[-1]
    else:
        return False

    duration = samples / sample_rate

    if duration < 0.5:
        print(f"⚠️ EditPostProcessor: Segment too short ({duration:.2f}s < 0.5s) - skipping edit")
        return False
    if duration > 30.0:
        print(f"⚠️ EditPostProcessor: Segment too long ({duration:.2f}s > 30s) - skipping edit")
        return False

    return True


def _save_audio_to_temp(audio_tensor: torch.Tensor, sample_rate: int) -> str:
    """Save audio tensor to temporary file for Step Audio EditX processing."""
    import torchaudio

    # Ensure correct shape for torchaudio
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    elif audio_tensor.dim() == 3:
        audio_tensor = audio_tensor.squeeze(0)

    # Ensure on CPU
    if audio_tensor.is_cuda:
        audio_tensor = audio_tensor.cpu()

    # Create temp file
    fd, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)

    torchaudio.save(temp_path, audio_tensor, sample_rate)
    return temp_path


def _apply_single_edit(
    engine,
    audio_tensor: torch.Tensor,
    sample_rate: int,
    transcript: str,
    edit_tag,  # EditTag dataclass
    progress_bar=None
) -> torch.Tensor:
    """
    Apply a single edit to audio.

    Args:
        engine: StepAudioEditX engine instance
        audio_tensor: Input audio tensor
        sample_rate: Sample rate
        transcript: Transcript of the audio (required for editing)
        edit_tag: EditTag object with edit_type, value, iterations
        progress_bar: Optional ComfyUI progress bar

    Returns:
        Edited audio tensor
    """
    # Save to temp file (Step Audio EditX requires file path)
    temp_path = _save_audio_to_temp(audio_tensor, sample_rate)

    try:
        # Map edit type to Step Audio EditX API
        edit_type = edit_tag.edit_type
        edit_info = edit_tag.value
        iterations = edit_tag.iterations
        text = None  # For paralinguistic, we need special handling

        # Handle paralinguistic - insert effect marker into transcript
        if edit_type == "paralinguistic":
            # Insert [Effect] at the specified position
            position = edit_tag.position if edit_tag.position is not None else len(transcript)
            text = transcript[:position] + f"[{edit_info}]" + transcript[position:]
            edit_info = edit_info  # The paralinguistic token name

        # Special handling for denoise/vad
        if edit_type in ("denoise", "vad"):
            edit_type = edit_tag.value  # "denoise" or "vad" IS the edit_type
            edit_info = None

        print(f"  🎨 Applying {edit_type}:{edit_info or 'default'} ({iterations} iteration{'s' if iterations > 1 else ''})")

        # Apply edits iteratively
        current_audio = audio_tensor
        current_path = temp_path

        for i in range(iterations):
            if model_management.interrupt_processing:
                raise InterruptedError("Edit processing interrupted by user")

            result = engine.edit_single(
                input_audio_path=current_path,
                audio_text=transcript,
                edit_type=edit_type,
                edit_info=edit_info,
                text=text,
                progress_bar=progress_bar
            )

            # If multiple iterations, save intermediate result
            if i < iterations - 1:
                os.unlink(current_path)
                current_path = _save_audio_to_temp(result, sample_rate)

            current_audio = result

        return current_audio

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def process_segments(
    segments: List[Dict],
    engine_config: Optional[Dict] = None,
    pre_loaded_engine = None
) -> List[Dict]:
    """
    Process all segments, applying Step Audio EditX edits where edit_tags exist.

    Called ONCE after all TTS generation completes.

    Args:
        segments: List of segment dicts with 'waveform', 'sample_rate', 'text', 'edit_tags' keys
        engine_config: Optional engine configuration (unused currently, for future extensions)
        pre_loaded_engine: Optional pre-loaded Step Audio EditX engine (from adapter)

    Returns:
        List of processed segments (modified in place for efficiency)
    """
    from utils.text.step_audio_editx_special_tags import (
        parse_edit_tags_with_iterations,
        sort_edit_tags_for_processing
    )

    # Find segments that need editing
    segments_to_edit = []
    for i, segment in enumerate(segments):
        # Check if segment already has edit_tags (pre-parsed)
        if 'edit_tags' in segment and segment['edit_tags']:
            segments_to_edit.append((i, segment, segment['edit_tags']))
        else:
            # Try to extract edit tags from text (fallback)
            text = segment.get('text', '')
            if text:
                clean_text, edit_tags = parse_edit_tags_with_iterations(text)
                if edit_tags:
                    # Update segment with clean text and tags
                    segment['original_text'] = text
                    segment['text'] = clean_text
                    segment['edit_tags'] = edit_tags
                    segments_to_edit.append((i, segment, edit_tags))

    if not segments_to_edit:
        return segments

    print(f"\n🎨 EditPostProcessor: Found {len(segments_to_edit)} segment(s) with edit tags")

    # Use pre-loaded engine if provided, otherwise lazy load
    if pre_loaded_engine is not None:
        print(f"🎨 EditPostProcessor: Using pre-loaded Step Audio EditX engine")
        engine = pre_loaded_engine
    else:
        engine = _get_step_audio_engine()
        if engine is None:
            print("⚠️ EditPostProcessor: Step Audio EditX engine not available - skipping edits")
            return segments

    # Process each segment with edits
    for idx, segment, edit_tags in segments_to_edit:
        if model_management.interrupt_processing:
            print("⚠️ EditPostProcessor: Processing interrupted")
            break

        waveform = segment.get('waveform')
        sample_rate = segment.get('sample_rate', 24000)
        transcript = segment.get('text', '')

        if waveform is None:
            print(f"⚠️ EditPostProcessor: Segment {idx} has no waveform - skipping")
            continue

        # Validate duration
        if not _validate_audio_duration(waveform, sample_rate):
            continue

        print(f"\n📝 Segment {idx + 1}: \"{transcript[:50]}{'...' if len(transcript) > 50 else ''}\"")

        # Sort tags for optimal processing order
        sorted_tags = sort_edit_tags_for_processing(edit_tags)

        # Apply each edit in sequence
        current_audio = waveform
        for tag in sorted_tags:
            try:
                current_audio = _apply_single_edit(
                    engine=engine,
                    audio_tensor=current_audio,
                    sample_rate=sample_rate,
                    transcript=transcript,
                    edit_tag=tag
                )
            except Exception as e:
                print(f"⚠️ EditPostProcessor: Failed to apply {tag}: {e}")
                # Continue with unmodified audio for this tag
                continue

        # Update segment with edited audio
        segment['waveform'] = current_audio

    print(f"\n✅ EditPostProcessor: Completed edit post-processing")
    return segments


def cleanup():
    """Clean up cached engine."""
    global _step_audio_engine, _step_audio_loaded

    if _step_audio_engine is not None:
        try:
            _step_audio_engine.to("cpu")
        except Exception:
            pass
        _step_audio_engine = None
    _step_audio_loaded = False
