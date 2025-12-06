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
    2. Uses the Step Audio EditX Audio Editor node for editing (reuses existing code)
    3. Applies edits in sequence (emotion → style → speed → paralinguistic)
    4. Returns modified segments for assembly
"""

import torch
from typing import List, Dict, Any, Optional
import comfy.model_management as model_management

# Cache for Audio Editor node instance
_audio_editor_node = None


def _get_audio_editor_node():
    """
    Get Step Audio EditX Audio Editor node instance.
    Lazy loads and caches the node for reuse.
    """
    global _audio_editor_node

    if _audio_editor_node is not None:
        return _audio_editor_node

    try:
        import os
        import sys
        import importlib.util

        # Get project root and construct path to audio editor node
        current_dir = os.path.dirname(__file__)  # utils/audio
        utils_dir = os.path.dirname(current_dir)  # utils
        project_root = os.path.dirname(utils_dir)  # project root

        audio_editor_path = os.path.join(
            project_root,
            "nodes",
            "step_audio_editx_special",
            "step_audio_editx_audio_editor_node.py"
        )

        # Load module from file path
        spec = importlib.util.spec_from_file_location("audio_editor_module", audio_editor_path)
        audio_editor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audio_editor_module)

        StepAudioEditXAudioEditorNode = audio_editor_module.StepAudioEditXAudioEditorNode
        _audio_editor_node = StepAudioEditXAudioEditorNode()
        return _audio_editor_node
    except Exception as e:
        print(f"❌ EditPostProcessor: Failed to load Audio Editor node: {e}")
        import traceback
        traceback.print_exc()
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


def _apply_edit_via_node(
    editor_node,
    audio_dict: dict,
    transcript: str,
    edit_tag  # EditTag dataclass
) -> torch.Tensor:
    """
    Apply a single edit using the Audio Editor node.

    Args:
        editor_node: StepAudioEditXAudioEditorNode instance
        audio_dict: ComfyUI audio dict with 'waveform' and 'sample_rate'
        transcript: Transcript of the audio (required for editing)
        edit_tag: EditTag object with edit_type, value, iterations

    Returns:
        Edited audio dict (ComfyUI format)
    """
    edit_type = edit_tag.edit_type
    value = edit_tag.value
    iterations = edit_tag.iterations

    # Prepare audio_text for the editor node
    # For paralinguistic, insert tag at position
    if edit_type == "paralinguistic":
        position = edit_tag.position if edit_tag.position is not None else len(transcript)
        audio_text = transcript[:position] + f"<{value}>" + transcript[position:]
    else:
        audio_text = transcript

    # Map edit parameters to node inputs
    emotion = value if edit_type == "emotion" else "none"
    style = value if edit_type == "style" else "none"
    speed = value if edit_type == "speed" else "none"

    print(f"  🎨 Applying {edit_type}:{value} ({iterations} iteration{'s' if iterations > 1 else ''})")

    # Call the Audio Editor node (has progress bar built-in)
    edited_audio, _ = editor_node.edit_audio(
        input_audio=audio_dict,
        audio_text=audio_text,
        edit_type=edit_type,
        emotion=emotion,
        style=style,
        speed=speed,
        n_edit_iterations=iterations,
        tts_engine=None  # Not needed, uses default
    )

    return edited_audio


def process_segments(
    segments: List[Dict],
    engine_config: Optional[Dict] = None,
    pre_loaded_engine = None  # Ignored - we use Audio Editor node now
) -> List[Dict]:
    """
    Process all segments, applying Step Audio EditX edits where edit_tags exist.

    Called ONCE after all TTS generation completes.

    Args:
        segments: List of segment dicts with 'waveform', 'sample_rate', 'text', 'edit_tags' keys
        engine_config: Optional engine configuration (unused)
        pre_loaded_engine: Ignored (we use Audio Editor node instead)

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

    # Get Audio Editor node
    editor_node = _get_audio_editor_node()
    if editor_node is None:
        print("⚠️ EditPostProcessor: Audio Editor node not available - skipping edits")
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

        # Create ComfyUI audio dict for the editor node
        current_audio_dict = {
            'waveform': waveform,
            'sample_rate': sample_rate
        }

        # Apply each edit in sequence via Audio Editor node
        for tag in sorted_tags:
            try:
                edited_audio_dict = _apply_edit_via_node(
                    editor_node=editor_node,
                    audio_dict=current_audio_dict,
                    transcript=transcript,
                    edit_tag=tag
                )
                # Use edited audio as input for next edit
                current_audio_dict = edited_audio_dict
            except Exception as e:
                print(f"⚠️ EditPostProcessor: Failed to apply {tag}: {e}")
                import traceback
                traceback.print_exc()
                # Continue with unmodified audio for this tag
                continue

        # Update segment with edited audio
        segment['waveform'] = current_audio_dict['waveform']

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
