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

# Global settings for inline edit tags (set from ComfyUI settings menu)
_inline_tag_settings = {
    "precision": "auto",
    "device": "auto"
}


def set_inline_tag_settings(precision: str = "auto", device: str = "auto"):
    """
    Set global settings for inline edit tag processing.
    Called from the API endpoint when user changes settings in ComfyUI menu.

    Args:
        precision: Model precision (auto, fp32, fp16, bf16, int8, int4)
        device: Device (auto, cuda, cpu, xpu)
    """
    global _inline_tag_settings
    _inline_tag_settings["precision"] = precision
    _inline_tag_settings["device"] = device
    print(f"🎨 Step Audio EditX inline tags: precision={precision}, device={device}")


def get_inline_tag_settings() -> Dict[str, str]:
    """Get current inline tag settings"""
    return _inline_tag_settings.copy()


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
    edit_tag,  # EditTag dataclass
    tts_engine_data=None  # Optional engine data to reuse pre-loaded engine
) -> torch.Tensor:
    """
    Apply a single edit using the Audio Editor node.

    Args:
        editor_node: StepAudioEditXAudioEditorNode instance
        audio_dict: ComfyUI audio dict with 'waveform' and 'sample_rate'
        transcript: Transcript of the audio (required for editing)
        edit_tag: EditTag object with edit_type, value, iterations
        tts_engine_data: Optional engine data to reuse pre-loaded engine (avoids duplicate loading)

    Returns:
        Edited audio dict (ComfyUI format)
    """
    edit_type = edit_tag.edit_type
    value = edit_tag.value
    iterations = edit_tag.iterations

    # Get inline tag settings from global config
    settings = get_inline_tag_settings()
    precision = settings.get("precision", "auto")
    device = settings.get("device", "auto")

    # Debug: ALWAYS print settings to verify they're being read
    print(f"  🎨 DEBUG: Inline tag settings: precision={precision}, device={device}")

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

    # Call the Audio Editor node (has progress bar built-in)
    # Pass inline tag settings and engine data for model loading
    edited_audio, _ = editor_node.edit_audio(
        input_audio=audio_dict,
        audio_text=audio_text,
        edit_type=edit_type,
        emotion=emotion,
        style=style,
        speed=speed,
        n_edit_iterations=iterations,
        tts_engine=tts_engine_data,  # Reuse pre-loaded engine to avoid duplicate loading
        suppress_progress=True,  # We show our own iteration progress
        inline_tag_precision=precision,
        inline_tag_device=device
    )

    return edited_audio


# Global cache for Voice Changer node (lazy-loaded)
_cached_vc_node = None

def _restore_voice_via_vc(edited_audio_dict, original_voice_dict, iterations=1, language="English"):
    """
    Apply ChatterBox Official 23-Lang Voice Changer to restore original voice.
    Reuses existing UnifiedVoiceChangerNode logic with its cache system.

    Args:
        edited_audio_dict: Audio after edits {waveform, sample_rate}
        original_voice_dict: Original voice reference {waveform, sample_rate}
        iterations: Number of VC refinement passes (1-5)
        language: Language for VC model (e.g., "English", "Polish", etc.)

    Returns:
        dict: Restored audio {waveform, sample_rate}
    """
    global _cached_vc_node

    # Lazy-load VC node instance (cache globally like Audio Editor)
    if _cached_vc_node is None:
        import os
        import sys
        import importlib.util

        # Get project root and construct path to VC node
        current_dir = os.path.dirname(__file__)  # utils/audio
        utils_dir = os.path.dirname(current_dir)  # utils
        project_root = os.path.dirname(utils_dir)  # project root

        vc_node_path = os.path.join(
            project_root, 'nodes', 'unified', 'voice_changer_node.py'
        )

        # Load module using importlib
        spec = importlib.util.spec_from_file_location("voice_changer_node", vc_node_path)
        vc_module = importlib.util.module_from_spec(spec)
        sys.modules["voice_changer_node"] = vc_module
        spec.loader.exec_module(vc_module)

        # Instantiate the VC node
        _cached_vc_node = vc_module.UnifiedVoiceChangerNode()

    # Create engine config for ChatterBox 23-Lang VC
    engine_config = {
        'engine_type': 'chatterbox_official_23lang',
        'config': {
            'device': 'auto',
            'language': language
        }
    }

    # Call existing VC node logic with refinement passes
    result = _cached_vc_node.convert_voice(
        TTS_engine=engine_config,
        source_audio=edited_audio_dict,
        narrator_target=original_voice_dict,
        refinement_passes=iterations,
        max_chunk_duration=30,
        chunk_method="smart"
    )

    # Return first element (audio dict) from tuple result
    return result[0] if isinstance(result, tuple) else result


def process_segments(
    segments: List[Dict],
    engine_config: Optional[Dict] = None,
    pre_loaded_engine = None  # Pre-loaded Step Audio EditX engine to reuse (avoids loading duplicate)
) -> List[Dict]:
    """
    Process all segments, applying Step Audio EditX edits where edit_tags exist.

    Called ONCE after all TTS generation completes.

    Args:
        segments: List of segment dicts with 'waveform', 'sample_rate', 'text', 'edit_tags' keys
        engine_config: Optional engine configuration
        pre_loaded_engine: Pre-loaded Step Audio EditX engine from TTS processor (for reuse)

    Returns:
        List of processed segments (modified in place for efficiency)
    """
    from utils.text.step_audio_editx_special_tags import (
        parse_edit_tags_with_iterations,
        sort_edit_tags_for_processing
    )

    # Get inline tag settings once at the start
    settings = get_inline_tag_settings()
    inline_precision = settings.get("precision", "auto")
    inline_device = settings.get("device", "auto")
    print(f"🔧 EditPostProcessor: Using inline tag settings - precision={inline_precision}, device={inline_device}")

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

    # Prepare engine data for Audio Editor (reuses pre-loaded engine if available)
    tts_engine_data = None
    if pre_loaded_engine is not None:
        tts_engine_data = {
            "engine_type": "step_audio_editx",
            "config": engine_config if engine_config else {},
            "pre_loaded_engine": pre_loaded_engine  # Pass engine directly to avoid re-loading
        }

    # Track segments that need voice restoration (collect for batch processing)
    segments_needing_restore = []

    # Process each segment with edits (NON-RESTORE edits only)
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

        # Store original PRE-EDIT audio as voice reference for restore tag
        import torch
        original_audio_dict = {
            'waveform': waveform.clone() if hasattr(waveform, 'clone') else waveform,
            'sample_rate': sample_rate
        }

        # Show segment being edited with clean formatting
        print(f"\n📝 Segment {idx + 1} - Applying edit tags:")
        print("="*60)
        print(transcript)
        print("="*60)

        # Sort tags for optimal processing order
        sorted_tags = sort_edit_tags_for_processing(edit_tags)

        # Group tags by type: non-paralinguistic → paralinguistic → restore
        restore_tags = [t for t in sorted_tags if t.edit_type == 'restore']
        paralinguistic_tags = [t for t in sorted_tags if t.edit_type == 'paralinguistic']
        non_paralinguistic_tags = [t for t in sorted_tags if t.edit_type not in ['paralinguistic', 'restore']]

        # Create ComfyUI audio dict for the editor node
        current_audio_dict = {
            'waveform': waveform,
            'sample_rate': sample_rate
        }

        # Track intermediate audio after each edit for restore reference
        iteration_audio_snapshots = {}  # iteration_number -> audio_dict
        current_iteration = 0

        # Apply non-paralinguistic edits first (emotion, style, speed)
        for tag in non_paralinguistic_tags:
            try:
                print(f"  🎨 Applying {tag.edit_type}:{tag.value} ({tag.iterations} iteration{'s' if tag.iterations > 1 else ''})")
                edited_audio_dict = _apply_edit_via_node(
                    editor_node=editor_node,
                    audio_dict=current_audio_dict,
                    transcript=transcript,
                    edit_tag=tag,
                    tts_engine_data=tts_engine_data
                )
                # Use edited audio as input for next edit
                current_audio_dict = edited_audio_dict

                # Store snapshot for each iteration
                for i in range(tag.iterations):
                    current_iteration += 1
                    iteration_audio_snapshots[current_iteration] = {
                        'waveform': current_audio_dict['waveform'].clone() if hasattr(current_audio_dict['waveform'], 'clone') else current_audio_dict['waveform'],
                        'sample_rate': current_audio_dict['sample_rate']
                    }
            except Exception as e:
                print(f"     ❌ Failed: {e}")

        # Apply all paralinguistic tags with proper iteration handling
        if paralinguistic_tags:
            try:
                # Sort by position (descending) for insertion
                sorted_para = sorted(paralinguistic_tags, key=lambda t: t.position or 0, reverse=True)
                max_iterations = max(tag.iterations for tag in paralinguistic_tags)

                # Print what we're applying
                tag_summary = ", ".join([f"{t.value}:{t.iterations}" for t in paralinguistic_tags])
                print(f"  🎨 Applying paralinguistic: {tag_summary} ({max_iterations} total iteration{'s' if max_iterations > 1 else ''})")

                # Apply iterations - each iteration includes tags that still need processing
                for iteration in range(1, max_iterations + 1):
                    # Determine which tags are active in this iteration
                    active_tags = [t for t in sorted_para if iteration <= t.iterations]

                    if not active_tags:
                        break  # All tags exhausted

                    # Build audio_text with active tags (use angle brackets for Audio Editor)
                    audio_text = transcript

                    # Track position offset as we insert tags (each insertion shifts positions)
                    position_offset = 0

                    for tag in active_tags:
                        position = tag.position if tag.position is not None else len(audio_text)
                        position = min(position, len(audio_text))

                        # Apply cumulative offset from previous insertions
                        adjusted_position = position + position_offset

                        # Check if we need space before tag (if previous char is alphanumeric)
                        needs_space_before = (adjusted_position > 0 and audio_text[adjusted_position - 1].isalnum())

                        # Check if we need space after tag (if next char is alphanumeric)
                        needs_space_after = (adjusted_position < len(audio_text) and audio_text[adjusted_position].isalnum())

                        # Build tag text with appropriate spacing
                        space_before = " " if needs_space_before else ""
                        space_after = " " if needs_space_after else ""
                        tag_text = f"{space_before}<{tag.value}>{space_after}"

                        audio_text = audio_text[:adjusted_position] + tag_text + audio_text[adjusted_position:]

                        # Update offset for next tag insertion
                        position_offset += len(tag_text)

                    active_summary = ", ".join([t.value for t in reversed(active_tags)])
                    print(f"     → Iteration {iteration}/{max_iterations}: [{active_summary}]")

                    # Apply this iteration via Audio Editor node
                    edited_audio_dict, _ = editor_node.edit_audio(
                        input_audio=current_audio_dict,
                        audio_text=audio_text,
                        edit_type="paralinguistic",
                        emotion="none",
                        style="none",
                        speed="none",
                        n_edit_iterations=1,  # Always 1 iteration, we loop ourselves
                        tts_engine=tts_engine_data,  # Reuse pre-loaded engine to avoid duplicate loading
                        suppress_progress=True,  # We show our own iteration progress
                        inline_tag_precision=inline_precision,
                        inline_tag_device=inline_device
                    )
                    current_audio_dict = edited_audio_dict

                    # Store snapshot after this iteration
                    current_iteration += 1
                    iteration_audio_snapshots[current_iteration] = {
                        'waveform': current_audio_dict['waveform'].clone() if hasattr(current_audio_dict['waveform'], 'clone') else current_audio_dict['waveform'],
                        'sample_rate': current_audio_dict['sample_rate']
                    }

            except Exception as e:
                print(f"     ❌ Failed: {e}")
                import traceback
                traceback.print_exc()
                # Continue with unmodified audio for this tag
                continue

        # If segment has restore tags, collect it for batch restoration later
        if restore_tags:
            segments_needing_restore.append({
                'idx': idx,
                'segment': segment,
                'restore_tags': restore_tags,
                'edited_audio': current_audio_dict,
                'original_audio': original_audio_dict,
                'iteration_snapshots': iteration_audio_snapshots  # Pass iteration snapshots
            })

        # Update segment with edited audio (pre-restore)
        segment['waveform'] = current_audio_dict['waveform']

    # BATCH RESTORE: Process all voice restorations at once (loads VC model only once)
    if segments_needing_restore:
        print(f"\n🎨🔄 Batch voice restoration for {len(segments_needing_restore)} segment(s)...")

        for restore_info in segments_needing_restore:
            if model_management.interrupt_processing:
                print("⚠️ Voice restoration interrupted")
                break

            idx = restore_info['idx']
            segment = restore_info['segment']
            restore_tags = restore_info['restore_tags']
            edited_audio = restore_info['edited_audio']
            original_audio = restore_info['original_audio']
            iteration_snapshots = restore_info['iteration_snapshots']

            for tag in restore_tags:
                try:
                    # Determine which audio to use as reference
                    reference_iteration = tag.position  # We stored reference iteration in position field

                    if reference_iteration is not None:
                        # Use specific iteration snapshot as reference
                        if reference_iteration in iteration_snapshots:
                            reference_audio = iteration_snapshots[reference_iteration]
                            print(f"  🔄 Segment {idx + 1}: Restoring voice using iteration {reference_iteration} as reference ({tag.iterations} VC pass{'es' if tag.iterations > 1 else ''})")
                        else:
                            print(f"  ⚠️ Segment {idx + 1}: Iteration {reference_iteration} not found, using original audio")
                            reference_audio = original_audio
                    else:
                        # Use PRE-EDIT audio as reference (default behavior)
                        reference_audio = original_audio
                        print(f"  🔄 Segment {idx + 1}: Restoring voice ({tag.iterations} VC pass{'es' if tag.iterations > 1 else ''})")

                    language = engine_config.get('language', 'English') if engine_config else 'English'

                    # Ensure reference audio waveform is 2D for VC compatibility
                    ref_wf = reference_audio['waveform']
                    if ref_wf.dim() == 1:
                        ref_wf = ref_wf.unsqueeze(0)
                    elif ref_wf.dim() == 3:
                        ref_wf = ref_wf.squeeze(0)

                    reference_audio_fixed = {
                        'waveform': ref_wf,
                        'sample_rate': reference_audio['sample_rate']
                    }

                    # Restore voice using selected reference
                    restored_audio = _restore_voice_via_vc(
                        edited_audio_dict=edited_audio,
                        original_voice_dict=reference_audio_fixed,
                        iterations=tag.iterations,
                        language=language
                    )
                    edited_audio = restored_audio  # Chain for multiple restore tags

                except Exception as e:
                    print(f"     ❌ Voice restoration failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with current audio (skip restoration)

            # Update segment with restored audio
            segment['waveform'] = edited_audio['waveform']

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
