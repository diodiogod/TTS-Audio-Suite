# SRT Edit Tag Batch Processing Pattern

## Problem
Edit post-processing was happening per-subtitle, causing the editing engine to load/unload multiple times per SRT generation, wasting time and VRAM.

## Solution
Batch all edit processing at the END of SRT generation, after ALL subtitles are generated.

## Implementation Pattern

### 1. Adapter Layer (e.g., `vibevoice_adapter.py`)
In `generate_with_pause_tags()`:
- Extract edit tags even when NO pause tags present
- Return segment dict format `[{waveform, text, edit_tags, ...}]` when edit tags exist
- Return plain tensor when NO edit tags (backwards compatibility)

### 2. Regular Processor (e.g., `step_audio_editx_processor.py`)
In `_process_character_switching()`:
```python
# Apply edit post-processing ONLY when NOT in SRT mode
if not self._srt_mode:
    audio_segments = apply_edit_post_processing(
        audio_segments,
        self.config,
        pre_loaded_engine=self.adapter.engine
    )
```

### 3. SRT Processor (e.g., `vibevoice_srt_processor.py`, `step_audio_editx_srt_processor.py`)

In `_process_all_subtitles()`:

```python
# Initialize collection for batch processing
all_segments_for_editing = []

# Loop through subtitles
for i, subtitle in enumerate(subtitles):
    segments = generate_subtitle(subtitle)

    # Check if segments have edit tags
    if isinstance(segments, list) or has_edit_tags(segments):
        # Tag with subtitle index for later reassembly
        for seg in segments:
            seg['subtitle_index'] = i
        all_segments_for_editing.extend(segments)
        audio_segments[i] = None  # Placeholder
    else:
        # No edit tags - store directly
        audio_segments[i] = combine_segments(segments)

# BATCH PROCESS all edits at once
if all_segments_for_editing:
    print(f"🎨 Applying edit post-processing to {len(all_segments_for_editing)} segment(s) from all subtitles...")
    processed = apply_edit_post_processing(
        all_segments_for_editing,
        config,
        pre_loaded_engine=engine
    )

    # Group back by subtitle_index
    segments_by_subtitle = group_by_subtitle_index(processed)

    # Combine and store
    for sub_idx, segs in segments_by_subtitle.items():
        audio_segments[sub_idx] = combine_segments(segs)
```

### 4. Subtitle Processing Functions
In functions like `_process_custom_character_switching_subtitle_with_params()`:
- Return segment dicts when edit tags present (don't apply edits)
- Return plain tensor when no edit tags
- Let caller handle batch processing

## Benefits
- Edit engine loads ONCE instead of once per subtitle
- Faster SRT generation
- Less VRAM thrashing
- Same quality output

## Engines Implemented
- ✅ VibeVoice SRT
- ✅ Step Audio EditX SRT
- ✅ Higgs Audio SRT

## TODO
- ChatterBox 23-Lang (needs edit tag support first)
