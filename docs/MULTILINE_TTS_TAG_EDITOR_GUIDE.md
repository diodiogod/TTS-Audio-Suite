# 🏷️ Multiline TTS Tag Editor Guide

This guide covers the practical user-facing workflow for the `🏷️ Multiline TTS Tag Editor` node.

For related topics, also see:

- `docs/CHARACTER_SWITCHING_GUIDE.md`
- `docs/PARAMETER_SWITCHING_GUIDE.md`
- `docs/INLINE_EDIT_TAGS_USER_GUIDE.md`

## What This Editor Is For

The editor is meant to make tag-heavy TTS text and SRT subtitle editing faster inside ComfyUI without manually rebuilding bracket syntax all the time.

It supports:

- Character switching tags
- Language switching tags
- Per-segment parameter overrides
- Inline Step Audio EditX tags
- Presets and edit history
- SRT-aware highlighting and timing editing

## Main Views

The top bar includes four views:

- `Editor` for direct text editing
- `History` for restoring earlier snapshots
- `Presets` for quick saved snippets or character setups
- `Library` for built-in reference material

## Basic Tag Editing

Use the left panel to insert or modify tags without writing every bracket manually.

Common examples:

```text
[Alice]
[en:Alice]
[Alice|seed:42|temperature:0.7]
[pause:1s]
```

Useful behavior:

- Character names are inserted at the caret or wrapped around the current selection
- Language and speaker can be combined in one tag
- Parameters can be stacked with `|`
- Presets can store either quick snippets or reusable speaker setups

## Inline Edit Tags

The editor also supports inline Step Audio EditX tags such as:

```text
<Laughter>
<emotion:happy>
<style:whisper:2>
<restore:1@2>
```

Use the dedicated inline-edit controls in the sidebar when you do not want to type these by hand.

## SRT Editing

When the text looks like valid SRT, the editor highlights subtitle numbers and timings differently and enables subtitle-specific editing tools.

Example:

```srt
1
00:00:01,000 --> 00:00:04,000
[Alice] Hello there.

2
00:00:04,500 --> 00:00:08,000
[Bob] This is the next subtitle.
```

### Drag Timing Controls

Timing lines are directly draggable:

- Drag the left timestamp to change cue start
- Drag the right timestamp to change cue end
- Drag the arrow to move the whole cue while keeping its duration

Modifiers:

- Hold `Shift` while dragging start or end to move the neighboring cue boundary by the same delta and preserve the existing gap
- Hold `Alt` while dragging for finer timing adjustment

### Merge Cues

Sometimes TTS sounds better when two short subtitles are spoken as one longer chunk.

Use:

- `Alt+click` on a cue number to merge that cue with the next cue
- `Alt+Shift+click` on a cue number to merge that cue with the previous cue

Merge behavior:

- Keeps the first cue start time
- Keeps the second cue end time
- Combines the text into one subtitle body
- Renumbers the remaining cues

### Split Cues

Sometimes a subtitle is too long and you want to split it into two timed parts.

Use:

- Place the caret inside subtitle text
- Press `Ctrl+Shift+Enter`

Split behavior:

- Splits the cue at the caret position
- Estimates the new timing boundary from the text proportion on each side
- Gives a small bias to the left side when the split happens after punctuation
- Enforces a minimum duration for both resulting cues
- Renumbers following cues

## Shortcuts

Core shortcuts:

- `Ctrl+Z` or `Cmd+Z` = Undo
- `Ctrl+Shift+Z` / `Cmd+Shift+Z` or `Ctrl+Y` = Redo
- `Alt+Z` = Undo
- `Alt+Shift+Z` = Redo
- `Ctrl+Shift+Enter` = Split current SRT cue at caret
- `Alt+click` on cue number = Merge with next cue
- `Alt+Shift+click` on cue number = Merge with previous cue

Sidebar insertion shortcuts:

- `Alt+C` = Add character tag
- `Alt+L` = Add language tag
- `Alt+P` = Add parameter tag
- `Alt+1`, `Alt+2`, `Alt+3` = Load preset slot

## Editing Tips

- Use merge when subtitles are too short for natural speech flow
- Use split when a single subtitle is too dense or needs better pacing
- Keep SRT text semantically clean before generation so later timing edits remain predictable
- Use presets for repeated speaker or parameter combinations
- Use the `Library` tab as a quick reminder, not as the full manual

## Known Limits

- SRT merge and split are designed for standard subtitle blocks, not malformed subtitle text
- Split timing is estimated from text distribution, so it is a good first pass, not perfect forced alignment
- If you make large structural SRT edits, review the final timings before generation
