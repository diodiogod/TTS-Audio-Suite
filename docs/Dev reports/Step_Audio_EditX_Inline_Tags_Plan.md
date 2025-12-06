# Step Audio EditX Inline Tags - Implementation Plan

## Overview

Add inline Step Audio EditX edit tags to the segment parameter system, enabling automatic post-processing edits on TTS-generated audio from ANY engine.

**Example:**
```
[Alice|seed:43] This is an example <Laughter:3>, I'm funny!
```

Result:
1. TTS generates "This is an example, I'm funny!" (tag stripped)
2. Step Audio EditX automatically edits the segment with paralinguistic `<Laughter>` in 3 iterations

## Syntax Design

### Current Segment Parameter Syntax
```
[Character|param:value|param2:value2]
```

### New Edit Tag Syntax (inline in text)

**Paralinguistic tags** (insert non-verbal sounds):
```
Hello <Laughter> world           # Default 1 iteration
Hello <Laughter:3> world         # 3 iterations
Hello <Sigh:2> goodbye           # 2 iterations
```

**Other edit types** - need different syntax to avoid confusion:

Option A - Angle brackets with type prefix:
```
<emotion:happy:2>      # emotion=happy, 2 iterations
<style:whisper:3>      # style=whisper, 3 iterations
<speed:faster:1>       # speed=faster, 1 iteration
```

Option B - Curly braces for non-paralinguistic:
```
{emotion:happy:2}
{style:whisper:3}
{speed:faster:1}
```

Option C - Unified angle brackets, context determines type:
```
<happy:2>              # detected as emotion
<whisper:3>            # detected as style
<faster:1>             # detected as speed
<Laughter:2>           # detected as paralinguistic (capitalized)
```

**Recommendation:** Option A - explicit type prefix. Clear, unambiguous, no magic detection.

### Multiple Tags on Same Segment

**Two syntax options for multiple edits:**

1. **Pipe-separated in single tag** (like segment parameters):
```
[Alice] Hello world! <Laughter:2|style:whisper:1|emotion:happy>
```

2. **Multiple separate tags:**
```
[Alice] Hello world! <Laughter:2><style:whisper:1><emotion:happy>
```

Both are equivalent and can be mixed.

**Processing order:**

- **Paralinguistic tags**: Position in text matters (where to insert sound)
  - `Hello <Laughter> world` → Laughter inserted after "Hello"

- **Other edit types** (emotion, style, speed): Position doesn't matter, gets stripped
  - `Hello <style:whisper> world` = `Hello world <style:whisper>` = `<style:whisper> Hello world`
  - All result in: Generate "Hello world", then apply whisper style

**Execution order for mixed tags:**
1. First: All non-paralinguistic edits (emotion → style → speed) in fixed order
2. Last: Paralinguistic edits (preserves position info for sound insertion)

**Example:**
```
[Alice] Hello <Laughter:2> world <style:whisper:1>
```
1. Generate TTS: "Hello world"
2. Edit pass 1: Apply whisper style (1 iteration) - position irrelevant
3. Edit pass 2: Add Laughter at position (2 iterations) - position matters

## Processing Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Input: "[Alice] Hello <Laughter:2> world!"                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Parse segment: character=Alice, text with tags          │
│     Extract edit tags: [("Laughter", "paralinguistic", 2)]  │
│     Clean text: "Hello world!"                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Generate TTS with clean text (any engine)               │
│     Output: audio segment                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Check duration: > 30s?                                  │
│     YES → Log warning, skip edit, return original audio     │
│     NO  → Continue to edit                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Load Step Audio EditX engine (lazy load on first use)   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. For each edit tag in order:                             │
│     - Save audio to temp file                               │
│     - Call edit_single() with appropriate edit_type         │
│     - Use output as input for next edit (if any)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Output: Edited audio segment                               │
└─────────────────────────────────────────────────────────────┘
```

## Architecture Decision: Processing Order

### Option A: Generate All TTS First, Then Edit (RECOMMENDED)

```
┌─────────────────────────────────────────────────────────────┐
│  1. Parse ALL segments, extract edit tags, store metadata   │
│  2. Generate ALL TTS segments (any engine) - clean text     │
│  3. Post-process: Apply edits to segments that have tags    │
│  4. Assemble final audio                                    │
└─────────────────────────────────────────────────────────────┘
```

**Pros:**
- Step Audio EditX engine loaded ONCE, only if needed
- All TTS generation completes before edit engine loads (memory efficient)
- Can batch edit operations
- Cleaner separation of concerns
- Progress reporting is clearer (TTS phase → Edit phase)

**Cons:**
- Need to track which segments have edit tags through the pipeline

### Option B: Edit Each Segment Immediately After Generation

```
For each segment:
  1. Parse segment, extract edit tags
  2. Generate TTS
  3. If has edit tags → load Step Audio EditX → edit
  4. Store result
Assemble final audio
```

**Pros:**
- Simpler data flow

**Cons:**
- Step Audio EditX loaded/unloaded repeatedly if interleaved
- Two heavy models potentially in memory at same time
- Harder to optimize

**Decision: Option A** - Generate all TTS first, then batch edit.

---

## Implementation Details - Modular Architecture

### Core Principle: NO CODE DUPLICATION

The edit tag system integrates with existing parsers. We don't modify adapters individually.

### Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        EXISTING PIPELINE                                  │
│                                                                          │
│  Text Input → Character Parser → Segment Parameters → TTS Generation     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      NEW: EDIT TAG EXTRACTION                            │
│                                                                          │
│  Happens at parsing stage (segment_parameters.py or new unified parser)  │
│  - Extract <tag:iter> from text                                          │
│  - Store in segment metadata: segment.edit_tags = [...]                  │
│  - Return clean text for TTS                                             │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      EXISTING: TTS GENERATION                            │
│                                                                          │
│  All engines generate audio from clean text (no edit tags)               │
│  No changes to adapters needed!                                          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      NEW: POST-PROCESSOR (single point)                  │
│                                                                          │
│  Called ONCE after all TTS generation, before assembly                   │
│  - Check each segment for edit_tags metadata                             │
│  - If any segments have tags → lazy load Step Audio EditX                │
│  - Apply edits to those segments                                         │
│  - Return modified segments for assembly                                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      EXISTING: AUDIO ASSEMBLY                            │
│                                                                          │
│  Combine segments (unchanged logic)                                      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Files to Modify/Create

#### 1. Extend Existing Parser: `utils/text/segment_parameters.py`

Add edit tag extraction alongside existing parameter parsing:

```python
# Existing: [Character|seed:42|temp:0.8]
# New addition: Also extract <Laughter:2|style:whisper>

class SegmentData:
    character: str
    text: str  # Clean text (tags stripped)
    parameters: dict  # seed, temp, etc.
    edit_tags: list  # NEW: [("Laughter", "paralinguistic", 2, position), ...]
```

**No duplication** - extend existing `parse_segment()` function.

#### 2. Extend: `utils/text/step_audio_editx_special_tags.py`

Add iteration parsing to existing tag functions:

```python
# Existing: convert_step_audio_editx_tags(), strip_paralinguistic_tags()
# Add: parse_edit_tags_with_iterations()

def parse_edit_tags_with_iterations(text: str) -> Tuple[str, List[EditTag]]:
    """
    Parse all edit tags from text, return clean text and tag list.

    Supports:
    - <Laughter> → paralinguistic, 1 iter
    - <Laughter:3> → paralinguistic, 3 iter
    - <style:whisper:2> → style, 2 iter
    - <emotion:happy> → emotion, 1 iter
    - <Laughter:2|style:whisper:1> → multiple in one tag
    """
```

#### 3. NEW: `utils/audio/edit_post_processor.py`

Single centralized post-processor:

```python
class EditPostProcessor:
    """
    Post-processes TTS segments with Step Audio EditX edits.
    Lazy loads engine only when needed.
    """

    _engine = None  # Lazy loaded

    @classmethod
    def process_segments(cls, segments: List[SegmentResult]) -> List[SegmentResult]:
        """
        Process all segments, applying edits where edit_tags exist.
        Called ONCE after all TTS generation.
        """
        # Find segments with edit tags
        segments_to_edit = [s for s in segments if s.edit_tags]

        if not segments_to_edit:
            return segments  # No edits needed

        # Lazy load engine
        if cls._engine is None:
            cls._engine = cls._load_engine()

        # Process each segment with tags
        for segment in segments_to_edit:
            segment.audio = cls._apply_edits(segment)

        return segments
```

#### 4. Integration Points (MINIMAL CHANGES)

**`nodes/unified/tts_text_node.py`** - Add one call:
```python
# After TTS generation, before return
segments = EditPostProcessor.process_segments(segments)
```

**`nodes/unified/tts_srt_node.py`** - Add one call:
```python
# After all segments generated, before assembly
segments = EditPostProcessor.process_segments(segments)
```

**Adapters: NO CHANGES NEEDED**
- Adapters receive clean text (tags already stripped at parse stage)
- Adapters return audio as usual
- Post-processor handles edits centrally

### Data Flow

```
Input: "[Alice|seed:42] Hello <Laughter:2> world! <style:whisper>"

Step 1 - Parse (segment_parameters.py):
  → character: "Alice"
  → parameters: {seed: 42}
  → clean_text: "Hello world!"
  → edit_tags: [
      EditTag(type="paralinguistic", value="Laughter", iterations=2, position=6),
      EditTag(type="style", value="whisper", iterations=1, position=None)
    ]

Step 2 - TTS Generation (any adapter):
  → Input: "Hello world!"
  → Output: audio tensor

Step 3 - Post-process (edit_post_processor.py):
  → Check: segment has edit_tags? YES
  → Validate: duration < 30s? YES
  → Apply style:whisper (1 iter) - position-independent first
  → Apply Laughter at position 6 (2 iter) - paralinguistic last
  → Output: edited audio tensor

Step 4 - Assembly (unchanged):
  → Combine all segments
```

### Tag Detection

**Paralinguistic tags** (existing set):
- Breathing, Laughter, Sigh, Uhm
- Surprise-oh, Surprise-ah, Surprise-wa
- Confirmation-en, Question-ei, Dissatisfaction-hnn

**Emotion tags** (14 options):
- happy, sad, angry, excited, calm, fearful, surprised, disgusted
- confusion, empathy, embarrass, depressed, coldness, admiration

**Style tags** (32 options):
- whisper, serious, child, older, girl, pure, sister, sweet
- exaggerated, ethereal, generous, recite, act_coy, warm, shy
- comfort, authority, chat, radio, soulful, gentle, story, vivid
- program, news, advertising, roar, murmur, shout, deeply, loudly
- arrogant, friendly

**Speed tags** (4 options):
- faster, slower, more_faster, more_slower

### Duration Validation

```python
def validate_for_edit(audio_tensor, sample_rate):
    duration = audio_tensor.shape[-1] / sample_rate
    if duration > 30.0:
        print(f"⚠️ Segment too long for Step Audio EditX edit ({duration:.1f}s > 30s). "
              f"Edit tags will be skipped. Split your text into shorter segments.")
        return False
    if duration < 0.5:
        print(f"⚠️ Segment too short for Step Audio EditX edit ({duration:.1f}s < 0.5s). "
              f"Edit tags will be skipped.")
        return False
    return True
```

### Transcript Handling

For edit mode, we need the transcript (what was spoken). This is the clean text BEFORE TTS:

```python
# Original: "[Alice] Hello <Laughter:2> world!"
# Clean text for TTS: "Hello world!"
# Transcript for edit: "Hello world!"  (same as clean text)

# For paralinguistic, target text has tags:
# Target: "Hello [Laughter] world!"
```

## Edge Cases

1. **No Step Audio EditX model downloaded**
   - First use triggers download (existing behavior)
   - User sees download progress

2. **Multiple segments, only some with tags**
   - Only segments with tags get post-processed
   - Others pass through unchanged

3. **Tag in middle of word** (malformed)
   - `Hel<Laughter>lo` → Strip tag, get "Hello"
   - Best effort, no error

4. **Unknown tag**
   - `<UnknownTag:2>` → Ignored, passed as plain text, stripped
   - Or warn user?

5. **ChatterBox 23-Lang v2 tag collision**
   - Their tags use different format: `<giggle>`, `<whisper>`, etc.
   - Our edit tags: `<Laughter>`, `<style:whisper>`
   - Paralinguistic overlap: need to check if text already has v2 tags
   - Solution: Process our tags first, before sending to ChatterBox

## Example Usage

### Simple paralinguistic
```
[Alice] I can't believe it <Laughter> that's hilarious!
```

### With iterations
```
[Alice] I can't believe it <Laughter:3> that's hilarious!
```

### Multiple effects - pipe syntax
```
[Bob] I'm so tired, goodnight <Sigh:2|style:whisper:1>
```

### Multiple effects - separate tags
```
[Bob] I'm so tired <Sigh:2> goodnight <style:whisper:1>
```

### Mixed syntax (equivalent to above)
```
[Bob] I'm so tired <Sigh:2|style:whisper:1> goodnight
```

### Combined with segment parameters
```
[Alice|seed:42|temperature:0.8] Hello <Laughter:3> friend!
```

### Full combo
```
[Alice|seed:42] Hello <Laughter:2|emotion:happy:1> my friend!
```

### SRT mode
```
1
00:00:00,000 --> 00:00:03,000
[Alice] Hello <Laughter> world!

2
00:00:03,500 --> 00:00:06,000
[Bob|seed:123] Hi there <style:whisper:2|emotion:calm>
```

### Position matters for paralinguistics
```
# Different results:
[Alice] Hello <Laughter> world!    # Laughter after "Hello"
[Alice] Hello world <Laughter>!    # Laughter after "world"

# Same result (style position irrelevant):
[Alice] Hello <style:whisper> world!
[Alice] <style:whisper> Hello world!
[Alice] Hello world! <style:whisper>
```

## Decisions Made

1. **Syntax for non-paralinguistic edits:** `<emotion:happy:2>` - explicit type prefix
2. **Multiple tags:** Support both `<tag1:2|tag2:3>` pipe syntax AND `<tag1:2><tag2:3>` separate tags
3. **Iteration default:** `<Laughter>` = 1 iteration (no explicit `:1` required)
4. **Warning behavior for >30s:** Skip edit with warning in console, return original audio
5. **Engine loading:** Lazy load ONCE after all TTS generation completes
6. **Processing order:** Generate ALL TTS first, then batch edit (Option A)
7. **Architecture:** Modular, no adapter changes, single post-processor

## Open Questions

1. **Position tracking for paralinguistics:**
   - How to track character position after tag stripping?
   - Store original position? Or store as "after word N"?

2. **Progress reporting:**
   - Show separate progress for TTS phase vs Edit phase?
   - How to report which segments are being edited?

3. **Cache interaction:**
   - Should edited segments be cached separately?
   - Cache key should include edit tags?

4. **Error handling:**
   - If one segment edit fails, continue with others?
   - Return original audio for failed edits?

## Implementation Order

1. **Phase 1: Tag Parsing**
   - Extend `step_audio_editx_special_tags.py` with iteration support
   - Add `parse_edit_tags_with_iterations()` function
   - Support pipe syntax `<tag1|tag2>` and separate tags

2. **Phase 2: Segment Integration**
   - Extend `segment_parameters.py` to extract and store edit_tags
   - Ensure clean text (tags stripped) flows to TTS

3. **Phase 3: Post-Processor**
   - Create `utils/audio/edit_post_processor.py`
   - Lazy engine loading
   - Duration validation with warnings
   - Sequential edit application

4. **Phase 4: Integration**
   - Add single call in `tts_text_node.py`
   - Add single call in `tts_srt_node.py`
   - Test with multiple engines

5. **Phase 5: Testing**
   - Single tag, multiple tags
   - Mixed syntax (pipe + separate)
   - Duration edge cases
   - SRT mode
   - Multiple engines
