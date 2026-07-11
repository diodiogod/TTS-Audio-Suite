# IndexTTS-2 Emotion Control Guide

IndexTTS-2 features advanced emotion control capabilities that allow you to precisely control the emotional expression of generated speech. This guide covers all available emotion control methods and their applications.

## Overview

IndexTTS-2 supports multiple emotion control methods that can be combined for sophisticated emotional expression:

- **Direct Audio Reference**: Use any audio file as an emotion reference
- **Character Voices**: Use character audio references from the Character Voices node
- **Emotion Vectors**: Manual 8-emotion slider control with precise values
- **Text Emotion**: AI-powered QwenEmotion analysis from text descriptions with dynamic templates
- **Character Tag Emotions**: Per-character emotion control using `[Character:emotion_ref]` syntax

## Emotion Control Priority

You can only connect to the Engine node one source of control emotion: Either audio, text, or vectors.

When using tags on the text iself, **Character tag emotions** (highest priority) - `[Alice:angry_bob]` overrides all other emotion control settings for that character segment

## Method 1: Direct Audio Reference

Connect any audio file directly to the IndexTTS-2 Engine's `emotion_control` input.

**How it works:**

- IndexTTS-2 analyzes the emotional characteristics of your reference audio
- The emotional style is applied to all generated speech
- Works with any audio format (WAV, MP3, etc.)

**Best practices:**

- Use audio clips with clear emotional expression
- Choose audio with consistent voice characteristics for best results
- Avoid background music or noise

**Example:**

```
AUDIO node → IndexTTS-2 Engine (emotion_control)
```

## Method 2: Character Voices Audio Reference

Use the `opt_narrator` output from the 🎭 Character Voices node as an emotion reference.

**Setup:**

1. Add a 🎭 Character Voices node
2. Select a voice with the desired emotional expression
3. Connect `opt_narrator` output to IndexTTS-2 Engine `emotion_control` input

**Advantages:**

- Leverages your existing voice library
- Consistent character-based emotions
- Easy to manage and organize

**Example workflow:**

```
🎭 Character Voices (David_Attenborough) → opt_narrator → IndexTTS-2 Engine (emotion_control)
```

## Method 3: Emotion Vectors

Use the 🌈 IndexTTS-2 Emotion Vectors node for precise manual control over 8 different emotions.

**Available emotions:**

- **Happy**: Joy, excitement, positivity (0.0-1.2)
- **Angry**: Aggression, frustration, intensity (0.0-1.2)
- **Sad**: Melancholy, sorrow, downcast tone (0.0-1.2)
- **Surprised**: Amazement, shock, wonder (0.0-1.2)
- **Afraid**: Fear, anxiety, nervousness (0.0-1.2)
- **Disgusted**: Revulsion, displeasure, rejection (0.0-1.2)
- **Calm**: Peaceful, relaxed, steady (0.0-1.2)
- **Melancholic**: Thoughtful sadness, wistfulness (0.0-1.2)

**Usage tips:**

- Values above 1.0 create more intense emotional expression BUT MAY interfear with the cloned voice resemblance
- Combine multiple emotions for complex feelings (e.g., 0.8 Happy + 0.3 Surprised = excited joy)
- Start with single emotions, then experiment with combinations
- Use the `random` buttom to get a completely random emotion pattern. Might be too strong.

## Method 4: Text Emotion (Dynamic Analysis)

Use the 🌈 IndexTTS-2 Text Emotion node for AI-powered emotion analysis with dynamic templates.

### Static Text Emotion

Provide a simple emotion description that applies to all text segments:

```
Input: "angry and frustrated"
Result: All speech generated with angry, frustrated emotion
```

### Dynamic Templates with {seg}

Use the `{seg}` placeholder for contextual, per-segment emotion analysis:

**Template examples:**

- `"Happy character speaking: {seg}"` - Cheerful narrator
- `"Angry boss yelling: {seg}"` - Aggressive authority figure
- `"Calm meditation guide: {seg}"` - Peaceful instructor
- `"Excited game show host: {seg}"` - Energetic presenter

**How dynamic templates work:**

1. IndexTTS-2 processes each text segment separately
2. `{seg}` gets replaced with the actual segment text
3. QwenEmotion analyzes the combined context + content
4. Unique emotion vector generated for each segment

**Example:**

```
Template: "Worried parent speaking: {seg}"
Segment: "Where have you been?"
Analysis: "Worried parent speaking: Where have you been?"
Result: Anxious, concerned vocal expression
```

---

## Inline Emotion Switching

Numeric emotion tags can replace or adjust the vector for one text segment:

```text
[sad:0.7|calm:0.2] This uses explicit absolute values.
[sad:+0.3|calm:-0.2] This modifies the connected vector.
[vector:0,0,0.7,0,0,0.4,0,0.2] This supplies all eight absolute values.
[vector:+0,+0,+0.3,+0,+0,+0,+0,-0.2] This supplies eight deltas.
```

The full-vector order is `happy, angry, sad, afraid, disgusted, melancholic,
surprised, calm`. Unsigned named values are absolute; `+` and `-` named values
are relative. A full vector is relative only when every value carries a sign.

Text-emotion controls support saved presets and quoted descriptions:

```text
[emotion:restrained_anger] Text using a saved preset.
[emotion:"Restrained anger masking disappointment"] Direct text control.
[emotion:"Analyze this delivery as nervous anticipation: {seg}"] Dynamic control.
```

Inline controls override a connected global emotion control. A character audio
emotion reference such as `[Alice:sad_reference]` remains the highest-priority
control for that segment. Inline settings revert at the next segment and do not
mutate the connected vector.

The TTS Tag Editor provides the same interactive radar used by the IndexTTS-2
Emotion Vectors node. Click an existing numeric emotion tag to open its radar
as a contextual popover beside the tag. Create tags from **Inline Tags →
IndexTTS-2**, and use **Manage Emotion Presets** in its Text Emotion section for
the preset library. Text and vector presets are stored in
`models/TTS/IndexTTS/emotion_presets.json`.

Radar changes appear in the editor text immediately. Intermediate drag/input
states are not added to undo history: closing the popover commits one undoable
change, while Cancel or Escape restores the tag exactly as it was when opened.

The editor's **Inline Tags** tab also includes an **IndexTTS-2** engine panel for
inserting full absolute/delta vectors, named emotion values, saved text presets,
quoted descriptions, and dynamic descriptions containing `{seg}`.

The named-emotion panel includes a magnitude slider and a press-drag-release
radial picker: direction chooses the emotion and distance chooses its value.
The operation dropdown determines whether that value is absolute, a positive
delta, or a negative delta. Saved text and vector presets refresh in the sidebar
immediately after they are changed in the preset manager.

Clicking a saved `[emotion:preset_name]` tag in the editor opens a small anchored
preset dropdown, allowing that line's preset to be swapped without opening the
full manager. Adding any IndexTTS emotion control while the caret is already
inside another IndexTTS emotion tag replaces that whole tag instead of nesting it.

Named tags remain readable while only some emotions are active. If radar editing
activates all eight emotions, the editor automatically converts the result to the
shorter ordered `[vector:...]` form.

## Character Tag Emotion Control

Control emotions per character using inline tags in your text: `[Character:emotion_ref]`

**Syntax:**

```
[CharacterName:emotion_reference]
```

**emotion_reference options:**

- Any character name from your voices library (uses that character's voice as emotion)
- Custom emotion references

**Examples:**

```
Hello everyone! [Alice:happy_sarah] I'm so excited to be here today!
[Bob:angry_tom] That's completely unacceptable behavior.
[Narrator:David] Meanwhile, in a distant galaxy...

*assuming happy_sarah, angry_tom and David are alias or character voices in yout folder with that name
```

\*assuming happy_sarah, angry_tom and David are alias or character voices in yout folder with that name



**Character tag priority:**

- Character tags override ALL other emotion settings for that specific character
- Other characters use global emotion settings
- Allows mixing different emotions in the same audio

## Emotion Alpha Control

The `emotion_alpha` parameter on the IndexTTS-2 Engine controls the intensity of emotion application:

**Values:**

- **0.0**: No emotion applied (neutral voice)
- **0.5**: 50% emotion blend (subtle emotional influence)
- **1.0**: Full emotion intensity (standard recommended setting)
- **1.5**: 150% enhanced emotion (more dramatic)
- **2.0**: Maximum emotion intensity (very dramatic)

## Practical Workflow Examples

### Example 1: Multi-Character Drama with Individual Emotions

```text
[Alice:happy_sarah] Welcome to our cooking show!
[Bob:serious_narrator] Today we'll be making pasta.
[Alice:excited_sarah] I can't wait to get started!
```

**Setup:**

- No global emotion control needed
- Each character gets individual emotion via tags
- `emotion_alpha=1.0` for more expressiveness

### Example 2: Mixed Emotion Control

**Global setup:**

- 🌈 IndexTTS-2 Text Emotion: `"Cheerful host presenting: {seg}"`
- `emotion_alpha=0.8`

**Text with overrides:**

```text
Welcome to our show! [Bob:serious_narrator] But first, a serious announcement.
[Alice:excited_sarah] Now back to our regular programming!
```

**Result:**

- Default segments use cheerful host emotion
- Bob's line uses serious narrator emotion (overrides global)
- Alice's line uses excited emotion (overrides global)

---

This comprehensive emotion control system gives you unprecedented flexibility in creating expressive, emotionally rich TTS audio for any application.
