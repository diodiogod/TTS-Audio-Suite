# IndexTTS-2 Emotion Control Guide

IndexTTS-2 features advanced emotion control capabilities that allow you to precisely control the emotional expression of generated speech. This guide covers all available emotion control methods and their applications.

## Overview

IndexTTS-2 supports multiple emotion control methods that can be combined for sophisticated emotional expression:

- **Direct Audio Reference**: Use any audio file as an emotion reference
- **Character Voices**: Use character audio references from the Character Voices node
- **Emotion Vectors**: Manual 8-emotion slider control with precise values
- **Text Emotion**: AI-powered QwenEmotion analysis from text descriptions with dynamic templates
- **Character Tag Emotions**: Per-character emotion control using `[Character:emotion_ref]` syntax

## Emotion Control Inputs and Blending

The IndexTTS-2 Engine has two emotion inputs:

- **`emotion_control`**: vector controls or Qwen text emotion
- **`emotion_audio`**: an audio reference from an AUDIO or Character Voices node

Both inputs may be connected at the same time. Text emotion is analyzed into an
8-value vector; audio emotion remains an audio-derived conditioning signal. The
engine blends the two signals in its latent emotion-conditioning space rather
than converting the audio into the eight visible vector values.

For backward compatibility, the original `emotion_control` socket still accepts
legacy audio connections, but new workflows should use `emotion_audio` for
audio references. A character tag such as `[Alice:angry_bob]` supplies a
segment-local audio reference and can also be combined with vector/text emotion
for that segment.

## Method 1: Direct Audio Reference

Connect any audio file directly to the IndexTTS-2 Engine's `emotion_audio` input.

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
AUDIO node → IndexTTS-2 Engine (emotion_audio)
```

## Method 2: Character Voices Audio Reference

Use the `opt_narrator` output from the 🎭 Character Voices node as an emotion reference.

**Setup:**

1. Add a 🎭 Character Voices node
2. Select a voice with the desired emotional expression
3. Connect `opt_narrator` output to IndexTTS-2 Engine `emotion_audio` input

**Advantages:**

- Leverages your existing voice library
- Consistent character-based emotions
- Easy to manage and organize

**Example workflow:**

```
🎭 Character Voices (David_Attenborough) → opt_narrator → IndexTTS-2 Engine (emotion_audio)
```

## Method 3: Emotion Vectors

Use the 🌈 IndexTTS-2 Emotion Vectors node for precise manual control over 8 different emotions.
Connect its `emotion_control` output to the IndexTTS-2 Engine's `emotion_control` input.

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
Connect its `emotion_control` output to the IndexTTS-2 Engine's `emotion_control` input.

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

## Combining Audio with Vectors or Text

Connect both emotion sources when you want an audio performance to provide the
base delivery while vectors or Qwen text analysis add a targeted emotional
direction:

```text
🎭 Character Voices (opt_narrator) ──→ emotion_audio
🌈 Emotion Vectors or Text Emotion ──→ emotion_control
                                      IndexTTS-2 Engine
```

`emotion_alpha` is the shared overall emotion-intensity control. The audio
reference and vector/text signal are blended during IndexTTS-2 conditioning;
they are not generated as two separate voices and mixed afterward.

For example, an audio reference can provide a natural speaking style while a
`[sad:+0.2|calm:-0.1]` inline adjustment adds a restrained sadness to one
segment. A Qwen text preset can be used the same way.

Character audio and inline emotion parameters can share one tag:

```text
[Bob:br_ivan_raiva3|sad:+0.25|calm:-0.10] Bob speaks with a restrained overlay.
[Bob:br_ivan_raiva3|emotion:"quiet grief masking frustration"] Bob uses Qwen text emotion too.
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

Inline controls override connected global vector/text values for that segment.
A character audio emotion reference such as `[Alice:sad_reference]` replaces the
global audio reference for that segment, but it can still blend with the
segment's vector/text control. Inline settings revert at the next segment and do
not mutate the connected vector.

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

Emotion controls can be composed directly on character tags: place the caret on
`[Bob:audio_reference]` and add a vector or text emotion to append it as a pipe
parameter.

The named-emotion panel includes a magnitude slider and a press-drag-release
radial picker: direction chooses the emotion and distance chooses its value.
The operation dropdown determines whether that value is absolute, a positive
delta, or a negative delta. Saved text and vector presets refresh in the sidebar
immediately after they are changed in the preset manager.

Clicking a saved `[emotion:preset_name]` tag in the editor opens a small anchored
preset dropdown, allowing that line's preset to be swapped without opening the
full manager. Adding an emotion control while the caret is inside a pure emotion
tag replaces that tag; when the caret is inside a character/audio tag, the
editor appends or updates the emotion parameter after the existing `|` fields.

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
[Bob:br_ivan_raiva3|sad:+0.25] Bob uses an audio reference plus a vector delta.

*assuming happy_sarah, angry_tom and David are alias or character voices in yout folder with that name
```

\*assuming happy_sarah, angry_tom and David are alias or character voices in yout folder with that name



**Character tag behavior:**

- Character tags select the speaker and can provide a segment-local audio emotion reference
- A segment-local audio reference replaces the global audio reference for that segment
- Global or inline vector/text controls can still blend with that audio reference
- Other characters use the global audio and vector/text settings
- Allows mixing different emotion sources in the same audio

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
