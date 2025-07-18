# 🎭 Character & Narrator Switching Guide

## Overview

ChatterBox Voice now supports seamless character and narrator switching for both **F5TTS** and **ChatterBox TTS** engines. Use `[CharacterName]` tags in your text to automatically switch between different voices, creating dynamic multi-character audio content.

---

## ✨ Key Features

- **🎭 Universal Character Parsing** - Works with both F5TTS and ChatterBox engines
- **🔄 Robust Fallback System** - No errors when characters not found
- **📁 Voice Folder Integration** - Organized character voice management
- **📺 SRT Support** - Character switching in subtitle timing
- **⚡ Performance Optimized** - Preserves all existing caching systems
- **🔙 Backward Compatible** - Existing workflows work unchanged

---

## 🚀 Quick Start

### 1. Text Format
Use `[CharacterName]` tags to switch voices:

```
Hello! This is the narrator speaking.
[Alice] Hi there! I'm Alice, nice to meet you.
[Bob] And I'm Bob! Great to meet you both.
Back to the narrator for the conclusion.
```

### 2. Voice File Structure
Organize character voices using filenames in `voices_examples/`:

```
voices_examples/
├── narrator.wav
├── narrator.reference.txt (for F5TTS only)
├── alice.wav
├── alice.reference.txt (for F5TTS only)
├── bob.wav
├── bob.reference.txt (for F5TTS only)
└── characters/          (folders for organization)
    ├── female_01.wav
    ├── female_01.reference.txt
    ├── male_01.wav
    └── male_01.reference.txt
```

**Character names are determined by the audio filename, not folder names. Folders are for organization only.**

### 3. Engine Differences
- **F5TTS**: Requires both `.wav` and `.reference.txt` files
- **ChatterBox**: Only needs `.wav` files (simpler setup)

---

## 📖 Detailed Usage

### Character Tag Syntax
- **Format**: `[CharacterName]Text content here`
- **Case**: Character names are case-insensitive (`[Alice]` = `[alice]`)
- **Punctuation**: Automatically cleaned (`[Alice:]` → `alice`)
- **Fallback**: Unknown characters use narrator voice automatically

### Example Multi-Character Script
```
Welcome to our story! This is the narrator.

[Alice] Hello everyone! I'm excited to be here.

[Bob] Nice to meet you, Alice. I'm Bob.

[Alice] Great to meet you too, Bob!

[Wizard] *mysterious voice* I am the ancient wizard...

And the narrator concludes the tale.
```

### SRT Subtitle Example
```srt
1
00:00:01,000 --> 00:00:04,000
Hello! This is F5-TTS SRT with character switching.

2
00:00:04,500 --> 00:00:09,500
[Alice] Hi there! I'm Alice speaking with precise timing.

3
00:00:10,000 --> 00:00:14,000
[Bob] And I'm Bob! The audio matches these exact SRT timings.
```

---

## 🛠️ Setup Instructions

### For F5TTS Nodes

1. **Add Character Voice Files**:
   ```
   voices_examples/alice.wav
   voices_examples/alice.reference.txt
   voices_examples/bob.wav
   voices_examples/bob.reference.txt
   ```

2. **Voice File Requirements**:
   - `alice.wav` - Audio sample of Alice's voice (5-15 seconds recommended)
   - `alice.reference.txt` - Transcript of what Alice says in the audio

3. **Reference Text Example**:
   ```
   Hello, this is Alice speaking clearly and naturally.
   ```

### For ChatterBox Nodes

1. **Add Audio Files Only**:
   ```
   voices_examples/alice.wav
   voices_examples/bob.wav
   ```
   - No text files needed!

2. **Flexible Organization**:
   ```
   voices_examples/
   ├── main_characters/
   │   ├── alice.wav
   │   └── bob.wav
   └── background_voices/
       ├── shopkeeper.wav
       └── guard.wav
   ```

### Alternative Voice Sources
- **ComfyUI Models**: `models/voices/` directory (same filename-based system)
- **Flexible Organization**: Any subfolder structure supported for organization

---

## 🔄 Fallback System

The system gracefully handles missing characters:

1. **Character Found**: Uses character-specific voice
2. **Character Not Found**: 
   - ⚠️ Warning message: `Using main voice for character 'Unknown' (not found in voice folders)`
   - 🔄 Automatically uses narrator/main reference voice
   - ✅ **No errors or workflow interruption**

### Example Fallback Behavior
```
[Alice] This uses Alice's voice (if available)
[UnknownCharacter] This falls back to narrator voice
[Bob] This uses Bob's voice (if available)
```

---

## ⚙️ Node-Specific Features

### 🎤 F5TTS Voice Generation
- **Input**: Text with `[Character]` tags
- **Voice Source**: Voice folders or direct reference audio/text
- **Features**: Character switching + all existing F5TTS features
- **Output**: Seamless multi-character audio

### 🎤 ChatterBox Voice TTS
- **Input**: Text with `[Character]` tags  
- **Voice Source**: Voice folders or direct reference audio
- **Features**: Character switching + all existing ChatterBox features
- **Simpler Setup**: No reference text files needed

### 📺 F5TTS SRT Voice Generation
- **Input**: SRT subtitles with `[Character]` tags
- **Features**: Character switching within precise timing
- **Benefits**: Perfect for dialogue with multiple characters

### 📺 ChatterBox SRT Voice TTS
- **Input**: SRT subtitles with `[Character]` tags
- **Features**: Character switching + SRT timing
- **Performance**: Maintains all caching optimizations

---

## 💡 Tips & Best Practices

### Voice Recording
- **Duration**: 5-15 seconds per character voice
- **Quality**: Clear, noise-free recordings
- **Content**: Natural speech that represents the character
- **Format**: WAV, MP3, FLAC supported

### Character Naming
- **Consistency**: Use the same character names throughout
- **Simplicity**: Avoid special characters in names
- **Organization**: Group related characters in subfolders

### Reference Text (F5TTS)
- **Accuracy**: Must match the audio exactly
- **Clarity**: Write exactly what is spoken
- **Length**: Should match audio duration

### Performance Optimization
- **Caching**: Character voices are cached automatically
- **Chunking**: Long character segments are chunked intelligently
- **Reuse**: Same character voices used across multiple generations

---

## 🐛 Troubleshooting

### Common Issues

#### "Character not found" warnings
- **Cause**: Character audio file missing or incorrectly named
- **Solution**: Check that audio filename matches character name used in `[Character]` tags
- **Result**: Uses fallback voice (no workflow break)

#### F5TTS missing reference text
- **Cause**: `.reference.txt` file missing for character
- **Solution**: Add reference text file matching audio
- **Alternative**: Use ChatterBox engine (no text required)

#### Audio quality inconsistent
- **Cause**: Different recording conditions per character
- **Solution**: Record all characters with similar setups
- **Tip**: Use consistent volume and background noise levels

### Debugging
Enable detailed logging to see character detection:
- Character switching mode messages: `🎭 F5-TTS: Character switching mode`
- Voice loading messages: `🎭 Using character voice for 'Alice'`
- Fallback messages: `🔄 Using main voice for character 'Unknown'`

---

## 📈 Advanced Usage

### Nested Character Organization
```
voices_examples/
├── story1/
│   ├── hero.wav
│   ├── hero.reference.txt
│   ├── villain.wav
│   └── villain.reference.txt
├── story2/
│   ├── alice.wav
│   ├── alice.reference.txt
│   ├── bob.wav
│   └── bob.reference.txt
└── narrator.wav
└── narrator.reference.txt
```

### Mixed Character Scenes
```
[Narrator] The scene opens in a busy marketplace.
[Merchant] Fresh apples! Get your fresh apples here!
[Customer] How much for a dozen?
[Merchant] Two coins, good sir!
[Narrator] The customer nodded and made the purchase.
```

### Dynamic Character Assignment
- Characters are detected automatically from text
- No pre-configuration needed
- Add new characters by adding audio files with matching names
- Remove characters by deleting audio files
- Character name = audio filename (without extension)

---

## 🎯 Integration Examples

### Story Narration
Perfect for audiobooks, stories, and educational content with multiple speakers.

### Dialogue Systems
Ideal for game dialogue, chatbots, and interactive applications.

### Educational Content
Great for language learning with different character voices.

### Accessibility
Helps distinguish speakers in audio content for better comprehension.

---

## 🔗 Related Features

- **[Voice Discovery System](../core/voice_discovery.py)**: Automatic character voice detection
- **[Audio Processing](../core/audio_processing.py)**: Smart audio chunking and combining
- **[SRT Integration](../chatterbox_srt/)**: Subtitle timing with character voices
- **[Caching System](../core/)**: Performance optimizations for character voices

---

## 📝 Version History

- **v3.0.13**: Initial character switching implementation
- **Future**: Enhanced character management UI, voice cloning improvements

---

*For technical support or feature requests, please check the main README or create an issue on GitHub.*