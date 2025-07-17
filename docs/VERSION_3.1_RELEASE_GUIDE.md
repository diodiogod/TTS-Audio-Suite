# Version 3.1 Release Guide

## Overview
Version 3.1 introduces major new features: **Character Switching System** and **Overlapping Subtitles Support**. This represents a significant enhancement to the TTS capabilities.

---

## 🎭 Major Features Added

### 1. Character Switching System
**Universal character switching for all TTS nodes using `[Character]` tags**

**Core Components:**
- `core/character_parser.py` - Universal text parsing with line-by-line processing
- Enhanced `core/voice_discovery.py` - Character voice discovery with flat file + folder support
- Character alias mapping system with `character_alias_map.json` files

**Integration:**
- ✅ F5TTS Voice Generation node
- ✅ ChatterBox Voice TTS node  
- ✅ F5TTS SRT Voice Generation node
- ✅ ChatterBox SRT Voice TTS node

**Features:**
- Line-by-line character parsing (fixes narrator fallback)
- Support for both folder structure and flat files
- Character alias mapping (`"Bob": "Clint_Eastwood CC3 (enhanced2)"`)
- Priority system: `models/voices/` overrides `voices_examples/`
- Robust fallback to narrator for unknown characters
- Preserves all existing functionality (caching, chunking)

### 2. Overlapping Subtitles Support
**Natural conversation patterns with overlapping dialogue**

**Implementation:**
- SRT parser now allows overlaps by default (`allow_overlaps=True`)
- Smart timing mode fallback system
- Enhanced audio assembly for overlapping segments

**Mode Behavior:**
- `stretch_to_fit`: Allows overlaps, stretches each segment to exact timing
- `pad_with_silence`: Naturally handles overlaps (places audio at start times)
- `smart_natural`: Auto-fallbacks to `pad_with_silence` when overlaps detected

---

## 📝 Changelog for v3.1

### 🎭 Character Switching System
- **NEW**: Universal `[Character]` tag support across all TTS nodes
- **NEW**: Character alias mapping with JSON configuration files
- **NEW**: Dual voice discovery (models/voices + voices_examples directories)
- **NEW**: Line-by-line character parsing for natural narrator fallback
- **NEW**: Robust fallback system for missing characters
- **ENHANCED**: Voice discovery with flat file and folder structure support
- **ENHANCED**: Character-aware caching system
- **DOCS**: Added comprehensive CHARACTER_SWITCHING_GUIDE.md

### 🎙️ Overlapping Subtitles Support
- **NEW**: Support for overlapping subtitles in SRT nodes
- **NEW**: Automatic mode switching (smart_natural → pad_with_silence)
- **NEW**: Enhanced audio mixing for conversation patterns
- **ENHANCED**: SRT parser with overlap detection and optional validation
- **ENHANCED**: Audio assembly with overlap-aware timing

### 🔧 Technical Improvements
- **ENHANCED**: SRT parser preserves newlines for character switching
- **ENHANCED**: Character parsing with punctuation normalization
- **ENHANCED**: Voice discovery initialization on startup
- **FIXED**: Line-by-line processing in SRT mode for proper narrator fallback
- **FIXED**: Character tag removal before TTS generation

---

## 🚀 Version Bump Commands

### Using the automated script:
```bash
python3 scripts/bump_version_enhanced.py 3.1.0 "Major release: Character switching system and overlapping subtitles support"
```

### Manual version updates:
```bash
# Update version in nodes.py
sed -i 's/VERSION = ".*"/VERSION = "3.1.0"/' nodes.py

# Update version in pyproject.toml  
sed -i 's/version = ".*"/version = "3.1.0"/' pyproject.toml

# Update README.md title
sed -i 's/v[0-9]\+\.[0-9]\+\.[0-9]\+/v3.1.0/' README.md
```

---

## 📋 Commit Message Template

```
Version 3.1.0: Character switching system and overlapping subtitles support

Major Features:
🎭 Character Switching System
- Universal [Character] tag support across all TTS nodes
- Character alias mapping with JSON configuration
- Dual voice discovery (models/voices + voices_examples)
- Line-by-line parsing with natural narrator fallback
- Robust fallback system for missing characters

🎙️ Overlapping Subtitles Support  
- Support for overlapping subtitles in SRT nodes
- Automatic smart_natural → pad_with_silence fallback
- Enhanced audio mixing for conversation patterns

Technical Improvements:
- Enhanced SRT parser with newline preservation
- Character-aware caching system
- Voice discovery initialization optimization
- Improved character parsing with alias resolution

Breaking Changes: None (fully backward compatible)

Fixes:
- Fixed narrator fallback in SRT mode
- Fixed character tag removal before TTS generation
- Fixed line-by-line processing in multiline subtitles
```

---

## 🏷️ Release Notes Template

```markdown
# 🎭 ChatterBox Voice v3.1.0 - Character Switching & Overlapping Subtitles

## 🌟 Major New Features

### Character Switching System
Transform your TTS with seamless character switching using simple `[Character]` tags!

**Example Usage:**
```
Hello! This is the narrator speaking.
[Alice] Hi there! I'm Alice with my unique voice.
[Bob] And I'm Bob! Great to meet you both.
Back to the narrator for the conclusion.
```

**Key Features:**
- **Universal Support**: Works across all TTS nodes (F5TTS, ChatterBox, SRT)
- **Character Aliases**: Map simple names to complex filenames
  ```json
  {
    "Bob": "Clint_Eastwood CC3 (enhanced2)",
    "Alice": "female_01"
  }
  ```
- **Flexible Voice Organization**: Support for both folder structure and flat files
- **Robust Fallback**: Unknown characters gracefully use narrator voice
- **Performance Optimized**: Character-aware caching system

### Overlapping Subtitles Support
Create natural conversation patterns with overlapping dialogue!

**Example SRT:**
```srt
1
00:00:00,000 --> 00:00:07,000
[BGpeople] Background chatter continues...

2
00:00:00,500 --> 00:00:02,500  
[Bob] Oh Hello Alice!

3
00:00:01,500 --> 00:00:04,000
[Alice] Hey Bob! How are you?
```

**Smart Mode Handling:**
- `stretch_to_fit`: Allows overlaps with exact timing
- `pad_with_silence`: Naturally handles overlapping audio
- `smart_natural`: Auto-switches to `pad_with_silence` when overlaps detected

## 📖 Documentation
- **[Complete Character Switching Guide](docs/CHARACTER_SWITCHING_GUIDE.md)**
- Updated README with new features
- Example character alias map included

## 🔧 Technical Details
- Fully backward compatible - existing workflows unchanged
- Enhanced SRT parser with overlap support
- Improved voice discovery with dual folder support
- Character-aware caching maintains performance

## 🎯 Use Cases
- **Audiobooks**: Multiple character voices in stories
- **Dialogue Systems**: Game characters and chatbots
- **Educational Content**: Different speakers for learning materials
- **Conversation Patterns**: Natural overlapping dialogue
- **Accessibility**: Voice distinction for better comprehension

Ready to create rich, multi-character audio content? Download v3.1.0 and transform your TTS workflows!
```

---

## 🧪 Pre-Release Testing Checklist

### Character Switching Tests
- [ ] F5TTS node with `[Character]` tags
- [ ] ChatterBox node with `[Character]` tags
- [ ] F5TTS SRT with character switching
- [ ] ChatterBox SRT with character switching
- [ ] Character alias mapping functionality
- [ ] Fallback to narrator for unknown characters
- [ ] Line-by-line processing in multiline text

### Overlapping Subtitles Tests
- [ ] SRT with overlapping timestamps
- [ ] stretch_to_fit mode with overlaps
- [ ] pad_with_silence mode with overlaps  
- [ ] smart_natural auto-fallback to pad_with_silence
- [ ] Conversation patterns (interruptions, background chatter)

### Regression Tests
- [ ] Existing workflows still work unchanged
- [ ] Caching system functions correctly
- [ ] Chunking system preserved
- [ ] Voice discovery works for existing voice files

---

## 🐛 Known Issues to Fix Before Release
*(Update this section based on testing)*

- [ ] Any character parsing edge cases
- [ ] Audio mixing artifacts in overlap mode
- [ ] Performance issues with large character sets
- [ ] SRT timing report accuracy with overlaps

---

**Release Readiness**: ⏳ Pending bug fixes and testing
**Target Release**: After thorough testing and bug resolution
**Impact**: Major feature release, fully backward compatible