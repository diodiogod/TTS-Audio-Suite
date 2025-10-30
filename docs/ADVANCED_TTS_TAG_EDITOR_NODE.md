# 🎨 Advanced TTS Tag Editor Node

## Overview

Advanced multiline string editor node that extends ComfyUI's standard multiline widget with a sidebar containing context-aware controls for TTS-specific tags (character switching, parameters, pauses). The node features real-time tag generation, preset management, and intelligent syntax support based on the CHARACTER_SWITCHING_GUIDE and PARAMETER_SWITCHING_GUIDE.

---

## Core Features

### 1. Editor Interface

- **Textarea**: Standard multiline editor (like ComfyUI's native string node)
- **Left Sidebar**: Vertical control panel with buttons and dropdowns
- **Smart Selection Detection**: Identify if selection is `[Character]text` or plain text
- **Caret Position Tracking**: Insert tags at cursor position or wrap selection
- **Mid-Sentence Tag Insertion**: Tags can be inserted anywhere in text, not just at the beginning (e.g., `"text text [alice|seed:2] more text"`)
- **Undo/Redo Stack**: Full history of all edits with visual indicators, accessible via buttons and keyboard shortcuts (Ctrl+Z / Ctrl+Shift+Z)

### 2. Sidebar Controls - Basic

#### Character Management
- Dropdown selector for available characters (auto-discovered from `voices_examples/`)
- Text input to specify custom character name
- Button: "Add Character" → wraps selection or inserts at caret as `[CharacterName]text`
- Shows current available characters with icons/indicators

#### Parameter Buttons
(appear if character is selected)

- `[Seed]` - Opens number input (0-4294967295) → adds `|seed:X`
- `[Temp]` - Opens slider (0.1-2.0) → adds `|temperature:X`
- `[CFG]` - Opens slider (engine-specific range) → adds `|cfg:X`
- `[Speed]` - Slider (0.5-2.0, F5-TTS only) → adds `|speed:X`
- `[Steps]` - Spinner (1-100, VibeVoice/IndexTTS) → adds `|inference_steps:X`
- `[Top-P]` / `[Top-K]` - For advanced engines

#### Pause Control
- Time input field with duration selector (milliseconds/seconds)
- Button: "Insert Pause" → adds `[pause:XXXms]` or `[pause:XXs]` at caret
- Quick presets: "0.5s", "1s", "2s" buttons

#### Language Control
- Dropdown for supported languages (en, de, fr, ja, es, it, pt, th, no)
- Works with character dropdown: `[language:Character]` format
- Shows which models are available/loaded

#### Undo/Redo Controls
- **Undo Button** (`[↶]`) - Revert last change
- **Redo Button** (`[↷]`) - Restore undone change
- **History Indicator** - Shows current position in edit history (e.g., "Step 5/10")
- Keyboard shortcuts: `Ctrl+Z` (Undo), `Ctrl+Shift+Z` (Redo)
- Full persistence: Undo/redo history survives session reload

### 3. Preset System

#### Preset Management UI
- Three preset slots (Preset 1, Preset 2, Preset 3)
- Each slot has:
  - Save button → captures configured tags
  - Load button → applies preset to selection/caret
  - Delete button → clears saved preset
  - Label field → custom name for the preset

#### Preset Content
- Stores: character, language, and all parameters
- Example: "Alice Calm" = `[de:Alice|seed:42|temp:0.3|cfg:0.7]`
- Can be applied as wrapper around selection or at caret

#### Preset Persistence
- Store in ComfyUI widget state (node.serialize/deserialize)
- Also optionally save to local config file for sharing
- **Full Session Persistence**: All presets, settings, and metadata survive ComfyUI session reload

### 4. Advanced Features

#### Syntax Validation
- Real-time visual indicators for tag syntax errors
- Highlight brackets, detect mismatched `[` `]`
- Warn on invalid parameter values (out of range)
- Suggest corrections (e.g., `temp` → `temperature`)

#### Multi-Selection Support
- Apply same tag to multiple selected segments
- Split by paragraph breaks or sentence punctuation
- Batch apply parameters across multiple `[Character]text` blocks

#### Tag Inspector
- Show existing tags in selection/current line
- Checkbox UI to toggle tags on/off temporarily
- Quick-edit dialog for existing tag values

#### Auto-Formatting
- Button: "Auto-Format Tags" → organize tags consistently
- Validates and cleans malformed tags
- Groups parameters in standard order: `[char|seed|temp|cfg|speed]`

#### Visual Feedback
- Color-code different tag types:
  - Character tags: **Blue** `[Character]`
  - Parameters: **Green** `|seed:value`
  - Pauses: **Orange** `[pause:1s]`
  - Language tags: **Purple** `[de:Character]`
- Hover tooltips explaining what each tag does

---

## SRT Support & Formatting

### SRT-Aware Mode Detection

The editor **automatically detects SRT format** and enables specialized visualization:

```srt
1
00:00:01,000 --> 00:00:04,000
[Alice] Hello! I'm Alice speaking with precise timing.

2
00:00:04,500 --> 00:00:09,500
[Bob|seed:42|temperature:0.5] And I'm Bob with parameters!

3
00:00:10,000 --> 00:00:14,000
[de:Narrator] Zurück zum Erzähler in Deutsch.
```

### SRT Color Coding

When SRT format is detected, the editor applies specialized syntax highlighting:

- **SRT Index**: `1`, `2`, `3` → **Gray** background (disabled/informational)
- **Timestamp**: `00:00:01,000 --> 00:00:04,000` → **Cyan** (timing information)
- **Text Content**: Plain text → Black
  - **Character Tags**: `[Character]` → **Blue**
  - **Language Tags**: `[de:Character]` → **Purple**
  - **Parameters**: `|seed:42|temp:0.5` → **Green**
  - **Pauses**: `[pause:1s]` → **Orange**
- **Blank Lines**: Between entries → Gray

### SRT-Specific Sidebar Options

When editing SRT content, the sidebar includes:

#### Subtitle Entry Navigation
- Previous/Next buttons to jump between SRT entries
- Current entry indicator: "Entry 2 of 5"
- Quick jump dropdown to select specific subtitle

#### Timing Tools
- **Insert Timing**: Auto-generates timestamps based on duration
- **Adjust Timing**: Shift all subsequent subtitles by offset
- **Validate Timing**: Check for overlaps and gaps

#### Per-Entry Parameter Application
- Apply parameters to **current entry only**
- Apply parameters to **selected entries**
- Apply parameters to **all entries**
- Clear all tags from current entry

#### SRT Validation
- Check syntax: proper index/timestamp/text format
- Warn on timing overlaps
- Suggest fixes for malformed entries
- Preview final SRT before saving

### SRT Tag Application Examples

#### Applying Character to Current Entry
```
Select entry 2
Click "Add Character" → [Alice]
Result:
2
00:00:04,500 --> 00:00:09,500
[Alice] And I'm Bob with parameters!
```

#### Adding Parameters to Selected Entries
```
Select entries 1-3
Click [Seed] → 42
Result:
1
00:00:01,000 --> 00:00:04,000
[Alice|seed:42] Hello! I'm Alice speaking...

2
00:00:04,500 --> 00:00:09,500
[Alice|seed:42] And I'm Bob with parameters!

3
00:00:10,000 --> 00:00:14,000
[Alice|seed:42] Zurück zum Erzähler in Deutsch.
```

---

## UI Layout

```
┌─ Advanced TTS Editor ─────────────────────────────┐
│ ┌───────────────┬─────────────────────────────────┐│
│ │   SIDEBAR     │      TEXTAREA                   ││
│ │               │      (multiline editor)         ││
│ │ ╔═══════════╗ │  1                              ││
│ │ ║ Character ║ │  00:00:01,000 --> 00:00:04,000 ││
│ │ ║ [Dropdown]║ │  [Alice] Hello there!          ││
│ │ ║[+ Add]    ║ │                                 ││
│ │ ╚═══════════╝ │  2                              ││
│ │               │  00:00:04,500 --> 00:00:09,500 ││
│ │ ╔═══════════╗ │  [Bob|seed:42] Nice to meet     ││
│ │ ║Parameters ║ │                                 ││
│ │ ║[Seed  ▼] ║ │  3                              ││
│ │ ║[Temp  -/-]║ │  00:00:10,000 --> 00:00:14,000 ││
│ │ ║[CFG   -/-]║ │  [pause:1s]                     ││
│ │ ║[Speed -/-]║ │                                 ││
│ │ ╚═══════════╝ │                                 ││
│ │               │                                 ││
│ │ ╔═══════════╗ │                                 ││
│ │ ║  Pause    ║ │                                 ││
│ │ ║[  1.0  ▼] │ │                                 ││
│ │ ║[Insert]  ║ │                                 ││
│ │ ╚═══════════╝ │                                 ││
│ │               │                                 ││
│ │ ╔═══════════╗ │                                 ││
│ │ ║ Presets   ║ │                                 ││
│ │ ║P1[Save]   ║ │                                 ││
│ │ ║   [Load]  ║ │                                 ││
│ │ ║   [Del ]  ║ │                                 ││
│ │ ║P2[Save]   ║ │                                 ││
│ │ ║   [Load]  ║ │                                 ││
│ │ ║   [Del ]  ║ │                                 ││
│ │ ║P3[Save]   ║ │                                 ││
│ │ ║   [Load]  ║ │                                 ││
│ │ ║   [Del ]  ║ │                                 ││
│ │ ╚═══════════╝ │                                 ││
│ │               │                                 ││
│ │ ╔═══════════╗ │                                 ││
│ │ ║ History   ║ │                                 ││
│ │ ║[↶][↷] 5/10║ │                                 ││
│ │ ╚═══════════╝ │                                 ││
│ │               │                                 ││
│ │ [Format]      │                                 ││
│ │ [Validate]    │                                 ││
│ └───────────────┴─────────────────────────────────┘│
│ Status: SRT Mode | Entry 2/5                     │
└────────────────────────────────────────────────────┘
```

---

## Session Persistence & State Management

**Complete State Persistence**:
- **Text Content**: Full edit history with all undo/redo states
- **Undo/Redo Stack**: Complete history serialized and restored on session reload
- **Preset Data**: All custom presets and their labels
- **Sidebar Settings**: Last used character, language, parameters, pause duration
- **UI State**: Sidebar panel expanded/collapsed state, last active tab
- **Cursor Position**: Last known caret position restored on reload
- **SRT Mode Settings**: Timing configuration, entry navigation state

**Storage Mechanism**:
- Serialize all state to ComfyUI node widget state
- Automatic backup to browser localStorage for redundancy
- JSON-based format for easy inspection and manual recovery
- No data loss on browser refresh, ComfyUI restart, or system reboot

---

## Technical Implementation

### Backend (Python Node)

**File**: `nodes/text/tts_tag_editor_node.py`

- Extend standard ComfyUI node pattern
- Discover available characters/voices from `utils/voice/discovery.py`
- Parse existing tags using `utils/text/character_parser.py` and `utils/text/segment_parameters.py`
- Validate parameters against engine specs
- **Full State Serialization**: Handle undo/redo stack, presets, and all sidebar settings in node state
- Detect and validate SRT format
- **History Persistence**: Serialize complete edit history to widget state for session recovery

### Frontend (TypeScript/Vue)

**Files**:
- `web/tts_tag_editor.js` - Core editor logic
- `web/tts_tag_editor_interface.js` - ComfyUI integration
- `web/tts_tag_editor_ui.js` - Sidebar components
- `web/tts_tag_editor.css` - Styling and color schemes
- `web/tts_tag_editor_srt.js` - SRT-specific formatting and validation

**Features**:
- Extend standard multiline widget with sidebar panel
- Implement event handlers for all buttons/inputs
- Add CSS for visual tag highlighting and styling
- **Undo/Redo Implementation**: Maintain immutable history stack with efficient state snapshots
- **Keyboard Shortcuts**: Ctrl+Z (Undo), Ctrl+Shift+Z (Redo), and other common editor shortcuts
- SRT format detection and specialized rendering
- Syntax highlighting for SRT entries
- **Persistent State Manager**: Auto-save to localStorage and node widget state

### Key Components

1. **Character Discovery**: Hook into existing voice system
2. **Parameter Validation**: Reuse existing parameter filtering logic
3. **Tag Parsing**: Leverage existing character parser and parameter parser (supports mid-sentence tags)
4. **Preset Storage**: Use node widget serialization with full persistence
5. **UI State Management**: Track cursor position, selection, active preset, all sidebar settings
6. **Undo/Redo Manager**: Immutable history stack with efficient snapshots, keyboard support
7. **History Persistence**: Serialize complete undo/redo stack to localStorage and widget state
8. **SRT Detection & Validation**: Detect SRT format, validate structure, highlight syntax
9. **SRT Navigation**: Jump between entries, manage per-entry operations
10. **State Recovery**: Auto-restore all editor state on session reload (text, history, presets, settings)

### Integration Points

- `utils/voice/discovery.py` - Character discovery
- `utils/text/character_parser.py` - Tag parsing
- `utils/text/segment_parameters.py` - Parameter validation
- `utils/timing/parser.py` - SRT format parsing and validation
- Existing language models for language support

---

## Features to Consider Later (v2.0)

- Syntax highlighting with language server protocol (LSP)
- Auto-complete for character names and parameters
- Export/import presets as JSON
- Macro system for complex tag sequences
- Real-time validation with detailed error messages
- Voice preview playback directly from tags
- Template system for common scenes/dialogues
- Regex tag replacement for bulk editing
- Keyboard shortcuts for common operations
- SRT subtitle duration calculator
- Automatic SRT timing generation from text audio duration
- Batch character/parameter application across multiple SRT files

---

## Files to Create

1. `nodes/text/tts_tag_editor_node.py` - Backend node
2. `web/tts_tag_editor.js` - Main editor implementation
3. `web/tts_tag_editor_ui.js` - Sidebar UI components
4. `web/tts_tag_editor_srt.js` - SRT-specific features
5. `web/tts_tag_editor.css` - Styling

---

## Usage Examples

### Basic Text Editing with Mid-Sentence Tags

```
Once upon a time, [Narrator] there was a beautiful princess.
[Alice] She lived in a [grand|seed:42] castle.
[Bob] Who [de:Alice] spoke many languages.
Back to [Bob|temperature:0.7] normal text.
```

### Multi-Parameter Example

```
[Alice|seed:42|temperature:0.3|cfg:0.7] She spoke softly.
[Bob|seed:100|temperature:0.8|cfg:1.0] He shouted loudly!
[de:Merchant|seed:55] Guten Tag, meine Damen!
```

### SRT with Multilingual Characters

```
1
00:00:01,000 --> 00:00:04,000
[Narrator] Welcome to our international story.

2
00:00:04,500 --> 00:00:09,500
[de:Alice|temperature:0.5] Hallo! Ich bin Alice.

3
00:00:09,600 --> 00:00:14,000
[fr:Bob|temperature:0.6] Bonjour! Je suis Bob.

4
00:00:14,100 --> 00:00:18,000
[Narrator] And they lived happily ever after.
```

### Preset Application Example

Saved Preset: "Action Scene" = `[|seed:random|temperature:0.8|cfg:1.2]`

Apply to selection:
```
Before:
[Hero] I will save you!
[Villain] Never!
[Hero] This ends now!

After:
[Hero|seed:42|temperature:0.8|cfg:1.2] I will save you!
[Villain|seed:100|temperature:0.8|cfg:1.2] Never!
[Hero|seed:55|temperature:0.8|cfg:1.2] This ends now!
```

---

## Related Documentation

- [Character Switching Guide](CHARACTER_SWITCHING_GUIDE.md) - Multi-character and language switching
- [Parameter Switching Guide](PARAMETER_SWITCHING_GUIDE.md) - Per-segment parameter control
- [ChatterBox Special Tokens Guide](CHATTERBOX_V2_SPECIAL_TOKENS.md) - Emotion and sound tokens
- README.md - Complete feature overview
