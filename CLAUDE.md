# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Focus

**Primary Focus: Audio Analyzer Node Enhancement**

We are actively developing and improving the Audio Analyzer node (`nodes/audio_analyzer_node.py`) and its interactive web interface (`web/audio_analyzer_*.js`). This is a sophisticated waveform visualization and timing extraction tool for ComfyUI specially usefull to get regions for f5-TTL audio edit node.

## Key Audio Analyzer Files

- **Core**: `nodes/audio_analyzer_node.py`, `core/audio_analysis.py`
- **Interface**: `web/audio_analyzer_core.js` (main coordinator)
- **Modules**: `web/audio_analyzer_events.js`, `web/audio_analyzer_visualization.js`, `web/audio_analyzer_controls.js`, `web/audio_analyzer_ui.js`
- **Integration**: `web/audio_analyzer_node_integration.js`

## Development Rules & Standards

### Git & Commit Policy

- **Auto-commit rule**: ALWAYS commit after big updates that I tested and confirmed working
- **Only commit when I say it works** - do not commit if I gave no feedback
- **Never push** unless explicitly requested

### Response Style

- **Be concise** - minimize tokens, avoid unnecessary explanations and long reports when task is simple
- **Direct answers** - no preamble/postamble unless asked
- **Use TodoWrite tool** for complex multi-step tasks
- **Few words for simple tweaks** to save context

## Project Context

ComfyUI ChatterBox Voice extension with Text-to-Speech and Voice Conversion capabilities. The Audio Analyzer provides precise timing extraction for F5-TTS integration with interactive waveform visualization.

## Testing Approach

- Manual testing via ComfyUI interface
- No formal testing files created
- User tests functionality and reports results
- Focus on backward compatibility and graceful degradation

## Successful Patterns

- **Modular JavaScript architecture** with clear separation of concerns
- Keep all files modular, 500-600 lines so LLMs can work with them