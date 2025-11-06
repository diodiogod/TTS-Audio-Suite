# Upstream TTS Engines Tracking Index

This document provides a centralized reference for monitoring updates to all bundled TTS engines. Each engine has its own update log file tracking changes, bug fixes, and new features from the official upstream repository.

**Last Updated:** 2025-11-06

---

## Quick Reference Table

| Engine | Bundled Source | Location | Official Repository | Update Log | Last Checked |
|--------|---------|----------|---------------------|------------|--------------|
| **IndexTTS-2** | ✅ Full | `engines/index_tts/indextts/` | [github.com/index-tts/index-tts](https://github.com/index-tts/index-tts) | [INDEXTTS2_UPDATE_LOG.md](./INDEXTTS2_UPDATE_LOG.md) | 2025-11-06 |
| **ChatterBox** | ✅ Full | `engines/chatterbox/` | [github.com/resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) | [CHATTERBOX_UPDATE_LOG.md](./CHATTERBOX_UPDATE_LOG.md) | TBD |
| **ChatterBox Multilingual (23-Lang)** | ✅ Full | `engines/chatterbox_official_23lang/` | [github.com/resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) | [CHATTERBOX_UPDATE_LOG.md](./CHATTERBOX_UPDATE_LOG.md) | TBD |
| **F5-TTS** | ✅ Full | `engines/f5_tts/` | [github.com/SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) | [F5TTS_UPDATE_LOG.md](./F5TTS_UPDATE_LOG.md) | TBD |
| **Higgs Audio** | ✅ Full | `engines/higgs_audio/boson_multimodal/` | [github.com/boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio) | [HIGGS_AUDIO_UPDATE_LOG.md](./HIGGS_AUDIO_UPDATE_LOG.md) | TBD |
| **RVC** | ✅ Full | `engines/rvc/` | [github.com/RVC-Project/Retrieval-based-Voice-Conversion](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion) | [RVC_UPDATE_LOG.md](./RVC_UPDATE_LOG.md) | TBD |
| **VibeVoice** | ⚠️ Wrapper Only | `engines/vibevoice_engine/` | [github.com/microsoft/VibeVoice](https://github.com/microsoft/VibeVoice) ⚠️ Disabled | [VIBEVOICE_UPDATE_LOG.md](./VIBEVOICE_UPDATE_LOG.md) | TBD |

---

## Engine Status Details

### ✅ Fully Bundled Engines (Source Code Included)

These engines have their complete source code bundled in the TTS Audio Suite repository:

1. **IndexTTS-2** - Full inference code and models bundled
2. **ChatterBox** - Full model and inference code bundled
3. **ChatterBox Multilingual** - Wrapper around same ChatterBox repo with language support
4. **F5-TTS** - Full source code with configs, models, and training scripts
5. **Higgs Audio** - Full boson_multimodal framework bundled
6. **RVC** - Complete voice conversion implementation

### ⚠️ Wrapper-Only Engines

These are external dependencies integrated via wrappers:

- **VibeVoice** - Wrapper around Microsoft's `vibevoice` PyPI package (⚠️ **Note**: Official repo is disabled by Microsoft as of Sept 2025)

---

## Update Checking Workflow

### Manual Checking (via slash command)
```bash
/check_tts_bundled_updates [engine_name] [--verbose]
```

Examples:
- `/check_tts_bundled_updates` - Check ALL engines
- `/check_tts_bundled_updates IndexTTS-2` - Check specific engine
- `/check_tts_bundled_updates ChatterBox --verbose` - Verbose output with detailed diffs

### What Gets Checked

For each engine, the process:
1. Clones/updates the official upstream repository
2. Compares commit history since last check
3. Identifies new features, bug fixes, breaking changes
4. Updates the engine's UPDATE_LOG.md file
5. Flags items requiring integration

### Priority Levels for Updates

- 🔴 **Critical**: Security fixes, breaking changes, major bug fixes
- 🟡 **Important**: New features, performance improvements, compatibility updates
- 🟢 **Nice-to-have**: Documentation, minor tweaks, non-critical improvements

---

## Individual Engine Logs

Each engine has its own update log tracking:
- **Date checked**: When upstream was last reviewed
- **New commits**: List of changes since last version
- **Actionable items**: What needs integration/testing
- **Integration status**: Whether changes have been applied

See individual UPDATE_LOG.md files in this directory for details.

---

## Integration Checklist

When integrating upstream updates:

- [ ] Read upstream changelog/commits
- [ ] Identify compatibility issues with our wrapper code
- [ ] Create backup of current bundled code
- [ ] Apply changes incrementally
- [ ] Run full test suite
- [ ] Document any conflicts or customizations we maintain
- [ ] Update version in nodes.py if integration warrants version bump
- [ ] Commit with reference to upstream changes

---

## Known Customizations We Maintain

### IndexTTS-2
- Audio normalization fix for Linux (v4.14.15) - **Not in upstream**
- Custom processor wrapper for unified interface
- Audio amplitude normalization to handle platform-specific torchaudio.load() behavior

### ChatterBox (both versions)
- Unified adapter pattern for consistent interface
- Integration with unified node architecture

### F5-TTS
- Unified adapter pattern for consistent interface

### Higgs Audio
- Unified adapter pattern for consistent interface

### RVC
- Complete custom implementation for TTS Audio Suite

### VibeVoice
- Wrapper around external vibevoice package
- ⚠️ **Status**: Microsoft disabled official repo in Sept 2025 - consider migration path

---

## Related Documentation

- [NEW_ENGINE_IMPLEMENTATION_GUIDE.md](../NEW_ENGINE_IMPLEMENTATION_GUIDE.md) - How to integrate new engines
- [INDEXTTS2_UPDATE_LOG.md](./INDEXTTS2_UPDATE_LOG.md) - IndexTTS-2 specific tracking
- [../DEPENDENCY_MANAGEMENT_GUIDE.md](../DEPENDENCY_MANAGEMENT_GUIDE.md) - Dependency compatibility
- [../BUMP_SCRIPT_INSTRUCTIONS.md](../BUMP_SCRIPT_INSTRUCTIONS.md) - Version management

---

## Source Code Location Details

### Where to Find the Bundled Engine Source Code

| Engine | Main Source Files | Config/Models | Processor/Adapter |
|--------|-------------------|---------------|-------------------|
| IndexTTS-2 | `engines/index_tts/indextts/` (infer.py, infer_v2.py, gpt/, s2mel/) | `engines/index_tts/indextts/` | `engines/processors/index_tts_processor.py` |
| ChatterBox | `engines/chatterbox/chatterbox/` (models, inference) | `engines/chatterbox/models/` | `engines/adapters/chatterbox_adapter.py` |
| ChatterBox Multilingual | `engines/chatterbox_official_23lang/` (tts.py, models/*, language_models.py) | `engines/chatterbox_official_23lang/models/` | `engines/processors/` |
| F5-TTS | `engines/f5_tts/` (src/, model/, infer/) | `engines/f5_tts/configs/` | `engines/f5tts/f5tts.py` (adapter) |
| Higgs Audio | `engines/higgs_audio/boson_multimodal/` (full framework) | `engines/higgs_audio/boson_multimodal/model/` | `engines/adapters/higgs_audio_adapter.py` |
| RVC | `engines/rvc/impl/` (inference, models) | `engines/rvc/impl/lib/` | `engines/rvc/rvc_engine.py` |
| VibeVoice | External package: `vibevoice` (PyPI) | Downloaded at runtime | `engines/vibevoice_engine/vibevoice_engine.py` |

### Key File Patterns to Check When Updating

**Inference/Generation Code:**
- IndexTTS-2: `engines/index_tts/indextts/infer_v2.py`
- ChatterBox: `engines/chatterbox/chatterbox/inference.py` or `tts.py`
- F5-TTS: `engines/f5_tts/src/f5_tts/infer/`
- Higgs Audio: `engines/higgs_audio/boson_multimodal/serve/serve_engine.py`
- RVC: `engines/rvc/impl/infer_pack/`

**Model Loading:**
- IndexTTS-2: `engines/index_tts/indextts/infer_v2.py` (model initialization)
- ChatterBox: `engines/chatterbox/chatterbox/models/`
- F5-TTS: `engines/f5_tts/src/f5_tts/model/`
- Higgs Audio: `engines/higgs_audio/boson_multimodal/model/higgs_audio.py`
- RVC: `engines/rvc/impl/infer_pack/modules.py`

**Audio Processing:**
- IndexTTS-2: `engines/index_tts/indextts/infer_v2.py` (audio normalization, vocoding)
- ChatterBox: Check audio post-processing in inference methods
- F5-TTS: `engines/f5_tts/` audio handling
- All: Look for clamp, normalize, scale operations

---

## Important Notes

- **ChatterBox and ChatterBox Multilingual** both use the same official repository: `github.com/resemble-ai/chatterbox`
  - The difference is in implementation: one focuses on single-language, the other on 23-language support
  - They share the same upstream to track for updates

- **VibeVoice** is NOT actively maintained - Microsoft disabled the official GitHub repository
  - Consider evaluating alternatives or pinning to a specific version
  - The wrapper depends on the external `vibevoice` PyPI package

- **F5-TTS** vs **f5tts** directories:
  - `f5_tts/` = Full bundled source code (official implementation from SWivid)
  - `f5tts/` = Wrapper/adapter layer only (NOT the official source code)

---

## Next Steps

1. ✅ Document all official upstream repositories
2. ✅ Identify which code is bundled vs wrapped
3. ✅ Implement `/check_tts_bundled_updates` slash command
4. ⏳ Create UPDATE_LOG.md for each engine (in progress)
5. ⏳ Establish regular checking schedule (weekly/monthly)
6. ⏳ Create integration PR template for upstream changes
