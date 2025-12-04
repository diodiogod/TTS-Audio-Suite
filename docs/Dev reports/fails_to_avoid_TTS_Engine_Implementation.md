# TTS Engine Implementation - Fails & Lessons Learned

## Step Audio EditX Implementation

### Import Errors
- **Missing import**: Always add `import folder_paths` when using `folder_paths.get_temp_directory()` in node files
- **Bundled code imports**: For complex bundled packages with internal cross-imports, add `sys.path.insert(0, impl_dir)` at top of main files instead of converting all imports

### Audio Utility Functions
- **Temp file creation**: Use `AudioProcessingUtils.save_audio_to_temp_file()` not `save_audio()` (doesn't exist)
- **ComfyUI audio format**: Extract waveform with `audio_tensor['waveform']` and `audio_tensor.get('sample_rate')` before passing to utilities

### Download Configuration
- **HuggingFace file list**: Always verify actual repo structure with `curl` before hardcoding file lists
- **Downloader API**: Use dict format `[{"remote": f, "local": f}]` for unified_downloader, not plain strings
- **File existence**: Check actual HuggingFace repo structure - don't assume file naming patterns (e.g., `model-00001.safetensors` not `model-00001-of-00002.safetensors`)
- **CRITICAL - No cache downloads**: NEVER allow models to auto-download to cache directories. ALL models/weights must be downloaded via our downloader to organized `models/TTS/` folders. Disable auto-download and add to downloader instead.

### Factory Registration
- **Unified model interface**: Always implement `register_<engine>_factory()` when using `unified_model_interface.load_model()`
- **Factory initialization**: Add factory to `initialize_all_factories()` or it won't be registered

### Node Registration
- **Engine branch**: Add engine-specific branch in `_create_engine_node_instance()` for each new engine
- **Variable naming**: Use consistent variable names (`config` not `engine_config`) throughout node code

### Voice References
- **Reference text requirement**: F5-TTS and Step Audio EditX REQUIRE `prompt_text` (transcript), ChatterBox/VibeVoice/Higgs don't
- **Narrator voice mapping**: Map narrator from TTS Text node input by saving `audio_tensor` to temp file with `reference_text`
- **Voice discovery**: Use `get_character_mapping()` not `discover_voices_for_engine()` (latter doesn't exist)

### Pause Tag Format
- **Correct format**: Use `[pause:2]`, `[pause:1.5s]`, `[pause:500ms]` - NOT `<pause_X>`
- **Parser usage**: Use `PauseTagProcessor.parse_pause_tags()` which returns `('text', content)` or `('pause', duration_seconds)` tuples

### Seed Control
- **Global torch state**: Some engines (Step Audio EditX) use global `torch.manual_seed()` for reproducibility, not function parameters
- **CUDA seeds**: Always set both `torch.manual_seed()` and `torch.cuda.manual_seed_all()` for GPU reproducibility

### Quantization Support
- **Device movement**: Bitsandbytes quantized models (int4/int8) CANNOT be moved with `.to()` - wrap in try-except
- **VRAM clearing**: Quantized models stay on GPU; "Clear VRAM" must unload completely (not move to CPU)
- **Error handling**: Catch `ValueError` with "is not supported for" and "8-bit"/"4-bit" in message

### Character Switching Implementation
- **Missing voice fallback**: When `get_character_mapping()` returns `(None, None)` for a character, MUST still add entry to `voice_mapping` with fallback to narrator/default voice
- **Voice mapping consistency**: All characters in parsed segments MUST have entries in `voice_mapping`, even if empty/None
- **Adapter validation**: Adapters that require voice references (Step Audio EditX, F5-TTS) need graceful fallback when voice_ref is None
- **Character parser interaction**: Character parser changes unknown characters to "narrator" - need to ensure original character names are preserved for voice mapping lookup
- **Parser fix**: Add text tag characters to `available_characters` (lowercase) + set language defaults like IndexTTS
- **Working pattern**: `all_available = set(get_available_characters())` + aliases + text tag chars (lowercase) + "narrator" → `set_available_characters(list(all_available))` + `set_character_language_default()` for each char
