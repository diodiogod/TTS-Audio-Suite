# F5-TTS (Bundled Version)

This is a bundled version of F5-TTS included with TTS Audio Suite to resolve dependency conflicts.

## Original Project

- **Original Repository**: https://github.com/SWivid/F5-TTS
- **Original Version**: 1.1.7
- **License**: MIT License (see LICENSE file)

## Modifications

This bundled version has been modified from the original:

1. **Dependency Fix**: Changed `numpy<=1.26.4` to `numpy>=1.24.0` to support numpy 2.x
2. **Package Name**: Renamed to `f5-tts-bundled` to avoid conflicts with pip-installed version

## Why Bundled?

The original F5-TTS 1.1.7 has a restrictive `numpy<=1.26.4` constraint that conflicts with modern ML packages requiring `numpy>=2.0.0`. However, F5-TTS works perfectly fine with numpy 2.x in practice.

This bundled version allows TTS Audio Suite users to:
- Use F5-TTS without dependency conflicts
- Install alongside other ML packages requiring numpy 2.x
- Avoid complex workarounds like `--no-deps` installations

## Copyright Notice

This bundled version complies with the MIT License of the original F5-TTS project.
All original copyright notices and attributions are preserved.

**Original Copyright**: Copyright (c) 2024 Yushen CHEN
**Bundled by**: TTS Audio Suite