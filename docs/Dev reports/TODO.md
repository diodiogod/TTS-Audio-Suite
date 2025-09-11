# Project Refactoring TODO

This document tracks architectural issues and inconsistencies that need refactoring to improve modularity and reduce code duplication.

## SRT Processing Architecture Issues

### Problem: Inconsistent SRT Implementation Approaches
Different engines use completely different patterns for SRT processing:

1. **ChatterBox (Old)**: Uses `ChatterboxSRTTTSNode` class (should be processor)
2. **VibeVoice**: Uses proper `VibeVoiceSRTProcessor` class with full implementation
3. **Higgs Audio**: Uses clean modular approach with timing utilities
4. **ChatterBox Official 23-Lang**: Follows Higgs Audio pattern (good)
5. **F5-TTS**: Uses `F5TTSSRTNode` class (should be processor)

**Impact**: Code duplication, inconsistent interfaces, hard to maintain
**Solution**: Standardize on processor pattern with shared utilities

### Problem: Node vs Processor Naming Inconsistency
- ChatterBox uses `ChatterboxSRTTTSNode` but should be `ChatterboxSRTProcessor`
- F5-TTS uses `F5TTSSRTNode` but should be `F5TTSSRTProcessor`
- Only VibeVoice and ChatterBox Official 23-Lang use proper processor naming

**Impact**: Confusing architecture, breaks expected patterns
**Solution**: Rename all SRT classes to use `*Processor` pattern

### Problem: Duplicated SRT Logic
Each engine reimplements:
- SRT parsing
- Character switching within SRT segments  
- Timing calculations
- Audio assembly
- Report generation

**Impact**: Massive code duplication, bugs in some implementations
**Solution**: Create shared SRT base class with common functionality

## Voice Reference Handling Issues

### Problem: Inconsistent Audio Reference Processing
- ChatterBox: Uses file paths only, converts tensors to temp files via `handle_reference_audio`
- F5-TTS: Uses both file paths and ComfyUI tensors directly
- Higgs Audio: Uses ComfyUI tensors with reference text
- VibeVoice: Uses ComfyUI tensors

**Impact**: Different interfaces, conversion logic scattered
**Solution**: Standardize voice reference handling in base class

## Character Parser Integration Issues

### Problem: Inconsistent Language Resolution
- Some engines hardcode language mappings
- Some use character parser properly
- Some bypass language mapper entirely
- ChatterBox Official 23-Lang now uses proper integration (good example)

**Impact**: Language switching behaves differently per engine
**Solution**: Mandate character parser usage in base class

### Problem: Engine-Specific Logic in Universal Character Parser
- **Issue**: Universal character parser contains ChatterBox-specific Italian prefix logic (`apply_italian_prefix_if_needed`)
- **Problem**: Engine-specific text processing belongs in engine adapters, not universal parser
- **Current State**: Italian `[it]` prefix filter has complex conditions but should only affect:
  - Explicit `[it:Alice]` or `[italian:Bob]` tags
  - Characters mapped to Italian in alias system
  - Should NOT affect normal English ChatterBox, other engines, or global model selection
- **Impact**: Violation of separation of concerns, coupling universal parser to specific engine quirks
- **Solution**: Move Italian prefix logic to ChatterBox adapter, keep character parser engine-agnostic
- **Priority**: Medium (works correctly but violates clean architecture)
- **Files Affected**: 
  - `utils/text/character_parser/` (remove Italian logic)
  - `engines/adapters/chatterbox_adapter.py` (add Italian logic)

## Cache System Issues

### Problem: Engine-Specific Cache Keys
- Each engine uses different cache identifiers
- Cache collision was discovered between ChatterBox engines
- No standardized cache invalidation

**Impact**: Cross-engine cache pollution, missed cache opportunities  
**Solution**: Implement unified cache system with proper namespacing

## Base Class Architecture Issues

### Problem: Inconsistent Base Class Usage
- Some engines properly inherit from `BaseTTSNode`
- Some engines duplicate base class functionality
- Method signatures vary between engines for same functionality

**Impact**: Code duplication, inconsistent interfaces
**Solution**: Enforce proper base class inheritance and standardize method signatures

## Import System Issues

### Problem: Different Import Patterns
- Some use direct imports
- Some use `importlib.util.spec_from_file_location`
- Some use robust_import
- Inconsistent path resolution

**Impact**: Import failures, maintenance overhead
**Solution**: Standardize import patterns project-wide

## Error Handling Inconsistencies

### Problem: Different Error Handling Approaches
- Some engines have comprehensive error handling
- Some engines crash on edge cases
- Different error message formats
- Inconsistent fallback mechanisms

**Impact**: Poor user experience, difficult debugging
**Solution**: Implement standardized error handling framework

## Configuration Management Issues

### Problem: Scattered Configuration Handling
- Engine configs stored differently
- Parameter validation inconsistent
- Default values scattered across files
- No centralized config schema

**Impact**: Bugs from invalid configs, hard to maintain defaults
**Solution**: Implement unified configuration management system

---

## Refactoring Priority

1. **High Priority**: SRT processing standardization (affects user experience directly)
2. **Medium Priority**: Voice reference handling (causes confusion)
3. **Medium Priority**: Cache system unification (performance impact)
4. **Low Priority**: Import system standardization (maintenance burden)

---

*Last Updated: 2025-01-XX*
*Add new issues to this file as they are discovered during development*