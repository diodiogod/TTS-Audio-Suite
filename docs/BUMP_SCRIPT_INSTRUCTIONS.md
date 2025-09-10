# Version Bump Instructions for ComfyUI ChatterBox Voice

## Quick Reference for Future Version Bumps

### Recommended Command (Separate Commit & Changelog)

**⚠️ IMPORTANT: Use positional arguments, NOT --commit/--changelog flags**

```bash
python3 scripts/bump_version_enhanced.py <version> "<commit_desc>" "<changelog_desc>"
```

### Examples

#### Multiline Format (Recommended Standard)

```bash
# Patch release (bug fixes) - CORRECT FORMAT
python3 scripts/bump_version_enhanced.py 3.2.9 "Fix character alias resolution

Technical details:
- Fix parser bypassing character tags in single mode
- Improve character name validation logic  
- Add fallback handling for unrecognized tags" "Fix character name handling issues

- Fix character tags not being removed from TTS output
- Improve character name recognition accuracy
- Better error handling for invalid character names"

# Minor release (new features) - CORRECT FORMAT  
python3 scripts/bump_version_enhanced.py 3.3.0 "Add Higgs Audio 2 TTS engine

Implementation details:
- Integrate boson_multimodal voice cloning system
- Add unified adapter for consistent interface
- Implement voice preset management" "Add Higgs Audio 2 TTS engine with voice cloning

- New realistic voice synthesis engine
- Voice cloning from short audio samples
- Multiple built-in voice presets available"

# Major release (breaking changes) - CORRECT FORMAT
python3 scripts/bump_version_enhanced.py 4.0.0 "Complete unified architecture implementation

Breaking changes:
- Migrate all nodes to unified interface pattern
- Consolidate engine adapters and processors
- Remove deprecated standalone variants" "Major architecture upgrade to unified system

- All TTS engines now use consistent interface
- Better performance and memory management  
- Simplified workflows with consolidated nodes
- Breaking: old standalone nodes removed"
```

#### Auto-Increment Examples (Recommended)
```bash
# Auto-increment patch version (4.5.25 → 4.5.26) - CORRECT FORMAT
python3 scripts/bump_version_enhanced.py patch "Fix character parsing issues" "Fix character name handling in TTS generation"

# Auto-increment minor version (4.5.25 → 4.6.0) - CORRECT FORMAT  
python3 scripts/bump_version_enhanced.py minor "Add new TTS engine support" "Add Higgs Audio 2 TTS engine with voice cloning"
```

#### Single-Line Format (Only for Super Minor Changes)
```bash
python3 scripts/bump_version_enhanced.py patch "Fix typo in node tooltip" "Fix typo in audio analyzer tooltip"
```

#### Dry-Run Preview (Test Before Committing)
```bash
python3 scripts/bump_version_enhanced.py patch "Fix preview issues" "Fix preview not reflecting filter parameters" --dry-run
```

#### Auto-Categorization System
**IMPORTANT**: The script automatically sorts each line of multiline changelog entries into categories based on keywords:

**Keywords for each category:**
- **Fixed** section: "fix", "bug", "error", "issue", "resolve", "correct", "patch", "crash", "problem", "broken", "compatibility"
- **Added** section: "add", "new", "implement", "feature", "create", "introduce", "support" (also default for unmatched lines)
- **Changed** section: "improve", "enhance", "optimize", "update", "modify", "better", "performance", "refactor"  
- **Removed** section: "remove", "delete", "deprecate", "drop", "eliminate"

**Writing Tips:**
- **Be intentional with word choice** - start bullet points with appropriate keywords
- **Line order doesn't matter** - script will reorganize lines by category automatically  
- **Each line categorized individually** - mix different change types in one changelog description
- **Commit vs Changelog focus:**
  - **Commit**: Technical implementation details for developers
  - **Changelog**: User-facing benefits and impacts

**Bash Syntax Notes:**
- Multiline strings need proper quoting (opening quote on first line, closing quote on last line)
- Use `\` (backslash) for line continuation in bash commands
- Don't add manual category prefixes like "Fixed:" - script handles categorization automatically!

### Interactive Mode (Recommended for Complex Changes)

```bash
python3 scripts/bump_version_enhanced.py 3.2.9 --interactive
```

### Legacy Mode (Same Description for Both)

```bash
python3 scripts/bump_version_enhanced.py 3.2.9 "Fix bugs and improve stability"
```

### What the Script Does

- Updates version in `nodes.py`, `README.md`, and `pyproject.toml` 
- Updates changelog with user-focused description
- Creates git commit with developer-focused description
- Follows semantic versioning (MAJOR.MINOR.PATCH)
- Supports separate commit/changelog descriptions for better communication

### When to Bump Versions

#### Patch (x.x.X)

- Bug fixes
- Documentation updates
- Performance improvements
- Security patches

#### Minor (x.X.0)

- New features
- New node types
- Enhanced functionality
- Backward-compatible changes

#### Major (X.0.0)

- Breaking changes
- API changes
- Architecture refactoring
- Incompatible updates

### Pre-Bump Checklist

1. ✅ All changes tested and working
2. ✅ User confirms functionality works
3. ✅ Git working directory is clean
4. ✅ All important changes committed

### Post-Bump Actions

- Script handles git commit automatically
- Consider pushing to remote if appropriate
- Update any external documentation
- Notify users of significant changes

### Important Notes

- **Never manually edit version files** - always use the script
- **Only bump when user confirms changes work**
- **Read detailed guide**: `docs/Dev reports/CLAUDE_VERSION_MANAGEMENT_GUIDE.md`
- **Follow project commit policy**: No Claude co-author credits in commits

### Commit vs Changelog Guidelines

#### Commit Description (--commit)

- **Developer diary**: What this specific version bump does
- **Include internal details**: Bug fixes, refactoring, technical changes
- **Development perspective**: "Fixed F5-TTS edit issues after refactoring"
- **Can mention temporary problems**: "Restore functionality broken by restructure"

#### Changelog Description (--changelog)

- **User perspective only**: Write for users upgrading from previous version
- **Use layman terms**: Explain WHAT the feature does, not just technical names
- **Don't document internal fixes**: Skip temporary issues introduced and fixed during development
- **Focus on net result**: What changed for the user, not the development process
- **Example**: If refactoring broke something then fixed it, only mention the refactoring benefit

**🎯 CRITICAL: Always Specify Engine/Component When Applicable**
- **Include engine name in titles**: Users need to know which TTS engine is affected
- **❌ Bad**: "Fix SRT processing issues" 
- **✅ Good**: "Fix ChatterBox SRT processing issues"
- **❌ Bad**: "Fix voice cloning errors"
- **✅ Good**: "Fix Higgs Audio voice cloning errors"
- **❌ Bad**: "Improve audio quality" 
- **✅ Good**: "Improve F5-TTS audio quality"

**Engine-Specific Title Examples:**
- "Fix Higgs Audio compatibility with transformers 4.46.3+"
- "Add VibeVoice multi-speaker support"  
- "Fix ChatterBox character switching in SRT mode"
- "Improve F5-TTS speech editing accuracy"
- "Fix RVC voice conversion model loading"
- "Add Audio Wave Analyzer timing extraction"

**User-Friendly Language Rules:**
- ❌ "Add support for MelBandRoFormer models" 
- ✅ "Add support for advanced audio separation models (vocal/instrumental isolation)"
- ❌ "Fix VibeVoice parameter propagation to cached engine instances"
- ✅ "Fix VibeVoice voice settings not applying correctly in some cases"
- ❌ "Resolve ChatterboxVC loading with .safetensors format"
- ✅ "Fix voice conversion failing to load newer model formats"

**Always explain obscure terms:**
- Model names → What they do (e.g., "audio separation", "voice conversion", "text-to-speech")
- Technical errors → User impact (e.g., "crashes when...", "fails to generate audio")
- Internal components → User-facing features (e.g., "character voice switching", "long audio generation")

#### Key Principle

**Commit = Development diary, Changelog = User release notes**

## Troubleshooting

### Common Issues

**"Git working directory is not clean"**
```bash
# Check what files are modified
git status

# Add/commit changes first, then run version bump
git add .
git commit -m "Prepare for version bump"
```

**"Error: Invalid version format"**
- Use semantic versioning: `4.5.25` (not `v4.5.25` or `4.5`)
- Or use auto-increment: `patch`, `minor`, `major`

**Bash syntax errors with multiline**
- Make sure opening quote is on same line as `--commit` or `--changelog`  
- Make sure closing quote is on its own line
- Use `\` for line continuation

**Want to see what will happen before committing?**
```bash
# Add --dry-run to preview changelog categorization
python3 scripts/bump_version_enhanced.py patch "description" "changelog" --dry-run
```