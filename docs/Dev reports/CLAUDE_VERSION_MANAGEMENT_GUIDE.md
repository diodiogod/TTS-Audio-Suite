# Claude Version Management Guide

**This file contains detailed instructions for Claude Code when performing version bumps.**

## Automated Version Bumping Instructions

### Primary Command
**NEW DEFAULT**: `python3 scripts/bump_version_enhanced.py <version> "<commit_desc>" "<changelog_desc>"`

**Auto-increment support**: Use `patch`, `minor`, or `major` instead of version numbers for automatic increment.

### Usage Examples

#### NEW Default Usage (RECOMMENDED)
```bash
# Auto-increment with separate descriptions
python3 scripts/bump_version_enhanced.py patch "Fix character tag removal bug in single character mode" "Fix unrecognized character tags not being removed from TTS output"

python3 scripts/bump_version_enhanced.py minor "Add RVC TTS engine support" "Add new RVC (Real-time Voice Conversion) TTS engine with voice cloning capabilities"

# Manual version with separate descriptions  
python3 scripts/bump_version_enhanced.py 3.4.3 "Fix crash in audio processing" "Fix crashes when processing certain audio file formats"
```

#### Legacy Usage (DEPRECATED)
```bash
python3 scripts/bump_version_enhanced.py patch --legacy "Fix character tag removal bug"
```

#### Multiline Description
```bash
python3 scripts/bump_version_enhanced.py 3.0.2 "Fix missing sounddevice dependency\nAdd comprehensive error handling\nUpdate installation documentation"
```

#### Interactive Mode (for detailed changelog entries)
```bash
python3 scripts/bump_version_enhanced.py 3.0.2 --interactive
```

#### From File
```bash
python3 scripts/bump_version_enhanced.py 3.0.2 --file changelog.txt
```

#### Dry Run First (RECOMMENDED)
```bash
python3 scripts/bump_version_enhanced.py 3.0.2 "Description" --dry-run
```

### When to Bump Versions
- **Patch (x.x.X)**: Bug fixes, dependency updates, small tweaks
- **Minor (x.X.x)**: New features, significant enhancements  
- **Major (X.x.x)**: Breaking changes, major architecture changes

### Files Auto-Updated by Script
- README.md (title version)
- nodes.py (VERSION constant)
- pyproject.toml (version field)
- srt/__init__.py (__version__)
- core/__init__.py (__version__)
- CHANGELOG.md (new entry with date)

### Script Features
- **Automatic categorization**: Sorts changelog entries into Added/Fixed/Changed/Removed
- **Multiline commit messages**: Properly formatted for git
- **Rollback protection**: Restores files if anything fails
- **Version validation**: Ensures semantic versioning format
- **Backup system**: Creates backup before making changes

### Changelog Entry Categories (IMPROVED v3.4.0+)
The script automatically categorizes based on enhanced keywords and context:
- **Added**: add, new, implement, feature, create, introduce, support, language switching, syntax, bracket, character, integration
- **Fixed**: fix, bug, error, issue, resolve, correct, patch
- **Changed**: update, enhance, improve, change, modify, optimize, performance, smart, efficient, reduced, eliminated, loading
- **Removed**: remove, delete, deprecate, drop

**Smart Context Detection**:
- Recognizes emoji section headers (🌍, 📋, 🚀, 🔧, etc.)
- Uses section context for ambiguous items
- Defaults to "Added" for feature releases instead of "Fixed"
- Skips main title line to avoid categorization errors

### Example Multiline Changelog Output
```markdown
## [3.0.2] - 2025-07-14

### Fixed
- Fix missing sounddevice dependency
- Add comprehensive error handling

### Changed
- Update installation documentation
- Simplify installation process by including all dependencies by default
```

### Fallback Process
If the automated script fails, refer to: `docs/Dev reports/VERSION_UPDATE_GUIDE.md`

### Important Notes
- **Never manually update version files** - the script handles all 6 files automatically
- **Script auto-commits** with proper multiline message format
- **Always dry run first** to preview changes
- **Use interactive mode** for complex changelog entries
- **The script validates** version format and provides backup/rollback
- **Commit messages**: Keep them clean and descriptive. NEVER add Claude co-author credits or "Generated with Claude Code" footers

### CRITICAL: Fix Poor Changelog Generation

**Problem**: Script generates detailed commit messages but poorly categorized changelog entries.

**Root Cause**: The categorization algorithm has flawed keyword matching that can misclassify bug fixes as "Added" items.

**Solutions**:

1. **Start descriptions with clear action words** for better categorization:
```bash
# Good - starts with "Fix" for proper categorization
python3 scripts/bump_version_enhanced.py patch "Fix character tag removal bug in single character mode\nRoot cause: TTS nodes bypassed character parser\nResult: Unrecognized character tags are now properly removed"

# Bad - ambiguous wording leads to miscategorization  
python3 scripts/bump_version_enhanced.py patch "Character tag removal bug in single character mode\nRoot cause: TTS nodes bypassed character parser\nResult: Character tags are now properly removed"
```

2. **Use separate commit and changelog descriptions** for precise control:
```bash
python3 scripts/bump_version_enhanced.py patch --commit "Fix character tag removal bug" --changelog "Fix character tags not being removed in single character mode"
```

**For complex changes, use interactive mode**:
```bash
python3 scripts/bump_version_enhanced.py 3.2.8 --interactive
```

This feeds the script proper technical details that get automatically categorized into good changelog entries.

### IMPORTANT: v3.4.0 Script Improvements

**Fixed Issues**: Enhanced multiline parsing with better keyword matching and context detection.

**Always verify changelog output** - if categorization is wrong, manually fix CHANGELOG.md before pushing.