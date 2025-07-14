# Version Update Guide

This guide explains how to update the version number across the ChatterBox Voice Extension.

## Files to Update

When updating the version, you need to update these **6 files**:

### 1. README.md
- Update the title: `# ComfyUI_ChatterBox_Voice v2.0.0`

### 2. CHANGELOG.md
- Add new version entry at the top with date and changes

### 3. nodes.py
- Update `VERSION = "2.0.0"`
- Set `IS_DEV = False` for release builds, `True` for development

### 4. pyproject.toml
- Update `version = "2.0.0"` in the [project] section

### 5. srt/__init__.py
- Update `__version__ = "2.0.0"`

### 6. core/__init__.py
- Update `__version__ = "2.0.0"`

## Automated Version Update Process (RECOMMENDED)

### Using the Automation Script

**Always use the automated script for version updates:**

```bash
python3 scripts/bump_version.py <version> "<description>"
```

**Examples:**
```bash
# Bug fix
python3 scripts/bump_version.py 3.0.2 "Fix missing sounddevice dependency"

# New feature
python3 scripts/bump_version.py 3.1.0 "Add audio waveform visualization"

# Breaking change
python3 scripts/bump_version.py 4.0.0 "Restructure API for better compatibility"
```

**Options:**
- `--dry-run`: Preview changes without making them
- `--no-commit`: Update files but don't commit to git

### What the Script Does Automatically

1. **Validates version format** (semantic versioning)
2. **Updates all 6 files** with the new version number
3. **Adds changelog entry** with date and description
4. **Commits changes** with proper message format
5. **Provides rollback** if anything fails

## Manual Version Update Process (FALLBACK ONLY)

**⚠️ Only use this if the automated script fails**

1. **Update the 6 files above** with the new version number
2. **Update CHANGELOG.md** with the new version entry and release notes
3. **Set IS_DEV flag appropriately** in nodes.py:
   - `IS_DEV = False` for release builds
   - `IS_DEV = True` for development builds
4. **Test the changes** to ensure everything works
5. **Commit and tag** the release

## Current Version: 3.0.1

Last updated: 2025-07-14