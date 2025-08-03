# Version Bump Instructions for ComfyUI ChatterBox Voice

## Quick Reference for Future Version Bumps

### Command
```bash
python3 scripts/bump_version_enhanced.py <version> "<description>"
```

### Examples
```bash
# Patch release (bug fixes)
python3 scripts/bump_version_enhanced.py 3.2.9 "Fix character alias resolution and F5-TTS imports"

# Minor release (new features)
python3 scripts/bump_version_enhanced.py 3.3.0 "Add character support for F5-TTS generation"

# Major release (breaking changes)
python3 scripts/bump_version_enhanced.py 4.0.0 "Complete architecture refactoring with new folder structure"
```

### What the Script Does
- Updates version in `__init__.py`
- Updates version in `pyproject.toml` 
- Updates changelog with new version entry
- Creates git commit with proper message format
- Follows semantic versioning (MAJOR.MINOR.PATCH)

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