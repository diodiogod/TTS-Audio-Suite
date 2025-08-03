# Version Bump Instructions for ComfyUI ChatterBox Voice

## Quick Reference for Future Version Bumps

### Recommended Command (Separate Commit & Changelog)
```bash
python3 scripts/bump_version_enhanced.py <version> --commit "<commit_desc>" --changelog "<changelog_desc>"
```

### Examples
```bash
# Patch release (bug fixes)
python3 scripts/bump_version_enhanced.py 3.2.9 \
  --commit "Fix character alias resolution and F5-TTS import issues" \
  --changelog "Bug fixes and stability improvements"

# Minor release (new features)  
python3 scripts/bump_version_enhanced.py 3.3.0 \
  --commit "Add character support system for F5-TTS generation" \
  --changelog "Add character support for F5-TTS generation"

# Major release (breaking changes)
python3 scripts/bump_version_enhanced.py 4.0.0 \
  --commit "Complete architecture refactoring with new modular structure" \
  --changelog "Project restructure for better maintainability"
```

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
- **Don't document internal fixes**: Skip temporary issues introduced and fixed during development
- **Focus on net result**: What changed for the user, not the development process
- **Example**: If refactoring broke something then fixed it, only mention the refactoring benefit

#### Key Principle
**Commit = Development diary, Changelog = User release notes**