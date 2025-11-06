# [ENGINE_NAME] Upstream Update Log

**Engine:** [ENGINE_NAME]
**Official Repository:** [UPSTREAM_URL]
**Bundled Location:** `engines/[engine_folder]/`
**Last Checked:** YYYY-MM-DD
**Last Checked By:** [username]

---

## Check History

| Date | Commits Checked | New Features | Bug Fixes | Breaking Changes | Notes |
|------|-----------------|--------------|-----------|------------------|-------|
| YYYY-MM-DD | N/A | - | - | - | Initial setup |

---

## Latest Upstream Changes (Since Last Check)

### New Features 🆕
- Feature 1
- Feature 2

### Bug Fixes 🐛
- Fix 1
- Fix 2

### Breaking Changes ⚠️
- Change 1
- Change 2

### Performance Improvements ⚡
- Improvement 1

### Documentation Updates 📝
- Update 1

---

## Integration Status

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| Feature/Fix name | `pending`/`integrated`/`skipped` | 🔴/🟡/🟢 | Integration notes |

### Legend
- `pending` - Identified but not yet integrated
- `integrated` - Applied to bundled code
- `skipped` - Reviewed but decided not to integrate
- `wontfix` - Not applicable to our implementation

---

## Integration Notes

### Conflicts with Our Customizations
- [List any conflicts between upstream changes and our custom modifications]

### Testing Required
- [ ] Test case 1
- [ ] Test case 2

### Version Bump Recommendation
- Current bundled version: X.Y.Z
- Latest upstream version: X.Y.Z
- Recommended bump: `patch` / `minor` / `major` / `none`

---

## Known Customizations We Maintain

List any intentional differences from upstream:
- Customization 1
- Customization 2

---

## Next Check

**Scheduled for:** YYYY-MM-DD
**Check interval:** Weekly / Bi-weekly / Monthly

---

## Previous Check Summary

Link to previous check details if available.

---

## Quick Commands

```bash
# Clone/update upstream repository
git clone [UPSTREAM_URL] /tmp/[engine_name]_upstream

# Compare with our bundled version
diff -r /tmp/[engine_name]_upstream/[source_folder] engines/[bundled_folder]/

# Show recent commits from upstream
git log --oneline -20 origin/main
```
