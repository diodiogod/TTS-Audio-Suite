# TTS Audio Suite Version Bump Instructions

Use the script to update version files and insert an authored changelog entry.
The agent writes and categorizes the release notes; the script preserves them.

## Release preparation

- Follow the shared `bump-version` skill: delegate changelog drafting to Luna when available, then review its output against the release changes.
- Cover the relevant changes since the previous release/version-bump commit, not just the latest commit. Use the net user-facing result, without duplicate entries or temporary development breakages.
- Name the affected engine, node, or component. Explain what users experience; keep implementation details in the separate commit description.
- Only release/commit after the user confirms the changes work, following `AGENTS.md`. Never push without an explicit request.
- The script reads the current version automatically. Use `patch`, `minor`, or `major`, or provide an explicit semantic version when needed. Do not manually edit version strings.

## Recommended command: UTF-8 file inputs

Write the developer-focused commit description and reviewed changelog body to separate UTF-8 files outside the repository. For example, use the following temporary file paths:

```powershell
$releaseCommitFile = Join-Path $env:TEMP 'tts-release-commit.txt'
$releaseNotesFile = Join-Path $env:TEMP 'tts-release-notes.md'

& 'J:\stablediffusion1111s2\Data\Packages\ComfyUIPy129\test_env_error\Scripts\python.exe' 'scripts\bump_version_enhanced.py' patch --commit-file $releaseCommitFile --changelog-file $releaseNotesFile --dry-run
```

Inspect the actual changelog preview. Remove `--dry-run` to perform the authorized release. Use `--no-commit` when the release should update files without creating a commit.

On Linux/macOS, use the available Python 3 interpreter:

```bash
python3 scripts/bump_version_enhanced.py patch --commit-file /tmp/tts-release-commit.txt --changelog-file /tmp/tts-release-notes.md --dry-run
```

File inputs preserve literal backslashes, Markdown formatting, section order, paragraphs, and nested bullets. The script only trims surrounding whitespace and supplies the version/date header and entry spacing.

## Changelog body format

Provide the body only, with explicit level-three category headings. For example:

```markdown
### Fixed

- ChatterBox character switching now selects the intended voice in SRT mode.
  - Cached segments follow the selected character and language.
- F5-TTS no longer shows an erroneous missing-model warning when weights are installed.

### Changed

- Vocal Removal explains which audio separation models isolate vocals and instruments.
```

Supported headings are `### Added`, `### Changed`, `### Deprecated`, `### Removed`, `### Fixed`, and `### Security`.

- Choose categories by the user-visible meaning. Removing an erroneous warning can be a fix; an internal code deletion does not necessarily remove a feature.
- Keep related explanations together, with the author's chosen order and grouping.
- Omit empty sections and the `## [version] - date` header. A release with one section and one bullet is valid.
- Unstructured descriptions, unsupported headings, and empty sections are rejected before version files change. There is no keyword categorization fallback.

Positional descriptions remain supported when using actual multiline strings:

```powershell
& 'J:\stablediffusion1111s2\Data\Packages\ComfyUIPy129\test_env_error\Scripts\python.exe' 'scripts\bump_version_enhanced.py' patch "Fix ChatterBox character switching

- Preserve selected character when cached SRT segments are reused" "### Fixed

- ChatterBox SRT generation now uses the selected character voice consistently." --dry-run
```

## Commit description

Write a clear title and normally a multiline developer-focused body. Do not add AI co-author credits or generated-by footers.

Only use `Fixes #X`, `Closes #X`, or `Resolves #X` after the user confirms the fix works. Otherwise use neutral references such as `Address issue #X`.

## Script behavior

- Updates `nodes.py`, `README.md`, and `pyproject.toml`.
- Adds the reviewed Markdown body to `CHANGELOG.md`, with a version/date header.
- Regenerates the engine documentation tables and README tables.
- By default runs `git add -A` and creates a version-bump commit. Inspect the working tree first and resolve unrelated changes before allowing that commit; do not discard or stash someone else's work automatically.
- `--dry-run` previews without changing files or committing; invalid changelog input exits with an error.
- `--no-commit` updates files without staging or committing.
- `--allow-downgrade` explicitly permits a non-increasing version when needed for a revert or correction.

## Version selection

- Patch: fixes and compatible maintenance changes.
- Minor: backward-compatible features.
- Major: breaking changes. Clearly describe user migration steps in the changelog.

## Troubleshooting

- Invalid changelog: provide an authored body starting with a supported `###` category and at least one entry in every section. A free-text title before the first category is not accepted.
- Invalid version: use `patch`, `minor`, `major`, or `X.Y.Z` without a `v` prefix.
- Multiline quoting problems: use both `--commit-file` and `--changelog-file` with UTF-8 files.
- A workflow that still sorts by keywords is outdated. Update the local integration rather than choosing words to influence category matching.
