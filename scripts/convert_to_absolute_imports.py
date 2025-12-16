#!/usr/bin/env python3
"""
Convert relative utils imports to absolute imports to prevent shadowing conflicts.

Issue #191: Other custom nodes with utils.py files shadow our utils/ directory.
Solution: Convert all "from utils.X import Y" to absolute imports.

This script:
1. Scans all Python files in the project
2. Finds lines with "from utils." imports
3. Converts them to absolute imports using the package name
4. Reports all changes for verification before applying
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# Package name for absolute imports
PACKAGE_NAME = "ComfyUI_TTS_Audio_Suite"

# Directories to exclude from conversion
EXCLUDE_DIRS = {
    "engines/step_audio_editx/step_audio_editx_impl",  # Third-party implementation
    "engines/rvc/impl/lib",  # Third-party RVC implementation
    "engines/higgs_audio/boson_multimodal",  # Third-party implementation
    "engines/chatterbox/models",  # Model architecture (shouldn't import utils)
    "engines/chatterbox_official_23lang/models",  # Model architecture
    "IgnoredForGitHubDocs",  # Reference files - don't modify
    "engines/index_tts/indextts",  # Third-party IndexTTS implementation
}

class ImportConverter:
    def __init__(self, project_root: Path, dry_run: bool = True):
        self.project_root = project_root
        self.dry_run = dry_run
        self.changes: List[Tuple[str, int, str, str]] = []

        # Regex patterns
        # Match: from utils.X import Y
        # Match: from utils.X.Y import Z
        self.import_pattern = re.compile(r'^(\s*)from utils\.([a-zA-Z0-9_.]+) import (.+)$')

        # Match: import utils.X
        self.direct_import_pattern = re.compile(r'^(\s*)import utils\.([a-zA-Z0-9_.]+)(.*)$')

    def should_skip_directory(self, file_path: Path) -> bool:
        """Check if file is in an excluded directory"""
        relative_path = file_path.relative_to(self.project_root)
        path_str = str(relative_path.parent).replace('\\', '/')

        for exclude_dir in EXCLUDE_DIRS:
            if path_str.startswith(exclude_dir) or exclude_dir in path_str:
                return True
        return False

    def convert_line(self, line: str) -> Tuple[str, bool]:
        """Convert a single line from relative to absolute import"""
        # Check for "from utils.X import Y" pattern
        match = self.import_pattern.match(line)
        if match:
            indent, module_path, imports = match.groups()
            new_line = f"{indent}from {PACKAGE_NAME}.utils.{module_path} import {imports}\n"
            return new_line, True

        # Check for "import utils.X" pattern
        match = self.direct_import_pattern.match(line)
        if match:
            indent, module_path, rest = match.groups()
            new_line = f"{indent}import {PACKAGE_NAME}.utils.{module_path}{rest}\n"
            return new_line, True

        return line, False

    def convert_file(self, file_path: Path) -> int:
        """Convert all imports in a single file. Returns number of changes."""
        if self.should_skip_directory(file_path):
            return 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
            return 0

        new_lines = []
        changes_in_file = 0

        for line_num, line in enumerate(lines, 1):
            new_line, changed = self.convert_line(line)
            new_lines.append(new_line)

            if changed:
                changes_in_file += 1
                relative_path = file_path.relative_to(self.project_root)
                self.changes.append((str(relative_path), line_num, line.strip(), new_line.strip()))

        # Write changes if not dry run
        if changes_in_file > 0 and not self.dry_run:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
            except Exception as e:
                print(f"⚠️  Error writing {file_path}: {e}")
                return 0

        return changes_in_file

    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project"""
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            # Skip __pycache__ and .git directories
            dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', '.venv', 'venv')]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    python_files.append(file_path)

        return python_files

    def run(self):
        """Run the conversion"""
        print(f"{'='*80}")
        print(f"Converting relative utils imports to absolute imports")
        print(f"Package name: {PACKAGE_NAME}")
        print(f"Mode: {'DRY RUN (no files changed)' if self.dry_run else 'LIVE (files will be modified)'}")
        print(f"{'='*80}\n")

        python_files = self.find_python_files()
        print(f"Found {len(python_files)} Python files\n")

        total_changes = 0
        files_changed = 0

        for file_path in python_files:
            changes = self.convert_file(file_path)
            if changes > 0:
                total_changes += changes
                files_changed += 1

        # Report results
        print(f"\n{'='*80}")
        print(f"CONVERSION SUMMARY")
        print(f"{'='*80}")
        print(f"Files scanned: {len(python_files)}")
        print(f"Files with changes: {files_changed}")
        print(f"Total import lines changed: {total_changes}")
        print(f"{'='*80}\n")

        if self.changes:
            print("DETAILED CHANGES:\n")

            # Group by file
            changes_by_file: Dict[str, List[Tuple[int, str, str]]] = {}
            for file_path, line_num, old_line, new_line in self.changes:
                if file_path not in changes_by_file:
                    changes_by_file[file_path] = []
                changes_by_file[file_path].append((line_num, old_line, new_line))

            # Print grouped changes
            for file_path in sorted(changes_by_file.keys()):
                print(f"\n📄 {file_path}")
                for line_num, old_line, new_line in changes_by_file[file_path]:
                    print(f"   Line {line_num}:")
                    print(f"   - {old_line}")
                    print(f"   + {new_line}")

        if self.dry_run:
            print(f"\n{'='*80}")
            print("This was a DRY RUN - no files were modified")
            print("To apply these changes, run with --apply flag")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print("✅ Changes have been applied to all files")
            print(f"{'='*80}\n")

        return files_changed, total_changes


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert relative utils imports to absolute imports")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")
    args = parser.parse_args()

    # Get project root (parent of scripts directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Run conversion
    converter = ImportConverter(project_root, dry_run=not args.apply)
    files_changed, total_changes = converter.run()

    # Exit with status code
    if files_changed > 0:
        sys.exit(0)  # Success, changes found
    else:
        sys.exit(1)  # No changes needed


if __name__ == "__main__":
    main()
