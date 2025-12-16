#!/usr/bin/env python3
"""
Fix relative imports to prevent shadowing by other custom nodes.

Issue #191: Other custom nodes with utils.py files shadow our utils/ directory.
Solution: Convert implicit relative imports to explicit relative imports using dots.

Example conversions:
  nodes/engines/file.py:     from utils.X import Y  →  from ...utils.X import Y
  engines/adapters/file.py:  from utils.X import Y  →  from ...utils.X import Y
  utils/models/file.py:      from utils.X import Y  →  from ..X import Y
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict

# Directories to exclude from conversion
EXCLUDE_DIRS = {
    "engines/step_audio_editx/step_audio_editx_impl",
    "engines/rvc/impl/lib",
    "engines/higgs_audio/boson_multimodal",
    "engines/chatterbox/models",
    "engines/chatterbox_official_23lang/models",
    "IgnoredForGitHubDocs",
    "engines/index_tts/indextts",
}

class RelativeImportFixer:
    def __init__(self, project_root: Path, dry_run: bool = True):
        self.project_root = project_root
        self.dry_run = dry_run
        self.changes: List[Tuple[str, int, str, str]] = []

        # Regex patterns for utils imports
        self.from_import_pattern = re.compile(r'^(\s*)from utils\.([a-zA-Z0-9_.]+) import (.+)$')
        self.direct_import_pattern = re.compile(r'^(\s*)import utils\.([a-zA-Z0-9_.]+)(.*)$')

    def calculate_relative_depth(self, file_path: Path) -> int:
        """Calculate how many levels up to go to reach project root"""
        relative_path = file_path.relative_to(self.project_root)

        # Get the parent directory path
        parent = relative_path.parent

        # Count directory depth (how many folders deep from root)
        # nodes/engines/file.py -> depth 2 -> need ...utils (3 dots)
        # engines/adapters/file.py -> depth 2 -> need ...utils (3 dots)
        # utils/models/file.py -> depth 2 -> need ..X (2 dots to go to utils/)
        # nodes.py (root) -> depth 0 -> need .utils (1 dot)

        depth = len(parent.parts)

        # Special case: files in utils/ directory
        if parent.parts and parent.parts[0] == 'utils':
            # From utils/models/file.py importing utils.audio -> ..audio
            # We're in utils/, so we just need .. to stay in utils/
            return 1

        # For all other files, we need to go up to root, then into utils/
        # depth 0 (root): .utils
        # depth 1 (nodes/): ..utils
        # depth 2 (nodes/engines/): ...utils
        return depth + 1

    def convert_line(self, line: str, depth: int, is_utils_file: bool) -> Tuple[str, bool]:
        """Convert a single import line"""
        # Handle "from utils.X import Y"
        match = self.from_import_pattern.match(line)
        if match:
            indent, module_path, imports = match.groups()

            if is_utils_file:
                # From utils/models/file.py: from utils.audio import X -> from ..audio import X
                dots = '.' * depth
                new_line = f"{indent}from {dots}{module_path} import {imports}\n"
            else:
                # From other files: need to reach utils/ from root
                dots = '.' * depth
                new_line = f"{indent}from {dots}utils.{module_path} import {imports}\n"

            return new_line, True

        # Handle "import utils.X"
        match = self.direct_import_pattern.match(line)
        if match:
            indent, module_path, rest = match.groups()

            # Cannot easily convert "import utils.X" to relative form
            # Would need to be "from .utils import X" which changes usage
            # Skip these for now - they're rare
            return line, False

        return line, False

    def should_skip_directory(self, file_path: Path) -> bool:
        """Check if file is in an excluded directory"""
        relative_path = file_path.relative_to(self.project_root)
        path_str = str(relative_path.parent).replace('\\', '/')

        for exclude_dir in EXCLUDE_DIRS:
            if path_str.startswith(exclude_dir) or exclude_dir in path_str:
                return True
        return False

    def convert_file(self, file_path: Path) -> int:
        """Convert all imports in a single file. Returns number of changes."""
        if self.should_skip_directory(file_path):
            return 0

        relative_path = file_path.relative_to(self.project_root)

        # Check if file is in utils/ directory
        is_utils_file = relative_path.parts and relative_path.parts[0] == 'utils'

        # Calculate relative depth
        depth = self.calculate_relative_depth(file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
            return 0

        new_lines = []
        changes_in_file = 0

        for line_num, line in enumerate(lines, 1):
            new_line, changed = self.convert_line(line, depth, is_utils_file)
            new_lines.append(new_line)

            if changed:
                changes_in_file += 1
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
            dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git', '.venv', 'venv')]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    python_files.append(file_path)

        return python_files

    def run(self):
        """Run the conversion"""
        print(f"{'='*80}")
        print(f"Fixing relative imports to prevent utils shadowing")
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

            # Print first 20 files as examples
            for idx, file_path in enumerate(sorted(changes_by_file.keys())):
                if idx >= 20:
                    print(f"\n... and {len(changes_by_file) - 20} more files")
                    break

                print(f"\n📄 {file_path}")
                for line_num, old_line, new_line in changes_by_file[file_path][:5]:
                    print(f"   Line {line_num}:")
                    print(f"   - {old_line}")
                    print(f"   + {new_line}")
                if len(changes_by_file[file_path]) > 5:
                    print(f"   ... and {len(changes_by_file[file_path]) - 5} more changes in this file")

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
    import sys

    parser = argparse.ArgumentParser(description="Fix relative imports to prevent shadowing")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry-run)")
    args = parser.parse_args()

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Run conversion
    fixer = RelativeImportFixer(project_root, dry_run=not args.apply)
    files_changed, total_changes = fixer.run()

    sys.exit(0 if files_changed >= 0 else 1)


if __name__ == "__main__":
    main()
