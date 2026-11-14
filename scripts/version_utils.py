#!/usr/bin/env python3
"""
Version Management Utilities for ComfyUI ChatterBox Voice
Provides centralized version reading/writing functionality
"""

import os
import re
from datetime import datetime
from typing import Dict, Optional

class VersionManager:
    """Manages version updates across multiple files"""
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.version_files = {
            'README.md': {
                'pattern': r'# TTS Audio Suite v(\d+\.\d+\.\d+)',
                'template': '# TTS Audio Suite v{version}'
            },
            'nodes.py': {
                'pattern': r'VERSION = "(\d+\.\d+\.\d+)"',
                'template': 'VERSION = "{version}"'
            },
            'pyproject.toml': {
                'pattern': r'version = "(\d+\.\d+\.\d+)"',
                'template': 'version = "{version}"'
            }
        }
    
    def validate_version(self, version: str) -> bool:
        """Validate semantic version format"""
        pattern = r'^\d+\.\d+\.\d+$'
        return bool(re.match(pattern, version))
    
    def get_current_version(self) -> Optional[str]:
        """Get current version from nodes.py"""
        try:
            nodes_file = os.path.join(self.project_root, 'nodes.py')
            with open(nodes_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            match = re.search(self.version_files['nodes.py']['pattern'], content)
            return match.group(1) if match else None
        except Exception as e:
            print(f"Error reading current version: {e}")
            return None
    
    def update_version_in_file(self, file_path: str, version: str) -> bool:
        """Update version in a specific file"""
        try:
            relative_path = os.path.relpath(file_path, self.project_root)
            if relative_path not in self.version_files:
                print(f"Warning: {relative_path} not in version files list")
                return False
            
            config = self.version_files[relative_path]
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find and replace version
            new_line = config['template'].format(version=version)
            updated_content = re.sub(config['pattern'], new_line, content)
            
            if updated_content == content:
                print(f"Warning: No version found in {relative_path}")
                return False
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"✓ Updated {relative_path}")
            return True
            
        except Exception as e:
            print(f"Error updating {file_path}: {e}")
            return False
    
    def update_all_versions(self, version: str) -> bool:
        """Update version in all files"""
        if not self.validate_version(version):
            print(f"Error: Invalid version format '{version}'. Use semantic versioning (e.g., 3.0.1)")
            return False
        
        success = True
        for file_path in self.version_files.keys():
            full_path = os.path.join(self.project_root, file_path)
            if os.path.exists(full_path):
                if not self.update_version_in_file(full_path, version):
                    success = False
            else:
                print(f"Warning: {file_path} not found")
                success = False
        
        return success
    
    def _generate_changelog_content(self, version: str, description: str) -> str:
        """Wrap authored Markdown without classifying or rewriting its contents."""
        body = description.strip()
        categories = {
            "### Added", "### Changed", "### Deprecated",
            "### Removed", "### Fixed", "### Security",
        }
        lines = body.splitlines()
        if not lines or lines[0] not in categories:
            raise ValueError(
                "Provide an authored changelog body with explicit category headings, "
                "for example: '### Fixed\n\n- Describe the user-facing fix'. "
                "The script no longer categorizes descriptions by keywords."
            )
        for line in lines:
            if re.match(r"^#{1,3}(?:\s|$)", line) and line not in categories:
                raise ValueError(
                    f"Unsupported changelog heading: {line!r}. Use Added, Changed, "
                    "Deprecated, Removed, Fixed, or Security at level 3; "
                    "the script supplies the version/date header."
                )
        sections = re.split(r"(?m)^### .*$", body)[1:]
        if any(not section.strip() for section in sections):
            raise ValueError("Omit empty changelog sections.")

        today = datetime.now().strftime("%Y-%m-%d")
        return f"## [{version}] - {today}\n\n{body}\n\n"

    def preview_changelog_entry(self, version: str, description: str) -> str:
        """Preview changelog entry content without writing to file (for dry-run)"""
        return self._generate_changelog_content(version, description)
    
    def add_changelog_entry(self, version: str, description: str) -> bool:
        """Add entry to CHANGELOG.md with proper formatting"""
        try:
            changelog_path = os.path.join(self.project_root, 'CHANGELOG.md')
            
            with open(changelog_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Generate changelog entry content using extracted method
            try:
                new_entry = self._generate_changelog_content(version, description)
            except ValueError as e:
                print(str(e))
                return False
            
            # Insert after the header (find first ## line)
            lines = content.splitlines()
            insert_index = len(lines)
            for i, line in enumerate(lines):
                if line.startswith('## [') and ']' in line:
                    insert_index = i
                    break
            
            # Insert the new entry with proper spacing
            if insert_index > 0 and lines[insert_index - 1] != "":
                lines.insert(insert_index, "")
                insert_index += 1
            lines.insert(insert_index, new_entry.rstrip() + "\n")
            
            with open(changelog_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines).rstrip() + '\n')
            
            print(f"✓ Added changelog entry for v{version}")
            return True
            
        except Exception as e:
            print(f"Error updating changelog: {e}")
            return False
    
    def backup_files(self) -> Dict[str, str]:
        """Create backup of all version files"""
        backups = {}
        for file_path in list(self.version_files) + ["CHANGELOG.md"]:
            full_path = os.path.join(self.project_root, file_path)
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    backups[file_path] = f.read()
        return backups
    
    def restore_files(self, backups: Dict[str, str]) -> bool:
        """Restore files from backup"""
        try:
            for file_path, content in backups.items():
                full_path = os.path.join(self.project_root, file_path)
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            return True
        except Exception as e:
            print(f"Error restoring files: {e}")
            return False
