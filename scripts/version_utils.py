#!/usr/bin/env python3
"""
Version Management Utilities for ComfyUI ChatterBox Voice
Provides centralized version reading/writing functionality
"""

import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class VersionManager:
    """Manages version updates across multiple files"""
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.version_files = {
            'README.md': {
                'pattern': r'# ComfyUI ChatterBox SRT Voice \(diogod\) v(\d+\.\d+\.\d+)',
                'template': '# ComfyUI ChatterBox SRT Voice (diogod) v{version}'
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
            with open(nodes_file, 'r') as f:
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
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Find and replace version
            new_line = config['template'].format(version=version)
            updated_content = re.sub(config['pattern'], new_line, content)
            
            if updated_content == content:
                print(f"Warning: No version found in {relative_path}")
                return False
            
            with open(file_path, 'w') as f:
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
    
    def add_changelog_entry(self, version: str, description: str, details: List[str] = None, simple_mode: bool = False) -> bool:
        """Add entry to CHANGELOG.md with support for multiline descriptions"""
        try:
            changelog_path = os.path.join(self.project_root, 'CHANGELOG.md')
            
            with open(changelog_path, 'r') as f:
                content = f.read()
            
            # Generate changelog entry
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Simple mode - use exact text without categorization
            if simple_mode:
                new_entry = f"""## [{version}] - {today}

### Fixed

- {description}
"""
            # Parse description for structured changelog
            elif '\n' in description or (details and len(details) > 0):
                # Multiline description - create detailed changelog
                lines = description.split('\n') if '\n' in description else [description]
                if details:
                    lines.extend(details)
                
                # Group by change type
                added_items = []
                fixed_items = []
                changed_items = []
                removed_items = []
                
                # Track if we're in a specific section (🌍, 📋, 🚀, 🔧, etc.)
                current_section = "added"  # Default to added for new features
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Skip the main title (first non-empty line)
                    if i == 0 and not line.startswith('-') and not line.startswith('•'):
                        continue
                    
                    # Remove leading bullet points or dashes
                    clean_line = re.sub(r'^[-*•]\s*', '', line)
                    
                    # Check for section headers (emoji-based sections)
                    if any(emoji in line for emoji in ['🌍', '📋', '🚀', '⚡', '🎯']):
                        if any(word in line.lower() for word in ['new', 'feature', 'added', 'support', 'language']):
                            current_section = "added"
                        elif any(word in line.lower() for word in ['performance', 'optimization', 'smart', 'improve']):
                            current_section = "changed"
                        elif any(word in line.lower() for word in ['technical', 'architecture', 'engine']):
                            current_section = "changed"
                        continue
                    
                    # Enhanced categorization based on keywords and context
                    line_lower = clean_line.lower()
                    
                    # More comprehensive keyword matching for Fixed (check first for bug fixes)
                    if any(word in line_lower for word in [
                        'fix', 'bug', 'error', 'issue', 'resolve', 'correct', 'patch', 
                        'crash', 'problem', 'fail', 'broken'
                    ]):
                        fixed_items.append(clean_line)
                    # More comprehensive keyword matching for Added
                    elif line_lower.startswith('add') or any(word in line_lower for word in [
                        'new', 'implement', 'feature', 'create', 'introduce', 'support',
                        'language switching', 'syntax', 'integration'
                    ]):
                        added_items.append(clean_line)
                    # More comprehensive keyword matching for Changed
                    elif any(word in line_lower for word in [
                        'update', 'enhance', 'improve', 'change', 'modify', 'optimize', 'performance',
                        'smart', 'efficient', 'reduced', 'eliminated', 'loading'
                    ]):
                        changed_items.append(clean_line)
                    # More comprehensive keyword matching for Removed
                    elif any(word in line_lower for word in [
                        'remove', 'delete', 'deprecate', 'drop'
                    ]):
                        removed_items.append(clean_line)
                    else:
                        # Use current section context instead of defaulting to Fixed
                        if current_section == "added":
                            added_items.append(clean_line)
                        elif current_section == "changed":
                            changed_items.append(clean_line)
                        else:
                            # Only default to fixed if it's clearly not a feature
                            if any(word in line_lower for word in ['processing', 'group', 'model', 'cache']):
                                changed_items.append(clean_line)
                            else:
                                added_items.append(clean_line)  # Default to added for new features
                
                # Build changelog entry
                entry_parts = [f"## [{version}] - {today}", ""]
                
                if added_items:
                    entry_parts.append("### Added")
                    entry_parts.append("")
                    for item in added_items:
                        entry_parts.append(f"- {item}")
                    entry_parts.append("")
                
                if fixed_items:
                    entry_parts.append("### Fixed")
                    entry_parts.append("")
                    for item in fixed_items:
                        entry_parts.append(f"- {item}")
                    entry_parts.append("")
                
                if changed_items:
                    entry_parts.append("### Changed")
                    entry_parts.append("")
                    for item in changed_items:
                        entry_parts.append(f"- {item}")
                    entry_parts.append("")
                
                if removed_items:
                    entry_parts.append("### Removed")
                    entry_parts.append("")
                    for item in removed_items:
                        entry_parts.append(f"- {item}")
                    entry_parts.append("")
                
                new_entry = "\n".join(entry_parts)
            else:
                # Single line description - use simple format
                change_type = "### Fixed"
                if any(word in description.lower() for word in ['add', 'new', 'implement', 'feature']):
                    change_type = "### Added"
                elif any(word in description.lower() for word in ['update', 'enhance', 'improve', 'change']):
                    change_type = "### Changed"
                elif any(word in description.lower() for word in ['remove', 'delete', 'deprecate']):
                    change_type = "### Removed"
                
                new_entry = f"""## [{version}] - {today}

{change_type}

- {description}

"""
            
            # Insert after the header (find first ## line)
            lines = content.split('\n')
            insert_index = 0
            for i, line in enumerate(lines):
                if line.startswith('## [') and ']' in line:
                    insert_index = i
                    break
            
            lines.insert(insert_index, new_entry.rstrip())
            
            with open(changelog_path, 'w') as f:
                f.write('\n'.join(lines))
            
            print(f"✓ Added changelog entry for v{version}")
            return True
            
        except Exception as e:
            print(f"Error updating changelog: {e}")
            return False
    
    def backup_files(self) -> Dict[str, str]:
        """Create backup of all version files"""
        backups = {}
        for file_path in self.version_files.keys():
            full_path = os.path.join(self.project_root, file_path)
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    backups[file_path] = f.read()
        return backups
    
    def restore_files(self, backups: Dict[str, str]) -> bool:
        """Restore files from backup"""
        try:
            for file_path, content in backups.items():
                full_path = os.path.join(self.project_root, file_path)
                with open(full_path, 'w') as f:
                    f.write(content)
            return True
        except Exception as e:
            print(f"Error restoring files: {e}")
            return False