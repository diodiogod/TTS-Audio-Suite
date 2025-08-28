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
    
    def _categorize_simple_description(self, description: str) -> str:
        """Categorize a simple description for changelog (legacy fallback)"""
        import re
        desc_lower = description.lower()
        
        # Helper function to check for whole words only (avoids "add" in "advanced")
        def has_whole_word(text, words):
            for word in words:
                # Use word boundary \b to match whole words only
                if re.search(rf'\b{re.escape(word)}\b', text):
                    return True
            return False
        
        # Check Fixed FIRST (most important - fixes take priority)
        if has_whole_word(desc_lower, [
            'fix', 'bug', 'error', 'issue', 'resolve', 'correct', 'patch',
            'crash', 'problem', 'fail', 'broken', 'dependency', 'missing'
        ]):
            return "### Fixed"
        # Check Added second (for new features/documentation)
        elif has_whole_word(desc_lower, [
            'add', 'new', 'implement', 'feature', 'create', 'introduce', 'support',
            'documentation', 'readme', 'guide', 'section', 'workflow', 'example'
        ]):
            return "### Added"
        # Check Changed third (for improvements/UI)
        elif has_whole_word(desc_lower, [
            'update', 'enhance', 'improve', 'change', 'modify', 'optimize',
            'ui', 'interface', 'slider', 'tooltip', 'dropdown', 'better', 'cleaner'
        ]):
            return "### Changed"
        # Check Removed last  
        elif has_whole_word(desc_lower, [
            'remove', 'delete', 'deprecate', 'drop', 'eliminate'
        ]):
            return "### Removed"
        else:
            return "### Added"  # Default to Added for new features
    
    def _generate_changelog_content(self, version: str, description: str, details: List[str] = None, simple_mode: bool = False) -> str:
        """Generate changelog entry content"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Simple mode - use exact text with proper formatting
        if simple_mode:
            # Smart categorization for simple mode
            change_type = self._categorize_simple_description(description)
            
            # Split multi-sentence descriptions into proper bullet points
            if '. ' in description and len(description) > 100:
                # Break long descriptions into bullet points
                sentences = [s.strip() for s in description.split('. ') if s.strip()]
                items = []
                for sentence in sentences:
                    if not sentence.endswith('.') and sentence != sentences[-1]:
                        sentence += '.'
                    items.append(f"- {sentence}")
                item_text = '\n'.join(items)
            else:
                # Single item
                item_text = f"- {description}"
            
            return f"""## [{version}] - {today}

{change_type}

{item_text}
"""
        # Parse description for structured changelog
        elif '\n' in description or (details and len(details) > 0):
            # Multiline description - create detailed changelog with better parsing
            lines = description.split('\n') if '\n' in description else [description]
            if details:
                lines.extend(details)
            
            # Group by change type
            added_items = []
            fixed_items = []
            changed_items = []
            removed_items = []
            
            # Process each line with better formatting
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Clean up the line - remove existing bullet points for consistent formatting
                clean_line = re.sub(r'^[-*•]\s*', '', line)
                if not clean_line:
                    continue
                
                # Skip section headers or emoji-only lines
                if re.match(r'^[🌍📋🚀⚡🎯🔧🎭🎙️📺🎵🏗️]+\s*[\w\s]*:?\s*$', line):
                    continue
                
                # Enhanced categorization based on keywords (whole words only)
                line_lower = clean_line.lower()
                
                # Helper function for whole word matching in multiline context
                def has_whole_word_in_line(text, words):
                    import re
                    for word in words:
                        # Use word boundary \b to match whole words only
                        if re.search(rf'\b{re.escape(word)}\b', text):
                            return True
                    return False
                
                # Check for specific patterns first
                if has_whole_word_in_line(line_lower, [
                    'fix', 'bug', 'error', 'issue', 'resolve', 'correct', 'patch',
                    'crash', 'problem', 'fail', 'broken', 'dependency', 'missing',
                    'compatibility', 'mismatch'
                ]):
                    fixed_items.append(clean_line)
                elif has_whole_word_in_line(line_lower, [
                    'remove', 'delete', 'deprecate', 'drop', 'eliminate'
                ]):
                    removed_items.append(clean_line)
                elif has_whole_word_in_line(line_lower, [
                    'update', 'enhance', 'improve', 'change', 'modify', 'optimize',
                    'performance', 'smart', 'efficient', 'better', 'cleaner',
                    'reorganize', 'refactor', 'streamline'
                ]):
                    changed_items.append(clean_line)
                else:
                    # Default to Added for new features, documentation, etc.
                    added_items.append(clean_line)
            
            # Build changelog entry with proper spacing
            entry_parts = [f"## [{version}] - {today}"]
            
            # Add sections in conventional order: Added, Changed, Fixed, Removed
            if added_items:
                entry_parts.extend(["", "### Added", ""])
                for item in added_items:
                    entry_parts.append(f"- {item}")
            
            if changed_items:
                entry_parts.extend(["", "### Changed", ""])
                for item in changed_items:
                    entry_parts.append(f"- {item}")
            
            if fixed_items:
                entry_parts.extend(["", "### Fixed", ""])
                for item in fixed_items:
                    entry_parts.append(f"- {item}")
            
            if removed_items:
                entry_parts.extend(["", "### Removed", ""])
                for item in removed_items:
                    entry_parts.append(f"- {item}")
            
            # Ensure proper ending
            entry_parts.append("")
            return "\n".join(entry_parts)
        else:
            # Single line description - provide helpful error message
            error_msg = f"""
❌ ERROR: Single-line descriptions are no longer supported for better changelog quality.

Please use the proper format with detailed descriptions:

CORRECT USAGE:
python3 scripts/bump_version_enhanced.py {version} "commit message" "changelog description

Explain what users will experience.
Focus on benefits and improvements.
Use bullet points for multiple changes."

EXAMPLE:
python3 scripts/bump_version_enhanced.py patch "Fix RVC dropdown" "Fix RVC model dropdown to show both downloadable and local models

The Load RVC Character Model node now properly displays both downloadable character models and local models in the same dropdown.
This matches the F5-TTS dropdown behavior for consistent user experience across all engines."
"""
            raise ValueError(error_msg)
    
    def preview_changelog_entry(self, version: str, description: str, details: List[str] = None, simple_mode: bool = False) -> str:
        """Preview changelog entry content without writing to file (for dry-run)"""
        return self._generate_changelog_content(version, description, details, simple_mode)
    
    def add_changelog_entry(self, version: str, description: str, details: List[str] = None, simple_mode: bool = False) -> bool:
        """Add entry to CHANGELOG.md with proper formatting"""
        try:
            changelog_path = os.path.join(self.project_root, 'CHANGELOG.md')
            
            with open(changelog_path, 'r') as f:
                content = f.read()
            
            # Generate changelog entry content using extracted method
            try:
                new_entry = self._generate_changelog_content(version, description, details, simple_mode)
            except ValueError as e:
                print(str(e))
                return False
            
            # Insert after the header (find first ## line)
            lines = content.split('\n')
            insert_index = 0
            for i, line in enumerate(lines):
                if line.startswith('## [') and ']' in line:
                    insert_index = i
                    break
            
            # Insert the new entry with proper spacing
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