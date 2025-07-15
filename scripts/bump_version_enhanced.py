#!/usr/bin/env python3
"""
Enhanced Automated Version Bumping Script for ComfyUI ChatterBox Voice
Usage: python scripts/bump_version_enhanced.py <version> "<description>"
Example: python scripts/bump_version_enhanced.py 3.0.2 "Add sounddevice dependency"

Supports multiline descriptions:
python scripts/bump_version_enhanced.py 3.0.2 "Fix missing sounddevice dependency\nAdd comprehensive error handling\nUpdate installation documentation"

Or read from file:
python scripts/bump_version_enhanced.py 3.0.2 --file changelog.txt
"""

import os
import sys
import subprocess
import argparse
from version_utils import VersionManager

def run_git_command(command: str) -> bool:
    """Run git command and return success status"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Git command failed: {command}")
            print(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Error running git command: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Bump version across all project files with enhanced multiline support')
    parser.add_argument('version', help='New version number (e.g., 3.0.2)')
    parser.add_argument('description', nargs='?', help='Description of changes for changelog (use quotes for multiline)')
    parser.add_argument('--no-commit', action='store_true', help='Skip git commit')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--file', help='Read description from file (supports multiline)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode for detailed changelog entry')
    
    args = parser.parse_args()
    
    # Initialize version manager
    vm = VersionManager()
    
    # Get current version
    current_version = vm.get_current_version()
    if not current_version:
        print("Error: Could not determine current version")
        sys.exit(1)
    
    # Handle description input
    description = ""
    if args.file:
        try:
            with open(args.file, 'r') as f:
                description = f.read().strip()
        except Exception as e:
            print(f"Error reading description file: {e}")
            sys.exit(1)
    elif args.interactive:
        print("Enter changelog description (press Ctrl+D on empty line to finish):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        description = '\n'.join(lines)
    elif args.description:
        description = args.description
    else:
        print("Error: Description required (use --file, --interactive, or provide as argument)")
        sys.exit(1)
    
    # Process newline characters in description
    description = description.replace('\\n', '\n')
    
    print(f"Current version: {current_version}")
    print(f"New version: {args.version}")
    print(f"Description preview:")
    print("=" * 50)
    print(description)
    print("=" * 50)
    
    if args.dry_run:
        print("\n[DRY RUN] Would update these files:")
        for file_path in vm.version_files.keys():
            full_path = os.path.join(vm.project_root, file_path)
            if os.path.exists(full_path):
                print(f"  ✓ {file_path}")
            else:
                print(f"  ✗ {file_path} (not found)")
        print(f"[DRY RUN] Would add detailed changelog entry for v{args.version}")
        if not args.no_commit:
            print(f"[DRY RUN] Would commit changes with detailed message")
        return
    
    # Validate version format
    if not vm.validate_version(args.version):
        print(f"Error: Invalid version format '{args.version}'. Use semantic versioning (e.g., 3.0.1)")
        sys.exit(1)
    
    # Check if version is newer than current
    try:
        current_parts = list(map(int, current_version.split('.')))
        new_parts = list(map(int, args.version.split('.')))
        
        if tuple(new_parts) <= tuple(current_parts):
            print(f"Warning: New version {args.version} is not newer than current {current_version}")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Version bump cancelled")
                sys.exit(0)
    except Exception as e:
        print(f"Warning: Could not compare versions: {e}")
    
    # Create backup
    print("\nCreating backup of current files...")
    backup = vm.backup_files()
    
    try:
        # Update all version files
        print("\nUpdating version files...")
        if not vm.update_all_versions(args.version):
            print("Error: Failed to update all version files")
            print("Restoring backup...")
            vm.restore_files(backup)
            sys.exit(1)
        
        # Add changelog entry with multiline support
        print("\nUpdating changelog...")
        if not vm.add_changelog_entry(args.version, description):
            print("Error: Failed to update changelog")
            print("Restoring backup...")
            vm.restore_files(backup)
            sys.exit(1)
        
        # Git operations
        if not args.no_commit:
            print("\nCommitting changes...")
            
            # Check if git repo exists
            if not os.path.exists(os.path.join(vm.project_root, '.git')):
                print("Warning: Not in a git repository, skipping commit")
            else:
                # Stage changes
                if not run_git_command("git add -A"):
                    print("Error: Failed to stage changes")
                    sys.exit(1)
                
                # Check if there are changes to commit
                result = subprocess.run("git diff --cached --quiet", shell=True)
                if result.returncode == 0:
                    print("No changes to commit")
                else:
                    # Create commit message
                    commit_title = f"Version {args.version}"
                    if '\n' in description:
                        # Multiline commit message
                        commit_message = f"{commit_title}\n\n{description}"
                    else:
                        # Single line commit message
                        commit_message = f"{commit_title}: {description}"
                    
                    # Write commit message to temp file for complex messages
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                        f.write(commit_message)
                        temp_file = f.name
                    
                    try:
                        # Use file for commit message to handle multiline properly
                        if not run_git_command(f'git commit -F "{temp_file}"'):
                            print("Error: Failed to commit changes")
                            sys.exit(1)
                    finally:
                        os.unlink(temp_file)
                    
                    print(f"✓ Committed changes with detailed message")
        
        print(f"\n🎉 Successfully bumped version to {args.version}!")
        print(f"📝 Changelog updated with detailed entry")
        
        if not args.no_commit:
            print("📦 Changes committed to git")
            print("\nNext steps:")
            print("1. Test the changes")
            print("2. Push to remote if ready: git push")
            print(f"3. Create release tag if needed: git tag v{args.version}")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        print("Restoring backup...")
        vm.restore_files(backup)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Restoring backup...")
        vm.restore_files(backup)
        sys.exit(1)

if __name__ == "__main__":
    main()