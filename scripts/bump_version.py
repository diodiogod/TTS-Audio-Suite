#!/usr/bin/env python3
"""
Automated Version Bumping Script for ComfyUI ChatterBox Voice
Usage: python scripts/bump_version.py <version> "<description>"
Example: python scripts/bump_version.py 3.0.2 "Add sounddevice dependency"
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
    parser = argparse.ArgumentParser(description='Bump version across all project files')
    parser.add_argument('version', help='New version number (e.g., 3.0.2)')
    parser.add_argument('description', help='Description of changes for changelog')
    parser.add_argument('--no-commit', action='store_true', help='Skip git commit')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    # Initialize version manager
    vm = VersionManager()
    
    # Get current version
    current_version = vm.get_current_version()
    if not current_version:
        print("Error: Could not determine current version")
        sys.exit(1)
    
    print(f"Current version: {current_version}")
    print(f"New version: {args.version}")
    print(f"Description: {args.description}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would update these files:")
        for file_path in vm.version_files.keys():
            full_path = os.path.join(vm.project_root, file_path)
            if os.path.exists(full_path):
                print(f"  ✓ {file_path}")
            else:
                print(f"  ✗ {file_path} (not found)")
        print(f"[DRY RUN] Would add changelog entry for v{args.version}")
        if not args.no_commit:
            print(f"[DRY RUN] Would commit changes with message:")
            print(f"  Version {args.version}: {args.description}")
        return
    
    # Validate version format
    if not vm.validate_version(args.version):
        print(f"Error: Invalid version format '{args.version}'. Use semantic versioning (e.g., 3.0.1)")
        sys.exit(1)
    
    # Check if version is newer than current
    try:
        current_parts = list(map(int, current_version.split('.')))
        new_parts = list(map(int, args.version.split('.')))
        
        if new_parts <= current_parts:
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
        
        # Add changelog entry
        print("\nUpdating changelog...")
        if not vm.add_changelog_entry(args.version, args.description):
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
                    # Commit changes
                    commit_message = f"Version {args.version}: {args.description}"
                    if not run_git_command(f'git commit -m "{commit_message}"'):
                        print("Error: Failed to commit changes")
                        sys.exit(1)
                    
                    print(f"✓ Committed changes with message: {commit_message}")
        
        print(f"\n🎉 Successfully bumped version to {args.version}!")
        print(f"📝 Changelog updated with: {args.description}")
        
        if not args.no_commit:
            print("📦 Changes committed to git")
            print("\nNext steps:")
            print("1. Test the changes")
            print("2. Push to remote if ready: git push")
            print("3. Create release tag if needed: git tag v{args.version}")
        
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