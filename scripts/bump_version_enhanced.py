#!/usr/bin/env python3
"""
Enhanced Automated Version Bumping Script for ComfyUI ChatterBox Voice

SEPARATE COMMIT & CHANGELOG MODE (recommended):
python scripts/bump_version_enhanced.py 3.0.2 --commit "Fix critical bugs" --changelog "Bug fixes and improvements"

LEGACY MODE (same description for both):
python scripts/bump_version_enhanced.py 3.0.2 "Add sounddevice dependency"

INTERACTIVE MODE:
python scripts/bump_version_enhanced.py 3.0.2 --interactive

FILE MODE:
python scripts/bump_version_enhanced.py 3.0.2 --changelog-file changelog.txt --commit-file commit.txt
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
    parser = argparse.ArgumentParser(description='Bump version with separate commit and changelog descriptions')
    parser.add_argument('version', nargs='?', help='New version number (e.g., 3.0.2) or auto-increment type (patch/minor/major)')
    parser.add_argument('commit_description', nargs='?', help='Commit message description (what this version bump does)')
    parser.add_argument('changelog_description', nargs='?', help='Changelog description (what users should know)')
    
    # Legacy mode (deprecated)
    parser.add_argument('--legacy', help='[DEPRECATED] Single description for both commit and changelog')
    parser.add_argument('--commit', help='[OVERRIDE] Commit message description')
    parser.add_argument('--changelog', help='[OVERRIDE] Changelog description')
    
    # File input options
    parser.add_argument('--commit-file', help='Read commit description from file')
    parser.add_argument('--changelog-file', help='Read changelog description from file')
    parser.add_argument('--file', help='[LEGACY] Read description from file for both commit and changelog')
    
    # Other options
    parser.add_argument('--interactive', action='store_true', help='Interactive mode for separate commit and changelog entry')
    parser.add_argument('--no-commit', action='store_true', help='Skip git commit')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--allow-downgrade', action='store_true', help='Allow bumping to a lower version number (for reverts/fixes)')
    
    args = parser.parse_args()
    
    # Initialize version manager
    vm = VersionManager()
    
    # Get current version
    current_version = vm.get_current_version()
    if not current_version:
        print("Error: Could not determine current version")
        sys.exit(1)
    
    # Handle auto-increment version types early
    if args.version and args.version.lower() in ['patch', 'minor', 'major']:
        current_parts = list(map(int, current_version.split('.')))
        if args.version.lower() == 'patch':
            current_parts[2] += 1
        elif args.version.lower() == 'minor':
            current_parts[1] += 1
            current_parts[2] = 0
        elif args.version.lower() == 'major':
            current_parts[0] += 1
            current_parts[1] = 0
            current_parts[2] = 0
        args.version = '.'.join(map(str, current_parts))
        print(f"Auto-increment: {current_version} → {args.version}")
    
    # Handle input for commit and changelog descriptions
    commit_description = ""
    changelog_description = ""
    
    # New default mode: separate commit and changelog descriptions
    if args.commit_description and args.changelog_description:
        commit_description = args.commit_description
        changelog_description = args.changelog_description
    elif args.commit_file and args.changelog_file:
        # File input mode
        try:
            with open(args.commit_file, 'r') as f:
                commit_description = f.read().strip()
            with open(args.changelog_file, 'r') as f:
                changelog_description = f.read().strip()
        except Exception as e:
            print(f"Error reading files: {e}")
            sys.exit(1)
    elif args.legacy:
        # Legacy mode - same description for both
        commit_description = args.legacy
        changelog_description = args.legacy
    else:
        print("Error: Please provide both commit and changelog descriptions and read BUMP_SCRIPT_INSTRUCTIONS.md if you have not yet")
        print("Usage: python3 scripts/bump_version_enhanced.py <version> \"<commit_desc>\" \"<changelog_desc>\"")
        print("Legacy: python3 scripts/bump_version_enhanced.py <version> --legacy \"<description>\"")
        sys.exit(1)
    
    # Override with explicit flags if provided
    if args.commit:
        commit_description = args.commit
    if args.changelog:
        changelog_description = args.changelog
    
    # Handle interactive mode
    if args.interactive:
        # Interactive mode for separate descriptions
        print("=== COMMIT DESCRIPTION ===")
        print("Enter commit description (what this version bump does):")
        print("(Press Ctrl+D on empty line to finish)")
        commit_lines = []
        try:
            while True:
                line = input()
                commit_lines.append(line)
        except EOFError:
            pass
        commit_description = '\n'.join(commit_lines)
        
        print("\n=== CHANGELOG DESCRIPTION ===")
        print("Enter changelog description (what users should know):")
        print("(Press Ctrl+D on empty line to finish)")
        changelog_lines = []
        try:
            while True:
                line = input()
                changelog_lines.append(line)
        except EOFError:
            pass
        changelog_description = '\n'.join(changelog_lines)
        
    elif args.file:
        # Legacy file mode - same description for both
        try:
            with open(args.file, 'r') as f:
                description = f.read().strip()
                commit_description = description
                changelog_description = description
        except Exception as e:
            print(f"Error reading description file: {e}")
            sys.exit(1)
    
    # Process newline characters
    commit_description = commit_description.replace('\\n', '\n')
    changelog_description = changelog_description.replace('\\n', '\n')
    
    print(f"Current version: {current_version}")
    print(f"New version: {args.version}")
    print(f"\nCommit description:")
    print("=" * 50)
    print(commit_description)
    print("=" * 50)
    print(f"\nChangelog description:")
    print("=" * 50)
    print(changelog_description)
    print("=" * 50)
    
    if args.dry_run:
        print("\n[DRY RUN] Would update these files:")
        for file_path in vm.version_files.keys():
            full_path = os.path.join(vm.project_root, file_path)
            if os.path.exists(full_path):
                print(f"  ✓ {file_path}")
            else:
                print(f"  ✗ {file_path} (not found)")
        
        # Generate and show the changelog entry preview using the REAL extracted method
        print(f"\n[DRY RUN] Changelog entry that would be added:")
        print("=" * 50)
        
        try:
            changelog_preview = vm.preview_changelog_entry(args.version, changelog_description)
            print(changelog_preview.rstrip())
        except Exception as e:
            print(f"Error generating changelog preview: {e}")
        
        print("=" * 50)
        
        if not args.no_commit:
            print(f"[DRY RUN] Would commit changes with detailed message")
        return
    
    # Validate version format
    if not vm.validate_version(args.version):
        print(f"Error: Invalid version format '{args.version}'. Use semantic versioning (e.g., 3.0.1)")
        sys.exit(1)
    
    # Check if version is newer than current (unless downgrade is explicitly allowed)
    if not args.allow_downgrade:
        try:
            current_parts = list(map(int, current_version.split('.')))
            new_parts = list(map(int, args.version.split('.')))
            
            if tuple(new_parts) <= tuple(current_parts):
                print(f"Error: New version {args.version} is not newer than current {current_version}")
                print("Cannot bump to an older or same version number.")
                print("Use a higher version number for the next release.")
                print("To force a downgrade, use --allow-downgrade flag.")
                sys.exit(1)
        except Exception as e:
            print(f"Warning: Could not compare versions: {e}")
            print("Proceeding with caution...")
    else:
        print("⚠️  Downgrade allowed - skipping version comparison check")
    
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
        if not vm.add_changelog_entry(args.version, changelog_description, simple_mode=False):
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
                    if '\n' in commit_description:
                        # Multiline commit message
                        commit_message = f"{commit_title}\n\n{commit_description}"
                    else:
                        # Single line commit message
                        commit_message = f"{commit_title}: {commit_description}"
                    
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