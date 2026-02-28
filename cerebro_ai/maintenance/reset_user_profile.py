"""
Reset User Profile
==================
Archives the current profile and creates a fresh, clean template.
User chose "fresh start" - they'll re-introduce themselves over time.
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

USER_DIR = DATA_DIR / "user"
PROFILE_FILE = USER_DIR / "profile.json"


def create_clean_profile_template() -> dict:
    """Create a clean, empty profile structure"""
    return {
        "identity": {
            "name": None,
            "username": None,
            "aliases": [],
            "roles": [],
            "location": None,
            "contact": {},
            "first_mentioned": None,
            "last_updated": datetime.now().isoformat()
        },
        "relationships": {
            "pets": [],
            "family": [],
            "colleagues": [],
            "friends": []
        },
        "projects": {
            "companies_owned": [],
            "clients": [],
            "active_projects": []
        },
        "goals": [],
        "preferences": {
            "technical": {},
            "personal": [],
            "dislikes": []
        },
        "technical_environment": {
            "operating_systems": [],
            "programming_languages": [],
            "tools": [],
            "infrastructure": {}
        },
        "_metadata": {
            "version": "2.0",
            "created": datetime.now().isoformat(),
            "note": "Fresh start profile - previous data archived"
        }
    }


def reset_profile(dry_run: bool = False) -> dict:
    """
    Archive current profile and create fresh template.

    Args:
        dry_run: If True, don't modify files

    Returns:
        Statistics about the operation
    """
    stats = {
        "archived": False,
        "archive_path": None,
        "new_profile_created": False,
        "old_profile_size": 0
    }

    print("=" * 60)
    print("AI Memory - Reset User Profile")
    print("=" * 60)
    print(f"Profile directory: {USER_DIR}")
    print()

    # Check current profile
    if PROFILE_FILE.exists():
        stats["old_profile_size"] = PROFILE_FILE.stat().st_size
        print(f"Current profile size: {stats['old_profile_size']:,} bytes")

        # Read current profile for summary
        try:
            with open(PROFILE_FILE, 'r', encoding='utf-8') as f:
                old_profile = json.load(f)

            # Show what's being archived
            identity = old_profile.get("identity", {})
            print(f"  Name: {identity.get('name', 'Unknown')}")
            print(f"  Location: {identity.get('location', 'Unknown')}")

            relationships = old_profile.get("relationships", {})
            print(f"  Pets: {len(relationships.get('pets', []))}")
            print(f"  Family: {len(relationships.get('family', []))}")

            projects = old_profile.get("projects", {})
            print(f"  Active projects: {len(projects.get('active_projects', []))}")

            print(f"  Goals: {len(old_profile.get('goals', []))}")
        except Exception as e:
            print(f"  (Could not read profile: {e})")

    else:
        print("No existing profile found")

    if dry_run:
        print("\n[DRY RUN] Would archive and create new profile")
        return stats

    # Archive current profile
    if PROFILE_FILE.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = USER_DIR / f"profile_backup_{timestamp}.json"

        print(f"\nArchiving to: {archive_path.name}")
        shutil.copy2(PROFILE_FILE, archive_path)
        stats["archived"] = True
        stats["archive_path"] = str(archive_path)

    # Create clean profile
    print("\nCreating fresh profile template...")
    clean_profile = create_clean_profile_template()

    with open(PROFILE_FILE, 'w', encoding='utf-8') as f:
        json.dump(clean_profile, f, indent=2, ensure_ascii=False)

    stats["new_profile_created"] = True
    new_size = PROFILE_FILE.stat().st_size
    print(f"New profile size: {new_size:,} bytes")

    # Also reset preferences.json
    preferences_file = USER_DIR / "preferences.json"
    if preferences_file.exists():
        # Archive it too
        pref_archive = USER_DIR / f"preferences_backup_{timestamp}.json"
        shutil.copy2(preferences_file, pref_archive)
        print(f"Archived preferences to: {pref_archive.name}")

        # Create clean preferences
        clean_preferences = {
            "communication_style": {
                "prefers": [],
                "dislikes": [],
                "tone": "professional"
            },
            "workflow_preferences": {
                "prefers": [],
                "dislikes": []
            },
            "technical_preferences": {
                "languages": {},
                "frameworks": {},
                "tools": {}
            },
            "response_format": {
                "code_style": "detailed",
                "explanation_depth": "balanced",
                "include_examples": True
            },
            "last_updated": datetime.now().isoformat()
        }
        with open(preferences_file, 'w', encoding='utf-8') as f:
            json.dump(clean_preferences, f, indent=2, ensure_ascii=False)
        print("Reset preferences.json")

    print()
    print("=" * 60)
    print("DONE - Fresh start complete!")
    print("=" * 60)
    print()
    print("Your profile is now empty. As we chat, I'll learn about you")
    print("again with improved extraction (no more garbage data).")
    print()

    return stats


if __name__ == "__main__":
    import sys

    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("[DRY RUN MODE]\n")

    reset_profile(dry_run=dry_run)
