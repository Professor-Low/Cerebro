"""
Validators - Validate project names and other entities to prevent malformed entries.

Prevents invalid project names like single characters, punctuation-only names,
and reserved system names from being stored.
"""

import json
import re
from pathlib import Path
from typing import List, Set, Tuple

# Reserved names that should not be used as project names
RESERVED_NAMES: Set[str] = {
    # System directories
    "home", "root", "tmp", "var", "etc", "usr", "bin", "lib", "opt",
    "proc", "sys", "dev", "run", "srv", "boot", "mnt", "media",

    # Common path segments that aren't project names
    "src", "lib", "bin", "include", "share", "local", "projects",
    "repos", "repositories", "workspace", "workspaces",

    # User directories
    "documents", "downloads", "desktop", "pictures", "music", "videos",
    ".local", ".config", ".cache", ".ssh", ".gnupg",

    # Generic names
    "test", "tests", "temp", "backup", "backups", "old", "new",
    "build", "dist", "node_modules", "__pycache__", ".git",

    # Single-character names
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
}


def is_valid_project_name(name: str) -> bool:
    """
    Validate if a string is a valid project name.

    Invalid patterns:
    - Single characters
    - Only punctuation/whitespace
    - Reserved system names
    - Names starting with . (hidden files/dirs)
    - Names that are just numbers
    - Names with path separators

    Args:
        name: Proposed project name

    Returns:
        True if valid project name, False otherwise
    """
    if not name:
        return False

    # Strip whitespace
    name = name.strip()

    # Check minimum length (at least 2 chars)
    if len(name) < 2:
        return False

    # Check for single character followed by punctuation (like ",")
    if len(name) <= 3 and not name.isalnum():
        return False

    # Check for only punctuation/whitespace
    if not any(c.isalnum() for c in name):
        return False

    # Check reserved names (case-insensitive)
    if name.lower() in RESERVED_NAMES:
        return False

    # Check for hidden files/dirs
    if name.startswith("."):
        return False

    # Check for pure numbers
    if name.isdigit():
        return False

    # Check for path separators
    if "/" in name or "\\" in name:
        return False

    # Check for common invalid patterns
    invalid_patterns = [
        r'^[,;:!@#$%^&*()]+$',  # Only punctuation
        r'^\s+$',               # Only whitespace
        r'^[\[\]{}]+$',         # Only brackets
        r'^[<>]+$',             # Only angle brackets
        r'^\d+[,;:]\d*$',       # Numbers with punctuation
    ]

    for pattern in invalid_patterns:
        if re.match(pattern, name):
            return False

    return True


def clean_project_tracker(tracker_path: str) -> Tuple[int, List[str]]:
    """
    Clean malformed entries from the project tracker file.

    Args:
        tracker_path: Path to the tracker.json file

    Returns:
        Tuple of (count of removed entries, list of removed names)
    """
    path = Path(tracker_path)

    if not path.exists():
        return (0, [])

    try:
        with open(path, "r", encoding="utf-8") as f:
            projects = json.load(f)
    except json.JSONDecodeError:
        return (0, [])

    # Collect invalid project names
    invalid_names = []
    for name in list(projects.keys()):
        if not is_valid_project_name(name):
            invalid_names.append(name)
            del projects[name]

    if invalid_names:
        # Backup original
        backup_path = path.with_suffix(".json.bak")
        with open(path, "r", encoding="utf-8") as f:
            original = f.read()
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(original)

        # Save cleaned version
        with open(path, "w", encoding="utf-8") as f:
            json.dump(projects, f, indent=2, ensure_ascii=False)

    return (len(invalid_names), invalid_names)


def validate_and_clean_name(name: str) -> str:
    """
    Attempt to clean a name and return a valid version.

    Args:
        name: Raw name that may need cleaning

    Returns:
        Cleaned name, or empty string if cannot be salvaged
    """
    if not name:
        return ""

    # Strip whitespace and common problematic chars
    cleaned = name.strip().strip(",;:!@#$%^&*()[]{}|\\\"'")

    # Remove path separators and get last meaningful component
    if "/" in cleaned or "\\" in cleaned:
        parts = re.split(r'[/\\]', cleaned)
        # Find last non-empty, valid part
        for part in reversed(parts):
            part = part.strip()
            if part and is_valid_project_name(part):
                return part
        return ""

    if is_valid_project_name(cleaned):
        return cleaned

    return ""


def is_valid_entity_name(name: str, entity_type: str = "general") -> bool:
    """
    Validate entity names based on type.

    Args:
        name: Entity name to validate
        entity_type: Type of entity (project, topic, tag, etc.)

    Returns:
        True if valid for the entity type
    """
    if not name or not name.strip():
        return False

    name = name.strip()

    # Type-specific validation
    if entity_type == "project":
        return is_valid_project_name(name)

    elif entity_type == "topic":
        # Topics can be shorter but still need content
        if len(name) < 2:
            return False
        if not any(c.isalnum() for c in name):
            return False
        return True

    elif entity_type == "tag":
        # Tags can be short but must be alphanumeric (with - or _)
        if len(name) < 1:
            return False
        if not re.match(r'^[\w\-_]+$', name):
            return False
        return True

    else:
        # General validation
        if len(name) < 2:
            return False
        if not any(c.isalnum() for c in name):
            return False
        return True


if __name__ == "__main__":
    # Test validation
    test_names = [
        "ai-memory-mcp",      # Valid
        ",",                  # Invalid - single punctuation
        "home",               # Invalid - reserved
        "a",                  # Invalid - single char
        ".hidden",            # Invalid - hidden
        "123",                # Invalid - pure numbers
        "my-project",         # Valid
        "   ",                # Invalid - whitespace only
        "src",                # Invalid - reserved
        "valid_project_123",  # Valid
        "",                   # Invalid - empty
        "sample-user",        # Valid
    ]

    print("=== Project Name Validation Test ===\n")

    for name in test_names:
        valid = is_valid_project_name(name)
        status = "VALID" if valid else "INVALID"
        display_name = repr(name)
        print(f"  {display_name:25} -> {status}")

    print("\n=== Cleanup Test ===")
    # Test cleanup (dry run, just show what would be removed)
    from config import PROJECTS_DIR
    tracker_path = PROJECTS_DIR / "tracker.json"
    if tracker_path.exists():
        # Just peek at what would be cleaned
        with open(tracker_path, "r") as f:
            projects = json.load(f)
        invalid = [n for n in projects.keys() if not is_valid_project_name(n)]
        print(f"\nFound {len(invalid)} invalid entries that would be cleaned:")
        for name in invalid[:10]:
            print(f"  - {repr(name)}")
        if len(invalid) > 10:
            print(f"  ... and {len(invalid) - 10} more")
    else:
        print(f"\nTracker file not found at {tracker_path}")
