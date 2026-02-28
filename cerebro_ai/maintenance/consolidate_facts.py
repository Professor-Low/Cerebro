"""
Consolidate Facts Script
========================
Reads all individual fact JSON files from data/facts/
and consolidates them into data/knowledge/facts.jsonl

This fixes the issue where facts.jsonl was empty (0 bytes)
while 1,633+ fact files existed as individual JSONs.
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

# Paths
FACTS_DIR = DATA_DIR / "facts"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
OUTPUT_FILE = KNOWLEDGE_DIR / "facts.jsonl"
BACKUP_FILE = KNOWLEDGE_DIR / f"facts_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"


def content_hash(content: str) -> str:
    """Generate hash for deduplication"""
    normalized = content.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


def load_fact_file(filepath: Path) -> Dict:
    """Load a single fact JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"  Error reading {filepath.name}: {e}")
        return None


def consolidate_facts(dry_run: bool = False) -> Dict:
    """
    Consolidate all fact files into a single JSONL.

    Args:
        dry_run: If True, don't write files, just report what would happen

    Returns:
        Statistics about the consolidation
    """
    stats = {
        "total_files": 0,
        "valid_facts": 0,
        "duplicates_removed": 0,
        "errors": 0,
        "output_file": str(OUTPUT_FILE)
    }

    # Ensure knowledge directory exists
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

    # Track seen content hashes for deduplication
    seen_hashes: Set[str] = set()
    unique_facts: List[Dict] = []

    # Get all fact files
    fact_files = list(FACTS_DIR.glob("*.json"))
    stats["total_files"] = len(fact_files)

    print(f"Found {len(fact_files)} fact files in {FACTS_DIR}")

    # Process each file
    for i, filepath in enumerate(fact_files):
        if i % 200 == 0:
            print(f"  Processing {i}/{len(fact_files)}...")

        fact = load_fact_file(filepath)
        if fact is None:
            stats["errors"] += 1
            continue

        # Get content for hashing
        content = fact.get("content", "")
        if not content:
            stats["errors"] += 1
            continue

        # Check for duplicates
        h = content_hash(content)
        if h in seen_hashes:
            stats["duplicates_removed"] += 1
            continue

        seen_hashes.add(h)

        # Ensure consistent structure
        consolidated_fact = {
            "id": fact.get("id", filepath.stem),
            "timestamp": fact.get("timestamp", datetime.now().isoformat()),
            "type": "fact",
            "content": content,
            "metadata": fact.get("metadata", {}),
            "word_count": fact.get("word_count", len(content.split()))
        }

        unique_facts.append(consolidated_fact)
        stats["valid_facts"] += 1

    print("\nConsolidation complete:")
    print(f"  Total files processed: {stats['total_files']}")
    print(f"  Valid unique facts: {stats['valid_facts']}")
    print(f"  Duplicates removed: {stats['duplicates_removed']}")
    print(f"  Errors: {stats['errors']}")

    if dry_run:
        print(f"\n[DRY RUN] Would write {len(unique_facts)} facts to {OUTPUT_FILE}")
        return stats

    # Backup existing file if it has content
    if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
        print(f"\nBacking up existing facts.jsonl to {BACKUP_FILE}")
        OUTPUT_FILE.rename(BACKUP_FILE)

    # Write consolidated JSONL
    print(f"\nWriting {len(unique_facts)} facts to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for fact in unique_facts:
            f.write(json.dumps(fact, ensure_ascii=False) + '\n')

    # Verify
    written_size = OUTPUT_FILE.stat().st_size
    print(f"  Written: {written_size:,} bytes")

    return stats


if __name__ == "__main__":
    import sys

    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print("AI Memory Facts Consolidation")
    print("=" * 60)
    print(f"Source: {FACTS_DIR}")
    print(f"Output: {OUTPUT_FILE}")
    if dry_run:
        print("[DRY RUN MODE - No files will be modified]")
    print("=" * 60)
    print()

    stats = consolidate_facts(dry_run=dry_run)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
