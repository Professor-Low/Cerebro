"""
Auto Maintenance for AI Memory System
======================================
Runs every 6 hours via Windows Task Scheduler to:
1. Check FAISS index freshness
2. Process unembedded conversations
3. Update facts.jsonl with new facts
4. Clean up stale sessions
5. Send Windows toast notifications on issues

Usage:
    python auto_maintenance.py           # Run full maintenance
    python auto_maintenance.py --check   # Check only, no fixes
    python auto_maintenance.py --quiet   # No toast notifications
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

# Import our modules
from toast_notifier import notify_issue

from config import DATA_DIR

# Paths
BASE_PATH = DATA_DIR
CONVERSATIONS_PATH = BASE_PATH / "conversations"
CHUNKS_PATH = BASE_PATH / "embeddings" / "chunks"
INDEX_PATH = BASE_PATH / "embeddings" / "indexes" / "faiss_index.bin"
FACTS_DIR = BASE_PATH / "facts"
FACTS_JSONL = BASE_PATH / "knowledge" / "facts.jsonl"
LOGS_PATH = BASE_PATH / "logs"


def check_index_freshness() -> Tuple[bool, int]:
    """
    Check if FAISS index is stale.

    Returns:
        (is_stale, hours_since_update)
    """
    if not INDEX_PATH.exists():
        return True, -1

    index_mtime = datetime.fromtimestamp(INDEX_PATH.stat().st_mtime)
    hours_old = (datetime.now() - index_mtime).total_seconds() / 3600

    # Count conversations newer than index
    newer_count = 0
    for conv_file in CONVERSATIONS_PATH.glob("*.json"):
        if conv_file.stat().st_mtime > INDEX_PATH.stat().st_mtime:
            newer_count += 1

    # Stale if > 10 new conversations or > 24 hours old
    is_stale = newer_count > 10 or hours_old > 24

    return is_stale, int(hours_old), newer_count


def find_missing_embeddings() -> List[str]:
    """Find conversations without chunk files."""
    conv_ids = set()
    for f in CONVERSATIONS_PATH.glob("*.json"):
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                conv_ids.add(data.get('id', f.stem))
        except:
            conv_ids.add(f.stem)

    chunk_ids = {f.stem for f in CHUNKS_PATH.glob("*.jsonl")}

    return [cid for cid in conv_ids if cid not in chunk_ids]


def check_facts_sync() -> Tuple[bool, int, int]:
    """
    Check if facts.jsonl is in sync with facts directory.

    Returns:
        (is_synced, facts_in_dir, facts_in_jsonl)
    """
    facts_in_dir = len(list(FACTS_DIR.glob("*.json")))

    if not FACTS_JSONL.exists():
        return False, facts_in_dir, 0

    with open(FACTS_JSONL, 'r', encoding='utf-8') as f:
        facts_in_jsonl = sum(1 for _ in f)

    # Consider synced if within 10% or jsonl has more (after dedup)
    is_synced = facts_in_jsonl > 0 and (facts_in_dir <= facts_in_jsonl * 1.5)

    return is_synced, facts_in_dir, facts_in_jsonl


def run_maintenance(check_only: bool = False, quiet: bool = False) -> Dict:
    """
    Run full maintenance cycle.

    Args:
        check_only: If True, only report issues without fixing
        quiet: If True, don't send toast notifications

    Returns:
        Maintenance report
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "checks": {},
        "actions": [],
        "issues": []
    }

    print("=" * 60)
    print("AI Memory Auto-Maintenance")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Check NAS connectivity
    print("\n[1/5] Checking NAS connectivity...")
    if not BASE_PATH.exists():
        report["checks"]["nas"] = "UNREACHABLE"
        report["issues"].append("Data directory unreachable")
        if not quiet:
            notify_issue("nas_unreachable", "Cannot access data directory - check configuration")
        print("  FAIL: NAS not accessible")
        return report

    report["checks"]["nas"] = "OK"
    print("  OK: NAS accessible")

    # 2. Check FAISS index freshness
    print("\n[2/5] Checking FAISS index freshness...")
    is_stale, hours_old, newer_count = check_index_freshness()
    report["checks"]["index"] = {
        "stale": is_stale,
        "hours_old": hours_old,
        "newer_conversations": newer_count
    }

    if is_stale:
        report["issues"].append(f"FAISS index is {hours_old}h old with {newer_count} newer conversations")
        print(f"  STALE: {hours_old}h old, {newer_count} newer conversations")

        if not check_only:
            print("  Rebuilding index...")
            try:
                os.environ['ENABLE_EMBEDDINGS'] = '1'
                from ai_embeddings_engine import EmbeddingsEngine
                engine = EmbeddingsEngine(base_path=str(BASE_PATH))
                engine.rebuild_index()
                report["actions"].append("Rebuilt FAISS index")
                print("  DONE: Index rebuilt")
            except Exception as e:
                print(f"  ERROR: {e}")
                report["issues"].append(f"Failed to rebuild index: {e}")
    else:
        print(f"  OK: {hours_old}h old")

    # 3. Check for missing embeddings
    print("\n[3/5] Checking for missing embeddings...")
    missing = find_missing_embeddings()
    report["checks"]["embeddings"] = {
        "missing_count": len(missing),
        "missing_ids": missing[:10]  # First 10
    }

    if missing:
        report["issues"].append(f"{len(missing)} conversations missing embeddings")
        print(f"  MISSING: {len(missing)} conversations")

        if not check_only and len(missing) > 0:
            print("  Processing missing embeddings...")
            try:
                os.environ['ENABLE_EMBEDDINGS'] = '1'
                from ai_embeddings_engine import EmbeddingsEngine
                engine = EmbeddingsEngine(base_path=str(BASE_PATH))

                processed = 0
                for conv_id in missing[:20]:  # Process max 20 per run
                    try:
                        conv_file = CONVERSATIONS_PATH / f"{conv_id}.json"
                        if conv_file.exists():
                            with open(conv_file, 'r', encoding='utf-8') as f:
                                conversation = json.load(f)
                            engine.process_conversation(conversation)
                            processed += 1
                    except Exception as e:
                        print(f"    Error processing {conv_id}: {e}")

                report["actions"].append(f"Processed {processed} missing embeddings")
                print(f"  DONE: Processed {processed} conversations")
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        print("  OK: All conversations have embeddings")

    # 4. Check facts sync
    print("\n[4/5] Checking facts synchronization...")
    is_synced, facts_dir, facts_jsonl = check_facts_sync()
    report["checks"]["facts"] = {
        "synced": is_synced,
        "in_directory": facts_dir,
        "in_jsonl": facts_jsonl
    }

    if not is_synced:
        report["issues"].append(f"Facts not synced: {facts_dir} in dir, {facts_jsonl} in JSONL")
        print(f"  OUT OF SYNC: {facts_dir} files, {facts_jsonl} in JSONL")

        if not check_only:
            print("  Running consolidation...")
            try:
                from consolidate_facts import consolidate_facts
                consolidate_facts()
                report["actions"].append("Consolidated facts")
                print("  DONE: Facts consolidated")
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        print(f"  OK: {facts_jsonl} facts in JSONL")

    # 5. Summary
    print("\n[5/5] Generating summary...")
    report["checks"]["overall"] = "HEALTHY" if not report["issues"] else "DEGRADED"

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Status: {report['checks']['overall']}")
    print(f"Issues: {len(report['issues'])}")
    print(f"Actions taken: {len(report['actions'])}")

    if report["issues"]:
        print("\nIssues found:")
        for issue in report["issues"]:
            print(f"  - {issue}")

    if report["actions"]:
        print("\nActions taken:")
        for action in report["actions"]:
            print(f"  - {action}")

    # Send notification
    if not quiet:
        if report["issues"]:
            notify_issue(
                "health_degraded",
                f"{len(report['issues'])} issues found. {len(report['actions'])} fixed."
            )
        elif report["actions"]:
            notify_issue(
                "maintenance_complete",
                f"Maintenance complete. {len(report['actions'])} actions taken."
            )

    # Save log
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_PATH / f"maintenance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nLog saved: {log_file}")

    return report


if __name__ == "__main__":
    check_only = "--check" in sys.argv
    quiet = "--quiet" in sys.argv

    if check_only:
        print("[CHECK MODE - No changes will be made]\n")

    run_maintenance(check_only=check_only, quiet=quiet)
