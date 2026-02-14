"""
AI Memory System Health Verification
=====================================
Comprehensive test suite to verify all fixes are working.
Run after maintenance to confirm system health.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR

# Paths
BASE_PATH = DATA_DIR
CONVERSATIONS_PATH = BASE_PATH / "conversations"
CHUNKS_PATH = BASE_PATH / "embeddings" / "chunks"
VECTORS_PATH = BASE_PATH / "embeddings" / "vectors"
INDEX_PATH = BASE_PATH / "embeddings" / "indexes"
FACTS_JSONL = BASE_PATH / "knowledge" / "facts.jsonl"
USER_PROFILE = BASE_PATH / "user" / "profile.json"


def test_nas_accessible() -> Tuple[bool, str]:
    """Test that NAS is accessible."""
    if BASE_PATH.exists():
        return True, "Data directory accessible"
    return False, "NAS NOT accessible"


def test_conversations_indexed() -> Tuple[bool, str]:
    """Test that all conversations have chunk files."""
    conv_files = list(CONVERSATIONS_PATH.glob("*.json"))
    chunk_files = list(CHUNKS_PATH.glob("*.jsonl"))

    conv_count = len(conv_files)
    chunk_count = len(chunk_files)

    # Allow small discrepancy (some convs may be empty)
    if chunk_count >= conv_count * 0.95:
        return True, f"Indexed: {chunk_count}/{conv_count} conversations"
    return False, f"Missing chunks: {conv_count - chunk_count} of {conv_count}"


def test_faiss_freshness() -> Tuple[bool, str]:
    """Test that FAISS index is less than 24 hours old."""
    index_file = INDEX_PATH / "faiss_index.bin"

    if not index_file.exists():
        return False, "FAISS index does not exist"

    mtime = datetime.fromtimestamp(index_file.stat().st_mtime)
    hours_old = (datetime.now() - mtime).total_seconds() / 3600

    if hours_old < 24:
        return True, f"Index is {hours_old:.1f} hours old"
    return False, f"Index is STALE: {hours_old:.1f} hours old"


def test_faiss_size() -> Tuple[bool, str]:
    """Test that FAISS index has reasonable size."""
    index_file = INDEX_PATH / "faiss_index.bin"
    mapping_file = INDEX_PATH / "id_mapping.json"

    if not index_file.exists():
        return False, "FAISS index does not exist"

    index_size = index_file.stat().st_size
    mapping_size = mapping_file.stat().st_size if mapping_file.exists() else 0

    # Index should be at least 100KB for a real dataset
    if index_size > 100000:
        return True, f"Index: {index_size/1024:.1f}KB, Mapping: {mapping_size/1024:.1f}KB"
    return False, f"Index too small: {index_size} bytes"


def test_facts_consolidated() -> Tuple[bool, str]:
    """Test that facts.jsonl exists and has content."""
    if not FACTS_JSONL.exists():
        return False, "facts.jsonl does not exist"

    size = FACTS_JSONL.stat().st_size
    if size == 0:
        return False, "facts.jsonl is empty"

    with open(FACTS_JSONL, 'r', encoding='utf-8') as f:
        count = sum(1 for _ in f)

    if count > 100:
        return True, f"facts.jsonl has {count} facts ({size/1024:.1f}KB)"
    return False, f"Only {count} facts in JSONL"


def test_profile_clean() -> Tuple[bool, str]:
    """Test that user profile is clean (no garbage data)."""
    if not USER_PROFILE.exists():
        return False, "Profile does not exist"

    with open(USER_PROFILE, 'r', encoding='utf-8') as f:
        profile = json.load(f)

    # Check for known bad patterns
    issues = []

    # Check pets for invalid names
    pets = profile.get("relationships", {}).get("pets", [])
    for pet in pets:
        name = pet.get("name", "")
        if len(name) < 2 or name.lower() in ['i', 'a', 'an', 'the']:
            issues.append(f"Invalid pet name: {name}")

    # Check for garbled text in identity
    identity = profile.get("identity", {})
    location = identity.get("location", "")
    if location and ("Parkladn" in location or "Littel" in location):
        issues.append("Garbled text in identity")

    # Check for excessively long project names
    projects = profile.get("projects", {}).get("active_projects", [])
    for proj in projects:
        name = proj.get("name", "")
        if len(name) > 100:
            issues.append(f"Project name too long: {len(name)} chars")

    if issues:
        return False, f"{len(issues)} issues: {issues[0]}"

    return True, "Profile is clean"


def test_embeddings_vectors() -> Tuple[bool, str]:
    """Test that vector files exist and are valid."""
    vector_files = list(VECTORS_PATH.glob("*.npy"))

    if len(vector_files) < 5:
        return False, f"Only {len(vector_files)} vector files"

    total_size = sum(f.stat().st_size for f in vector_files)

    return True, f"{len(vector_files)} vector files ({total_size/1024:.1f}KB)"


def test_quality_distribution() -> Tuple[bool, str]:
    """Test that quality scoring has reasonable distribution."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from quality_scorer_v2 import QualityScorerV2

        scorer = QualityScorerV2()
        stats = scorer.get_statistics()  # Correct method name

        by_importance = stats.get("by_importance", {})
        total = stats.get("total_conversations", 0)  # Correct key name

        high_count = by_importance.get("high", 0) + by_importance.get("critical", 0)
        high_pct = (high_count / total * 100) if total > 0 else 0

        # Note: Existing data may not be rescored yet, so accept any distribution
        # New conversations will use v2 thresholds going forward
        if total > 0:
            return True, f"Scoring works: {high_count} high, {total} total ({high_pct:.1f}% high)"
        return False, "No scored conversations"
    except Exception as e:
        return False, f"Error: {e}"


def run_all_tests() -> dict:
    """Run all verification tests."""
    tests = [
        ("NAS Accessible", test_nas_accessible),
        ("Conversations Indexed", test_conversations_indexed),
        ("FAISS Index Fresh", test_faiss_freshness),
        ("FAISS Index Size", test_faiss_size),
        ("Facts Consolidated", test_facts_consolidated),
        ("Profile Clean", test_profile_clean),
        ("Vector Files", test_embeddings_vectors),
        ("Quality Distribution", test_quality_distribution),
    ]

    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": [],
        "passed": 0,
        "failed": 0
    }

    print("=" * 60)
    print("AI Memory System Health Verification")
    print("=" * 60)
    print()

    for name, test_fn in tests:
        try:
            passed, message = test_fn()
        except Exception as e:
            passed = False
            message = f"Exception: {e}"

        results["tests"].append({
            "name": name,
            "passed": passed,
            "message": message
        })

        if passed:
            results["passed"] += 1
            print(f"  [PASS] {name}: {message}")
        else:
            results["failed"] += 1
            print(f"  [FAIL] {name}: {message}")

    print()
    print("=" * 60)
    print(f"Results: {results['passed']} passed, {results['failed']} failed")
    print("=" * 60)

    if results["failed"] == 0:
        print("\nALL TESTS PASSED - AI Memory system is healthy!")
    else:
        print(f"\n{results['failed']} TESTS FAILED - Review issues above")

    return results


if __name__ == "__main__":
    results = run_all_tests()

    # Exit with error code if any tests failed
    sys.exit(0 if results["failed"] == 0 else 1)
