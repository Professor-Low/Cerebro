#!/usr/bin/env python3
"""
Decay Pipeline - Compresses and eventually removes old memories.
Decay is a feature, not a bug. Forgetting the irrelevant is as important as remembering the important.

Phase 4 of Brain Evolution - Extended to conversations and facts.

Summary Decay Rules:
- 0-7 days:   Full summary (200 tokens max)
- 7-30 days:  Compressed (50 tokens max)
- 30-90 days: Minimal (10 tokens) or merged
- 90+ days:   Archived or deleted

Conversation Decay Rules:
- 0-90 days:   Full storage (all fields)
- 90-180 days: Compressed (keep extracted_data, drop raw messages)
- 180-365 days: Minimal (keep facts + metadata only)
- 365+ days:   Cold storage

Fact Decay Rules:
- Low confidence (<0.4) + never accessed + >180 days: Archive
- Superseded + >365 days: Delete
- Golden facts: NEVER decay

Corrections are NEVER decayed - they're sacred.

Run: Daily via cron or on-startup check.

Author: Claude
Updated: 2026-01-18 (Phase 4)
"""

import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent))
from config import AI_MEMORY_BASE, ARCHIVE_DIR, CONVERSATIONS_DIR, DECAY_STATE_FILE, LOG_DIR, SUMMARIES_DIR

# Debug log path (from config)
DEBUG_LOG = LOG_DIR / "decay_pipeline_debug.log"

# Directories
FACTS_DIR = AI_MEMORY_BASE / "facts"
COLD_STORAGE_DIR = AI_MEMORY_BASE / "cold_storage"
DECAY_LOG_FILE = AI_MEMORY_BASE / "logs" / "decay_actions.jsonl"

# Summary decay rules (existing)
SUMMARY_DECAY_RULES = {
    7: {"max_words": 50, "level": "full"},      # 0-7 days: ~65 tokens
    30: {"max_words": 15, "level": "brief"},    # 7-30 days: ~20 tokens
    90: {"max_words": 5, "level": "minimal"},   # 30-90 days: ~7 tokens
    365: {"max_words": 0, "level": "archive"},  # 90+ days: archive/delete
}

# Conversation decay rules (new)
CONVERSATION_DECAY_RULES = {
    90: "full",           # 0-90 days: keep everything
    180: "compressed",    # 90-180 days: drop raw messages, keep extracted
    365: "minimal",       # 180-365 days: keep facts + metadata only
    730: "cold_storage",  # 365+ days: move to cold storage
}

# Fact decay thresholds (Phase 3.3 - More aggressive config)
FACT_DECAY_CONFIG = {
    "low_confidence_threshold": 0.50,  # Raised from 0.40 - catch more low-quality facts
    "min_age_days": 90,                # Reduced from 180 - decay faster
    "max_access_count": 1,             # Max accesses to be "never used"
    "never_accessed_days": 60,         # NEW: Archive facts never accessed after 60 days
    "superseded_delete_days": 365,     # Delete superseded after this many days
}

# Keep alias for backward compat
DECAY_RULES = SUMMARY_DECAY_RULES


def log_debug(msg: str):
    """Write debug info to log file."""
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(f"{datetime.now().isoformat()} - {msg}\n")
    except:
        pass


def load_decay_state() -> Dict[str, Any]:
    """Load decay pipeline state (last run time, stats)."""
    if DECAY_STATE_FILE.exists():
        try:
            with open(DECAY_STATE_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {"last_run": None, "runs": 0, "total_compressed": 0, "total_archived": 0}


def save_decay_state(state: Dict[str, Any]):
    """Save decay pipeline state."""
    try:
        DECAY_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DECAY_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log_debug(f"Error saving decay state: {e}")


def should_run() -> bool:
    """Check if decay pipeline should run (once per day max)."""
    state = load_decay_state()
    last_run = state.get("last_run")

    if not last_run:
        return True

    try:
        last_run_dt = datetime.fromisoformat(last_run)
        hours_since = (datetime.now() - last_run_dt).total_seconds() / 3600
        return hours_since >= 24
    except:
        return True


def get_age_days(timestamp_str: str) -> int:
    """Get age in days from ISO timestamp string."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return (datetime.now() - dt.replace(tzinfo=None)).days
    except:
        return 0


def compress_summary(summary: str, max_words: int) -> str:
    """Compress summary to max_words."""
    if max_words <= 0:
        return ""

    words = summary.split()
    if len(words) <= max_words:
        return summary

    return " ".join(words[:max_words]) + "..."


def process_summary_file(file_path: Path, stats: Dict[str, int]) -> bool:
    """
    Process a single summary file, applying decay rules.
    Returns True if file was modified.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            summaries = json.load(f)
    except Exception as e:
        log_debug(f"Error reading {file_path}: {e}")
        return False

    modified = False

    for summary in summaries:
        timestamp = summary.get("timestamp", "")
        age_days = get_age_days(timestamp)

        # Skip already minimally compressed
        current_words = summary.get("word_count", len(summary.get("summary", "").split()))
        decay_level = summary.get("decay_level", "full")

        # Determine target decay level based on age
        target_words = 50
        target_level = "full"

        for threshold_days, rule in sorted(DECAY_RULES.items()):
            if age_days >= threshold_days:
                target_words = rule["max_words"]
                target_level = rule["level"]

        # Skip if already at or below target
        if current_words <= target_words and decay_level == target_level:
            continue

        # Apply compression
        if target_level == "archive":
            # Mark for archival
            summary["archived"] = True
            summary["archived_at"] = datetime.now().isoformat()
            stats["archived"] += 1
            modified = True
        elif target_words > 0 and current_words > target_words:
            # Compress
            summary["summary"] = compress_summary(summary.get("summary", ""), target_words)
            summary["word_count"] = target_words
            summary["decay_level"] = target_level
            summary["decayed_at"] = datetime.now().isoformat()
            stats["compressed"] += 1
            modified = True

    if modified:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log_debug(f"Error writing {file_path}: {e}")
            return False

    return modified


def archive_old_summaries(stats: Dict[str, int]):
    """Move archived summaries to archive directory."""
    if not SUMMARIES_DIR.exists():
        return

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in SUMMARIES_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                summaries = json.load(f)

            # Separate archived from active
            active = []
            archived = []

            for s in summaries:
                if s.get("archived"):
                    archived.append(s)
                else:
                    active.append(s)

            if archived:
                # Save archived to archive directory
                archive_file = ARCHIVE_DIR / file_path.name
                existing_archive = []
                if archive_file.exists():
                    with open(archive_file, "r", encoding="utf-8") as f:
                        existing_archive = json.load(f)

                existing_archive.extend(archived)

                with open(archive_file, "w", encoding="utf-8") as f:
                    json.dump(existing_archive, f, indent=2, ensure_ascii=False)

                # Update original file with only active summaries
                if active:
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(active, f, indent=2, ensure_ascii=False)
                else:
                    # No active summaries left, delete file
                    file_path.unlink()

                stats["files_archived"] += 1

        except Exception as e:
            log_debug(f"Error archiving {file_path}: {e}")


def delete_very_old_archives(max_age_days: int = 365) -> int:
    """Delete archives older than max_age_days. Returns count deleted."""
    if not ARCHIVE_DIR.exists():
        return 0

    deleted = 0
    cutoff = datetime.now() - timedelta(days=max_age_days)

    for file_path in ARCHIVE_DIR.glob("*.json"):
        try:
            # Check file modification time
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_mtime < cutoff:
                file_path.unlink()
                deleted += 1
                log_debug(f"Deleted old archive: {file_path}")
        except Exception as e:
            log_debug(f"Error deleting {file_path}: {e}")

    return deleted


# =============================================================================
# Decay Action Logging
# =============================================================================

@dataclass
class DecayAction:
    """Record of a decay action for audit trail."""
    timestamp: str
    action_type: str        # compressed, archived, cold_storage, deleted
    item_type: str          # summary, conversation, fact
    item_id: str
    reason: str
    details: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def log_decay_action(action: DecayAction):
    """Log a decay action to the audit log."""
    try:
        DECAY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(DECAY_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(action.to_dict()) + "\n")
    except Exception as e:
        log_debug(f"Error logging decay action: {e}")


# =============================================================================
# Conversation Decay (Phase 4)
# =============================================================================

def get_conversation_decay_level(age_days: int) -> str:
    """Determine decay level for a conversation based on age."""
    for threshold, level in sorted(CONVERSATION_DECAY_RULES.items()):
        if age_days < threshold:
            return level
    return "cold_storage"


def compress_conversation(conv: Dict, level: str) -> Dict:
    """
    Compress a conversation to the specified decay level.

    Levels:
    - full: Keep everything
    - compressed: Drop raw messages, keep extracted_data
    - minimal: Keep only facts, metadata, and basic info
    """
    if level == "full":
        return conv

    result = {
        "id": conv.get("id"),
        "timestamp": conv.get("timestamp"),
        "type": conv.get("type", "conversation"),
        "decay_level": level,
        "decayed_at": datetime.now().isoformat(),
    }

    if level == "compressed":
        # Keep extracted_data, metadata, search_index
        # Drop: messages, content (raw text)
        result["extracted_data"] = conv.get("extracted_data", {})
        result["metadata"] = conv.get("metadata", {})
        result["search_index"] = conv.get("search_index", {})
        # Keep a summary if available
        if "search_index" in conv and "summary" in conv["search_index"]:
            result["summary"] = conv["search_index"]["summary"]

    elif level == "minimal":
        # Keep only essential metadata
        result["metadata"] = {
            "topics": conv.get("metadata", {}).get("topics", []),
            "tags": conv.get("metadata", {}).get("tags", []),
            "message_count": conv.get("metadata", {}).get("message_count", 0),
        }
        # Keep extracted facts
        extracted = conv.get("extracted_data", {})
        result["facts"] = extracted.get("facts", [])
        result["entities"] = extracted.get("entities", {})
        # Keep brief summary
        if "search_index" in conv and "summary" in conv["search_index"]:
            result["summary"] = conv["search_index"]["summary"][:100]

    result["original_size"] = len(json.dumps(conv))
    result["compressed_size"] = len(json.dumps(result))

    return result


def process_conversation_decay(stats: Dict[str, int]) -> None:
    """
    Process all conversations for decay.
    Compresses old conversations and moves very old ones to cold storage.
    """
    if not CONVERSATIONS_DIR.exists():
        return

    # Ensure cold storage exists
    cold_conv_dir = COLD_STORAGE_DIR / "conversations"
    cold_conv_dir.mkdir(parents=True, exist_ok=True)

    for file_path in CONVERSATIONS_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                conv = json.load(f)

            timestamp = conv.get("timestamp", "")
            age_days = get_age_days(timestamp)
            current_level = conv.get("decay_level", "full")
            target_level = get_conversation_decay_level(age_days)

            # Skip if already at target level
            if current_level == target_level:
                continue

            # Skip if already more decayed than target
            level_order = ["full", "compressed", "minimal", "cold_storage"]
            if level_order.index(current_level) >= level_order.index(target_level):
                continue

            conv_id = conv.get("id", file_path.stem)

            if target_level == "cold_storage":
                # Move to cold storage
                dest = cold_conv_dir / file_path.name
                shutil.move(str(file_path), str(dest))
                stats["conversations_cold_storage"] += 1

                log_decay_action(DecayAction(
                    timestamp=datetime.now().isoformat(),
                    action_type="cold_storage",
                    item_type="conversation",
                    item_id=conv_id,
                    reason=f"Age {age_days} days exceeds threshold",
                    details={"age_days": age_days}
                ))
                log_debug(f"Moved conversation to cold storage: {conv_id}")

            else:
                # Compress in place
                compressed = compress_conversation(conv, target_level)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(compressed, f, indent=2, ensure_ascii=False)
                stats["conversations_compressed"] += 1

                log_decay_action(DecayAction(
                    timestamp=datetime.now().isoformat(),
                    action_type="compressed",
                    item_type="conversation",
                    item_id=conv_id,
                    reason=f"Age {age_days} days - compressed to {target_level}",
                    details={
                        "age_days": age_days,
                        "from_level": current_level,
                        "to_level": target_level,
                        "size_reduction": compressed.get("original_size", 0) - compressed.get("compressed_size", 0)
                    }
                ))
                log_debug(f"Compressed conversation {conv_id} to {target_level}")

        except Exception as e:
            log_debug(f"Error processing conversation {file_path}: {e}")
            stats["conversation_errors"] += 1


# =============================================================================
# Fact Decay (Phase 4)
# =============================================================================

def get_fact_confidence(fact_id: str) -> float:
    """Get confidence score for a fact from the confidence tracker."""
    try:
        from confidence_tracker import ConfidenceTracker
        tracker = ConfidenceTracker(AI_MEMORY_BASE)
        record = tracker.get_record(fact_id)
        if record:
            return record.current_confidence
    except Exception as e:
        log_debug(f"Error getting confidence for {fact_id}: {e}")
    return 0.5  # Default to medium confidence


def is_fact_superseded(fact: Dict) -> bool:
    """Check if a fact has been superseded."""
    return fact.get("superseded", False) or fact.get("metadata", {}).get("superseded", False)


def should_decay_fact(fact: Dict, usage_tracker=None) -> Tuple[bool, str]:
    """
    Determine if a fact should be decayed.

    Returns:
        (should_decay, reason)
    """
    fact_id = fact.get("id", "")
    timestamp = fact.get("timestamp", "")
    age_days = get_age_days(timestamp)

    # Check if it's a golden fact (never decay)
    if usage_tracker:
        if usage_tracker.is_golden(fact_id):
            return False, "golden_fact"

        # Check if frequently accessed
        if usage_tracker.is_frequently_accessed(fact_id):
            return False, "frequently_accessed"

        # Check if recently accessed
        if usage_tracker.is_recently_accessed(fact_id):
            return False, "recently_accessed"

    # Rule 1: Superseded facts older than threshold -> delete
    if is_fact_superseded(fact):
        if age_days >= FACT_DECAY_CONFIG["superseded_delete_days"]:
            return True, "superseded_old"
        return False, "superseded_recent"

    # Rule 2: Never accessed + older than threshold -> archive (Phase 3.3)
    never_accessed_days = FACT_DECAY_CONFIG.get("never_accessed_days", 60)
    if age_days >= never_accessed_days:
        access_count = 0
        if usage_tracker:
            access_count = usage_tracker.get_access_count(fact_id)
        if access_count == 0:
            return True, f"never_accessed_age_{age_days}"

    # Rule 3: Low confidence + old + rarely accessed -> archive
    min_age = FACT_DECAY_CONFIG["min_age_days"]
    if age_days < min_age:
        return False, "too_young"

    confidence = get_fact_confidence(fact_id)
    if confidence < FACT_DECAY_CONFIG["low_confidence_threshold"]:
        access_count = 0
        if usage_tracker:
            access_count = usage_tracker.get_access_count(fact_id)
        if access_count <= FACT_DECAY_CONFIG["max_access_count"]:
            return True, f"low_confidence_{confidence:.2f}_low_access_{access_count}"

    return False, "retained"


def process_fact_decay(stats: Dict[str, int]) -> None:
    """
    Process all facts for decay.
    Archives low-value facts and deletes old superseded ones.
    """
    if not FACTS_DIR.exists():
        return

    # Ensure cold storage exists
    cold_facts_dir = COLD_STORAGE_DIR / "facts"
    cold_facts_dir.mkdir(parents=True, exist_ok=True)

    # Try to get usage tracker
    usage_tracker = None
    try:
        from usage_tracker import UsageTracker
        usage_tracker = UsageTracker(AI_MEMORY_BASE)
    except Exception as e:
        log_debug(f"Could not load usage tracker: {e}")

    for file_path in FACTS_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                fact = json.load(f)

            fact_id = fact.get("id", file_path.stem)
            should_decay, reason = should_decay_fact(fact, usage_tracker)

            if not should_decay:
                continue

            if reason == "superseded_old":
                # Delete old superseded facts
                file_path.unlink()
                stats["facts_deleted"] += 1

                log_decay_action(DecayAction(
                    timestamp=datetime.now().isoformat(),
                    action_type="deleted",
                    item_type="fact",
                    item_id=fact_id,
                    reason=reason,
                    details={"age_days": get_age_days(fact.get("timestamp", ""))}
                ))
                log_debug(f"Deleted superseded fact: {fact_id}")

            else:
                # Archive low-value facts to cold storage
                dest = cold_facts_dir / file_path.name
                shutil.move(str(file_path), str(dest))
                stats["facts_archived"] += 1

                log_decay_action(DecayAction(
                    timestamp=datetime.now().isoformat(),
                    action_type="archived",
                    item_type="fact",
                    item_id=fact_id,
                    reason=reason,
                    details={
                        "age_days": get_age_days(fact.get("timestamp", "")),
                        "content_preview": fact.get("content", "")[:50]
                    }
                ))
                log_debug(f"Archived fact to cold storage: {fact_id}")

        except Exception as e:
            log_debug(f"Error processing fact {file_path}: {e}")
            stats["fact_errors"] += 1


# =============================================================================
# Main Decay Pipeline
# =============================================================================

def run_decay(
    force: bool = False,
    include_summaries: bool = True,
    include_conversations: bool = True,
    include_facts: bool = True
) -> Dict[str, Any]:
    """
    Run the decay pipeline.

    Args:
        force: Run even if already ran today
        include_summaries: Process summary decay (original behavior)
        include_conversations: Process conversation tier decay (Phase 4)
        include_facts: Process fact decay (Phase 4)

    Returns:
        Stats dict with counts of actions taken
    """
    log_debug("Decay pipeline started")

    if not force and not should_run():
        log_debug("Skipping - already ran within 24 hours")
        return {"skipped": True, "reason": "already_ran_today"}

    stats = {
        # Summary decay stats (original)
        "summaries_compressed": 0,
        "summaries_archived": 0,
        "files_processed": 0,
        "files_archived": 0,
        "files_deleted": 0,
        # Conversation decay stats (Phase 4)
        "conversations_compressed": 0,
        "conversations_cold_storage": 0,
        "conversation_errors": 0,
        # Fact decay stats (Phase 4)
        "facts_archived": 0,
        "facts_deleted": 0,
        "fact_errors": 0,
    }

    # Keep backward compat aliases
    stats["compressed"] = 0
    stats["archived"] = 0

    # 1. Process summary files (original behavior)
    if include_summaries and SUMMARIES_DIR.exists():
        for file_path in SUMMARIES_DIR.glob("*.json"):
            if process_summary_file(file_path, stats):
                stats["files_processed"] += 1
        stats["compressed"] = stats.get("compressed", 0) + stats["summaries_compressed"]
        stats["archived"] = stats.get("archived", 0) + stats["summaries_archived"]

        # Move archived summaries to archive directory
        archive_old_summaries(stats)

        # Delete very old archives (365+ days)
        stats["files_deleted"] = delete_very_old_archives(max_age_days=365)

    # 2. Process conversation decay (Phase 4)
    if include_conversations:
        log_debug("Processing conversation decay...")
        process_conversation_decay(stats)

    # 3. Process fact decay (Phase 4)
    if include_facts:
        log_debug("Processing fact decay...")
        process_fact_decay(stats)

    # Update state
    state = load_decay_state()
    state["last_run"] = datetime.now().isoformat()
    state["runs"] = state.get("runs", 0) + 1
    state["total_compressed"] = state.get("total_compressed", 0) + stats["compressed"]
    state["total_archived"] = state.get("total_archived", 0) + stats["archived"]

    # Add Phase 4 stats to state
    state["total_conversations_compressed"] = state.get("total_conversations_compressed", 0) + stats["conversations_compressed"]
    state["total_conversations_cold_storage"] = state.get("total_conversations_cold_storage", 0) + stats["conversations_cold_storage"]
    state["total_facts_archived"] = state.get("total_facts_archived", 0) + stats["facts_archived"]
    state["total_facts_deleted"] = state.get("total_facts_deleted", 0) + stats["facts_deleted"]

    save_decay_state(state)

    log_debug(f"Decay complete: {stats}")
    return stats


def get_decay_stats() -> Dict[str, Any]:
    """Get decay pipeline statistics."""
    state = load_decay_state()

    # Count summaries by decay level
    summary_levels = {"full": 0, "brief": 0, "minimal": 0, "archived": 0}

    if SUMMARIES_DIR.exists():
        for file_path in SUMMARIES_DIR.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    summaries = json.load(f)
                for s in summaries:
                    level = s.get("decay_level", "full")
                    summary_levels[level] = summary_levels.get(level, 0) + 1
            except:
                pass

    # Count archived summaries
    if ARCHIVE_DIR.exists():
        for file_path in ARCHIVE_DIR.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    summaries = json.load(f)
                summary_levels["archived"] += len(summaries)
            except:
                pass

    # Count conversations by decay level
    conv_levels = {"full": 0, "compressed": 0, "minimal": 0, "cold_storage": 0}

    if CONVERSATIONS_DIR.exists():
        for file_path in CONVERSATIONS_DIR.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    conv = json.load(f)
                level = conv.get("decay_level", "full")
                conv_levels[level] = conv_levels.get(level, 0) + 1
            except:
                pass

    # Count cold storage conversations
    cold_conv_dir = COLD_STORAGE_DIR / "conversations"
    if cold_conv_dir.exists():
        conv_levels["cold_storage"] = len(list(cold_conv_dir.glob("*.json")))

    # Count facts
    fact_counts = {"active": 0, "cold_storage": 0}
    if FACTS_DIR.exists():
        fact_counts["active"] = len(list(FACTS_DIR.glob("*.json")))
    cold_facts_dir = COLD_STORAGE_DIR / "facts"
    if cold_facts_dir.exists():
        fact_counts["cold_storage"] = len(list(cold_facts_dir.glob("*.json")))

    return {
        "last_run": state.get("last_run"),
        "total_runs": state.get("runs", 0),
        # Summary stats
        "summaries": {
            "total_compressed": state.get("total_compressed", 0),
            "total_archived": state.get("total_archived", 0),
            "by_level": summary_levels
        },
        # Conversation stats (Phase 4)
        "conversations": {
            "total_compressed": state.get("total_conversations_compressed", 0),
            "total_cold_storage": state.get("total_conversations_cold_storage", 0),
            "by_level": conv_levels
        },
        # Fact stats (Phase 4)
        "facts": {
            "total_archived": state.get("total_facts_archived", 0),
            "total_deleted": state.get("total_facts_deleted", 0),
            "counts": fact_counts
        },
        # Legacy format for backward compat
        "total_compressed": state.get("total_compressed", 0),
        "total_archived": state.get("total_archived", 0),
        "summaries_by_level": summary_levels
    }


def get_decay_preview(dry_run: bool = True) -> Dict[str, Any]:
    """
    Preview what would be decayed without actually doing it.

    Returns:
        Dict with lists of items that would be affected
    """
    preview = {
        "conversations": {
            "would_compress": [],
            "would_cold_storage": []
        },
        "facts": {
            "would_archive": [],
            "would_delete": []
        }
    }

    # Preview conversation decay
    if CONVERSATIONS_DIR.exists():
        for file_path in CONVERSATIONS_DIR.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    conv = json.load(f)

                timestamp = conv.get("timestamp", "")
                age_days = get_age_days(timestamp)
                current_level = conv.get("decay_level", "full")
                target_level = get_conversation_decay_level(age_days)

                if current_level != target_level:
                    level_order = ["full", "compressed", "minimal", "cold_storage"]
                    if level_order.index(current_level) < level_order.index(target_level):
                        entry = {
                            "id": conv.get("id", file_path.stem),
                            "age_days": age_days,
                            "from_level": current_level,
                            "to_level": target_level
                        }
                        if target_level == "cold_storage":
                            preview["conversations"]["would_cold_storage"].append(entry)
                        else:
                            preview["conversations"]["would_compress"].append(entry)
            except:
                pass

    # Preview fact decay
    usage_tracker = None
    try:
        from usage_tracker import UsageTracker
        usage_tracker = UsageTracker(AI_MEMORY_BASE)
    except:
        pass

    if FACTS_DIR.exists():
        for file_path in FACTS_DIR.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    fact = json.load(f)

                fact_id = fact.get("id", file_path.stem)
                should_decay, reason = should_decay_fact(fact, usage_tracker)

                if should_decay:
                    entry = {
                        "id": fact_id,
                        "age_days": get_age_days(fact.get("timestamp", "")),
                        "reason": reason,
                        "content_preview": fact.get("content", "")[:50]
                    }
                    if reason == "superseded_old":
                        preview["facts"]["would_delete"].append(entry)
                    else:
                        preview["facts"]["would_archive"].append(entry)
            except:
                pass

    preview["summary"] = {
        "conversations_to_compress": len(preview["conversations"]["would_compress"]),
        "conversations_to_cold_storage": len(preview["conversations"]["would_cold_storage"]),
        "facts_to_archive": len(preview["facts"]["would_archive"]),
        "facts_to_delete": len(preview["facts"]["would_delete"])
    }

    return preview


if __name__ == "__main__":
    import sys

    def print_help():
        print("Decay Pipeline - Phase 4")
        print("Usage: python decay_pipeline.py [command]")
        print("")
        print("Commands:")
        print("  --force       Run decay even if already ran today")
        print("  --stats       Show decay statistics")
        print("  --preview     Preview what would be decayed (dry run)")
        print("  --summaries   Only process summary decay")
        print("  --conversations  Only process conversation decay")
        print("  --facts       Only process fact decay")
        print("  --help        Show this help message")

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "--help":
            print_help()
            sys.exit(0)

        elif cmd == "--force":
            print("Running decay pipeline (forced)...")
            result = run_decay(force=True)
            print(json.dumps(result, indent=2))

        elif cmd == "--stats":
            print("Decay pipeline stats:")
            stats = get_decay_stats()
            print(json.dumps(stats, indent=2))

        elif cmd == "--preview":
            print("Decay preview (dry run):")
            preview = get_decay_preview()
            print(json.dumps(preview, indent=2))

        elif cmd == "--summaries":
            print("Running summary decay only...")
            result = run_decay(force=True, include_conversations=False, include_facts=False)
            print(json.dumps(result, indent=2))

        elif cmd == "--conversations":
            print("Running conversation decay only...")
            result = run_decay(force=True, include_summaries=False, include_facts=False)
            print(json.dumps(result, indent=2))

        elif cmd == "--facts":
            print("Running fact decay only...")
            result = run_decay(force=True, include_summaries=False, include_conversations=False)
            print(json.dumps(result, indent=2))

        else:
            print(f"Unknown command: {cmd}")
            print_help()
            sys.exit(1)
    else:
        print("Running decay pipeline...")
        result = run_decay()
        print(json.dumps(result, indent=2))
