"""
Usage Tracking System
Track access patterns for intelligent decay.

Features:
- Track last_accessed timestamp for facts and conversations
- Track access_count for frequency analysis
- Golden facts tier for permanent retention
- Usage-based decay decisions

Author: Claude
Created: 2026-01-18
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Golden facts are NEVER decayed - they're critical knowledge
GOLDEN_FACTS_FILE = "golden_facts.json"

# Access tracking settings
FREQUENT_ACCESS_THRESHOLD = 5      # Access count to be considered "frequently used"
RECENT_ACCESS_DAYS = 30            # Days to consider an access "recent"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class UsageRecord:
    """Usage tracking for a fact or conversation."""
    item_id: str
    item_type: str                  # "fact" or "conversation"
    first_accessed: str             # ISO timestamp
    last_accessed: str              # ISO timestamp
    access_count: int = 1
    access_history: List[str] = field(default_factory=list)  # Last 10 access timestamps

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'UsageRecord':
        return cls(**data)


# =============================================================================
# Usage Tracker
# =============================================================================

class UsageTracker:
    """
    Tracks access patterns for facts and conversations.
    Used to make intelligent decay decisions.
    """

    def __init__(self, base_path: Path):
        """
        Initialize usage tracker.

        Args:
            base_path: Base AI_MEMORY directory
        """
        self.base_path = Path(base_path)
        self.usage_dir = self.base_path / "usage"
        self.usage_dir.mkdir(parents=True, exist_ok=True)

        # Usage index file - maps item_id to usage record
        self.usage_index_file = self.usage_dir / "usage_index.json"

        # Golden facts file - items that should never decay
        self.golden_facts_file = self.usage_dir / GOLDEN_FACTS_FILE

        # Load existing data
        self._usage_index: Dict[str, UsageRecord] = {}
        self._golden_items: Set[str] = set()
        self._load()

    def _load(self):
        """Load usage index and golden facts from disk."""
        # Load usage index
        if self.usage_index_file.exists():
            try:
                with open(self.usage_index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._usage_index = {
                        k: UsageRecord.from_dict(v) for k, v in data.items()
                    }
                logger.debug(f"Loaded {len(self._usage_index)} usage records")
            except Exception as e:
                logger.error(f"Error loading usage index: {e}")
                self._usage_index = {}

        # Load golden facts
        if self.golden_facts_file.exists():
            try:
                with open(self.golden_facts_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._golden_items = set(data.get("items", []))
                logger.debug(f"Loaded {len(self._golden_items)} golden items")
            except Exception as e:
                logger.error(f"Error loading golden facts: {e}")
                self._golden_items = set()

    def _save_usage_index(self):
        """Save usage index to disk."""
        try:
            data = {k: v.to_dict() for k, v in self._usage_index.items()}
            with open(self.usage_index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving usage index: {e}")

    def _save_golden_facts(self):
        """Save golden facts to disk."""
        try:
            data = {
                "items": list(self._golden_items),
                "updated_at": datetime.now().isoformat(),
                "count": len(self._golden_items)
            }
            with open(self.golden_facts_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving golden facts: {e}")

    # -------------------------------------------------------------------------
    # Core Tracking Methods
    # -------------------------------------------------------------------------

    def record_access(self, item_id: str, item_type: str = "fact") -> UsageRecord:
        """
        Record an access to an item.

        Args:
            item_id: The fact or conversation ID
            item_type: "fact" or "conversation"

        Returns:
            Updated UsageRecord
        """
        now = datetime.now().isoformat()

        if item_id in self._usage_index:
            record = self._usage_index[item_id]
            record.last_accessed = now
            record.access_count += 1

            # Keep last 10 accesses in history
            record.access_history.append(now)
            if len(record.access_history) > 10:
                record.access_history = record.access_history[-10:]
        else:
            record = UsageRecord(
                item_id=item_id,
                item_type=item_type,
                first_accessed=now,
                last_accessed=now,
                access_count=1,
                access_history=[now]
            )
            self._usage_index[item_id] = record

        self._save_usage_index()
        return record

    def get_usage(self, item_id: str) -> Optional[UsageRecord]:
        """Get usage record for an item."""
        return self._usage_index.get(item_id)

    def get_last_accessed(self, item_id: str) -> Optional[str]:
        """Get last accessed timestamp for an item."""
        record = self._usage_index.get(item_id)
        return record.last_accessed if record else None

    def get_access_count(self, item_id: str) -> int:
        """Get access count for an item."""
        record = self._usage_index.get(item_id)
        return record.access_count if record else 0

    # -------------------------------------------------------------------------
    # Golden Facts Management
    # -------------------------------------------------------------------------

    def mark_golden(self, item_id: str, reason: Optional[str] = None) -> bool:
        """
        Mark an item as golden (never decay).

        Args:
            item_id: The item to mark as golden
            reason: Optional reason for marking as golden

        Returns:
            True if successful
        """
        self._golden_items.add(item_id)
        self._save_golden_facts()

        # Log the action
        logger.info(f"Marked {item_id} as golden: {reason or 'no reason given'}")

        return True

    def unmark_golden(self, item_id: str) -> bool:
        """Remove golden status from an item."""
        if item_id in self._golden_items:
            self._golden_items.remove(item_id)
            self._save_golden_facts()
            return True
        return False

    def is_golden(self, item_id: str) -> bool:
        """Check if an item is marked as golden."""
        return item_id in self._golden_items

    def get_golden_items(self) -> List[str]:
        """Get all golden item IDs."""
        return list(self._golden_items)

    # -------------------------------------------------------------------------
    # Decay Decision Support
    # -------------------------------------------------------------------------

    def is_frequently_accessed(self, item_id: str) -> bool:
        """Check if an item is frequently accessed."""
        return self.get_access_count(item_id) >= FREQUENT_ACCESS_THRESHOLD

    def is_recently_accessed(self, item_id: str, days: int = RECENT_ACCESS_DAYS) -> bool:
        """Check if an item was accessed within the given days."""
        last = self.get_last_accessed(item_id)
        if not last:
            return False

        try:
            last_dt = datetime.fromisoformat(last)
            cutoff = datetime.now() - timedelta(days=days)
            return last_dt > cutoff
        except:
            return False

    def should_protect_from_decay(self, item_id: str) -> tuple[bool, str]:
        """
        Determine if an item should be protected from decay.

        Returns:
            (should_protect, reason)
        """
        # Golden items are always protected
        if self.is_golden(item_id):
            return True, "golden_fact"

        # Frequently accessed items are protected
        if self.is_frequently_accessed(item_id):
            return True, f"frequently_accessed (count={self.get_access_count(item_id)})"

        # Recently accessed items are protected
        if self.is_recently_accessed(item_id):
            return True, "recently_accessed"

        return False, "no_protection"

    def get_decay_candidates(
        self,
        item_type: str = "fact",
        min_age_days: int = 180,
        max_access_count: int = 2
    ) -> List[str]:
        """
        Get items that are candidates for decay.

        Args:
            item_type: "fact" or "conversation"
            min_age_days: Minimum age in days to consider for decay
            max_access_count: Maximum access count to consider for decay

        Returns:
            List of item IDs that are decay candidates
        """
        candidates = []
        cutoff = datetime.now() - timedelta(days=min_age_days)

        for item_id, record in self._usage_index.items():
            if record.item_type != item_type:
                continue

            # Skip golden items
            if self.is_golden(item_id):
                continue

            # Skip frequently accessed
            if record.access_count > max_access_count:
                continue

            # Check age
            try:
                first_dt = datetime.fromisoformat(record.first_accessed)
                if first_dt < cutoff:
                    candidates.append(item_id)
            except:
                continue

        return candidates

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get usage tracking statistics."""
        fact_records = [r for r in self._usage_index.values() if r.item_type == "fact"]
        conv_records = [r for r in self._usage_index.values() if r.item_type == "conversation"]

        stats = {
            "total_tracked": len(self._usage_index),
            "facts_tracked": len(fact_records),
            "conversations_tracked": len(conv_records),
            "golden_items": len(self._golden_items),
            "frequently_accessed": sum(
                1 for r in self._usage_index.values()
                if r.access_count >= FREQUENT_ACCESS_THRESHOLD
            ),
        }

        if fact_records:
            stats["fact_avg_access_count"] = round(
                sum(r.access_count for r in fact_records) / len(fact_records), 2
            )

        if conv_records:
            stats["conv_avg_access_count"] = round(
                sum(r.access_count for r in conv_records) / len(conv_records), 2
            )

        return stats

    def get_top_accessed(self, limit: int = 20, item_type: Optional[str] = None) -> List[Dict]:
        """
        Get the most accessed items.

        Args:
            limit: Maximum number of items to return
            item_type: Filter by type ("fact" or "conversation"), or None for all

        Returns:
            List of usage records sorted by access count descending
        """
        records = list(self._usage_index.values())

        if item_type:
            records = [r for r in records if r.item_type == item_type]

        records.sort(key=lambda r: r.access_count, reverse=True)

        return [r.to_dict() for r in records[:limit]]

    def get_never_accessed(self, item_type: Optional[str] = None) -> List[str]:
        """
        Get items that have never been accessed (not in usage index).
        Note: This returns items we know about but have 0 or 1 access.

        For truly never-accessed items, compare with facts/conversations directories.
        """
        records = list(self._usage_index.values())

        if item_type:
            records = [r for r in records if r.item_type == item_type]

        # Items with only 1 access (the initial save) are effectively never "used"
        return [r.item_id for r in records if r.access_count <= 1]


# =============================================================================
# Module-level convenience functions
# =============================================================================

_tracker: Optional[UsageTracker] = None

def get_tracker(base_path: Optional[Path] = None) -> UsageTracker:
    """Get or create the global usage tracker instance."""
    global _tracker
    if _tracker is None:
        if base_path is None:
            from config import AI_MEMORY_BASE
            base_path = AI_MEMORY_BASE
        _tracker = UsageTracker(base_path)
    return _tracker


def record_fact_access(fact_id: str) -> UsageRecord:
    """Convenience function to record fact access."""
    return get_tracker().record_access(fact_id, "fact")


def record_conversation_access(conv_id: str) -> UsageRecord:
    """Convenience function to record conversation access."""
    return get_tracker().record_access(conv_id, "conversation")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    # Initialize with config
    from config import AI_MEMORY_BASE
    tracker = UsageTracker(AI_MEMORY_BASE)

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "stats":
            print("Usage Tracking Statistics:")
            print(json.dumps(tracker.get_stats(), indent=2))

        elif cmd == "top":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
            print(f"Top {limit} Most Accessed Items:")
            for item in tracker.get_top_accessed(limit):
                print(f"  {item['item_id']}: {item['access_count']} accesses")

        elif cmd == "golden":
            print("Golden Items (never decay):")
            for item_id in tracker.get_golden_items():
                print(f"  {item_id}")

        elif cmd == "mark-golden" and len(sys.argv) > 2:
            item_id = sys.argv[2]
            reason = sys.argv[3] if len(sys.argv) > 3 else None
            tracker.mark_golden(item_id, reason)
            print(f"Marked {item_id} as golden")

        elif cmd == "decay-candidates":
            print("Decay Candidates (old + rarely accessed):")
            for item_id in tracker.get_decay_candidates():
                print(f"  {item_id}")

        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python usage_tracker.py [stats|top|golden|mark-golden|decay-candidates]")
    else:
        print("Usage Tracker initialized")
        print(json.dumps(tracker.get_stats(), indent=2))
