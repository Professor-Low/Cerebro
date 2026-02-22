"""
Preference Evolution System
Makes preferences evolve over time.

Features:
- Timestamps on all preferences (created_at, last_reinforced)
- Decay detection (90 days without reinforcement → stale)
- Contradiction detection and supersession
- Bounded lists with FIFO eviction

Author: Claude
Created: 2026-01-18
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Configuration
DEFAULT_STALE_DAYS = 90  # Days without reinforcement before marked stale
DEFAULT_MAX_PREFERENCES = 50  # Max preferences per category
SIMILARITY_THRESHOLD = 0.7  # For contradiction detection


@dataclass
class EvolvedPreference:
    """A preference with evolution metadata."""
    content: str
    category: str  # communication_style, workflow, technical
    positive: bool  # True = prefers, False = dislikes
    created_at: str  # ISO timestamp
    last_reinforced: str  # ISO timestamp
    reinforcement_count: int = 1
    stale: bool = False
    superseded: bool = False
    superseded_by: Optional[str] = None  # content of superseding preference
    superseded_at: Optional[str] = None
    source: str = "extracted"  # extracted, manual, inferred
    confidence: float = 0.7

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'EvolvedPreference':
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_legacy(cls, content: str, category: str, positive: bool) -> 'EvolvedPreference':
        """Create from legacy (string-only) preference."""
        now = datetime.now().isoformat()
        return cls(
            content=content,
            category=category,
            positive=positive,
            created_at=now,
            last_reinforced=now,
            reinforcement_count=1,
            stale=False,
            superseded=False,
            source="migrated"
        )


class PreferenceEvolution:
    """
    Manages preference evolution: timestamps, decay, supersession, and bounds.
    """

    # Known contradiction patterns
    CONTRADICTION_PATTERNS = [
        # Format: (pattern_a, pattern_b) - if both match, they're contradictory
        (r'\btabs?\b', r'\bspaces?\b'),
        (r'\bdark\s*(mode|theme)\b', r'\blight\s*(mode|theme)\b'),
        (r'\bverbose\b', r'\bconcise\b'),
        (r'\bdetailed\b', r'\bbrief\b'),
        (r'\blong\b', r'\bshort\b'),
        (r'\bmanual\b', r'\bautomat'),
        (r'\bcode\s*over\s*explanation', r'\bexplanation\s*over\s*code'),
        (r'\bminimal', r'\bcomprehensive'),
    ]

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.preferences_dir = self.base_path / "preferences"
        self.evolved_file = self.preferences_dir / "evolved_preferences.json"
        self.supersession_log = self.preferences_dir / "preference_supersession.jsonl"

        # Ensure directory exists
        self.preferences_dir.mkdir(parents=True, exist_ok=True)

        # Load existing evolved preferences
        self.preferences: Dict[str, List[EvolvedPreference]] = self._load_preferences()

        # Configuration
        self.stale_days = DEFAULT_STALE_DAYS
        self.max_preferences = DEFAULT_MAX_PREFERENCES

    def _load_preferences(self) -> Dict[str, List[EvolvedPreference]]:
        """Load evolved preferences from disk."""
        if self.evolved_file.exists():
            try:
                with open(self.evolved_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                result = {}
                for category, prefs in data.get("preferences", {}).items():
                    result[category] = [EvolvedPreference.from_dict(p) for p in prefs]
                return result
            except Exception as e:
                logger.error(f"Error loading evolved preferences: {e}")

        # Default structure
        return {
            "communication_style": [],
            "workflow": [],
            "technical": []
        }

    def _save_preferences(self):
        """Save evolved preferences to disk."""
        data = {
            "version": "2.0",
            "last_updated": datetime.now().isoformat(),
            "preferences": {
                category: [p.to_dict() for p in prefs]
                for category, prefs in self.preferences.items()
            },
            "stats": self.get_stats()
        }

        with open(self.evolved_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _log_supersession(self, old_pref: EvolvedPreference, new_pref: EvolvedPreference, reason: str):
        """Log a supersession event for audit trail."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "old_content": old_pref.content,
            "new_content": new_pref.content,
            "category": old_pref.category,
            "reason": reason
        }

        with open(self.supersession_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + "\n")

    # =========================================================================
    # Core Operations
    # =========================================================================

    def add_preference(
        self,
        content: str,
        category: str,
        positive: bool,
        source: str = "extracted",
        confidence: float = 0.7
    ) -> Dict[str, Any]:
        """
        Add a preference with full evolution tracking.

        Returns dict with:
        - added: bool - whether preference was added
        - reinforced: bool - whether existing preference was reinforced
        - superseded: list - any preferences that were superseded
        - message: str - human-readable result
        """
        now = datetime.now().isoformat()

        # Ensure category exists
        if category not in self.preferences:
            self.preferences[category] = []

        # Check for existing identical preference
        existing = self._find_exact_match(content, category, positive)
        if existing:
            # Reinforce existing preference
            existing.last_reinforced = now
            existing.reinforcement_count += 1
            existing.stale = False  # No longer stale if reinforced

            # Boost confidence on reinforcement
            existing.confidence = min(1.0, existing.confidence + 0.05)

            self._save_preferences()
            return {
                "added": False,
                "reinforced": True,
                "preference": existing.to_dict(),
                "message": f"Reinforced existing preference (count: {existing.reinforcement_count})"
            }

        # Check for contradictions
        contradictions = self._find_contradictions(content, category, positive)
        superseded_list = []

        for contradiction in contradictions:
            # Supersede the old preference
            contradiction.superseded = True
            contradiction.superseded_by = content
            contradiction.superseded_at = now

            self._log_supersession(contradiction,
                                   EvolvedPreference(content, category, positive, now, now),
                                   "contradiction_detected")
            superseded_list.append(contradiction.to_dict())

        # Create new preference
        new_pref = EvolvedPreference(
            content=content,
            category=category,
            positive=positive,
            created_at=now,
            last_reinforced=now,
            reinforcement_count=1,
            source=source,
            confidence=confidence
        )

        self.preferences[category].append(new_pref)

        # Enforce bounds
        evicted = self._enforce_bounds(category)

        self._save_preferences()

        result = {
            "added": True,
            "reinforced": False,
            "preference": new_pref.to_dict(),
            "superseded": superseded_list,
            "evicted": evicted,
            "message": f"Added new preference to {category}"
        }

        if superseded_list:
            result["message"] += f" (superseded {len(superseded_list)} contradicting preference(s))"

        return result

    def reinforce_preference(self, content: str, category: str) -> Dict[str, Any]:
        """
        Reinforce an existing preference by content match.
        """
        # Try to find it
        for pref in self.preferences.get(category, []):
            if pref.content.lower() == content.lower() and not pref.superseded:
                pref.last_reinforced = datetime.now().isoformat()
                pref.reinforcement_count += 1
                pref.stale = False
                pref.confidence = min(1.0, pref.confidence + 0.05)

                self._save_preferences()
                return {
                    "success": True,
                    "preference": pref.to_dict(),
                    "message": f"Reinforced preference (count: {pref.reinforcement_count})"
                }

        return {
            "success": False,
            "message": f"Preference not found in {category}"
        }

    def mark_decay(self) -> Dict[str, Any]:
        """
        Check all preferences and mark stale ones.
        Call this periodically (e.g., daily or on session start).

        Returns stats about what was marked stale.
        """
        now = datetime.now()
        stale_threshold = now - timedelta(days=self.stale_days)

        newly_stale = []

        for category, prefs in self.preferences.items():
            for pref in prefs:
                if pref.superseded:
                    continue  # Already superseded, don't care about staleness

                last_reinforced = datetime.fromisoformat(pref.last_reinforced)

                if last_reinforced < stale_threshold and not pref.stale:
                    pref.stale = True
                    newly_stale.append({
                        "category": category,
                        "content": pref.content,
                        "last_reinforced": pref.last_reinforced,
                        "days_since": (now - last_reinforced).days
                    })

        if newly_stale:
            self._save_preferences()

        return {
            "checked": sum(len(p) for p in self.preferences.values()),
            "newly_stale": len(newly_stale),
            "stale_preferences": newly_stale
        }

    def get_active_preferences(self, include_stale: bool = False) -> Dict[str, List[Dict]]:
        """
        Get all active (non-superseded) preferences.

        Args:
            include_stale: If False, excludes stale preferences

        Returns:
            Dict of category -> list of preference dicts
        """
        result = {}

        for category, prefs in self.preferences.items():
            active = []
            for pref in prefs:
                if pref.superseded:
                    continue
                if not include_stale and pref.stale:
                    continue
                active.append(pref.to_dict())

            # Sort by reinforcement count (most reinforced first)
            active.sort(key=lambda x: x["reinforcement_count"], reverse=True)
            result[category] = active

        return result

    def get_weighted_preferences(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get preferences weighted by recency, reinforcement, and staleness.

        Returns:
            Dict of category -> list of (content, weight) tuples
        """
        result = {}
        now = datetime.now()

        for category, prefs in self.preferences.items():
            weighted = []

            for pref in prefs:
                if pref.superseded:
                    continue

                # Base weight from confidence
                weight = pref.confidence

                # Reinforcement bonus (log scale)
                import math
                weight *= (1 + 0.1 * math.log(pref.reinforcement_count + 1))

                # Recency factor
                last_reinforced = datetime.fromisoformat(pref.last_reinforced)
                days_old = (now - last_reinforced).days
                recency_factor = 1.0 / (1 + days_old / 30)  # Half-life of 30 days
                weight *= recency_factor

                # Staleness penalty
                if pref.stale:
                    weight *= 0.3  # 70% penalty for stale

                weighted.append((pref.content, round(weight, 3)))

            # Sort by weight descending
            weighted.sort(key=lambda x: x[1], reverse=True)
            result[category] = weighted

        return result

    # =========================================================================
    # Contradiction Detection
    # =========================================================================

    def _find_exact_match(
        self,
        content: str,
        category: str,
        positive: bool
    ) -> Optional[EvolvedPreference]:
        """Find an exact (or near-exact) match for a preference."""
        content_lower = content.lower().strip()

        for pref in self.preferences.get(category, []):
            if pref.superseded:
                continue
            if pref.positive != positive:
                continue

            # Check for exact match
            if pref.content.lower().strip() == content_lower:
                return pref

            # Check for high similarity
            similarity = SequenceMatcher(None, pref.content.lower(), content_lower).ratio()
            if similarity > 0.9:
                return pref

        return None

    def _find_contradictions(
        self,
        content: str,
        category: str,
        positive: bool
    ) -> List[EvolvedPreference]:
        """
        Find preferences that contradict the new one.

        Contradictions:
        1. Same category, opposite polarity, similar content
        2. Known contradiction patterns
        """
        contradictions = []
        content_lower = content.lower()

        for pref in self.preferences.get(category, []):
            if pref.superseded:
                continue

            # Type 1: Opposite polarity with similar content
            if pref.positive != positive:
                similarity = SequenceMatcher(None, pref.content.lower(), content_lower).ratio()
                if similarity > SIMILARITY_THRESHOLD:
                    contradictions.append(pref)
                    continue

            # Type 2: Known contradiction patterns (same polarity but contradicting values)
            if pref.positive == positive:
                for pattern_a, pattern_b in self.CONTRADICTION_PATTERNS:
                    # Check if one matches pattern_a and other matches pattern_b
                    a_in_old = re.search(pattern_a, pref.content.lower())
                    b_in_old = re.search(pattern_b, pref.content.lower())
                    a_in_new = re.search(pattern_a, content_lower)
                    b_in_new = re.search(pattern_b, content_lower)

                    # Contradiction if one has A and other has B
                    if (a_in_old and b_in_new) or (b_in_old and a_in_new):
                        contradictions.append(pref)
                        break

        return contradictions

    def detect_contradictions_all(self) -> List[Dict]:
        """
        Scan all preferences for contradictions.
        Useful for cleanup/audit.
        """
        all_contradictions = []

        for category, prefs in self.preferences.items():
            active_prefs = [p for p in prefs if not p.superseded]

            for i, pref_a in enumerate(active_prefs):
                for pref_b in active_prefs[i+1:]:
                    # Check for contradiction
                    if self._are_contradictory(pref_a, pref_b):
                        all_contradictions.append({
                            "category": category,
                            "pref_a": pref_a.content,
                            "pref_b": pref_b.content,
                            "pref_a_date": pref_a.last_reinforced,
                            "pref_b_date": pref_b.last_reinforced
                        })

        return all_contradictions

    def _are_contradictory(self, pref_a: EvolvedPreference, pref_b: EvolvedPreference) -> bool:
        """Check if two preferences contradict each other."""
        # Opposite polarity with similar content
        if pref_a.positive != pref_b.positive:
            similarity = SequenceMatcher(
                None,
                pref_a.content.lower(),
                pref_b.content.lower()
            ).ratio()
            if similarity > SIMILARITY_THRESHOLD:
                return True

        # Known patterns
        for pattern_a, pattern_b in self.CONTRADICTION_PATTERNS:
            a_has_a = re.search(pattern_a, pref_a.content.lower())
            a_has_b = re.search(pattern_b, pref_a.content.lower())
            b_has_a = re.search(pattern_a, pref_b.content.lower())
            b_has_b = re.search(pattern_b, pref_b.content.lower())

            if (a_has_a and b_has_b) or (a_has_b and b_has_a):
                return True

        return False

    # =========================================================================
    # Bounded Lists / Eviction
    # =========================================================================

    def _enforce_bounds(self, category: str) -> List[Dict]:
        """
        Enforce max preferences per category.
        Evicts lowest-priority preferences using FIFO with priority weighting.

        Priority (lowest evicted first):
        1. Stale + low reinforcement
        2. Stale + higher reinforcement
        3. Active + low reinforcement
        4. Never evict: high reinforcement active preferences

        Returns list of evicted preferences.
        """
        prefs = self.preferences.get(category, [])
        active_prefs = [p for p in prefs if not p.superseded]

        if len(active_prefs) <= self.max_preferences:
            return []

        # Sort by eviction priority (lowest priority = evict first)
        def eviction_priority(p: EvolvedPreference) -> float:
            # Higher score = keep, lower score = evict
            score = 0

            # Staleness is bad
            if p.stale:
                score -= 100

            # Reinforcement is good
            score += p.reinforcement_count * 10

            # Confidence is good
            score += p.confidence * 50

            # Recency is good
            last = datetime.fromisoformat(p.last_reinforced)
            days_old = (datetime.now() - last).days
            score -= days_old * 0.1

            return score

        active_prefs.sort(key=eviction_priority)

        # Evict lowest priority until under limit
        evicted = []
        to_evict = len(active_prefs) - self.max_preferences

        for pref in active_prefs[:to_evict]:
            pref.superseded = True
            pref.superseded_by = "[EVICTED: list bounds]"
            pref.superseded_at = datetime.now().isoformat()
            evicted.append(pref.to_dict())

            self._log_supersession(
                pref,
                EvolvedPreference("[BOUNDS]", category, pref.positive, "", ""),
                "evicted_for_bounds"
            )

        return evicted

    # =========================================================================
    # Migration
    # =========================================================================

    def migrate_from_legacy(self, legacy_file: Path) -> Dict[str, Any]:
        """
        Migrate preferences from legacy format (string lists) to evolved format.
        """
        if not legacy_file.exists():
            return {"success": False, "error": "Legacy file not found"}

        try:
            with open(legacy_file, 'r', encoding='utf-8') as f:
                legacy = json.load(f)
        except Exception as e:
            return {"success": False, "error": str(e)}

        migrated = 0
        skipped = 0

        # Map legacy structure to categories
        category_mapping = {
            "communication_style": "communication_style",
            "workflow_preferences": "workflow",
            "workflow": "workflow",
            "technical_preferences": "technical",
            "technical": "technical"
        }

        for legacy_cat, evolved_cat in category_mapping.items():
            if legacy_cat not in legacy:
                continue

            cat_data = legacy[legacy_cat]

            # Handle both structures: {"prefers": [...], "dislikes": [...]} and {"languages": {...}}
            if isinstance(cat_data, dict):
                # Process prefers
                for pref_content in cat_data.get("prefers", []):
                    if isinstance(pref_content, str) and pref_content.strip():
                        existing = self._find_exact_match(pref_content, evolved_cat, True)
                        if not existing:
                            new_pref = EvolvedPreference.from_legacy(pref_content, evolved_cat, True)
                            self.preferences[evolved_cat].append(new_pref)
                            migrated += 1
                        else:
                            skipped += 1

                # Process dislikes
                for pref_content in cat_data.get("dislikes", []):
                    if isinstance(pref_content, str) and pref_content.strip():
                        existing = self._find_exact_match(pref_content, evolved_cat, False)
                        if not existing:
                            new_pref = EvolvedPreference.from_legacy(pref_content, evolved_cat, False)
                            self.preferences[evolved_cat].append(new_pref)
                            migrated += 1
                        else:
                            skipped += 1

        self._save_preferences()

        return {
            "success": True,
            "migrated": migrated,
            "skipped_duplicates": skipped,
            "total_preferences": sum(len(p) for p in self.preferences.values())
        }

    # =========================================================================
    # Stats & Reporting
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about preference evolution."""
        total = 0
        active = 0
        stale = 0
        superseded = 0

        by_category = {}

        for category, prefs in self.preferences.items():
            cat_stats = {
                "total": len(prefs),
                "active": 0,
                "stale": 0,
                "superseded": 0,
                "avg_reinforcement": 0
            }

            reinforcement_sum = 0

            for pref in prefs:
                total += 1

                if pref.superseded:
                    superseded += 1
                    cat_stats["superseded"] += 1
                elif pref.stale:
                    stale += 1
                    cat_stats["stale"] += 1
                else:
                    active += 1
                    cat_stats["active"] += 1

                reinforcement_sum += pref.reinforcement_count

            if prefs:
                cat_stats["avg_reinforcement"] = round(reinforcement_sum / len(prefs), 2)

            by_category[category] = cat_stats

        return {
            "total": total,
            "active": active,
            "stale": stale,
            "superseded": superseded,
            "by_category": by_category,
            "stale_threshold_days": self.stale_days,
            "max_per_category": self.max_preferences
        }
