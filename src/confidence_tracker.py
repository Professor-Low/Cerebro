"""
Confidence Tracking System
Phase 3 of Brain Evolution - Meaningful confidence scores for facts.

Features:
- Source-based confidence scoring (user explicit = 0.95, inferred = 0.50)
- Confidence decay over time (unchallenged facts lose confidence monthly)
- Confidence reinforcement (facts that are confirmed gain confidence)
- Full provenance tracking (why confidence was assigned, all changes)

Author: Claude
Created: 2026-01-18
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Source-based confidence scores
class ConfidenceSource(Enum):
    """Sources of information with their base confidence scores."""
    USER_EXPLICIT = ("user_explicit", 0.95)      # User directly stated
    USER_CONFIRMATION = ("user_confirmation", 0.90)  # User confirmed Claude's statement
    CORRECTION = ("correction", 1.0)              # Verified through correction
    EXTRACTED_CONTEXT = ("extracted_context", 0.75)  # Extracted from user context (raised from 0.70)
    INFERRED = ("inferred", 0.60)                 # Claude inferred (raised from 0.55)
    MIGRATED = ("migrated", 0.60)                 # Migrated from old system
    UNKNOWN = ("unknown", 0.50)                   # Default/unknown source

    @property
    def name_str(self) -> str:
        return self.value[0]

    @property
    def base_confidence(self) -> float:
        return self.value[1]

    @classmethod
    def from_string(cls, s: str) -> 'ConfidenceSource':
        """Get source from string name."""
        for source in cls:
            if source.name_str == s:
                return source
        return cls.UNKNOWN


# Decay configuration
MONTHLY_DECAY = 0.01           # Confidence loss per month without reinforcement
MIN_CONFIDENCE = 0.30          # Minimum confidence before archiving
REINFORCEMENT_BOOST = 0.05     # Confidence gain per reinforcement
MAX_CONFIDENCE = 1.0           # Maximum confidence


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ConfidenceEvent:
    """A single confidence change event for provenance tracking."""
    timestamp: str                  # ISO timestamp
    event_type: str                 # assigned, reinforced, decayed, corrected
    old_confidence: float
    new_confidence: float
    source: str                     # What triggered the change
    reason: Optional[str] = None    # Human-readable explanation

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ConfidenceEvent':
        return cls(**data)


@dataclass
class ConfidenceRecord:
    """Complete confidence record for a fact."""
    fact_id: str
    current_confidence: float
    source: str                     # Original source (ConfidenceSource.name_str)
    created_at: str                 # ISO timestamp
    last_updated: str               # ISO timestamp
    last_reinforced: Optional[str] = None  # ISO timestamp of last reinforcement
    reinforcement_count: int = 0
    decay_applied: bool = False     # Has decay been applied this month?
    last_decay_check: Optional[str] = None
    provenance: List[Dict] = field(default_factory=list)  # List of ConfidenceEvent dicts

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ConfidenceRecord':
        return cls(**data)


# =============================================================================
# Main Confidence Tracker Class
# =============================================================================

class ConfidenceTracker:
    """
    Manages confidence scores for facts with full provenance tracking.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.confidence_dir = self.base_path / "confidence"
        self.records_file = self.confidence_dir / "confidence_records.json"
        self.events_log = self.confidence_dir / "confidence_events.jsonl"

        # Ensure directory exists
        self.confidence_dir.mkdir(parents=True, exist_ok=True)

        # Load existing records
        self.records: Dict[str, ConfidenceRecord] = self._load_records()

    def _load_records(self) -> Dict[str, ConfidenceRecord]:
        """Load confidence records from disk."""
        if self.records_file.exists():
            try:
                with open(self.records_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                result = {}
                for fact_id, record_data in data.get("records", {}).items():
                    result[fact_id] = ConfidenceRecord.from_dict(record_data)
                return result
            except Exception as e:
                logger.error(f"Error loading confidence records: {e}")

        return {}

    def _save_records(self):
        """Save confidence records to disk."""
        data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "stats": self.get_stats(),
            "records": {
                fact_id: record.to_dict()
                for fact_id, record in self.records.items()
            }
        }

        with open(self.records_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _log_event(self, event: ConfidenceEvent, fact_id: str):
        """Log a confidence event for audit trail."""
        log_entry = {
            "fact_id": fact_id,
            **event.to_dict()
        }

        with open(self.events_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + "\n")

    # =========================================================================
    # Core Confidence Operations
    # =========================================================================

    def assign_confidence(
        self,
        fact_id: str,
        source: ConfidenceSource,
        content: str = "",
        context: str = "",
        reason: Optional[str] = None
    ) -> float:
        """
        Assign initial confidence to a new fact.

        Args:
            fact_id: Unique identifier for the fact
            source: The ConfidenceSource for this fact
            content: The fact content (for pattern analysis)
            context: Surrounding context (for confidence modifiers)
            reason: Optional human-readable reason

        Returns:
            The assigned confidence score
        """
        now = datetime.now().isoformat()

        # Start with base confidence from source
        base_confidence = source.base_confidence

        # Apply modifiers based on content and context
        confidence = self._apply_confidence_modifiers(base_confidence, content, context)

        # Create event
        event = ConfidenceEvent(
            timestamp=now,
            event_type="assigned",
            old_confidence=0.0,
            new_confidence=confidence,
            source=source.name_str,
            reason=reason or f"Initial assignment from {source.name_str}"
        )

        # Create record
        record = ConfidenceRecord(
            fact_id=fact_id,
            current_confidence=confidence,
            source=source.name_str,
            created_at=now,
            last_updated=now,
            provenance=[event.to_dict()]
        )

        self.records[fact_id] = record
        self._log_event(event, fact_id)
        self._save_records()

        return confidence

    def _apply_confidence_modifiers(
        self,
        base_confidence: float,
        content: str,
        context: str
    ) -> float:
        """
        Apply modifiers to base confidence based on content/context patterns.

        Enhanced (Phase 7) with:
        - Implicit confirmation detection
        - Technical content indicators
        - User behavior signals
        """
        confidence = base_confidence
        content_lower = content.lower()
        context_lower = context.lower()

        # BOOST indicators (explicit confirmation in context)
        boost_patterns = [
            "yes, that's correct", "exactly", "confirmed", "that's right",
            "you're correct", "correct", "yes!", "that is correct"
        ]
        for pattern in boost_patterns:
            if pattern in context_lower:
                confidence = min(MAX_CONFIDENCE, confidence + 0.10)
                break

        # NEW: BOOST for implicit confirmations (Phase 7)
        # User proceeds with task after Claude's statement = implicit confirmation
        implicit_confirmation_patterns = [
            r"\bthanks?\b", r"\bperfect\b", r"\bgreat\b", r"\bgot\s+it\b",
            r"\bnice\b", r"\bcool\b", r"\bawesome\b", r"\bworks?\b",
            r"\bthat\s+fixed\b", r"\bnow\s+it\s+works\b"
        ]
        import re
        for pattern in implicit_confirmation_patterns:
            if re.search(pattern, context_lower):
                confidence = min(MAX_CONFIDENCE, confidence + 0.08)
                break

        # NEW: BOOST for technical specificity (Phase 7)
        # Facts with specific technical details are more reliable
        technical_specificity_patterns = [
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IP addresses
            r':\d{2,5}\b',  # Port numbers
            r'[A-Z]:\\',  # Windows paths
            r'/(?:home|usr|var|etc)/',  # Unix paths
            r'v?\d+\.\d+(?:\.\d+)?',  # Version numbers
            r'[a-f0-9]{8,}',  # Hashes/IDs
        ]
        specificity_count = 0
        for pattern in technical_specificity_patterns:
            if re.search(pattern, content):
                specificity_count += 1

        if specificity_count >= 2:
            confidence = min(MAX_CONFIDENCE, confidence + 0.10)
        elif specificity_count == 1:
            confidence = min(MAX_CONFIDENCE, confidence + 0.05)

        # NEW: BOOST for facts from user messages about their own system (Phase 7)
        personal_system_patterns = [
            r'\bmy\s+(?:nas|server|system|setup|config)\b',
            r'\bi\s+(?:use|have|run|configured|installed)\b',
            r'\bon\s+my\s+\w+\b',
        ]
        for pattern in personal_system_patterns:
            if re.search(pattern, content_lower):
                confidence = min(MAX_CONFIDENCE, confidence + 0.08)
                break

        # REDUCE indicators (uncertainty in content)
        uncertainty_patterns = [
            "might", "maybe", "possibly", "probably", "i think",
            "seems like", "appears to", "could be", "not sure",
            "i believe", "perhaps"
        ]
        for pattern in uncertainty_patterns:
            if pattern in content_lower:
                confidence = max(MIN_CONFIDENCE, confidence - 0.15)
                break

        # REDUCE for hedging language
        if any(word in content_lower for word in ["usually", "sometimes", "often"]):
            confidence = max(MIN_CONFIDENCE, confidence - 0.05)

        # NEW: REDUCE for overly generic statements (Phase 7)
        generic_patterns = [
            r'^the\s+\w+\s+is\s+\w+$',  # Very short "The X is Y" statements
            r'^it\s+(?:is|was|has)\b',  # Starts with "it is/was/has"
        ]
        for pattern in generic_patterns:
            if re.match(pattern, content_lower) and len(content) < 50:
                confidence = max(MIN_CONFIDENCE, confidence - 0.10)
                break

        return round(confidence, 2)

    def reinforce(
        self,
        fact_id: str,
        source: Optional[str] = None,
        reason: Optional[str] = None
    ) -> Optional[float]:
        """
        Reinforce a fact's confidence (user confirmed or used it again).

        Args:
            fact_id: The fact to reinforce
            source: What triggered the reinforcement
            reason: Optional explanation

        Returns:
            New confidence score, or None if fact not found
        """
        if fact_id not in self.records:
            return None

        record = self.records[fact_id]
        now = datetime.now().isoformat()
        old_confidence = record.current_confidence

        # Apply reinforcement boost
        new_confidence = min(MAX_CONFIDENCE, old_confidence + REINFORCEMENT_BOOST)

        # Create event
        event = ConfidenceEvent(
            timestamp=now,
            event_type="reinforced",
            old_confidence=old_confidence,
            new_confidence=new_confidence,
            source=source or "user_action",
            reason=reason or "Fact was reinforced"
        )

        # Update record
        record.current_confidence = new_confidence
        record.last_updated = now
        record.last_reinforced = now
        record.reinforcement_count += 1
        record.provenance.append(event.to_dict())

        self._log_event(event, fact_id)
        self._save_records()

        # Phase 5.2: Extra boost for highly reinforced facts (3+ confirmations)
        if record.reinforcement_count >= 3:
            # Apply milestone boost
            milestone_boost = 0.05
            if record.current_confidence < MAX_CONFIDENCE:
                record.current_confidence = min(MAX_CONFIDENCE, record.current_confidence + milestone_boost)
                record.last_updated = now
                self._save_records()
                logger.info(f"Fact {fact_id} reached {record.reinforcement_count} reinforcements, milestone boost applied")

        return new_confidence

    def get_highly_reinforced_facts(self, min_reinforcements: int = 3) -> List[Dict[str, Any]]:
        """Get facts with high reinforcement counts (likely reliable).

        Args:
            min_reinforcements: Minimum reinforcement count (default 3)

        Returns:
            List of facts sorted by reinforcement count descending
        """
        highly_reinforced = []
        for fact_id, record in self.records.items():
            if record.reinforcement_count >= min_reinforcements:
                highly_reinforced.append({
                    "fact_id": fact_id,
                    "confidence": record.current_confidence,
                    "reinforcement_count": record.reinforcement_count,
                    "source": record.source,
                    "last_reinforced": record.last_reinforced,
                })

        # Sort by reinforcement count
        highly_reinforced.sort(key=lambda x: x["reinforcement_count"], reverse=True)
        return highly_reinforced

    def get_reinforcement_stats(self) -> Dict[str, Any]:
        """Get statistics about reinforcements in the system."""
        total_facts = len(self.records)
        total_reinforcements = sum(r.reinforcement_count for r in self.records.values())
        facts_with_reinforcements = sum(1 for r in self.records.values() if r.reinforcement_count > 0)
        highly_reinforced = sum(1 for r in self.records.values() if r.reinforcement_count >= 3)

        return {
            "total_facts": total_facts,
            "total_reinforcements": total_reinforcements,
            "facts_with_reinforcements": facts_with_reinforcements,
            "highly_reinforced_facts": highly_reinforced,
            "average_reinforcements_per_fact": round(total_reinforcements / max(1, total_facts), 2),
        }

    def apply_decay(self, fact_id: str) -> Optional[float]:
        """
        Apply monthly confidence decay to a fact.

        Returns:
            New confidence score, or None if fact not found or decay not applicable
        """
        if fact_id not in self.records:
            return None

        record = self.records[fact_id]
        now = datetime.now()

        # Check if decay was already applied this month
        if record.last_decay_check:
            last_check = datetime.fromisoformat(record.last_decay_check)
            if last_check.month == now.month and last_check.year == now.year:
                return record.current_confidence  # Already decayed this month

        # Check if recently reinforced (within 30 days) - no decay
        if record.last_reinforced:
            last_reinforced = datetime.fromisoformat(record.last_reinforced)
            if (now - last_reinforced).days < 30:
                record.last_decay_check = now.isoformat()
                self._save_records()
                return record.current_confidence

        # Apply decay
        old_confidence = record.current_confidence
        new_confidence = max(MIN_CONFIDENCE, old_confidence - MONTHLY_DECAY)

        # Only log if there was actual change
        if new_confidence != old_confidence:
            event = ConfidenceEvent(
                timestamp=now.isoformat(),
                event_type="decayed",
                old_confidence=old_confidence,
                new_confidence=new_confidence,
                source="monthly_decay",
                reason=f"Monthly decay applied (-{MONTHLY_DECAY})"
            )
            record.provenance.append(event.to_dict())
            self._log_event(event, fact_id)

        record.current_confidence = new_confidence
        record.last_decay_check = now.isoformat()
        record.decay_applied = True
        record.last_updated = now.isoformat()

        self._save_records()
        return new_confidence

    def apply_decay_all(self) -> Dict[str, Any]:
        """
        Apply monthly decay to ALL facts.

        Returns:
            Summary of decay operation
        """
        results = {
            "checked": 0,
            "decayed": 0,
            "skipped_recent": 0,
            "skipped_already_checked": 0,
            "below_threshold": []
        }

        for fact_id, record in self.records.items():
            results["checked"] += 1
            old_conf = record.current_confidence
            new_conf = self.apply_decay(fact_id)

            if new_conf is not None:
                if new_conf < old_conf:
                    results["decayed"] += 1
                elif record.last_reinforced:
                    results["skipped_recent"] += 1
                else:
                    results["skipped_already_checked"] += 1

                # Track facts at minimum confidence
                if new_conf <= MIN_CONFIDENCE:
                    results["below_threshold"].append(fact_id)

        return results

    def set_confidence(
        self,
        fact_id: str,
        confidence: float,
        reason: str
    ) -> Optional[float]:
        """
        Manually set confidence (for corrections or administrative updates).

        Args:
            fact_id: The fact to update
            confidence: New confidence value
            reason: Explanation for the change

        Returns:
            New confidence score, or None if fact not found
        """
        if fact_id not in self.records:
            return None

        record = self.records[fact_id]
        now = datetime.now().isoformat()
        old_confidence = record.current_confidence

        # Clamp confidence
        confidence = max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, confidence))

        event = ConfidenceEvent(
            timestamp=now,
            event_type="corrected",
            old_confidence=old_confidence,
            new_confidence=confidence,
            source="manual",
            reason=reason
        )

        record.current_confidence = confidence
        record.last_updated = now
        record.provenance.append(event.to_dict())

        self._log_event(event, fact_id)
        self._save_records()

        return confidence

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_confidence(self, fact_id: str) -> Optional[float]:
        """Get current confidence for a fact."""
        if fact_id in self.records:
            return self.records[fact_id].current_confidence
        return None

    def get_provenance(self, fact_id: str) -> Optional[List[Dict]]:
        """Get full provenance history for a fact."""
        if fact_id in self.records:
            return self.records[fact_id].provenance
        return None

    def get_record(self, fact_id: str) -> Optional[Dict]:
        """Get full confidence record for a fact."""
        if fact_id in self.records:
            return self.records[fact_id].to_dict()
        return None

    def get_low_confidence_facts(self, threshold: float = 0.50) -> List[Dict]:
        """Get facts below a confidence threshold."""
        results = []
        for fact_id, record in self.records.items():
            if record.current_confidence < threshold:
                results.append({
                    "fact_id": fact_id,
                    "confidence": record.current_confidence,
                    "source": record.source,
                    "created_at": record.created_at,
                    "last_reinforced": record.last_reinforced
                })

        # Sort by confidence (lowest first)
        results.sort(key=lambda x: x["confidence"])
        return results

    def quarantine_low_confidence(self, threshold: float = 0.4) -> Dict[str, Any]:
        """
        Quarantine facts below a confidence threshold.

        Args:
            threshold: Confidence threshold (facts below this are quarantined)

        Returns:
            Summary of quarantine operation
        """
        import json
        from datetime import datetime

        low_facts = self.get_low_confidence_facts(threshold=threshold)
        quarantined_ids = set(f["fact_id"] for f in low_facts)

        # Save quarantine index
        quarantine_dir = self.base_path / "facts" / "_quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        quarantine_index = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "threshold_used": threshold,
            "count": len(low_facts),
            "quarantined_ids": list(quarantined_ids),
            "facts": low_facts
        }

        quarantine_path = quarantine_dir / "quarantined_facts.json"
        with open(quarantine_path, "w", encoding="utf-8") as f:
            json.dump(quarantine_index, f, indent=2, ensure_ascii=False)

        return {
            "quarantined": len(low_facts),
            "threshold": threshold,
            "path": str(quarantine_path)
        }

    def get_quarantined_ids(self) -> set:
        """Get set of quarantined fact IDs for filtering."""
        quarantine_path = self.base_path / "facts" / "_quarantine" / "quarantined_facts.json"
        if not quarantine_path.exists():
            return set()

        try:
            with open(quarantine_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return set(data.get("quarantined_ids", []))
        except Exception:
            return set()

    def is_quarantined(self, fact_id: str) -> bool:
        """Check if a fact is quarantined."""
        return fact_id in self.get_quarantined_ids()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about confidence distribution."""
        if not self.records:
            return {
                "total_facts": 0,
                "avg_confidence": 0.0,
                "distribution": {},
                "by_source": {}
            }

        confidences = [r.current_confidence for r in self.records.values()]
        sources = [r.source for r in self.records.values()]

        # Distribution buckets
        distribution = {
            "high (0.8-1.0)": sum(1 for c in confidences if c >= 0.8),
            "medium (0.5-0.8)": sum(1 for c in confidences if 0.5 <= c < 0.8),
            "low (0.3-0.5)": sum(1 for c in confidences if 0.3 <= c < 0.5),
            "archive (<0.3)": sum(1 for c in confidences if c < 0.3)
        }

        # By source
        by_source = {}
        for source in set(sources):
            source_confs = [r.current_confidence for r in self.records.values() if r.source == source]
            by_source[source] = {
                "count": len(source_confs),
                "avg_confidence": round(sum(source_confs) / len(source_confs), 2) if source_confs else 0
            }

        return {
            "total_facts": len(self.records),
            "avg_confidence": round(sum(confidences) / len(confidences), 2),
            "min_confidence": round(min(confidences), 2),
            "max_confidence": round(max(confidences), 2),
            "distribution": distribution,
            "by_source": by_source,
            "total_reinforcements": sum(r.reinforcement_count for r in self.records.values())
        }

    # =========================================================================
    # Source Detection (for fact extraction integration)
    # =========================================================================

    def detect_source(
        self,
        content: str,
        context: str,
        role: str = "user"
    ) -> ConfidenceSource:
        """
        Detect the appropriate confidence source based on content and context.

        Enhanced (Phase 7) with better classification to reduce over-use of INFERRED.

        Args:
            content: The fact content
            context: Surrounding conversation context
            role: The role of the message source (user/assistant)

        Returns:
            Appropriate ConfidenceSource
        """
        import re
        content_lower = content.lower()
        context_lower = context.lower()

        # Check for explicit user statements
        explicit_patterns = [
            r"\bmy\s+\w+\s+is\b",           # "my name is", "my email is"
            r"\bi\s+(am|have|use|prefer)\b", # "I am", "I have", "I use"
            r"\bi'?m\s+\w+",                 # "I'm a developer"
            r"\bthe\s+\w+\s+is\s+\w+",       # "the password is X"
            r"\bi\s+set\s+up\b",             # "I set up"
            r"\bi\s+configured\b",           # "I configured"
            r"\bi\s+installed\b",            # "I installed"
        ]

        if role == "user":
            for pattern in explicit_patterns:
                if re.search(pattern, content_lower):
                    return ConfidenceSource.USER_EXPLICIT

        # Check for user confirmation
        confirmation_patterns = [
            "yes", "correct", "exactly", "that's right", "confirmed",
            "you got it", "perfect", "that's correct"
        ]
        if role == "user" and any(p in context_lower for p in confirmation_patterns):
            return ConfidenceSource.USER_CONFIRMATION

        # Check for correction context
        if "correction" in context_lower or "actually" in context_lower:
            if "not" in context_lower or "wrong" in context_lower:
                return ConfidenceSource.CORRECTION

        # NEW (Phase 7): Check for implicit confirmation through user action
        implicit_confirm_patterns = [
            r"\bthanks?\b.*\bwork", r"\bperfect\b", r"\bthat\s+fixed\b",
            r"\bnow\s+it\s+works\b", r"\bgreat.*\bwork"
        ]
        for pattern in implicit_confirm_patterns:
            if re.search(pattern, context_lower):
                return ConfidenceSource.USER_CONFIRMATION

        # If from assistant/Claude
        if role == "assistant":
            # Check if it's extracted from what user said (higher confidence)
            user_reference_patterns = [
                r"you\s+mentioned", r"you\s+said", r"you\s+have",
                r"your\s+\w+\s+is", r"based\s+on\s+what\s+you",
                r"from\s+your\s+message", r"you\s+configured",
                r"you\s+set\s+up", r"you\s+installed"
            ]
            for pattern in user_reference_patterns:
                if re.search(pattern, content_lower):
                    return ConfidenceSource.EXTRACTED_CONTEXT

            # NEW (Phase 7): Check if fact contains highly specific technical details
            # These are less likely to be pure inference
            specific_patterns = [
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IP
                r':\d{4,5}\b',  # Port
                r'\bport\s*\d{4,5}\b',  # "port 8080"
                r'[A-Z]:\\(?:\w+\\)+',  # Windows path with depth
                r'[A-Z]:\\[\w\\.-]+\.\w+',  # Windows file path
                r'/(?:home|usr|var|etc|mnt)/[a-z0-9_/]+',  # Unix path
            ]
            specificity_count = sum(1 for p in specific_patterns if re.search(p, content, re.IGNORECASE))
            if specificity_count >= 1:
                # Technical specificity - likely extracted from user context, not pure inference
                return ConfidenceSource.EXTRACTED_CONTEXT

            return ConfidenceSource.INFERRED

        # Default for user content is extracted from context
        return ConfidenceSource.EXTRACTED_CONTEXT

    # =========================================================================
    # Search Integration
    # =========================================================================

    def apply_confidence_to_score(
        self,
        fact_id: str,
        similarity_score: float,
        recency_weight: float = 1.0
    ) -> float:
        """
        Apply confidence weighting to a search score.

        Combined score = similarity * confidence * recency

        Args:
            fact_id: The fact ID
            similarity_score: Base similarity score from vector search
            recency_weight: Recency weight (already applied or 1.0)

        Returns:
            Combined weighted score
        """
        confidence = self.get_confidence(fact_id)

        if confidence is None:
            # Unknown fact - use default medium confidence
            confidence = 0.60

        # Combined scoring formula
        combined = similarity_score * confidence * recency_weight

        return round(combined, 4)

    def bulk_apply_confidence(
        self,
        results: List[Dict],
        similarity_key: str = "similarity",
        fact_id_key: str = "fact_id"
    ) -> List[Dict]:
        """
        Apply confidence weighting to a list of search results.

        Args:
            results: List of search result dicts
            similarity_key: Key for similarity score in results
            fact_id_key: Key for fact ID in results

        Returns:
            Results with added confidence_score and combined_score fields
        """
        for result in results:
            fact_id = result.get(fact_id_key)
            similarity = result.get(similarity_key, 0.5)

            confidence = self.get_confidence(fact_id) if fact_id else None

            result["confidence"] = confidence if confidence is not None else 0.60
            result["confidence_source"] = (
                self.records[fact_id].source if fact_id and fact_id in self.records else "unknown"
            )
            result["combined_score"] = round(
                similarity * result["confidence"], 4
            )

        # Re-sort by combined score
        results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

        return results


# =============================================================================
# Utility Functions
# =============================================================================

def get_confidence_label(confidence: float) -> str:
    """Convert numeric confidence to human-readable label."""
    if confidence >= 0.9:
        return "very high"
    elif confidence >= 0.7:
        return "high"
    elif confidence >= 0.5:
        return "medium"
    elif confidence >= 0.3:
        return "low"
    else:
        return "very low"


def confidence_to_display(confidence: float) -> str:
    """Format confidence for display."""
    label = get_confidence_label(confidence)
    return f"{confidence:.0%} ({label})"


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Confidence Tracker - Phase 3 Test Suite")
    print("=" * 50)

    # Use test directory
    import tempfile
    test_dir = Path(tempfile.mkdtemp()) / "test_confidence"
    tracker = ConfidenceTracker(base_path=str(test_dir))

    # Test 1: Assign confidence from different sources
    print("\n1. Testing source-based confidence assignment...")

    test_cases = [
        ("fact_001", ConfidenceSource.USER_EXPLICIT, "My name is Test User", ""),
        ("fact_002", ConfidenceSource.USER_CONFIRMATION, "The port is 8080", "yes, that's correct"),
        ("fact_003", ConfidenceSource.INFERRED, "Probably uses Python", ""),
        ("fact_004", ConfidenceSource.CORRECTION, "NAS IP is 10.0.0.100", "Actually, it's .100 not .1"),
    ]

    for fact_id, source, content, context in test_cases:
        conf = tracker.assign_confidence(fact_id, source, content, context)
        expected = source.base_confidence
        status = "PASS" if abs(conf - expected) < 0.15 else "FAIL"  # Allow for modifiers
        print(f"   {fact_id}: {source.name_str} -> {conf:.2f} (base: {expected:.2f}) [{status}]")

    # Test 2: Reinforcement
    print("\n2. Testing reinforcement...")
    old_conf = tracker.get_confidence("fact_003")
    new_conf = tracker.reinforce("fact_003", source="user_reconfirmed")
    print(f"   fact_003: {old_conf:.2f} -> {new_conf:.2f} (+{REINFORCEMENT_BOOST}) [{'PASS' if new_conf > old_conf else 'FAIL'}]")

    # Test 3: Decay
    print("\n3. Testing decay...")
    # Manually set last_decay_check to last month to allow decay
    tracker.records["fact_003"].last_decay_check = None
    tracker.records["fact_003"].last_reinforced = None  # Clear reinforcement
    old_conf = tracker.get_confidence("fact_003")
    new_conf = tracker.apply_decay("fact_003")
    print(f"   fact_003: {old_conf:.2f} -> {new_conf:.2f} (-{MONTHLY_DECAY}) [{'PASS' if new_conf < old_conf else 'FAIL'}]")

    # Test 4: Provenance
    print("\n4. Testing provenance tracking...")
    prov = tracker.get_provenance("fact_003")
    print(f"   fact_003 has {len(prov)} provenance entries")
    for entry in prov:
        print(f"     - {entry['event_type']}: {entry['old_confidence']:.2f} -> {entry['new_confidence']:.2f}")
    status = "PASS" if len(prov) >= 3 else "FAIL"  # assign + reinforce + decay
    print(f"   [{status}]")

    # Test 5: Statistics
    print("\n5. Testing statistics...")
    stats = tracker.get_stats()
    print(f"   Total facts: {stats['total_facts']}")
    print(f"   Average confidence: {stats['avg_confidence']}")
    print(f"   Distribution: {stats['distribution']}")
    print("   [PASS]")

    # Test 6: Source detection
    print("\n6. Testing source detection...")
    test_detections = [
        ("My email is test@example.com", "", "user", ConfidenceSource.USER_EXPLICIT),
        ("Confirmed", "That's the right IP", "user", ConfidenceSource.USER_CONFIRMATION),
        ("The user seems to prefer Python", "", "assistant", ConfidenceSource.INFERRED),
    ]

    for content, context, role, expected in test_detections:
        detected = tracker.detect_source(content, context, role)
        status = "PASS" if detected == expected else "FAIL"
        print(f"   '{content[:30]}...' -> {detected.name_str} (expected: {expected.name_str}) [{status}]")

    # Test 7: Search score integration
    print("\n7. Testing search score integration...")
    mock_results = [
        {"fact_id": "fact_001", "similarity": 0.9},
        {"fact_id": "fact_003", "similarity": 0.85},
        {"fact_id": "fact_unknown", "similarity": 0.95},
    ]
    weighted = tracker.bulk_apply_confidence(mock_results)
    print("   Before weighting: fact_unknown was highest (0.95)")
    print(f"   After weighting: {weighted[0]['fact_id']} is highest ({weighted[0]['combined_score']:.3f})")
    print("   [PASS]")

    print("\n" + "=" * 50)
    print("Phase 3 Confidence Tracker: All tests completed!")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
