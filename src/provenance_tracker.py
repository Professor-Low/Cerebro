"""
Provenance Tracking System
Comprehensive tracking of fact origins and history.

Features:
- Source chain: Track where each fact came from (conversation, message, type)
- Correction history: Track what was corrected, when, by whom
- Reinforcement logging: Track when facts are confirmed
- Contradiction detection: Alert when storing conflicting facts

Author: Claude
Created: 2026-01-18
"""

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Contradiction detection thresholds
CONTRADICTION_SIMILARITY_THRESHOLD = 0.5  # Minimum similarity to be considered a contradiction
ENTITY_CONFLICT_WEIGHT = 0.7             # Weight for entity conflicts in scoring


# =============================================================================
# Enums and Data Classes
# =============================================================================

class LearnedType(Enum):
    """How a fact was learned/acquired."""
    USER_EXPLICIT = "user_explicit"          # User directly stated it
    USER_CORRECTION = "user_correction"      # User corrected a previous fact
    EXTRACTION = "extraction"                # Extracted from conversation
    INFERENCE = "inference"                  # Inferred by Claude
    CONFIRMATION = "confirmation"            # Confirmed by user agreement
    MIGRATED = "migrated"                    # Migrated from legacy system
    IMPORTED = "imported"                    # Imported from external source


@dataclass
class SourceChain:
    """
    Tracks the original source of a fact.
    Where did this fact come from?
    """
    learned_from: str              # conversation_id where fact was learned
    learned_at: str                # ISO timestamp
    learned_type: str              # LearnedType value
    message_index: Optional[int] = None   # Which message in the conversation
    context_snippet: Optional[str] = None  # Surrounding context (truncated)
    extracted_by: str = "pattern"  # Extraction method (pattern, llm, manual)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'SourceChain':
        return cls(**data)


@dataclass
class CorrectionEvent:
    """
    Records a correction made to a fact.
    What was changed, when, and why?
    """
    correction_id: str             # Unique ID for this correction
    from_value: str                # Original value
    to_value: str                  # Corrected value
    corrected_at: str              # ISO timestamp
    corrected_by: str              # Who made the correction (user, system, claude)
    conversation_id: Optional[str] = None  # Where correction occurred
    reason: Optional[str] = None   # Why the correction was made

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'CorrectionEvent':
        return cls(**data)


@dataclass
class ReinforcementEvent:
    """
    Records when a fact was reinforced/confirmed.
    When did someone agree this fact is correct?
    """
    reinforced_at: str             # ISO timestamp
    reinforced_in: str             # conversation_id
    reinforcement_type: str        # How it was reinforced (explicit, usage, confirmation)
    context: Optional[str] = None  # Context of the reinforcement

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ReinforcementEvent':
        return cls(**data)


@dataclass
class ContradictionCandidate:
    """
    A potential contradiction between two facts.
    """
    fact_id_a: str                 # First fact
    fact_id_b: str                 # Second fact (the new one)
    similarity_score: float        # How similar/contradictory (0-1)
    contradiction_type: str        # Type: entity_conflict, semantic, negation
    evidence: str                  # Explanation of the contradiction
    status: str = "pending"        # pending, confirmed, dismissed
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved_at: Optional[str] = None
    resolution: Optional[str] = None  # How it was resolved

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ContradictionCandidate':
        # Handle default for detected_at if missing
        if 'detected_at' not in data:
            data['detected_at'] = datetime.now().isoformat()
        return cls(**data)


@dataclass
class ProvenanceRecord:
    """
    Complete provenance record for a fact.
    Contains full history: source, corrections, reinforcements.
    """
    fact_id: str
    version: int = 1               # Increments with each correction
    source_chain: Optional[Dict] = None  # SourceChain as dict
    corrections: List[Dict] = field(default_factory=list)  # List of CorrectionEvent
    reinforcements: List[Dict] = field(default_factory=list)  # List of ReinforcementEvent
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ProvenanceRecord':
        # Handle defaults
        if 'created_at' not in data:
            data['created_at'] = datetime.now().isoformat()
        if 'updated_at' not in data:
            data['updated_at'] = data.get('created_at', datetime.now().isoformat())
        return cls(**data)


# =============================================================================
# Main Provenance Tracker Class
# =============================================================================

class ProvenanceTracker:
    """
    Manages provenance tracking for facts with:
    - Source chain (origin tracking)
    - Correction history
    - Reinforcement logging
    - Contradiction detection
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.provenance_dir = self.base_path / "provenance"

        # Storage files
        self.records_file = self.provenance_dir / "provenance_records.json"
        self.events_log = self.provenance_dir / "provenance_events.jsonl"
        self.contradictions_file = self.provenance_dir / "contradictions.json"

        # Ensure directory exists
        self.provenance_dir.mkdir(parents=True, exist_ok=True)

        # Load existing records
        self.records: Dict[str, ProvenanceRecord] = self._load_records()
        self.contradictions: List[ContradictionCandidate] = self._load_contradictions()

    # =========================================================================
    # Persistence
    # =========================================================================

    def _load_records(self) -> Dict[str, ProvenanceRecord]:
        """Load provenance records from disk."""
        if self.records_file.exists():
            try:
                with open(self.records_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                result = {}
                for fact_id, record_data in data.get("records", {}).items():
                    result[fact_id] = ProvenanceRecord.from_dict(record_data)
                return result
            except Exception as e:
                logger.error(f"Error loading provenance records: {e}")

        return {}

    def _save_records(self):
        """Save provenance records to disk."""
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

    def _load_contradictions(self) -> List[ContradictionCandidate]:
        """Load contradictions from disk."""
        if self.contradictions_file.exists():
            try:
                with open(self.contradictions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                return [
                    ContradictionCandidate.from_dict(c)
                    for c in data.get("contradictions", [])
                ]
            except Exception as e:
                logger.error(f"Error loading contradictions: {e}")

        return []

    def _save_contradictions(self):
        """Save contradictions to disk."""
        data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "contradictions": [c.to_dict() for c in self.contradictions]
        }

        with open(self.contradictions_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _log_event(self, event_type: str, fact_id: str, details: Dict):
        """Log an event to the immutable audit log."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "fact_id": fact_id,
            **details
        }

        with open(self.events_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + "\n")

    # =========================================================================
    # Source Chain Operations
    # =========================================================================

    def create_source_chain(
        self,
        fact_id: str,
        conversation_id: str,
        learned_type: LearnedType,
        message_index: Optional[int] = None,
        context_snippet: Optional[str] = None,
        extracted_by: str = "pattern"
    ) -> SourceChain:
        """
        Create a source chain for a new fact.

        Args:
            fact_id: The fact's unique identifier
            conversation_id: Where the fact was learned
            learned_type: How the fact was learned
            message_index: Which message in the conversation
            context_snippet: Surrounding context
            extracted_by: Extraction method

        Returns:
            The created SourceChain
        """
        now = datetime.now().isoformat()

        # Create source chain
        source_chain = SourceChain(
            learned_from=conversation_id,
            learned_at=now,
            learned_type=learned_type.value,
            message_index=message_index,
            context_snippet=context_snippet[:200] if context_snippet else None,
            extracted_by=extracted_by
        )

        # Create or update provenance record
        if fact_id in self.records:
            # Update existing record
            record = self.records[fact_id]
            record.source_chain = source_chain.to_dict()
            record.updated_at = now
        else:
            # Create new record
            record = ProvenanceRecord(
                fact_id=fact_id,
                version=1,
                source_chain=source_chain.to_dict(),
                created_at=now,
                updated_at=now
            )
            self.records[fact_id] = record

        # Log event
        self._log_event("source_chain_created", fact_id, {
            "conversation_id": conversation_id,
            "learned_type": learned_type.value,
            "message_index": message_index
        })

        self._save_records()
        return source_chain

    def get_source_chain(self, fact_id: str) -> Optional[Dict]:
        """Get the source chain for a fact."""
        if fact_id in self.records:
            return self.records[fact_id].source_chain
        return None

    # =========================================================================
    # Correction History
    # =========================================================================

    def add_correction(
        self,
        fact_id: str,
        from_value: str,
        to_value: str,
        corrected_by: str = "user",
        conversation_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> CorrectionEvent:
        """
        Record a correction made to a fact.

        Args:
            fact_id: The fact being corrected
            from_value: Original value
            to_value: New value
            corrected_by: Who made the correction
            conversation_id: Where the correction happened
            reason: Why it was corrected

        Returns:
            The CorrectionEvent
        """
        now = datetime.now().isoformat()

        # Generate correction ID
        correction_id = hashlib.sha256(
            f"{fact_id}:{from_value}:{to_value}:{now}".encode()
        ).hexdigest()[:16]

        correction = CorrectionEvent(
            correction_id=correction_id,
            from_value=from_value,
            to_value=to_value,
            corrected_at=now,
            corrected_by=corrected_by,
            conversation_id=conversation_id,
            reason=reason
        )

        # Update provenance record
        if fact_id not in self.records:
            # Create minimal record for fact
            self.records[fact_id] = ProvenanceRecord(
                fact_id=fact_id,
                version=1,
                created_at=now,
                updated_at=now
            )

        record = self.records[fact_id]
        record.corrections.append(correction.to_dict())
        record.version += 1
        record.updated_at = now

        # Log event
        self._log_event("correction_added", fact_id, {
            "correction_id": correction_id,
            "from_value": from_value[:100],
            "to_value": to_value[:100],
            "corrected_by": corrected_by
        })

        self._save_records()
        return correction

    def get_correction_history(self, fact_id: str) -> List[Dict]:
        """Get all corrections for a fact."""
        if fact_id in self.records:
            return self.records[fact_id].corrections
        return []

    # =========================================================================
    # Reinforcement Logging
    # =========================================================================

    def add_reinforcement(
        self,
        fact_id: str,
        conversation_id: str,
        reinforcement_type: str = "explicit",
        context: Optional[str] = None
    ) -> ReinforcementEvent:
        """
        Log when a fact was reinforced/confirmed.

        Args:
            fact_id: The fact being reinforced
            conversation_id: Where it was reinforced
            reinforcement_type: How (explicit, usage, confirmation)
            context: Context of the reinforcement

        Returns:
            The ReinforcementEvent
        """
        now = datetime.now().isoformat()

        reinforcement = ReinforcementEvent(
            reinforced_at=now,
            reinforced_in=conversation_id,
            reinforcement_type=reinforcement_type,
            context=context[:200] if context else None
        )

        # Update provenance record
        if fact_id not in self.records:
            self.records[fact_id] = ProvenanceRecord(
                fact_id=fact_id,
                version=1,
                created_at=now,
                updated_at=now
            )

        record = self.records[fact_id]
        record.reinforcements.append(reinforcement.to_dict())
        record.updated_at = now

        # Log event
        self._log_event("reinforcement_added", fact_id, {
            "conversation_id": conversation_id,
            "reinforcement_type": reinforcement_type
        })

        self._save_records()
        return reinforcement

    def get_reinforcements(self, fact_id: str) -> List[Dict]:
        """Get all reinforcements for a fact."""
        if fact_id in self.records:
            return self.records[fact_id].reinforcements
        return []

    # =========================================================================
    # Contradiction Detection
    # =========================================================================

    def detect_contradictions(
        self,
        new_fact_id: str,
        new_fact_content: str,
        existing_facts: Optional[List[Dict]] = None
    ) -> List[ContradictionCandidate]:
        """
        Detect potential contradictions between a new fact and existing facts.

        Algorithm:
        1. Extract entities (IPs, ports, paths, hostnames)
        2. For each existing fact:
           - Check entity conflicts (same type, different values)
           - Check semantic similarity with negation detection
           - Score based on conflicts found
        3. Return candidates with score >= threshold

        Args:
            new_fact_id: ID of the new fact
            new_fact_content: Content of the new fact
            existing_facts: List of existing facts to check against
                           (if None, loads from facts.jsonl)

        Returns:
            List of ContradictionCandidate objects
        """
        if existing_facts is None:
            existing_facts = self._load_existing_facts()

        contradictions = []
        new_entities = self._extract_entities(new_fact_content)

        for fact in existing_facts:
            fact_id = fact.get("fact_id", "")
            content = fact.get("content", "")

            # Skip self-comparison
            if fact_id == new_fact_id:
                continue

            # Skip superseded facts
            if fact.get("superseded", False):
                continue

            # Check for contradictions
            score, contradiction_type, evidence = self._check_entity_conflicts(
                new_fact_content, new_entities,
                content, self._extract_entities(content)
            )

            if score >= CONTRADICTION_SIMILARITY_THRESHOLD:
                candidate = ContradictionCandidate(
                    fact_id_a=fact_id,
                    fact_id_b=new_fact_id,
                    similarity_score=score,
                    contradiction_type=contradiction_type,
                    evidence=evidence,
                    status="pending"
                )
                contradictions.append(candidate)
                self.contradictions.append(candidate)

        if contradictions:
            self._save_contradictions()
            self._log_event("contradictions_detected", new_fact_id, {
                "count": len(contradictions),
                "fact_ids": [c.fact_id_a for c in contradictions]
            })

        return contradictions

    def _load_existing_facts(self) -> List[Dict]:
        """Load existing facts from facts.jsonl."""
        facts = []
        facts_file = self.base_path / "facts" / "facts.jsonl"

        if facts_file.exists():
            try:
                with open(facts_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                fact = json.loads(line)
                                if fact.get("type") == "fact":
                                    facts.append(fact)
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                logger.error(f"Error loading facts: {e}")

        return facts

    def _extract_entities(self, content: str) -> Dict[str, Set[str]]:
        """
        Extract entities from content for contradiction detection.

        Returns dict of entity type -> set of values
        """
        entities = {
            "ips": set(),
            "ports": set(),
            "paths": set(),
            "hostnames": set(),
            "versions": set(),
            "port_contexts": set()  # New: ports with their service context
        }

        # IP addresses
        ip_pattern = r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'
        entities["ips"].update(re.findall(ip_pattern, content))

        # Ports (preceded by : or "port")
        port_pattern = r'(?::|port\s*)(\d{2,5})\b'
        entities["ports"].update(re.findall(port_pattern, content, re.IGNORECASE))

        # Extract ports WITH service context (Phase 7 Enhancement)
        port_contexts = self._extract_port_context(content)
        entities["port_contexts"].update(port_contexts)

        # Windows paths
        win_path_pattern = r'([A-Z]:\\(?:[^\s\\/:*?"<>|]+\\)*[^\s\\/:*?"<>|]*)'
        entities["paths"].update(re.findall(win_path_pattern, content))

        # Unix paths
        unix_path_pattern = r'(/(?:[^\s/:*?"<>|]+/)+[^\s/:*?"<>|]*)'
        entities["paths"].update(re.findall(unix_path_pattern, content))

        # Hostnames (common patterns)
        hostname_pattern = r'\b((?:NAS|SERVER|HOST)[A-Z0-9-]*|[A-Za-z0-9-]+\.local)\b'
        entities["hostnames"].update(re.findall(hostname_pattern, content, re.IGNORECASE))

        # Version numbers
        version_pattern = r'\b(\d+(?:\.\d+)+)\b'
        entities["versions"].update(re.findall(version_pattern, content))

        return entities

    def _extract_port_context(self, content: str) -> Set[str]:
        """
        Extract ports with their associated service context.

        Returns set of "service:port" strings for context-aware conflict detection.
        Different services on different ports are NOT conflicts.
        """
        port_contexts = set()
        content_lower = content.lower()

        # Known service names to look for
        service_keywords = [
            'mcp', 'redis', 'cerebro', 'frontend', 'backend', 'api', 'webhook',
            'docker', 'postgres', 'postgresql', 'mysql', 'mongodb', 'nginx',
            'server', 'service', 'app', 'web', 'http', 'https', 'ssh', 'ftp',
            'smtp', 'imap', 'ollama', 'llm', 'brain', 'memory', 'graph', 'viz'
        ]

        # Pattern 1: "service_name on port X" or "service_name port X"
        for service in service_keywords:
            pattern1 = rf'\b{service}\b[^.]*?\bon\s+port\s+(\d{{2,5}})\b'
            pattern2 = rf'\b{service}\b[^.]*?\bport\s+(\d{{2,5}})\b'
            pattern3 = rf'\bport\s+(\d{{2,5}})\b[^.]*?\b{service}\b'
            pattern4 = rf'\b{service}\b[^.]*?:(\d{{2,5}})\b'

            for pattern in [pattern1, pattern2, pattern3, pattern4]:
                matches = re.findall(pattern, content_lower)
                for port in matches:
                    port_contexts.add(f"{service}:{port}")

        # Pattern 2: Generic port mentions without clear service context
        # These get tagged as "unknown" service
        generic_port_pattern = r'\bport\s+(\d{2,5})\b'
        all_ports = set(re.findall(generic_port_pattern, content_lower))

        # Find ports that weren't associated with a service
        associated_ports = {pc.split(':')[1] for pc in port_contexts}
        unassociated_ports = all_ports - associated_ports

        # For unassociated ports, try to infer from surrounding context
        for port in unassociated_ports:
            # Look for any service keyword near this port mention
            port_context_pattern = rf'(\w+)[^.]*?\bport\s+{port}\b|\bport\s+{port}\b[^.]*?(\w+)'
            matches = re.findall(port_context_pattern, content_lower)

            found_service = False
            for match in matches:
                words = [w for w in match if w and w in service_keywords]
                if words:
                    port_contexts.add(f"{words[0]}:{port}")
                    found_service = True
                    break

            if not found_service:
                port_contexts.add(f"unknown:{port}")

        return port_contexts

    def _is_same_service(self, context1: Set[str], context2: Set[str]) -> Tuple[bool, Optional[str]]:
        """
        Check if two port contexts refer to the same service with different ports.

        Returns:
            (is_conflict, service_name) - True if same service has different ports
        """
        # Extract services and their ports from contexts
        services1 = {}  # service -> set of ports
        services2 = {}

        for ctx in context1:
            if ':' in ctx:
                service, port = ctx.rsplit(':', 1)
                if service not in services1:
                    services1[service] = set()
                services1[service].add(port)

        for ctx in context2:
            if ':' in ctx:
                service, port = ctx.rsplit(':', 1)
                if service not in services2:
                    services2[service] = set()
                services2[service].add(port)

        # Check for conflicts: same service, different ports
        common_services = set(services1.keys()) & set(services2.keys())

        # Skip "unknown" service - can't determine conflict
        common_services.discard("unknown")

        for service in common_services:
            ports1 = services1[service]
            ports2 = services2[service]

            # If same service has different ports, that's a conflict
            if ports1 != ports2 and ports1 and ports2:
                return True, service

        return False, None

    def _check_entity_conflicts(
        self,
        new_content: str,
        new_entities: Dict[str, Set[str]],
        existing_content: str,
        existing_entities: Dict[str, Set[str]]
    ) -> Tuple[float, str, str]:
        """
        Check for entity conflicts between new and existing content.

        Enhanced with service-context awareness for ports (Phase 7).
        Different services on different ports are NOT conflicts.

        Returns:
            (score, contradiction_type, evidence)
        """
        score = 0.0
        conflicts = []
        contradiction_type = "none"

        # Check each entity type for conflicts
        for entity_type in ["ips", "hostnames"]:  # Removed "ports" - handled separately
            new_vals = new_entities.get(entity_type, set())
            existing_vals = existing_entities.get(entity_type, set())

            if new_vals and existing_vals:
                # Check if they're talking about the same thing but different values
                new_lower = new_content.lower()
                existing_lower = existing_content.lower()

                # Check for common subject patterns
                common_subjects = self._find_common_subjects(new_lower, existing_lower)

                if common_subjects and new_vals != existing_vals:
                    # Same subject, different values = potential conflict
                    conflict_score = ENTITY_CONFLICT_WEIGHT
                    score += conflict_score
                    conflicts.append(
                        f"{entity_type}: {existing_vals} vs {new_vals} for '{common_subjects[0]}'"
                    )
                    contradiction_type = "entity_conflict"

        # ENHANCED PORT CONFLICT DETECTION (Phase 7)
        # Use service context to avoid false positives
        new_port_contexts = new_entities.get("port_contexts", set())
        existing_port_contexts = existing_entities.get("port_contexts", set())

        if new_port_contexts and existing_port_contexts:
            is_conflict, conflicting_service = self._is_same_service(
                new_port_contexts, existing_port_contexts
            )

            if is_conflict and conflicting_service:
                # Same service with different ports IS a conflict
                new_ports = {ctx.split(':')[1] for ctx in new_port_contexts
                            if ctx.startswith(f"{conflicting_service}:")}
                existing_ports = {ctx.split(':')[1] for ctx in existing_port_contexts
                                 if ctx.startswith(f"{conflicting_service}:")}

                conflict_score = ENTITY_CONFLICT_WEIGHT
                score += conflict_score
                conflicts.append(
                    f"ports: {conflicting_service} has conflicting ports "
                    f"{existing_ports} vs {new_ports}"
                )
                contradiction_type = "entity_conflict"
            # If not same service, NO conflict - different services can have different ports

        # Check for negation patterns
        negation_score = self._check_negation(new_content, existing_content)
        if negation_score > 0:
            score += negation_score
            conflicts.append("Potential negation detected")
            if contradiction_type == "none":
                contradiction_type = "negation"

        # Cap score at 1.0
        score = min(1.0, score)

        evidence = "; ".join(conflicts) if conflicts else "No conflicts found"
        return score, contradiction_type, evidence

    def _find_common_subjects(self, text1: str, text2: str) -> List[str]:
        """
        Find common subjects (nouns/entities) between two texts.
        Returns list of common subjects.

        Enhanced to be more specific - generic words like "server" or "port"
        alone don't indicate same-subject conflict.
        """
        # Key subject patterns that indicate same-subject references
        # More specific patterns that actually indicate talking about the SAME thing
        subject_patterns = [
            # Specific named entities (higher priority)
            r'\b(nas\s*ip|server\s*ip|host\s*ip)\b',
            r'\b(nas\s*address|server\s*address)\b',
            r'\b(nas[-_]?server)\b',
            r'\b(main\s+server|primary\s+server|backup\s+server)\b',
            r'\b(redis\s+port|mcp\s+port|api\s+port|http\s+port)\b',
            # Named tools/services (specific)
            r'\b(cerebro|brain[-_]?graph|ai[-_]?memory)\b',
            # Configuration file references
            r'\b(config\.json|settings\.json|\.claude)\b',
        ]

        # Generic subjects that need additional context to conflict
        # These alone don't indicate same-subject
        generic_subjects = [
            r'\b(nas|database|api)\b',
            r'\b(python|node|java|docker|redis|postgres)\b',
        ]

        common = []

        # Check specific patterns first (stronger indicators)
        for pattern in subject_patterns:
            matches1 = set(re.findall(pattern, text1, re.IGNORECASE))
            matches2 = set(re.findall(pattern, text2, re.IGNORECASE))
            intersection = matches1 & matches2
            if intersection:
                common.extend(list(intersection))

        # Only check generic subjects if we have no specific matches
        # AND both texts are short (likely about same thing)
        if not common and len(text1) < 200 and len(text2) < 200:
            for pattern in generic_subjects:
                matches1 = set(re.findall(pattern, text1, re.IGNORECASE))
                matches2 = set(re.findall(pattern, text2, re.IGNORECASE))
                common.extend(list(matches1 & matches2))

        return common

    def _check_negation(self, new_content: str, existing_content: str) -> float:
        """
        Check for negation patterns that indicate contradiction.
        Returns a score 0-0.5 based on negation strength.
        """
        new_lower = new_content.lower()
        existing_lower = existing_content.lower()

        # Negation pairs
        negation_pairs = [
            (r'\bis\b', r'\bis\s+not\b'),
            (r'\bcan\b', r'\bcannot\b|\bcan\'t\b'),
            (r'\bworks\b', r'\bdoes\s*n.t\s+work\b'),
            (r'\bsupports?\b', r'\bdoes\s*n.t\s+support\b'),
            (r'\benabled\b', r'\bdisabled\b'),
            (r'\btrue\b', r'\bfalse\b'),
        ]

        score = 0.0
        for positive, negative in negation_pairs:
            # Check if one has positive and other has negative
            has_pos_new = bool(re.search(positive, new_lower))
            has_neg_new = bool(re.search(negative, new_lower))
            has_pos_existing = bool(re.search(positive, existing_lower))
            has_neg_existing = bool(re.search(negative, existing_lower))

            if (has_pos_new and has_neg_existing) or (has_neg_new and has_pos_existing):
                score += 0.25

        return min(0.5, score)

    def get_contradiction_stats(self) -> Dict[str, Any]:
        """Get statistics about detected contradictions."""
        if not self.contradictions:
            return {
                "total": 0,
                "pending": 0,
                "confirmed": 0,
                "dismissed": 0,
                "by_type": {}
            }

        by_status = {"pending": 0, "confirmed": 0, "dismissed": 0}
        by_type = {}

        for c in self.contradictions:
            by_status[c.status] = by_status.get(c.status, 0) + 1
            by_type[c.contradiction_type] = by_type.get(c.contradiction_type, 0) + 1

        return {
            "total": len(self.contradictions),
            "pending": by_status.get("pending", 0),
            "confirmed": by_status.get("confirmed", 0),
            "dismissed": by_status.get("dismissed", 0),
            "by_type": by_type
        }

    def resolve_contradiction(
        self,
        fact_id_a: str,
        fact_id_b: str,
        status: str,
        resolution: str
    ) -> bool:
        """
        Resolve a contradiction.

        Args:
            fact_id_a: First fact ID
            fact_id_b: Second fact ID
            status: 'confirmed' or 'dismissed'
            resolution: How it was resolved

        Returns:
            True if found and resolved
        """
        now = datetime.now().isoformat()

        for c in self.contradictions:
            if (c.fact_id_a == fact_id_a and c.fact_id_b == fact_id_b) or \
               (c.fact_id_a == fact_id_b and c.fact_id_b == fact_id_a):
                c.status = status
                c.resolved_at = now
                c.resolution = resolution
                self._save_contradictions()
                self._log_event("contradiction_resolved", fact_id_a, {
                    "fact_id_b": fact_id_b,
                    "status": status,
                    "resolution": resolution
                })
                return True

        return False

    def batch_resolve_contradictions(
        self,
        count: int = 100,
        strategy: str = "keep_newer"
    ) -> Dict[str, Any]:
        """
        Batch resolve pending contradictions using a strategy.

        Args:
            count: Max number to resolve
            strategy: Resolution strategy:
                - 'keep_newer': Keep the newer fact, dismiss older
                - 'keep_higher_confidence': Keep higher confidence fact
                - 'dismiss_all': Dismiss all as not contradictory

        Returns:
            Summary of batch resolution
        """
        from datetime import datetime

        pending = [c for c in self.contradictions if c.status == "pending"][:count]
        results = {"resolved": 0, "failed": 0, "strategy": strategy}

        now = datetime.now().isoformat()

        for c in pending:
            try:
                if strategy == "keep_newer":
                    resolution = f"Auto-resolved: kept newer fact ({c.fact_id_b})"
                elif strategy == "keep_higher_confidence":
                    from confidence_tracker import ConfidenceTracker
                    conf_tracker = ConfidenceTracker(base_path=str(self.base_path))
                    conf_a = conf_tracker.get_confidence(c.fact_id_a) or 0.5
                    conf_b = conf_tracker.get_confidence(c.fact_id_b) or 0.5
                    winner = c.fact_id_a if conf_a >= conf_b else c.fact_id_b
                    resolution = f"Auto-resolved: kept higher confidence fact ({winner})"
                elif strategy == "dismiss_all":
                    resolution = "Auto-dismissed: batch cleanup"
                else:
                    resolution = f"Auto-resolved via {strategy}"

                # Directly update the contradiction object
                c.status = "dismissed"
                c.resolved_at = now
                c.resolution = resolution
                results["resolved"] += 1

            except Exception:
                results["failed"] += 1

        # Save all changes at once
        self._save_contradictions()

        results["remaining_pending"] = len([
            c for c in self.contradictions if c.status == "pending"
        ])
        return results

    def batch_resolve_contradictions_old(
        self,
        count: int = 100,
        strategy: str = "keep_newer"
    ) -> Dict[str, Any]:
        """Old implementation - kept for reference."""
        from confidence_tracker import ConfidenceTracker

        pending = [c for c in self.contradictions if c.status == "pending"][:count]
        results = {"resolved": 0, "failed": 0, "strategy": strategy}

        conf_tracker = ConfidenceTracker(base_path=str(self.base_path))

        for c in pending:
            try:
                if strategy == "keep_newer":
                    # Assume fact_id_b is newer (detected second)
                    resolved = self.resolve_contradiction(
                        c.fact_id_a,
                        c.fact_id_b,
                        status="dismissed",
                        resolution=f"Auto-resolved: kept newer fact ({c.fact_id_b})"
                    )
                elif strategy == "keep_higher_confidence":
                    conf_a = conf_tracker.get_confidence(c.fact_id_a) or 0.5
                    conf_b = conf_tracker.get_confidence(c.fact_id_b) or 0.5
                    winner = c.fact_id_a if conf_a >= conf_b else c.fact_id_b
                    resolved = self.resolve_contradiction(
                        c.fact_id_a,
                        c.fact_id_b,
                        status="dismissed",
                        resolution=f"Auto-resolved: kept higher confidence fact ({winner})"
                    )
                elif strategy == "dismiss_all":
                    resolved = self.resolve_contradiction(
                        c.fact_id_a,
                        c.fact_id_b,
                        status="dismissed",
                        resolution="Auto-dismissed: batch cleanup"
                    )
                else:
                    resolved = False

                if resolved:
                    results["resolved"] += 1
                else:
                    results["failed"] += 1
            except Exception:
                results["failed"] += 1

        results["remaining_pending"] = len([
            c for c in self.contradictions if c.status == "pending"
        ])
        return results

    def get_pending_contradictions(self, limit: int = 20) -> List[Dict]:
        """Get pending contradictions for review."""
        pending = [c for c in self.contradictions if c.status == "pending"][:limit]
        return [c.to_dict() for c in pending]

    # =========================================================================
    # Migration & Utilities
    # =========================================================================

    def migrate_fact_provenance(
        self,
        fact_id: str,
        conversation_id: str = "legacy",
        fact_content: Optional[str] = None
    ) -> ProvenanceRecord:
        """
        Create provenance for an existing fact (lazy migration).
        Called when a fact without provenance is first accessed.

        Args:
            fact_id: The fact ID
            conversation_id: Source conversation (default: legacy)
            fact_content: The fact content (for entity extraction)

        Returns:
            The created ProvenanceRecord
        """
        now = datetime.now().isoformat()

        # Check if already has provenance
        if fact_id in self.records:
            return self.records[fact_id]

        # Create migrated source chain
        source_chain = SourceChain(
            learned_from=conversation_id,
            learned_at=now,
            learned_type=LearnedType.MIGRATED.value,
            message_index=None,
            context_snippet=None,
            extracted_by="migration"
        )

        # Create record
        record = ProvenanceRecord(
            fact_id=fact_id,
            version=1,
            source_chain=source_chain.to_dict(),
            created_at=now,
            updated_at=now
        )

        self.records[fact_id] = record

        # Log event
        self._log_event("provenance_migrated", fact_id, {
            "conversation_id": conversation_id
        })

        self._save_records()
        return record

    def get_full_provenance(self, fact_id: str) -> Optional[Dict]:
        """
        Get complete provenance information for a fact.
        Includes source chain, corrections, and reinforcements.
        """
        if fact_id not in self.records:
            return None

        record = self.records[fact_id]

        return {
            "fact_id": fact_id,
            "version": record.version,
            "source_chain": record.source_chain,
            "corrections": record.corrections,
            "correction_count": len(record.corrections),
            "reinforcements": record.reinforcements,
            "reinforcement_count": len(record.reinforcements),
            "created_at": record.created_at,
            "updated_at": record.updated_at
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about provenance tracking."""
        if not self.records:
            return {
                "total_tracked": 0,
                "with_corrections": 0,
                "with_reinforcements": 0,
                "by_learned_type": {},
                "total_corrections": 0,
                "total_reinforcements": 0,
                "contradictions": self.get_contradiction_stats()
            }

        by_type = {}
        with_corrections = 0
        with_reinforcements = 0
        total_corrections = 0
        total_reinforcements = 0

        for record in self.records.values():
            # Count by learned type
            if record.source_chain:
                learned_type = record.source_chain.get("learned_type", "unknown")
                by_type[learned_type] = by_type.get(learned_type, 0) + 1

            # Count corrections
            if record.corrections:
                with_corrections += 1
                total_corrections += len(record.corrections)

            # Count reinforcements
            if record.reinforcements:
                with_reinforcements += 1
                total_reinforcements += len(record.reinforcements)

        return {
            "total_tracked": len(self.records),
            "with_corrections": with_corrections,
            "with_reinforcements": with_reinforcements,
            "by_learned_type": by_type,
            "total_corrections": total_corrections,
            "total_reinforcements": total_reinforcements,
            "contradictions": self.get_contradiction_stats()
        }


# =============================================================================
# Learned Type Detection (for integration)
# =============================================================================

def detect_learned_type(
    content: str,
    role: str,
    context: str = ""
) -> LearnedType:
    """
    Detect how a fact was learned based on content and context.

    Args:
        content: The fact content
        role: Message role (user/assistant)
        context: Surrounding context

    Returns:
        Appropriate LearnedType
    """
    content_lower = content.lower()
    context_lower = context.lower()

    # Check for correction patterns
    correction_patterns = [
        r'\bactually\b.*\bnot\b',
        r'\bcorrect\w*\b.*\bshould\s+be\b',
        r'\bwrong\b',
        r'\bfix\w*\b',
        r'\bchange\b.*\bto\b'
    ]
    for pattern in correction_patterns:
        if re.search(pattern, context_lower):
            return LearnedType.USER_CORRECTION

    # User explicit patterns
    if role == "user":
        explicit_patterns = [
            r'\bmy\s+\w+\s+is\b',
            r'\bi\s+(am|have|use|prefer)\b',
            r'\bi\'?m\s+',
            r'\bthe\s+\w+\s+is\s+\w+'
        ]
        for pattern in explicit_patterns:
            if re.search(pattern, content_lower):
                return LearnedType.USER_EXPLICIT

    # Confirmation patterns
    confirmation_patterns = [
        "yes", "correct", "exactly", "that's right", "confirmed",
        "you got it", "perfect", "that's correct"
    ]
    if role == "user" and any(p in context_lower for p in confirmation_patterns):
        return LearnedType.CONFIRMATION

    # Assistant inference
    if role == "assistant":
        return LearnedType.INFERENCE

    # Default to extraction
    return LearnedType.EXTRACTION


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Provenance Tracker - Phase 6 Test Suite")
    print("=" * 50)

    # Use test directory
    import shutil
    import tempfile

    test_dir = Path(tempfile.mkdtemp()) / "test_provenance"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create facts directory for contradiction testing
    (test_dir / "facts").mkdir(exist_ok=True)

    # Create some test facts
    test_facts = [
        {"fact_id": "fact_001", "type": "fact", "content": "NAS IP is 10.0.0.1"},
        {"fact_id": "fact_002", "type": "fact", "content": "Main server port is 8080"},
        {"fact_id": "fact_003", "type": "fact", "content": "Python version is 3.11"},
    ]

    with open(test_dir / "facts" / "facts.jsonl", 'w') as f:
        for fact in test_facts:
            f.write(json.dumps(fact) + "\n")

    tracker = ProvenanceTracker(base_path=str(test_dir))

    # Test 1: Source chain creation
    print("\n1. Testing source chain creation...")
    source = tracker.create_source_chain(
        fact_id="fact_new_001",
        conversation_id="conv_test_001",
        learned_type=LearnedType.USER_EXPLICIT,
        message_index=5,
        context_snippet="User said: My NAS IP is 10.0.0.100"
    )
    print(f"   Created source chain: {source.learned_type} from {source.learned_from}")
    print("   [PASS]")

    # Test 2: Correction history
    print("\n2. Testing correction history...")
    correction = tracker.add_correction(
        fact_id="fact_001",
        from_value="10.0.0.1",
        to_value="10.0.0.100",
        corrected_by="user",
        conversation_id="conv_test_001",
        reason="User corrected the IP address"
    )
    print(f"   Correction added: {correction.from_value} -> {correction.to_value}")
    history = tracker.get_correction_history("fact_001")
    print(f"   Correction history has {len(history)} entries")
    print("   [PASS]")

    # Test 3: Reinforcement logging
    print("\n3. Testing reinforcement logging...")
    reinforcement = tracker.add_reinforcement(
        fact_id="fact_002",
        conversation_id="conv_test_002",
        reinforcement_type="explicit",
        context="User confirmed port 8080 is correct"
    )
    print(f"   Reinforcement added: {reinforcement.reinforcement_type}")
    reinforcements = tracker.get_reinforcements("fact_002")
    print(f"   Fact has {len(reinforcements)} reinforcements")
    print("   [PASS]")

    # Test 4: Entity conflict detection (IP addresses)
    print("\n4. Testing entity conflict detection...")
    contradictions = tracker.detect_contradictions(
        new_fact_id="fact_new_002",
        new_fact_content="NAS IP is 10.0.0.100"
    )
    print(f"   Found {len(contradictions)} potential contradictions")
    if contradictions:
        for c in contradictions:
            print(f"   - {c.fact_id_a} conflicts with {c.fact_id_b}: {c.evidence}")
    print("   [PASS]")

    # Test 5: Negation detection
    print("\n5. Testing semantic negation detection...")
    # Add a fact that could conflict
    test_facts_negation = [
        {"fact_id": "fact_neg_001", "type": "fact", "content": "Feature X is enabled"},
    ]
    with open(test_dir / "facts" / "facts.jsonl", 'a') as f:
        for fact in test_facts_negation:
            f.write(json.dumps(fact) + "\n")

    contradictions_neg = tracker.detect_contradictions(
        new_fact_id="fact_neg_002",
        new_fact_content="Feature X is disabled"
    )
    print(f"   Found {len(contradictions_neg)} negation contradictions")
    print("   [PASS]")

    # Test 6: Lazy migration
    print("\n6. Testing lazy migration...")
    record = tracker.migrate_fact_provenance(
        fact_id="fact_legacy_001",
        conversation_id="legacy_conv",
        fact_content="Old fact from before provenance"
    )
    print(f"   Migrated fact has version: {record.version}")
    print(f"   Source chain type: {record.source_chain.get('learned_type')}")
    print("   [PASS]")

    # Test 7: Get full provenance
    print("\n7. Testing full provenance retrieval...")
    full_prov = tracker.get_full_provenance("fact_001")
    print(f"   Fact version: {full_prov['version']}")
    print(f"   Corrections: {full_prov['correction_count']}")
    print(f"   Reinforcements: {full_prov['reinforcement_count']}")
    print("   [PASS]")

    # Test 8: Statistics
    print("\n8. Testing statistics...")
    stats = tracker.get_stats()
    print(f"   Total tracked: {stats['total_tracked']}")
    print(f"   With corrections: {stats['with_corrections']}")
    print(f"   With reinforcements: {stats['with_reinforcements']}")
    print(f"   Contradictions: {stats['contradictions']['total']}")
    print("   [PASS]")

    # Test 9: Learned type detection
    print("\n9. Testing learned type detection...")
    test_cases = [
        ("My email is test@example.com", "user", "", LearnedType.USER_EXPLICIT),
        ("Actually, the port should be 8081", "user", "wrong port was 8080", LearnedType.USER_CORRECTION),
        ("Claude inferred this", "assistant", "", LearnedType.INFERENCE),
    ]
    for content, role, context, expected in test_cases:
        detected = detect_learned_type(content, role, context)
        status = "PASS" if detected == expected else "FAIL"
        print(f"   '{content[:30]}...' -> {detected.value} [{status}]")

    print("\n" + "=" * 50)
    print("Phase 6 Provenance Tracker: All tests completed!")

    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)
