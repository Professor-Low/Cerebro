"""
Memory Type Definitions - Cerebro AI Memory
Distinguishes episodic (what happened) from semantic (general knowledge) memory.

Part of Phase 1: Episodic vs Semantic Separation
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryType(Enum):
    """Core memory types for AGI-level cognition"""
    EPISODIC = "episodic"      # Events with context ("what happened on date X")
    SEMANTIC = "semantic"      # General facts ("NAS timeout is 60s")
    PROCEDURAL = "procedural"  # How-to knowledge ("steps to debug MCP")
    WORKING = "working"        # Active reasoning state (temporary)


class EmotionalState(Enum):
    """Emotional context markers for episodic memories"""
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    RELIEVED = "relieved"
    EXCITED = "excited"
    CONFUSED = "confused"
    CONFIDENT = "confident"


@dataclass
class EpisodicMemory:
    """
    An episodic memory - a specific event with full context.

    Examples:
    - "On 2026-01-30, User debugged NAS timeout for 45 min, fixed by increasing to 60s"
    - "Session where we implemented emotion tracker - User was initially frustrated"
    """
    id: str
    timestamp: str
    event: str
    context: Dict[str, Any] = field(default_factory=dict)
    actors: List[str] = field(default_factory=list)
    outcome: Optional[str] = None
    emotional_state: Optional[str] = None
    duration_minutes: Optional[int] = None
    linked_semantic_ids: List[str] = field(default_factory=list)
    conversation_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": MemoryType.EPISODIC.value,
            "timestamp": self.timestamp,
            "event": self.event,
            "context": self.context,
            "actors": self.actors,
            "outcome": self.outcome,
            "emotional_state": self.emotional_state,
            "duration_minutes": self.duration_minutes,
            "linked_semantic_ids": self.linked_semantic_ids,
            "conversation_id": self.conversation_id
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EpisodicMemory":
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            event=data["event"],
            context=data.get("context", {}),
            actors=data.get("actors", []),
            outcome=data.get("outcome"),
            emotional_state=data.get("emotional_state"),
            duration_minutes=data.get("duration_minutes"),
            linked_semantic_ids=data.get("linked_semantic_ids", []),
            conversation_id=data.get("conversation_id")
        )


@dataclass
class SemanticMemory:
    """
    A semantic memory - general knowledge abstracted from episodes.

    Examples:
    - "NAS operations require 60s timeout due to SMB latency"
    - "User prefers direct communication without pleasantries"
    """
    id: str
    fact: str
    generalized_from: List[str] = field(default_factory=list)  # episodic IDs
    confidence: float = 0.8
    domain: str = "general"
    access_count: int = 0
    last_accessed: Optional[str] = None
    created_at: Optional[str] = None
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": MemoryType.SEMANTIC.value,
            "fact": self.fact,
            "generalized_from": self.generalized_from,
            "confidence": self.confidence,
            "domain": self.domain,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "created_at": self.created_at or datetime.now().isoformat(),
            "keywords": self.keywords
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SemanticMemory":
        return cls(
            id=data["id"],
            fact=data["fact"],
            generalized_from=data.get("generalized_from", []),
            confidence=data.get("confidence", 0.8),
            domain=data.get("domain", "general"),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed"),
            created_at=data.get("created_at"),
            keywords=data.get("keywords", [])
        )


@dataclass
class ProceduralMemory:
    """
    A procedural memory - how-to knowledge.

    Examples:
    - "To debug MCP timeouts: 1) Check NAS, 2) Verify socket, 3) Check filesystem"
    - "Python bytes escaping: use \\x5c for literal backslash"
    """
    id: str
    title: str
    steps: List[str]
    domain: str = "general"
    prerequisites: List[str] = field(default_factory=list)
    success_rate: float = 1.0
    last_used: Optional[str] = None
    linked_episodic_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": MemoryType.PROCEDURAL.value,
            "title": self.title,
            "steps": self.steps,
            "domain": self.domain,
            "prerequisites": self.prerequisites,
            "success_rate": self.success_rate,
            "last_used": self.last_used,
            "linked_episodic_ids": self.linked_episodic_ids
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ProceduralMemory":
        return cls(
            id=data["id"],
            title=data["title"],
            steps=data["steps"],
            domain=data.get("domain", "general"),
            prerequisites=data.get("prerequisites", []),
            success_rate=data.get("success_rate", 1.0),
            last_used=data.get("last_used"),
            linked_episodic_ids=data.get("linked_episodic_ids", [])
        )


# Domain categories for semantic memories
SEMANTIC_DOMAINS = [
    "infrastructure",      # NAS, networking, servers
    "development",         # Code patterns, languages, tools
    "configuration",       # Settings, paths, environment
    "user_preference",     # User's likes/dislikes
    "debugging",           # Common issues and fixes
    "workflow",            # Process knowledge
    "general"              # Uncategorized
]


def classify_memory_type(content: str, context: Dict) -> MemoryType:
    """
    Classify whether content is episodic or semantic.

    Episodic indicators: specific dates, "I did", "we tried", specific outcomes
    Semantic indicators: general statements, "always", "never", definitions
    """
    content_lower = content.lower()

    # Episodic indicators
    episodic_patterns = [
        "yesterday", "today", "last week", "on january", "on february",
        "i tried", "we tried", "i did", "we did", "i found", "we found",
        "this morning", "this afternoon", "when i", "when we",
        "debugged", "fixed", "resolved", "discovered", "realized"
    ]

    # Semantic indicators
    semantic_patterns = [
        "always", "never", "typically", "usually", "is a", "are a",
        "requires", "should be", "must be", "needs to",
        "by default", "in general", "the correct", "the standard"
    ]

    episodic_score = sum(1 for p in episodic_patterns if p in content_lower)
    semantic_score = sum(1 for p in semantic_patterns if p in content_lower)

    # Context-based boosting
    if context.get("has_timestamp"):
        episodic_score += 2
    if context.get("has_conversation_id"):
        episodic_score += 1
    if context.get("is_generalized"):
        semantic_score += 2

    return MemoryType.EPISODIC if episodic_score > semantic_score else MemoryType.SEMANTIC


def extract_domain(content: str) -> str:
    """Extract the domain category from content."""
    content_lower = content.lower()

    domain_keywords = {
        "infrastructure": ["nas", "network", "server", "ip", "port", "smb", "ssh"],
        "development": ["code", "function", "class", "python", "javascript", "api"],
        "configuration": ["config", "setting", "path", "environment", "variable"],
        "user_preference": ["prefer", "like", "dislike", "want", "style"],
        "debugging": ["error", "bug", "fix", "debug", "issue", "problem"],
        "workflow": ["process", "step", "workflow", "procedure", "routine"]
    }

    scores = {}
    for domain, keywords in domain_keywords.items():
        scores[domain] = sum(1 for kw in keywords if kw in content_lower)

    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return "general"
