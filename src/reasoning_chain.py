"""
Reasoning Chain - Claude.Me v6.0
Represents a chain of thought during problem-solving.

Part of Phase 2: Working Memory Integration
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class EvaluationStatus(Enum):
    """Status of hypothesis evaluation"""
    UNTESTED = "untested"
    TESTING = "testing"
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class EvidenceType(Enum):
    """Type of evidence in a reasoning chain"""
    OBSERVATION = "observation"       # Something we observed/noticed
    TEST_RESULT = "test_result"       # Result of a test we ran
    MEMORY_RECALL = "memory_recall"   # Information from memory
    INFERENCE = "inference"           # Logical deduction
    USER_INPUT = "user_input"         # Information from user
    ASSUMPTION = "assumption"         # Something we're assuming


@dataclass
class Evidence:
    """A piece of evidence in a reasoning chain."""
    type: str  # EvidenceType value
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 0.8
    source: Optional[str] = None  # Where this evidence came from

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "source": self.source
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Evidence":
        return cls(
            type=data["type"],
            content=data["content"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            confidence=data.get("confidence", 0.8),
            source=data.get("source")
        )


@dataclass
class ReasoningChain:
    """
    A chain of reasoning - hypothesis, evidence, evaluation.

    Example chain:
    - Hypothesis: "NAS is unreachable"
    - Evidence: [observation: socket timeout, test_result: ping succeeds]
    - Evaluation: partially_supported
    - Next step: check SMB port specifically
    """
    chain_id: str
    hypothesis: str
    evidence: List[Evidence] = field(default_factory=list)
    evaluation: str = "untested"  # EvaluationStatus value
    next_step: Optional[str] = None
    outcome: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    parent_chain_id: Optional[str] = None  # If this chain branches from another

    def add_evidence(self, evidence_type: str, content: str, confidence: float = 0.8, source: str = None):
        """Add evidence to the chain."""
        self.evidence.append(Evidence(
            type=evidence_type,
            content=content,
            confidence=confidence,
            source=source
        ))
        self.updated_at = datetime.now().isoformat()

    def evaluate(self, status: str, next_step: str = None, outcome: str = None):
        """Update the evaluation of this chain."""
        self.evaluation = status
        if next_step:
            self.next_step = next_step
        if outcome:
            self.outcome = outcome
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "chain_id": self.chain_id,
            "hypothesis": self.hypothesis,
            "evidence": [e.to_dict() for e in self.evidence],
            "evaluation": self.evaluation,
            "next_step": self.next_step,
            "outcome": self.outcome,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "parent_chain_id": self.parent_chain_id
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ReasoningChain":
        return cls(
            chain_id=data["chain_id"],
            hypothesis=data["hypothesis"],
            evidence=[Evidence.from_dict(e) for e in data.get("evidence", [])],
            evaluation=data.get("evaluation", "untested"),
            next_step=data.get("next_step"),
            outcome=data.get("outcome"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            parent_chain_id=data.get("parent_chain_id")
        )

    def summarize(self) -> str:
        """Generate a brief summary of this reasoning chain."""
        evidence_summary = f"{len(self.evidence)} pieces of evidence"
        status = self.evaluation
        if self.outcome:
            return f"Hypothesis: {self.hypothesis[:50]}... | Status: {status} | Outcome: {self.outcome[:50]}..."
        elif self.next_step:
            return f"Hypothesis: {self.hypothesis[:50]}... | Status: {status} | Next: {self.next_step[:50]}..."
        else:
            return f"Hypothesis: {self.hypothesis[:50]}... | Status: {status} | {evidence_summary}"
