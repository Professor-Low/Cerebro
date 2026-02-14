"""
Self Model - Claude.Me v6.0
Track Claude's state during reasoning for continuous self-awareness.

Part of Phase 9: Continuous Self-Modeling
"""
import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class CurrentState:
    """Represents Claude's current cognitive state."""

    def __init__(
        self,
        confidence_level: float = 0.5,
        uncertainty_topics: List[str] = None,
        known_limitations: List[str] = None,
        strengths_today: List[str] = None
    ):
        self.confidence_level = confidence_level
        self.uncertainty_topics = uncertainty_topics or []
        self.known_limitations = known_limitations or []
        self.strengths_today = strengths_today or []

    def to_dict(self) -> Dict:
        return {
            "confidence_level": self.confidence_level,
            "uncertainty_topics": self.uncertainty_topics,
            "known_limitations": self.known_limitations,
            "strengths_today": self.strengths_today
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CurrentState":
        return cls(
            confidence_level=data.get("confidence_level", 0.5),
            uncertainty_topics=data.get("uncertainty_topics", []),
            known_limitations=data.get("known_limitations", []),
            strengths_today=data.get("strengths_today", [])
        )


class ReasoningQuality:
    """Represents reasoning quality metrics."""

    def __init__(
        self,
        task_clarity: float = 0.5,
        evidence_sufficiency: float = 0.5,
        hallucination_risk: float = 0.5
    ):
        self.task_clarity = task_clarity
        self.evidence_sufficiency = evidence_sufficiency
        self.hallucination_risk = hallucination_risk

    def to_dict(self) -> Dict:
        return {
            "task_clarity": self.task_clarity,
            "evidence_sufficiency": self.evidence_sufficiency,
            "hallucination_risk": self.hallucination_risk
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ReasoningQuality":
        return cls(
            task_clarity=data.get("task_clarity", 0.5),
            evidence_sufficiency=data.get("evidence_sufficiency", 0.5),
            hallucination_risk=data.get("hallucination_risk", 0.5)
        )


class Behaviors:
    """Recommended behaviors based on self-model."""

    def __init__(
        self,
        should_ask_clarification: bool = False,
        should_verify_facts: bool = False,
        should_slow_down: bool = False
    ):
        self.should_ask_clarification = should_ask_clarification
        self.should_verify_facts = should_verify_facts
        self.should_slow_down = should_slow_down

    def to_dict(self) -> Dict:
        return {
            "should_ask_clarification": self.should_ask_clarification,
            "should_verify_facts": self.should_verify_facts,
            "should_slow_down": self.should_slow_down
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Behaviors":
        return cls(
            should_ask_clarification=data.get("should_ask_clarification", False),
            should_verify_facts=data.get("should_verify_facts", False),
            should_slow_down=data.get("should_slow_down", False)
        )


class SelfModelSnapshot:
    """A snapshot of the self-model at a point in time."""

    def __init__(
        self,
        snapshot_id: str,
        current_state: CurrentState,
        reasoning_quality: ReasoningQuality,
        behaviors: Behaviors,
        context: str = None
    ):
        self.snapshot_id = snapshot_id
        self.current_state = current_state
        self.reasoning_quality = reasoning_quality
        self.behaviors = behaviors
        self.context = context
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "snapshot_id": self.snapshot_id,
            "current_state": self.current_state.to_dict(),
            "reasoning_quality": self.reasoning_quality.to_dict(),
            "behaviors": self.behaviors.to_dict(),
            "context": self.context,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SelfModelSnapshot":
        snapshot = cls(
            snapshot_id=data["snapshot_id"],
            current_state=CurrentState.from_dict(data.get("current_state", {})),
            reasoning_quality=ReasoningQuality.from_dict(data.get("reasoning_quality", {})),
            behaviors=Behaviors.from_dict(data.get("behaviors", {})),
            context=data.get("context")
        )
        snapshot.timestamp = data.get("timestamp", snapshot.timestamp)
        return snapshot


class SelfModelManager:
    """
    Track and update Claude's self-model.

    Capabilities:
    - Track confidence and uncertainty
    - Monitor reasoning quality
    - Suggest behavioral adjustments
    - Detect potential hallucination risk
    - Historical self-awareness
    """

    # Uncertainty indicators in text
    UNCERTAINTY_INDICATORS = [
        "i think", "might be", "probably", "not sure", "unclear",
        "possibly", "maybe", "could be", "seems like", "appears to",
        "i believe", "i'm uncertain", "hard to say", "don't know"
    ]

    # High-risk topic patterns
    HIGH_RISK_TOPICS = [
        "specific date", "exact number", "precise quote", "url", "email",
        "phone number", "address", "code execution", "system command"
    ]

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.self_model_path = self.base_path / "self_model"
        self.current_file = self.self_model_path / "current.json"
        self.history_path = self.self_model_path / "history"
        self._lock = threading.Lock()
        self._ensure_directories()

    def _ensure_directories(self):
        """Create directories if they don't exist."""
        self.self_model_path.mkdir(parents=True, exist_ok=True)
        self.history_path.mkdir(parents=True, exist_ok=True)

    def _generate_id(self) -> str:
        """Generate snapshot ID."""
        ts = datetime.now().isoformat()
        return f"sm_{hashlib.sha256(ts.encode()).hexdigest()[:10]}"

    def _load_current(self) -> Dict:
        """Load current self-model state."""
        if self.current_file.exists():
            try:
                with open(self.current_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return self._default_state()

    def _default_state(self) -> Dict:
        """Get default self-model state."""
        return {
            "last_updated": datetime.now().isoformat(),
            "current_state": CurrentState().to_dict(),
            "reasoning_quality": ReasoningQuality().to_dict(),
            "behaviors": Behaviors().to_dict()
        }

    def _save_current(self, data: Dict):
        """Save current self-model state."""
        data["last_updated"] = datetime.now().isoformat()
        with self._lock:
            with open(self.current_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

    def get_current_state(self) -> Dict:
        """Get current self-model state."""
        return self._load_current()

    def update_confidence(self, confidence: float, topic: str = None) -> Dict:
        """
        Update confidence level.

        Args:
            confidence: New confidence level (0-1)
            topic: Optional topic this relates to
        """
        data = self._load_current()
        data["current_state"]["confidence_level"] = max(0, min(1, confidence))

        if topic and confidence < 0.5:
            uncertainties = data["current_state"].get("uncertainty_topics", [])
            if topic not in uncertainties:
                uncertainties.append(topic)
                data["current_state"]["uncertainty_topics"] = uncertainties[-10:]

        self._recalculate_behaviors(data)
        self._save_current(data)

        return {
            "success": True,
            "confidence": data["current_state"]["confidence_level"],
            "behaviors": data["behaviors"]
        }

    def add_uncertainty(self, topic: str) -> Dict:
        """Add a topic to uncertainty list."""
        data = self._load_current()
        uncertainties = data["current_state"].get("uncertainty_topics", [])

        if topic not in uncertainties:
            uncertainties.append(topic)
            data["current_state"]["uncertainty_topics"] = uncertainties[-10:]

        # Lower confidence when adding uncertainty
        data["current_state"]["confidence_level"] = max(
            0.3, data["current_state"]["confidence_level"] - 0.1
        )

        self._recalculate_behaviors(data)
        self._save_current(data)

        return {"success": True, "uncertainty_topics": uncertainties}

    def add_limitation(self, limitation: str) -> Dict:
        """Add a known limitation."""
        data = self._load_current()
        limitations = data["current_state"].get("known_limitations", [])

        if limitation not in limitations:
            limitations.append(limitation)
            data["current_state"]["known_limitations"] = limitations[-10:]

        self._save_current(data)
        return {"success": True, "known_limitations": limitations}

    def add_strength(self, strength: str) -> Dict:
        """Add a recognized strength."""
        data = self._load_current()
        strengths = data["current_state"].get("strengths_today", [])

        if strength not in strengths:
            strengths.append(strength)
            data["current_state"]["strengths_today"] = strengths[-10:]

        # Slightly boost confidence when recognizing strength
        data["current_state"]["confidence_level"] = min(
            0.9, data["current_state"]["confidence_level"] + 0.05
        )

        self._save_current(data)
        return {"success": True, "strengths_today": strengths}

    def update_reasoning_quality(
        self,
        task_clarity: float = None,
        evidence_sufficiency: float = None,
        hallucination_risk: float = None
    ) -> Dict:
        """Update reasoning quality metrics."""
        data = self._load_current()

        if task_clarity is not None:
            data["reasoning_quality"]["task_clarity"] = max(0, min(1, task_clarity))
        if evidence_sufficiency is not None:
            data["reasoning_quality"]["evidence_sufficiency"] = max(0, min(1, evidence_sufficiency))
        if hallucination_risk is not None:
            data["reasoning_quality"]["hallucination_risk"] = max(0, min(1, hallucination_risk))

        self._recalculate_behaviors(data)
        self._save_current(data)

        return {
            "success": True,
            "reasoning_quality": data["reasoning_quality"],
            "behaviors": data["behaviors"]
        }

    def _recalculate_behaviors(self, data: Dict):
        """Recalculate recommended behaviors based on state."""
        state = data.get("current_state", {})
        quality = data.get("reasoning_quality", {})

        behaviors = {
            "should_ask_clarification": False,
            "should_verify_facts": False,
            "should_slow_down": False
        }

        # Low task clarity -> ask clarification
        if quality.get("task_clarity", 0.5) < 0.5:
            behaviors["should_ask_clarification"] = True

        # Low confidence or evidence -> verify facts
        if (state.get("confidence_level", 0.5) < 0.5 or
                quality.get("evidence_sufficiency", 0.5) < 0.5):
            behaviors["should_verify_facts"] = True

        # High hallucination risk -> slow down and verify
        if quality.get("hallucination_risk", 0.5) > 0.6:
            behaviors["should_slow_down"] = True
            behaviors["should_verify_facts"] = True

        # Many uncertainties -> slow down
        if len(state.get("uncertainty_topics", [])) >= 3:
            behaviors["should_slow_down"] = True

        data["behaviors"] = behaviors

    def assess_text(self, text: str) -> Dict:
        """
        Assess text for uncertainty indicators and risk factors.
        Updates self-model based on assessment.
        """
        text_lower = text.lower()

        # Count uncertainty indicators
        uncertainty_count = sum(
            1 for indicator in self.UNCERTAINTY_INDICATORS
            if indicator in text_lower
        )

        # Check for high-risk topics
        risk_topics = [
            topic for topic in self.HIGH_RISK_TOPICS
            if topic in text_lower
        ]

        # Calculate metrics
        uncertainty_ratio = min(1.0, uncertainty_count * 0.15)
        hallucination_risk = min(1.0, len(risk_topics) * 0.2 + uncertainty_ratio * 0.3)

        # Update reasoning quality
        data = self._load_current()
        data["reasoning_quality"]["hallucination_risk"] = hallucination_risk

        # Adjust confidence based on uncertainty
        if uncertainty_ratio > 0.3:
            data["current_state"]["confidence_level"] = max(
                0.3,
                data["current_state"]["confidence_level"] - uncertainty_ratio * 0.2
            )

        self._recalculate_behaviors(data)
        self._save_current(data)

        return {
            "uncertainty_indicators_found": uncertainty_count,
            "risk_topics_found": risk_topics,
            "hallucination_risk": hallucination_risk,
            "behaviors": data["behaviors"]
        }

    def take_snapshot(self, context: str = None) -> str:
        """Take a snapshot of current self-model state."""
        data = self._load_current()

        snapshot = SelfModelSnapshot(
            snapshot_id=self._generate_id(),
            current_state=CurrentState.from_dict(data.get("current_state", {})),
            reasoning_quality=ReasoningQuality.from_dict(data.get("reasoning_quality", {})),
            behaviors=Behaviors.from_dict(data.get("behaviors", {})),
            context=context
        )

        # Save snapshot
        snapshot_file = self.history_path / f"{snapshot.snapshot_id}.json"
        with open(snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(snapshot.to_dict(), f, indent=2)

        return snapshot.snapshot_id

    def get_snapshot(self, snapshot_id: str) -> Optional[Dict]:
        """Get a specific snapshot."""
        snapshot_file = self.history_path / f"{snapshot_id}.json"
        if not snapshot_file.exists():
            return None

        try:
            with open(snapshot_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None

    def get_recent_snapshots(self, limit: int = 10) -> List[Dict]:
        """Get recent snapshots."""
        snapshots = []

        for snapshot_file in sorted(self.history_path.glob("sm_*.json"), reverse=True):
            if len(snapshots) >= limit:
                break

            try:
                with open(snapshot_file, 'r', encoding='utf-8') as f:
                    snapshots.append(json.load(f))
            except:
                continue

        return snapshots

    def reset_daily(self) -> Dict:
        """Reset daily metrics (call at start of new day/session)."""
        data = self._load_current()

        # Keep limitations but reset daily tracking
        data["current_state"]["confidence_level"] = 0.7
        data["current_state"]["uncertainty_topics"] = []
        data["current_state"]["strengths_today"] = []

        data["reasoning_quality"]["task_clarity"] = 0.7
        data["reasoning_quality"]["evidence_sufficiency"] = 0.7
        data["reasoning_quality"]["hallucination_risk"] = 0.2

        self._recalculate_behaviors(data)
        self._save_current(data)

        return {"success": True, "state": "reset", "current": data}

    def get_stats(self) -> Dict:
        """Get self-model statistics."""
        data = self._load_current()

        snapshot_count = len(list(self.history_path.glob("sm_*.json")))

        return {
            "current_confidence": data.get("current_state", {}).get("confidence_level", 0.5),
            "uncertainty_count": len(data.get("current_state", {}).get("uncertainty_topics", [])),
            "limitation_count": len(data.get("current_state", {}).get("known_limitations", [])),
            "hallucination_risk": data.get("reasoning_quality", {}).get("hallucination_risk", 0.5),
            "behaviors": data.get("behaviors", {}),
            "total_snapshots": snapshot_count,
            "last_updated": data.get("last_updated")
        }
