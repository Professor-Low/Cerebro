"""
Goal Tracker - Claude.Me v6.0
Track and infer user goals for proactive memory retrieval.

Part of Phase 6: Goal-Directed Memory Access
"""
import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class Goal:
    """Represents a user goal or intention."""

    def __init__(
        self,
        goal_id: str,
        description: str,
        inferred_from: str = "explicit",
        priority: str = "medium",
        subgoals: List[str] = None,
        relevant_memory_ids: List[str] = None,
        known_blockers: List[str] = None,
        status: str = "active"
    ):
        self.goal_id = goal_id
        self.description = description
        self.inferred_from = inferred_from  # explicit, implicit, context
        self.priority = priority  # high, medium, low
        self.subgoals = subgoals or []
        self.relevant_memory_ids = relevant_memory_ids or []
        self.known_blockers = known_blockers or []
        self.status = status  # active, paused, completed, abandoned
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at

    def to_dict(self) -> Dict:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "inferred_from": self.inferred_from,
            "priority": self.priority,
            "subgoals": self.subgoals,
            "relevant_memory_ids": self.relevant_memory_ids,
            "known_blockers": self.known_blockers,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Goal":
        goal = cls(
            goal_id=data["goal_id"],
            description=data["description"],
            inferred_from=data.get("inferred_from", "explicit"),
            priority=data.get("priority", "medium"),
            subgoals=data.get("subgoals", []),
            relevant_memory_ids=data.get("relevant_memory_ids", []),
            known_blockers=data.get("known_blockers", []),
            status=data.get("status", "active")
        )
        goal.created_at = data.get("created_at", goal.created_at)
        goal.updated_at = data.get("updated_at", goal.updated_at)
        return goal


class GoalTracker:
    """
    Track user goals and intentions.

    Capabilities:
    - Detect goals from user statements
    - Track goal progress
    - Link goals to relevant memories
    - Identify blockers
    - Suggest proactive retrieval
    """

    # Goal detection patterns
    GOAL_PATTERNS = [
        (r'\b(?:I|we)\s+(?:want|need|would\s+like|\'d\s+like)\s+to\s+(.+?)(?:\.|,|!|$)', 'explicit'),
        (r'\b(?:goal|objective|aim|purpose)\s+is\s+to\s+(.+?)(?:\.|,|!|$)', 'explicit'),
        (r'\b(?:trying|attempting|working\s+on)\s+(?:to\s+)?(.+?)(?:\.|,|!|$)', 'implicit'),
        (r'\b(?:plan|planning|intend(?:ing)?)\s+to\s+(.+?)(?:\.|,|!|$)', 'explicit'),
        (r'\blet\'?s\s+(.+?)(?:\.|,|!|$)', 'implicit'),
        (r'\bhelp\s+(?:me\s+)?(.+?)(?:\.|,|!|$)', 'explicit'),
    ]

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.goals_path = self.base_path / "goals"
        self.active_file = self.goals_path / "active.json"
        self._lock = threading.Lock()
        self._ensure_directories()

    def _ensure_directories(self):
        """Create directories if they don't exist."""
        self.goals_path.mkdir(parents=True, exist_ok=True)

    def _generate_id(self) -> str:
        """Generate goal ID."""
        import hashlib
        ts = datetime.now().isoformat()
        return f"goal_{hashlib.sha256(ts.encode()).hexdigest()[:10]}"

    def _load_active(self) -> Dict:
        """Load active goals."""
        if self.active_file.exists():
            try:
                with open(self.active_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"goals": [], "updated_at": datetime.now().isoformat()}

    def _save_active(self, data: Dict):
        """Save active goals."""
        data["updated_at"] = datetime.now().isoformat()
        with self._lock:
            with open(self.active_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

    def detect_goals(self, text: str) -> List[Goal]:
        """
        Detect goals from user text.

        Returns list of detected goals.
        """
        detected = []

        for pattern, inference_type in self.GOAL_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                goal_text = match.group(1).strip()

                # Skip very short or generic matches
                if len(goal_text) < 10:
                    continue

                # Skip if it looks like a question
                if goal_text.endswith("?"):
                    continue

                goal = Goal(
                    goal_id=self._generate_id(),
                    description=goal_text[:200],
                    inferred_from=inference_type,
                    priority=self._infer_priority(goal_text)
                )
                detected.append(goal)

        return detected

    def _infer_priority(self, goal_text: str) -> str:
        """Infer priority from goal text."""
        text_lower = goal_text.lower()

        high_indicators = ["urgent", "asap", "immediately", "critical", "important", "must"]
        low_indicators = ["eventually", "someday", "might", "could", "maybe", "later"]

        if any(ind in text_lower for ind in high_indicators):
            return "high"
        if any(ind in text_lower for ind in low_indicators):
            return "low"
        return "medium"

    def add_goal(self, goal: Goal) -> str:
        """Add a goal to tracking."""
        data = self._load_active()

        # Check for duplicates
        for existing in data["goals"]:
            if existing["description"].lower() == goal.description.lower():
                return existing["goal_id"]  # Already exists

        data["goals"].append(goal.to_dict())
        self._save_active(data)

        # Save individual goal file
        goal_file = self.goals_path / f"{goal.goal_id}.json"
        with open(goal_file, 'w', encoding='utf-8') as f:
            json.dump(goal.to_dict(), f, indent=2)

        return goal.goal_id

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        goal_file = self.goals_path / f"{goal_id}.json"
        if not goal_file.exists():
            return None

        try:
            with open(goal_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Goal.from_dict(data)
        except:
            return None

    def get_active_goals(self) -> List[Goal]:
        """Get all active goals."""
        data = self._load_active()
        goals = []
        for g in data.get("goals", []):
            if g.get("status") == "active":
                goals.append(Goal.from_dict(g))
        return goals

    def update_goal(
        self,
        goal_id: str,
        status: str = None,
        add_subgoal: str = None,
        add_blocker: str = None,
        add_memory: str = None
    ) -> Optional[Goal]:
        """Update a goal."""
        goal = self.get_goal(goal_id)
        if not goal:
            return None

        if status:
            goal.status = status
        if add_subgoal:
            goal.subgoals.append(add_subgoal)
        if add_blocker:
            goal.known_blockers.append(add_blocker)
        if add_memory:
            goal.relevant_memory_ids.append(add_memory)

        goal.updated_at = datetime.now().isoformat()

        # Save individual file
        goal_file = self.goals_path / f"{goal_id}.json"
        with open(goal_file, 'w', encoding='utf-8') as f:
            json.dump(goal.to_dict(), f, indent=2)

        # Update active list
        data = self._load_active()
        for i, g in enumerate(data.get("goals", [])):
            if g["goal_id"] == goal_id:
                data["goals"][i] = goal.to_dict()
                break
        self._save_active(data)

        return goal

    def complete_goal(self, goal_id: str) -> bool:
        """Mark a goal as completed."""
        goal = self.update_goal(goal_id, status="completed")
        return goal is not None

    def find_relevant_goals(self, context: str) -> List[Goal]:
        """Find goals relevant to the current context."""
        context_lower = context.lower()
        context_words = set(context_lower.split())

        active_goals = self.get_active_goals()
        scored_goals = []

        for goal in active_goals:
            goal_words = set(goal.description.lower().split())
            overlap = len(context_words.intersection(goal_words))

            if overlap >= 2:
                scored_goals.append((goal, overlap))

        # Sort by relevance
        scored_goals.sort(key=lambda x: x[1], reverse=True)
        return [g[0] for g in scored_goals]

    def get_blockers_for_context(self, context: str) -> List[Dict]:
        """Get known blockers relevant to current context."""
        relevant_goals = self.find_relevant_goals(context)
        blockers = []

        for goal in relevant_goals:
            for blocker in goal.known_blockers:
                blockers.append({
                    "blocker": blocker,
                    "goal": goal.description,
                    "goal_id": goal.goal_id
                })

        return blockers

    def get_stats(self) -> Dict:
        """Get goal tracking statistics."""
        data = self._load_active()
        goals = data.get("goals", [])

        by_status = {}
        by_priority = {}

        for g in goals:
            status = g.get("status", "unknown")
            by_status[status] = by_status.get(status, 0) + 1

            priority = g.get("priority", "medium")
            by_priority[priority] = by_priority.get(priority, 0) + 1

        return {
            "total_goals": len(goals),
            "by_status": by_status,
            "by_priority": by_priority
        }
