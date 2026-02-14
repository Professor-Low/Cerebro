"""
Proactive Retrieval - Claude.Me v6.0
Pre-fetch relevant memories based on user goals.

Part of Phase 6: Goal-Directed Memory Access
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from goal_tracker import GoalTracker


class ProactiveRetrieval:
    """
    Proactively retrieve memories based on user goals.

    Capabilities:
    - Pre-fetch memories relevant to active goals
    - Surface warnings about known pitfalls
    - Suggest relevant context before user asks
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.goal_tracker = GoalTracker(base_path)

    def get_proactive_context(self, user_message: str) -> Dict:
        """
        Get proactive context for a user message.

        Returns relevant memories, warnings, and suggestions.
        """
        result = {
            "relevant_goals": [],
            "relevant_memories": [],
            "warnings": [],
            "suggestions": [],
            "timestamp": datetime.now().isoformat()
        }

        # Find relevant goals
        relevant_goals = self.goal_tracker.find_relevant_goals(user_message)
        for goal in relevant_goals[:3]:
            result["relevant_goals"].append({
                "goal_id": goal.goal_id,
                "description": goal.description[:100],
                "priority": goal.priority,
                "known_blockers": goal.known_blockers[:3]
            })

        # Get blockers as warnings
        blockers = self.goal_tracker.get_blockers_for_context(user_message)
        for blocker in blockers[:5]:
            result["warnings"].append({
                "warning": f"Known blocker: {blocker['blocker']}",
                "related_goal": blocker['goal'][:50]
            })

        # Search for relevant semantic memories
        semantic_path = self.base_path / "semantic"
        if semantic_path.exists():
            keywords = set(user_message.lower().split())
            for sem_file in semantic_path.glob("sem_*.json"):
                try:
                    with open(sem_file, 'r', encoding='utf-8') as f:
                        sem = json.load(f)

                    fact_words = set(sem.get("fact", "").lower().split())
                    if len(keywords.intersection(fact_words)) >= 2:
                        result["relevant_memories"].append({
                            "type": "semantic",
                            "id": sem.get("id"),
                            "content": sem.get("fact", "")[:150],
                            "confidence": sem.get("confidence", 0.5)
                        })
                        if len(result["relevant_memories"]) >= 5:
                            break
                except:
                    continue

        # Search causal model for relevant causes/effects
        causal_path = self.base_path / "causal" / "links"
        if causal_path.exists():
            for link_file in causal_path.glob("cl_*.json"):
                try:
                    with open(link_file, 'r', encoding='utf-8') as f:
                        link = json.load(f)

                    searchable = f"{link.get('cause', '')} {link.get('effect', '')}".lower()
                    if any(kw in searchable for kw in keywords):
                        result["suggestions"].append({
                            "type": "causal",
                            "content": f"If '{link.get('cause', '')[:50]}' then '{link.get('effect', '')[:50]}'",
                            "interventions": link.get("interventions", [])[:2]
                        })
                        if len(result["suggestions"]) >= 3:
                            break
                except:
                    continue

        return result

    def detect_and_track_goal(self, user_message: str) -> Dict:
        """
        Detect goals in user message and track them.

        Returns detected goals and status.
        """
        detected_goals = self.goal_tracker.detect_goals(user_message)
        added_goals = []

        for goal in detected_goals:
            goal_id = self.goal_tracker.add_goal(goal)
            added_goals.append({
                "goal_id": goal_id,
                "description": goal.description,
                "priority": goal.priority,
                "inferred_from": goal.inferred_from
            })

        return {
            "goals_detected": len(detected_goals),
            "goals_added": added_goals
        }

    def get_goal_focused_memories(self, goal_id: str) -> List[Dict]:
        """Get memories specifically relevant to a goal."""
        goal = self.goal_tracker.get_goal(goal_id)
        if not goal:
            return []

        memories = []

        # Get linked memories
        for mem_id in goal.relevant_memory_ids:
            semantic_path = self.base_path / "semantic" / f"{mem_id}.json"
            if semantic_path.exists():
                try:
                    with open(semantic_path) as f:
                        memories.append(json.load(f))
                except:
                    pass

        # Search for more relevant memories
        keywords = set(goal.description.lower().split())
        semantic_path = self.base_path / "semantic"

        if semantic_path.exists():
            for sem_file in semantic_path.glob("sem_*.json"):
                if len(memories) >= 10:
                    break
                try:
                    with open(sem_file) as f:
                        sem = json.load(f)

                    if sem.get("id") in goal.relevant_memory_ids:
                        continue

                    fact_words = set(sem.get("fact", "").lower().split())
                    if len(keywords.intersection(fact_words)) >= 2:
                        memories.append(sem)
                except:
                    continue

        return memories

    def surface_pitfall_warning(self, context: str) -> Optional[Dict]:
        """
        Check if user is approaching a known pitfall.

        Returns warning if pitfall detected, None otherwise.
        """
        context_lower = context.lower()

        # Check failure memory
        failure_path = self.base_path / "failure_memory" / "failures_index.json"
        if failure_path.exists():
            try:
                with open(failure_path) as f:
                    failures = json.load(f)

                for failure in failures.get("failures", []):
                    problem = failure.get("problem", "").lower()
                    # Check for keyword overlap
                    problem_words = set(problem.split())
                    context_words = set(context_lower.split())

                    if len(problem_words.intersection(context_words)) >= 2:
                        return {
                            "warning": f"Previous failure with similar context: {failure.get('problem', '')[:100]}",
                            "what_failed": failure.get("what_didnt_work", ""),
                            "successful_fix": failure.get("what_worked", ""),
                            "severity": "high" if failure.get("frequency", 1) > 2 else "medium"
                        }
            except:
                pass

        return None

    def get_stats(self) -> Dict:
        """Get proactive retrieval statistics."""
        goal_stats = self.goal_tracker.get_stats()

        return {
            "goals": goal_stats,
            "service": "proactive_retrieval"
        }
