"""
Preference & Emotional Context Manager
Manages user preferences and emotional context extraction.

AGENT 8: TEMPORAL & PREFERENCE INTELLIGENCE

Updated 2026-01-18: Integrated with PreferenceEvolution for Phase 2 Brain Evolution
- Timestamps on all preferences
- Decay detection (90 days → stale)
- Contradiction detection and supersession
- Bounded lists
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import evolution system
try:
    from preference_evolution import PreferenceEvolution
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False


class PreferenceManager:
    """
    Manages user preferences and emotional context extraction.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.user_path = self.base_path / "user"
        self.preferences_file = self.user_path / "preferences.json"
        self.emotional_file = self.user_path / "emotional_context.json"

        # Ensure directories exist
        self.user_path.mkdir(parents=True, exist_ok=True)

        # Load existing data
        self.preferences = self._load_preferences()
        self.emotional_context = self._load_emotional_context()

        # Initialize evolution system if available
        self.evolution: Optional[PreferenceEvolution] = None
        if EVOLUTION_AVAILABLE:
            try:
                self.evolution = PreferenceEvolution(base_path)
            except Exception as e:
                print(f"Warning: Could not initialize PreferenceEvolution: {e}")

    def _load_preferences(self) -> Dict:
        """Load existing preferences or create default"""
        if self.preferences_file.exists():
            with open(self.preferences_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Default structure
        return {
            "communication_style": {
                "prefers": [],
                "dislikes": [],
                "tone": "professional but friendly"
            },
            "workflow_preferences": {
                "prefers": [],
                "dislikes": []
            },
            "technical_preferences": {
                "languages": {},
                "frameworks": {},
                "tools": {}
            },
            "response_format": {
                "code_style": "detailed",
                "explanation_depth": "concise",
                "include_examples": True
            },
            "last_updated": datetime.now().isoformat()
        }

    def _load_emotional_context(self) -> Dict:
        """Load emotional context or create default"""
        if self.emotional_file.exists():
            with open(self.emotional_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        return {
            "frustrations": [],
            "excitements": [],
            "working_style": {
                "prefers_automation": True,
                "likes_visual_feedback": True,
                "values_real_time": True
            },
            "interaction_patterns": {
                "frequency": {},
                "peak_hours": [],
                "common_topics": []
            },
            "last_updated": datetime.now().isoformat()
        }

    def extract_preferences_from_conversation(self, conversation: Dict):
        """
        Extract preferences from a conversation.

        Detects patterns like:
        - "I prefer X"
        - "I don't like Y"
        - "Can you always Z?"
        """
        messages = conversation.get('messages', [])

        # Analyze user messages
        for msg in messages:
            if msg.get('role') != 'user':
                continue

            content = msg.get('content', '').lower()

            # Detect preferences
            if 'prefer' in content or 'like' in content:
                self._extract_preference(content, positive=True)

            if "don't" in content or 'dislike' in content or 'hate' in content:
                self._extract_preference(content, positive=False)

            # Detect frustrations
            if any(word in content for word in ['frustrating', 'annoying', 'slow', 'takes too long']):
                self._extract_frustration(content)

            # Detect excitements
            if any(word in content for word in ['love', 'awesome', 'great', 'perfect', 'exactly']):
                self._extract_excitement(content)

        self._save_preferences()
        self._save_emotional_context()

    def _extract_preference(self, text: str, positive: bool):
        """Extract a specific preference statement"""
        # Simple keyword-based extraction (can be enhanced with NLP)

        if positive:
            # Look for "I prefer/like X" patterns
            if 'code' in text and 'explanation' not in text:
                if "code over explanation" not in self.preferences["communication_style"]["prefers"]:
                    self.preferences["communication_style"]["prefers"].append("code over explanation")

            if 'direct' in text or 'concise' in text:
                if "direct answers" not in self.preferences["communication_style"]["prefers"]:
                    self.preferences["communication_style"]["prefers"].append("direct answers")

            if 'automat' in text:
                self.emotional_context["working_style"]["prefers_automation"] = True

        else:
            # Look for "I don't like Y" patterns
            if 'long' in text and ('response' in text or 'explanation' in text):
                if "long-winded responses" not in self.preferences["communication_style"]["dislikes"]:
                    self.preferences["communication_style"]["dislikes"].append("long-winded responses")

            if 'manual' in text:
                if "manual processes" not in self.preferences["workflow_preferences"]["dislikes"]:
                    self.preferences["workflow_preferences"]["dislikes"].append("manual processes")

    def _extract_frustration(self, text: str):
        """Extract frustration statements"""
        # Detect what's frustrating
        frustration = {
            "text": text[:200],  # Keep it short
            "timestamp": datetime.now().isoformat(),
            "category": self._categorize_frustration(text)
        }

        # Avoid duplicates
        if frustration not in self.emotional_context["frustrations"]:
            self.emotional_context["frustrations"].append(frustration)

            # Keep only last 20 frustrations
            self.emotional_context["frustrations"] = self.emotional_context["frustrations"][-20:]

    def _extract_excitement(self, text: str):
        """Extract excitement statements"""
        excitement = {
            "text": text[:200],
            "timestamp": datetime.now().isoformat(),
            "category": self._categorize_excitement(text)
        }

        if excitement not in self.emotional_context["excitements"]:
            self.emotional_context["excitements"].append(excitement)
            self.emotional_context["excitements"] = self.emotional_context["excitements"][-20:]

    def _categorize_frustration(self, text: str) -> str:
        """Categorize type of frustration"""
        if any(word in text for word in ['slow', 'lag', 'wait', 'long time']):
            return 'performance'
        elif any(word in text for word in ['manual', 'repeat', 'again']):
            return 'automation'
        elif any(word in text for word in ['work', 'broken', 'error', 'fail']):
            return 'reliability'
        else:
            return 'general'

    def _categorize_excitement(self, text: str) -> str:
        """Categorize type of excitement"""
        if any(word in text for word in ['visual', 'animation', 'graph', 'see']):
            return 'visualization'
        elif any(word in text for word in ['automat', 'automatic', 'background']):
            return 'automation'
        elif any(word in text for word in ['work', 'perfect', 'exactly']):
            return 'functionality'
        else:
            return 'general'

    def _save_preferences(self):
        """Save preferences to disk"""
        self.preferences["last_updated"] = datetime.now().isoformat()
        with open(self.preferences_file, 'w', encoding='utf-8') as f:
            json.dump(self.preferences, f, indent=2, ensure_ascii=False)

    def _save_emotional_context(self):
        """Save emotional context to disk"""
        self.emotional_context["last_updated"] = datetime.now().isoformat()
        with open(self.emotional_file, 'w', encoding='utf-8') as f:
            json.dump(self.emotional_context, f, indent=2, ensure_ascii=False)

    def get_preferences_summary(self) -> Dict:
        """Get summary of user preferences"""
        return {
            "communication_style": self.preferences["communication_style"],
            "workflow_preferences": self.preferences["workflow_preferences"],
            "top_frustrations": self._get_top_frustrations(5),
            "top_excitements": self._get_top_excitements(5),
            "working_style": self.emotional_context["working_style"]
        }

    def _get_top_frustrations(self, limit: int) -> List[Dict]:
        """Get most recent frustrations"""
        return sorted(
            self.emotional_context["frustrations"],
            key=lambda x: x['timestamp'],
            reverse=True
        )[:limit]

    def _get_top_excitements(self, limit: int) -> List[Dict]:
        """Get most recent excitements"""
        return sorted(
            self.emotional_context["excitements"],
            key=lambda x: x['timestamp'],
            reverse=True
        )[:limit]

    def update_preference_manual(self, category: str, preference: str, positive: bool) -> Dict:
        """
        Manually update a user preference.

        Args:
            category: 'communication_style', 'workflow', or 'technical'
            preference: The preference text to add
            positive: True for 'prefers', False for 'dislikes'

        Returns:
            Updated preference dict with success status
        """
        try:
            # Use evolution system if available (Phase 2)
            if self.evolution:
                result = self.evolution.add_preference(
                    content=preference,
                    category=category,
                    positive=positive,
                    source="manual",
                    confidence=0.95  # Manual preferences are high confidence
                )
                result["success"] = result.get("added", False) or result.get("reinforced", False)
                return result

            # Fallback to legacy system
            # Map category to preference key
            if category == "communication_style":
                target = self.preferences["communication_style"]
            elif category == "workflow":
                target = self.preferences["workflow_preferences"]
            elif category == "technical":
                # Technical preferences have a different structure
                return {
                    "success": False,
                    "error": "Technical preferences use a different update method"
                }
            else:
                return {
                    "success": False,
                    "error": f"Invalid category: {category}"
                }

            # Add to appropriate list
            key = "prefers" if positive else "dislikes"
            if preference not in target[key]:
                target[key].append(preference)
                self._save_preferences()

                return {
                    "success": True,
                    "category": category,
                    "preference": preference,
                    "positive": positive,
                    "message": f"Added to {category}.{key}"
                }
            else:
                return {
                    "success": True,
                    "message": "Preference already exists",
                    "category": category,
                    "preference": preference
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    # =========================================================================
    # Evolution System Methods (Phase 2)
    # =========================================================================

    def get_evolved_preferences(self, include_stale: bool = False) -> Dict:
        """
        Get preferences from the evolution system.
        Returns weighted preferences if evolution is available.
        """
        if not self.evolution:
            return {"error": "Evolution system not available", "legacy": self.get_preferences_summary()}

        return {
            "active": self.evolution.get_active_preferences(include_stale=include_stale),
            "weighted": self.evolution.get_weighted_preferences(),
            "stats": self.evolution.get_stats()
        }

    def check_preference_decay(self) -> Dict:
        """
        Check for stale preferences and mark them.
        Call periodically or on session start.
        """
        if not self.evolution:
            return {"error": "Evolution system not available"}

        return self.evolution.mark_decay()

    def detect_contradictions(self) -> List[Dict]:
        """
        Scan all preferences for contradictions.
        """
        if not self.evolution:
            return []

        return self.evolution.detect_contradictions_all()

    def migrate_to_evolution(self) -> Dict:
        """
        Migrate legacy preferences to the evolution system.
        """
        if not self.evolution:
            return {"error": "Evolution system not available"}

        results = {}

        # Migrate from user/preferences.json
        if self.preferences_file.exists():
            results["user_preferences_legacy"] = self.evolution.migrate_from_legacy(self.preferences_file)

        # Migrate from preferences/user_preferences.json
        user_prefs_file = self.base_path / "preferences" / "user_preferences.json"
        if user_prefs_file.exists():
            results["user_preferences"] = self.evolution.migrate_from_legacy(user_prefs_file)

        return results

    def get_evolution_stats(self) -> Dict:
        """Get statistics about preference evolution."""
        if not self.evolution:
            return {"error": "Evolution system not available"}

        return self.evolution.get_stats()
