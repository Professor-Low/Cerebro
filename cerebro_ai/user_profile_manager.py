#!/usr/bin/env python3
"""
User Profile Manager - Enhanced with Emotional Context
Manages a comprehensive user personal profile including
identity, relationships, projects, preferences, goals, AND emotional patterns.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

DEFAULT_PROFILE_STRUCTURE = {
    "identity": {
        "name": None,
        "username": None,
        "aliases": [],
        "roles": [],
        "location": None,
        "contact": {
            "email": None,
            "phone": None
        }
    },
    "relationships": {
        "pets": [],
        "family": [],
        "colleagues": [],
        "friends": []
    },
    "projects": {
        "companies_owned": [],
        "active_projects": [],
        "clients": []
    },
    "preferences": {
        "technical": [],
        "personal": [],
        "dislikes": []
    },
    "goals": [],
    "technical_environment": {
        "nas": None,
        "operating_system": None,
        "servers": [],
        "networks": []
    },
    # NEW: Communication Style
    "communication_style": {
        "tone_preference": "professional but friendly",
        "detail_level": "balanced",  # concise / detailed / balanced
        "code_focus": "balanced",  # code_focused / explanation_focused / balanced
        "prefers": [],
        "dislikes": [],
        "code_style": {
            "language_preferences": [],
            "comment_style": "minimal",
            "naming_convention": "snake_case"
        },
        "confidence": 0.0  # 0-1 based on signals detected
    },
    # NEW: Emotional Patterns
    "emotional_patterns": {
        "frustrations": [],
        "excitements": [],
        "working_style": {
            "preferred_hours": None,  # morning/afternoon/evening/night
            "session_length": None,  # short/medium/long
            "multitasking_preference": None  # single_task/multitask
        }
    },
    "metadata": {
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "total_conversations_analyzed": 0,
        "completeness_score": 0.0,
        "version": "2.0"  # Updated to v2 with emotional context
    }
}


class UserProfileManager:
    """Manages user's comprehensive personal profile with emotional context."""

    def __init__(self, base_path: str = ""):
        if not base_path:
            from config import DATA_DIR
            base_path = str(DATA_DIR)
        self.base_path = Path(base_path)
        self.profile_path = self.base_path / "user" / "profile.json"
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)

    def load_profile(self) -> Dict:
        """Load user profile, create if doesn't exist."""
        if not self.profile_path.exists():
            return self._create_default_profile()

        try:
            with open(self.profile_path, 'r', encoding='utf-8') as f:
                profile = json.load(f)

            # Migrate old profiles to v2 if needed
            if profile.get('metadata', {}).get('version', '1.0') < '2.0':
                profile = self._migrate_to_v2(profile)

            return profile

        except json.JSONDecodeError as e:
            print(f"Error loading profile: {e}")
            return self._create_default_profile()

    def save_profile(self, profile: Dict):
        """Save user profile to disk."""
        profile['metadata']['last_updated'] = datetime.now().isoformat()
        profile['metadata']['completeness_score'] = self._calculate_completeness(profile)

        with open(self.profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

    def _create_default_profile(self) -> Dict:
        """Create default profile structure."""
        profile = DEFAULT_PROFILE_STRUCTURE.copy()
        self.save_profile(profile)
        return profile

    def _migrate_to_v2(self, old_profile: Dict) -> Dict:
        """Migrate v1 profile to v2 with emotional context."""
        print("Migrating profile to v2 with emotional context...")

        new_profile = DEFAULT_PROFILE_STRUCTURE.copy()

        # Copy over existing fields
        for key in ['identity', 'relationships', 'projects', 'preferences', 'goals', 'technical_environment']:
            if key in old_profile:
                new_profile[key] = old_profile[key]

        # Preserve metadata
        if 'metadata' in old_profile:
            for key, value in old_profile['metadata'].items():
                if key in new_profile['metadata']:
                    new_profile['metadata'][key] = value

        new_profile['metadata']['version'] = '2.0'
        new_profile['metadata']['migrated_at'] = datetime.now().isoformat()

        self.save_profile(new_profile)
        return new_profile

    def update_emotional_patterns(self, profile: Dict, new_patterns: Dict) -> Dict:
        """
        Update emotional patterns in profile.

        Args:
            profile: Current profile
            new_patterns: New patterns from emotional_detector
                {
                    'frustrations': [...],
                    'excitements': [...],
                    'preferences': {'prefers': [...], 'dislikes': [...]}
                }
        """
        # Update frustrations (deduplicate by trigger)
        existing_frustrations = {f['trigger']: f for f in profile['emotional_patterns']['frustrations']}

        for new_frust in new_patterns.get('frustrations', []):
            trigger = new_frust['trigger']
            if trigger in existing_frustrations:
                # Increment occurrences
                existing_frustrations[trigger]['occurrences'] += 1
                existing_frustrations[trigger]['last_noted'] = new_frust.get('timestamp', datetime.now().isoformat())
                # Update intensity if higher
                if self._intensity_value(new_frust['intensity']) > self._intensity_value(existing_frustrations[trigger]['intensity']):
                    existing_frustrations[trigger]['intensity'] = new_frust['intensity']
            else:
                existing_frustrations[trigger] = {
                    'trigger': trigger,
                    'intensity': new_frust['intensity'],
                    'occurrences': 1,
                    'first_noted': new_frust.get('timestamp', datetime.now().isoformat()),
                    'last_noted': new_frust.get('timestamp', datetime.now().isoformat())
                }

        profile['emotional_patterns']['frustrations'] = list(existing_frustrations.values())

        # Update excitements (deduplicate by trigger)
        existing_excitements = {e['trigger']: e for e in profile['emotional_patterns']['excitements']}

        for new_excite in new_patterns.get('excitements', []):
            trigger = new_excite['trigger']
            if trigger in existing_excitements:
                existing_excitements[trigger]['occurrences'] += 1
                existing_excitements[trigger]['last_noted'] = new_excite.get('timestamp', datetime.now().isoformat())
                if self._intensity_value(new_excite['intensity']) > self._intensity_value(existing_excitements[trigger]['intensity']):
                    existing_excitements[trigger]['intensity'] = new_excite['intensity']
            else:
                existing_excitements[trigger] = {
                    'trigger': trigger,
                    'intensity': new_excite['intensity'],
                    'occurrences': 1,
                    'first_noted': new_excite.get('timestamp', datetime.now().isoformat()),
                    'last_noted': new_excite.get('timestamp', datetime.now().isoformat())
                }

        profile['emotional_patterns']['excitements'] = list(existing_excitements.values())

        # Update communication preferences
        for pref in new_patterns.get('preferences', {}).get('prefers', []):
            if pref not in profile['communication_style']['prefers']:
                profile['communication_style']['prefers'].append(pref)

        for dislike in new_patterns.get('preferences', {}).get('dislikes', []):
            if dislike not in profile['communication_style']['dislikes']:
                profile['communication_style']['dislikes'].append(dislike)

        return profile

    def update_communication_style(self, profile: Dict, style_data: Dict) -> Dict:
        """
        Update communication style in profile.

        Args:
            profile: Current profile
            style_data: Style data from emotional_detector
        """
        MAX_PREFS = 20  # Cap list sizes to prevent bloat

        # Update detected preferences
        if 'detail_level' in style_data and style_data['detail_level'] != 'balanced':
            profile['communication_style']['detail_level'] = style_data['detail_level']

        if 'code_focus' in style_data and style_data['code_focus'] != 'balanced':
            profile['communication_style']['code_focus'] = style_data['code_focus']

        # Calculate confidence based on signal counts
        signals = style_data.get('signals_detected', {})
        total_signals = sum(signals.values())
        if total_signals > 0:
            # Higher confidence with more signals
            confidence = min(1.0, total_signals / 20.0)  # 20 signals = 100% confidence
            profile['communication_style']['confidence'] = confidence

        # DEDUPLICATION: Clean up prefers/dislikes lists after update
        if 'communication_style' in profile:
            for key in ['prefers', 'dislikes']:
                if key in profile['communication_style']:
                    items = profile['communication_style'][key]
                    # Filter valid items (min 3 chars, max 100 chars)
                    items = [x for x in items if isinstance(x, str) and 3 <= len(x) <= 100]
                    # Dedupe preserving order and cap
                    profile['communication_style'][key] = list(dict.fromkeys(items))[:MAX_PREFS]

        return profile

    def get_emotional_context_for_entity(self, profile: Dict, entity: str) -> Optional[str]:
        """
        Get emotional context related to an entity.

        Args:
            profile: User profile
            entity: Entity name (e.g., "NAS", "project name")

        Returns:
            Emotional context string or None
        """
        entity_lower = entity.lower()

        # Check frustrations
        for frust in profile['emotional_patterns']['frustrations']:
            if entity_lower in frust['trigger'].lower():
                return f"NOTE: User finds {frust['trigger']} frustrating (intensity: {frust['intensity']}, {frust['occurrences']} times)"

        # Check excitements
        for excite in profile['emotional_patterns']['excitements']:
            if entity_lower in excite['trigger'].lower():
                return f"NOTE: User loves {excite['trigger']} (intensity: {excite['intensity']}, {excite['occurrences']} times) - emphasize this!"

        return None

    def get_communication_preferences_summary(self, profile: Dict) -> str:
        """Get summary of communication preferences for context injection."""
        style = profile['communication_style']

        summary_parts = []

        if style['detail_level'] == 'concise':
            summary_parts.append("User prefers concise, brief responses")
        elif style['detail_level'] == 'detailed':
            summary_parts.append("User prefers detailed explanations")

        if style['code_focus'] == 'code_focused':
            summary_parts.append("Focus on code, minimal explanation")
        elif style['code_focus'] == 'explanation_focused':
            summary_parts.append("Provide thorough explanations")

        if style['prefers']:
            summary_parts.append(f"Preferences: {', '.join(style['prefers'][:3])}")

        if style['dislikes']:
            summary_parts.append(f"Avoid: {', '.join(style['dislikes'][:3])}")

        return " | ".join(summary_parts) if summary_parts else None

    def _intensity_value(self, intensity: str) -> int:
        """Convert intensity string to numeric value."""
        return {'low': 1, 'medium': 2, 'high': 3}.get(intensity, 1)

    def _calculate_completeness(self, profile: Dict) -> float:
        """Calculate profile completeness score (0-1)."""
        max_score = 0
        current_score = 0

        # Identity (20 points)
        max_score += 20
        if profile['identity'].get('name'):
            current_score += 5
        if profile['identity'].get('roles'):
            current_score += 5
        if profile['identity'].get('location'):
            current_score += 5
        if profile['identity'].get('contact', {}).get('email'):
            current_score += 5

        # Relationships (15 points)
        max_score += 15
        current_score += min(15, len(profile['relationships'].get('pets', [])) * 5)

        # Projects (20 points)
        max_score += 20
        current_score += min(10, len(profile['projects'].get('companies_owned', [])) * 10)
        current_score += min(10, len(profile['projects'].get('active_projects', [])) * 2)

        # Preferences (10 points)
        max_score += 10
        current_score += min(10, len(profile['preferences'].get('technical', [])) * 2)

        # Goals (10 points)
        max_score += 10
        current_score += min(10, len(profile.get('goals', [])) * 5)

        # Technical Environment (10 points)
        max_score += 10
        if profile['technical_environment'].get('nas'):
            current_score += 10

        # Emotional Patterns (10 points)
        max_score += 10
        current_score += min(5, len(profile['emotional_patterns'].get('frustrations', [])))
        current_score += min(5, len(profile['emotional_patterns'].get('excitements', [])))

        # Communication Style (5 points)
        max_score += 5
        if profile['communication_style'].get('detail_level') != 'balanced':
            current_score += 2.5
        if len(profile['communication_style'].get('prefers', [])) > 0:
            current_score += 2.5

        return current_score / max_score if max_score > 0 else 0.0


if __name__ == "__main__":
    # Test the profile manager
    manager = UserProfileManager()

    profile = manager.load_profile()
    print("Current Profile:")
    print(json.dumps(profile, indent=2, ensure_ascii=False))

    print(f"\nCompleteness: {profile['metadata']['completeness_score']:.1%}")

    # Test emotional context
    context = manager.get_emotional_context_for_entity(profile, "NAS")
    if context:
        print(f"\nEmotional context for 'NAS': {context}")

    # Test communication preferences
    comm_summary = manager.get_communication_preferences_summary(profile)
    if comm_summary:
        print(f"\nCommunication preferences: {comm_summary}")
