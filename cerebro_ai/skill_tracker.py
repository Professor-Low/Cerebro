"""
Skill Tracker - Track technology usage and skill development over time
Part of Agent 9: Code Understanding & Pattern Detection
"""
import json
from pathlib import Path
from typing import Dict, List


class SkillTracker:
    """
    Track technology usage and skill development over time.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.patterns_path = self.base_path / "patterns"
        self.conversations_path = self.base_path / "conversations"
        self.skill_file = self.patterns_path / "skill_progression.json"

        self.patterns_path.mkdir(parents=True, exist_ok=True)

    def track_skill_development(self, skill: str) -> Dict:
        """
        Track usage and complexity of a skill over time.

        Args:
            skill: Technology/skill to track (e.g., "Python", "Docker")

        Returns:
            Timeline of skill usage with complexity analysis
        """
        timeline = []
        skill_lower = skill.lower()

        for conv_file in sorted(self.conversations_path.glob('*.json')):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv = json.load(f)

                # Check if skill is mentioned
                entities = conv.get('extracted_data', {}).get('entities', {})
                technologies = entities.get('technologies', [])

                if not any(skill_lower in tech.lower() for tech in technologies):
                    continue

                # Skill was used in this conversation
                entry = {
                    'date': conv['timestamp'],
                    'conversation_id': conv['id'],
                    'complexity': self._analyze_complexity(conv, skill),
                    'topics': conv.get('metadata', {}).get('topics', []),
                    'actions': len(conv.get('extracted_data', {}).get('actions_taken', [])),
                    'decisions': len(conv.get('extracted_data', {}).get('decisions_made', []))
                }

                timeline.append(entry)

            except (json.JSONDecodeError, KeyError):
                continue

        # Calculate progression
        result = {
            'skill': skill,
            'timeline': timeline,
            'first_used': timeline[0]['date'] if timeline else None,
            'last_used': timeline[-1]['date'] if timeline else None,
            'usage_count': len(timeline),
            'complexity_trend': self._calculate_trend(timeline),
            'progression': self._assess_progression(timeline)
        }

        return result

    def _analyze_complexity(self, conversation: Dict, skill: str) -> str:
        """Analyze complexity level of skill usage"""
        # Simple heuristic based on conversation content
        message_count = len(conversation.get('messages', []))
        actions = len(conversation.get('extracted_data', {}).get('actions_taken', []))
        code_snippets = len(conversation.get('extracted_data', {}).get('code_snippets', []))

        complexity_score = message_count + (actions * 2) + (code_snippets * 3)

        if complexity_score < 10:
            return 'basic'
        elif complexity_score < 30:
            return 'intermediate'
        else:
            return 'advanced'

    def _calculate_trend(self, timeline: List[Dict]) -> str:
        """Calculate complexity trend"""
        if len(timeline) < 2:
            return 'insufficient_data'

        complexity_map = {'basic': 1, 'intermediate': 2, 'advanced': 3}

        first_half = timeline[:len(timeline)//2]
        second_half = timeline[len(timeline)//2:]

        avg_first = sum(complexity_map[e['complexity']] for e in first_half) / len(first_half)
        avg_second = sum(complexity_map[e['complexity']] for e in second_half) / len(second_half)

        if avg_second > avg_first * 1.2:
            return 'improving'
        elif avg_second < avg_first * 0.8:
            return 'declining'
        else:
            return 'stable'

    def _assess_progression(self, timeline: List[Dict]) -> str:
        """Overall assessment of skill progression"""
        if not timeline:
            return 'no_data'

        # Check recent vs early complexity
        early = [e for e in timeline[:3]]
        recent = [e for e in timeline[-3:]]

        early_complex = sum(1 for e in early if e['complexity'] == 'advanced')
        recent_complex = sum(1 for e in recent if e['complexity'] == 'advanced')

        if recent_complex > early_complex:
            return 'skill_improving'
        elif len(timeline) > 10 and recent[0]['complexity'] == 'advanced':
            return 'proficient'
        else:
            return 'developing'

    def get_all_skills(self) -> List[str]:
        """Get list of all technologies used"""
        skills = set()

        for conv_file in self.conversations_path.glob('*.json'):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv = json.load(f)

                entities = conv.get('extracted_data', {}).get('entities', {})
                technologies = entities.get('technologies', [])
                skills.update(technologies)

            except (json.JSONDecodeError, KeyError):
                continue

        return sorted(list(skills))
