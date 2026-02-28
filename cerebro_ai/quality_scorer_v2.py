"""
Quality Scorer V2 - Production Ready
Improved scoring with better signals and code indexer integration
"""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class QualityScorerV2:
    """
    Production-grade quality scoring with comprehensive signals.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.conversations_path = self.base_path / "conversations"
        self.code_index_path = self.base_path / "code_index"

    def score_conversation(self, conversation: Dict) -> Dict:
        """
        Score conversation with comprehensive quality signals.

        Scoring components (0-110 total):
        - Length & Engagement (15 points)
        - Decisions Made (15 points)
        - Actions Taken (10 points)
        - Problems Solved (20 points)
        - Code Quality (15 points)
        - User Preferences (10 points)
        - Goals & Planning (10 points)
        - Recency Bonus (5 points)
        - Breakthrough Bonus (10 points) - NEW
        """
        scores = {
            'length_engagement': 0,
            'decisions': 0,
            'actions': 0,
            'problems_solved': 0,
            'code_quality': 0,
            'user_preferences': 0,
            'goals': 0,
            'recency': 0,
            'breakthrough': 0  # NEW
        }

        messages = conversation.get('messages', [])
        ed = conversation.get('extracted_data', {})

        # 1. Length & Engagement Score (15 points)
        message_count = len(messages)
        user_messages = sum(1 for m in messages if m.get('role') == 'user')
        assistant_messages = sum(1 for m in messages if m.get('role') == 'assistant')

        # Base length score
        length_score = min(8, message_count * 0.5)  # Up to 8 points for length

        # Engagement bonus (back-and-forth conversation)
        if user_messages > 0 and assistant_messages > 0:
            engagement = min(user_messages, assistant_messages) / max(user_messages, assistant_messages)
            engagement_score = engagement * 7  # Up to 7 points for balanced exchange
        else:
            engagement_score = 0

        scores['length_engagement'] = min(15, length_score + engagement_score)

        # 2. Decisions Score (15 points)
        decisions = ed.get('decisions_made', [])
        scores['decisions'] = min(15, len(decisions) * 5)  # 5 points each, cap 15

        # 3. Actions Score (10 points)
        actions = ed.get('actions_taken', [])
        # Give more weight to important action types
        action_points = 0
        for action in actions:
            action_type = action.get('action_type', 'other')
            if action_type in ['implement', 'create', 'fix']:
                action_points += 3
            elif action_type in ['configure', 'modify']:
                action_points += 2
            else:
                action_points += 1
        scores['actions'] = min(10, action_points)

        # 4. Problems Solved Score (20 points)
        problems = ed.get('problems_solved', [])
        # Higher value for problems with explicit solutions
        problem_points = 0
        for problem in problems:
            if problem.get('solution') and problem['solution'] != "Not explicitly stated":
                problem_points += 12  # Full points for solved problems
            else:
                problem_points += 5   # Partial points for identified problems
        scores['problems_solved'] = min(20, problem_points)

        # 5. Code Quality Score (15 points)
        code_snippets = ed.get('code_snippets', [])
        if code_snippets:
            # Base points
            code_points = min(10, len(code_snippets) * 2)

            # Bonus for diverse languages
            languages = set(c.get('language', 'unknown') for c in code_snippets)
            if len(languages) > 1:
                code_points += 3

            # Bonus for substantial code (more lines)
            total_lines = sum(c.get('lines', 0) for c in code_snippets)
            if total_lines > 50:
                code_points += 2

            scores['code_quality'] = min(15, code_points)
        else:
            scores['code_quality'] = 0

        # 6. User Preferences Score (10 points)
        preferences = ed.get('user_preferences', [])
        scores['user_preferences'] = min(10, len(preferences) * 5)

        # 7. Goals & Planning Score (10 points)
        goals = ed.get('goals_and_intentions', [])
        scores['goals'] = min(10, len(goals) * 5)

        # 8. Recency Bonus (5 points)
        # Newer conversations are slightly more valuable
        try:
            timestamp_str = conversation.get('timestamp', '')
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
                age_days = (datetime.now() - timestamp).days

                if age_days <= 7:
                    scores['recency'] = 5
                elif age_days <= 30:
                    scores['recency'] = 3
                elif age_days <= 90:
                    scores['recency'] = 1
                else:
                    scores['recency'] = 0
        except:
            scores['recency'] = 0

        # 9. Breakthrough Detection Bonus (10 points) - NEW
        # Detect conversations with breakthrough moments
        breakthrough_score = 0
        full_text = ' '.join([m.get('content', '') for m in messages])
        full_text.lower()

        # Breakthrough indicators in user messages
        breakthrough_phrases = [
            'it works', 'finally', 'we did it', 'perfect', 'that fixed it',
            'oh my god', 'yes!', 'awesome', 'brilliant', 'exactly what i needed',
            'thank you so much', 'this is great', 'solved', 'working now'
        ]
        user_text = ' '.join([m.get('content', '') for m in messages if m.get('role') == 'user']).lower()

        # Count breakthrough indicators
        breakthrough_count = sum(1 for phrase in breakthrough_phrases if phrase in user_text)
        if breakthrough_count >= 3:
            breakthrough_score = 10  # Strong breakthrough
        elif breakthrough_count >= 2:
            breakthrough_score = 7
        elif breakthrough_count >= 1:
            breakthrough_score = 4

        # Bonus for long debugging sessions that succeeded
        if len(messages) >= 10 and breakthrough_count >= 1:
            breakthrough_score = min(10, breakthrough_score + 3)

        # Check metadata for explicit breakthrough tags
        if conversation.get('metadata', {}).get('breakthrough'):
            breakthrough_score = 10

        scores['breakthrough'] = breakthrough_score

        # Calculate overall score
        total_score = sum(scores.values())

        # Importance classification with more granular levels
        if total_score >= 75:
            importance = 'critical'
            multiplier = 1.8
        elif total_score >= 55:
            importance = 'very_high'
            multiplier = 1.5
        elif total_score >= 40:
            importance = 'high'
            multiplier = 1.3
        elif total_score >= 25:
            importance = 'medium'
            multiplier = 1.1
        elif total_score >= 12:
            importance = 'low'
            multiplier = 0.9
        else:
            importance = 'trivial'
            multiplier = 0.7

        return {
            'overall_score': round(total_score, 1),
            'importance': importance,
            'search_boost_multiplier': multiplier,
            'breakdown': {k: round(v, 1) for k, v in scores.items()},
            'has_breakthrough': breakthrough_score >= 7
        }

    def score_all_conversations(self) -> List[Dict]:
        """Score all conversations and return sorted by quality"""
        scored = []

        for conv_file in self.conversations_path.glob("*.json"):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conversation = json.load(f)

                score_data = self.score_conversation(conversation)
                scored.append({
                    'conversation_id': conversation['id'],
                    'file': conv_file.name,
                    'score': score_data['overall_score'],
                    'importance': score_data['importance'],
                    'breakdown': score_data['breakdown']
                })

            except Exception as e:
                print(f"Error scoring {conv_file.name}: {e}")
                continue

        # Sort by score descending
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored

    def get_statistics(self) -> Dict:
        """Get quality statistics across all conversations"""
        scores = []
        importance_counts = defaultdict(int)

        for conv_file in self.conversations_path.glob("*.json"):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conversation = json.load(f)

                score_data = self.score_conversation(conversation)
                scores.append(score_data['overall_score'])
                importance_counts[score_data['importance']] += 1

            except:
                continue

        if not scores:
            return {}

        return {
            'total_conversations': len(scores),
            'average_score': round(sum(scores) / len(scores), 1),
            'min_score': min(scores),
            'max_score': max(scores),
            'by_importance': dict(importance_counts),
            'top_10_average': round(sum(sorted(scores, reverse=True)[:10]) / min(10, len(scores)), 1)
        }

    # Alias for backwards compatibility
    def get_quality_stats(self) -> Dict:
        """Alias for get_statistics() - used by MCP quality tool"""
        return self.get_statistics()
