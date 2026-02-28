"""
Quality Scorer - Agent 11
Scores memory quality and importance for search ranking
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


class QualityScorer:
    """
    Score memory quality and importance for search ranking.
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

    def score_conversation(self, conversation: Dict) -> Dict:
        """
        Score a conversation's importance/quality.

        Returns:
            Scoring breakdown with overall score
        """
        scores = {
            'length': 0,
            'decisions': 0,
            'actions': 0,
            'code': 0,
            'problems_solved': 0,
            'user_preferences': 0,
            'goals': 0,
            'corrections': 0
        }

        # Length score (longer = more substance)
        message_count = len(conversation.get('messages', []))
        scores['length'] = min(10, message_count)  # Cap at 10

        # Decisions score
        decisions = len(conversation.get('extracted_data', {}).get('decisions_made', []))
        scores['decisions'] = min(15, decisions * 5)  # 5 points each, cap 15

        # Actions score
        actions = len(conversation.get('extracted_data', {}).get('actions_taken', []))
        scores['actions'] = min(10, actions * 2)  # 2 points each, cap 10

        # Code score
        code_snippets = len(conversation.get('extracted_data', {}).get('code_snippets', []))
        scores['code'] = min(15, code_snippets * 3)  # 3 points each, cap 15

        # Problems solved score
        problems = len(conversation.get('extracted_data', {}).get('problems_solved', []))
        scores['problems_solved'] = min(20, problems * 10)  # 10 points each, cap 20

        # User preferences score
        preferences = len(conversation.get('extracted_data', {}).get('user_preferences', []))
        scores['user_preferences'] = min(10, preferences * 5)  # 5 points each

        # Goals score
        goals = len(conversation.get('extracted_data', {}).get('goals_and_intentions', []))
        scores['goals'] = min(10, goals * 5)  # 5 points each

        # Corrections score (if user corrected Claude)
        corrections = sum(1 for msg in conversation.get('messages', [])
                         if msg.get('role') == 'user' and
                         any(word in msg.get('content', '').lower() for word in ['no', 'wrong', 'incorrect', 'actually']))
        scores['corrections'] = min(10, corrections * 5)

        # Calculate overall score (0-100)
        total_score = sum(scores.values())

        # Importance level
        if total_score >= 70:
            importance = 'critical'
            multiplier = 1.5
        elif total_score >= 40:
            importance = 'high'
            multiplier = 1.3
        elif total_score >= 20:
            importance = 'medium'
            multiplier = 1.0
        else:
            importance = 'low'
            multiplier = 0.8

        return {
            'overall_score': total_score,
            'importance': importance,
            'search_boost_multiplier': multiplier,
            'breakdown': scores
        }

    def score_all_conversations(self) -> List[Dict]:
        """Score all conversations and return sorted by quality"""
        scored = []

        for conv_file in self.conversations_path.glob('*.json'):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv = json.load(f)

                quality = self.score_conversation(conv)

                scored.append({
                    'conversation_id': conv.get('id', conv_file.stem),
                    'timestamp': conv.get('timestamp', ''),
                    'summary': conv.get('search_index', {}).get('summary', '')[:100],
                    'quality_score': quality['overall_score'],
                    'importance': quality['importance']
                })

            except Exception as e:
                print(f"[QualityScorer] Error scoring {conv_file}: {e}")
                continue

        # Sort by quality score
        scored.sort(key=lambda x: x['quality_score'], reverse=True)

        return scored

    def get_quality_stats(self) -> Dict:
        """Get quality statistics across all conversations"""
        stats = {
            'total': 0,
            'by_importance': defaultdict(int),
            'avg_score': 0,
            'top_10_avg': 0
        }

        scores = []

        for conv_file in self.conversations_path.glob('*.json'):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv = json.load(f)

                quality = self.score_conversation(conv)
                scores.append(quality['overall_score'])
                stats['by_importance'][quality['importance']] += 1
                stats['total'] += 1

            except Exception:
                continue

        if scores:
            stats['avg_score'] = round(sum(scores) / len(scores), 2)
            top_10 = sorted(scores, reverse=True)[:10]
            stats['top_10_avg'] = round(sum(top_10) / len(top_10), 2) if top_10 else 0

        stats['by_importance'] = dict(stats['by_importance'])

        return stats


# Example usage
if __name__ == "__main__":
    scorer = QualityScorer()

    # Score all conversations
    print("Scoring all conversations...")
    scored = scorer.score_all_conversations()

    print("\nTop 10 Quality Conversations:")
    for i, conv in enumerate(scored[:10], 1):
        print(f"\n{i}. {conv['conversation_id']}")
        print(f"   Score: {conv['quality_score']}, Importance: {conv['importance']}")
        print(f"   Summary: {conv['summary']}")

    # Get quality stats
    print("\n\nQuality Statistics:")
    stats = scorer.get_quality_stats()
    print(f"Total conversations: {stats['total']}")
    print(f"Average score: {stats['avg_score']}")
    print(f"Top 10 average: {stats['top_10_avg']}")
    print("\nBy importance:")
    for importance, count in stats['by_importance'].items():
        print(f"  {importance}: {count}")
