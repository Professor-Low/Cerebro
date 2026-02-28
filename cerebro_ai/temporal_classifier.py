"""
Temporal Classification System
Classifies memories by temporal categories and detects staleness.

AGENT 8: TEMPORAL & PREFERENCE INTELLIGENCE
"""

from datetime import datetime, timedelta
from typing import Dict, List


class TemporalClassifier:
    """
    Classifies memories by temporal categories and detects staleness.
    """

    # Temporal category thresholds
    RECENT_DAYS = 7
    CURRENT_DAYS = 30
    HISTORICAL_DAYS = 180

    def __init__(self):
        pass

    def classify_conversation(self, conversation: Dict) -> str:
        """
        Classify a conversation into temporal category.

        Args:
            conversation: Conversation dict with timestamp

        Returns:
            Category: RECENT, CURRENT, HISTORICAL, or ARCHIVED
        """
        timestamp_str = conversation.get('timestamp')
        if not timestamp_str:
            return 'UNKNOWN'

        timestamp = datetime.fromisoformat(timestamp_str)
        age_days = (datetime.now() - timestamp).days

        if age_days < self.RECENT_DAYS:
            return 'RECENT'
        elif age_days < self.CURRENT_DAYS:
            return 'CURRENT'
        elif age_days < self.HISTORICAL_DAYS:
            return 'HISTORICAL'
        else:
            return 'ARCHIVED'

    def get_age_description(self, timestamp_str: str) -> str:
        """
        Get human-readable age description.

        Args:
            timestamp_str: ISO format timestamp

        Returns:
            Description like "2 hours ago", "3 days ago", "2 weeks ago"
        """
        timestamp = datetime.fromisoformat(timestamp_str)
        delta = datetime.now() - timestamp

        if delta.days == 0:
            if delta.seconds < 3600:
                minutes = delta.seconds // 60
                return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            else:
                hours = delta.seconds // 3600
                return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif delta.days == 1:
            return "yesterday"
        elif delta.days < 7:
            return f"{delta.days} days ago"
        elif delta.days < 30:
            weeks = delta.days // 7
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        elif delta.days < 365:
            months = delta.days // 30
            return f"{months} month{'s' if months != 1 else ''} ago"
        else:
            years = delta.days // 365
            return f"{years} year{'s' if years != 1 else ''} ago"

    def detect_stale_projects(self, projects: Dict, stale_days: int = 14) -> List[Dict]:
        """
        Detect projects that haven't been updated recently.

        Args:
            projects: Dict of project_id -> project data
            stale_days: Days of inactivity to consider stale

        Returns:
            List of stale projects with staleness info
        """
        stale_projects = []
        threshold = datetime.now() - timedelta(days=stale_days)

        for project_id, project in projects.items():
            last_worked_str = project.get('last_worked')
            if not last_worked_str:
                continue

            last_worked = datetime.fromisoformat(last_worked_str)

            if last_worked < threshold and project.get('status') == 'active':
                stale_days_count = (datetime.now() - last_worked).days
                stale_projects.append({
                    'project_id': project_id,
                    'name': project['name'],
                    'last_worked': last_worked_str,
                    'stale_days': stale_days_count,
                    'status': 'stale',
                    'message': f"No activity for {stale_days_count} days"
                })

        return sorted(stale_projects, key=lambda x: x['stale_days'], reverse=True)

    def add_temporal_context_to_result(self, result: Dict) -> Dict:
        """
        Add temporal context to a search result.

        Args:
            result: Search result with timestamp

        Returns:
            Result enhanced with temporal info
        """
        timestamp_str = result.get('timestamp')
        if not timestamp_str:
            return result

        # Add temporal category
        result['temporal_category'] = self.classify_conversation({'timestamp': timestamp_str})

        # Add age description
        result['age_description'] = self.get_age_description(timestamp_str)

        # Add recency score (1.0 = today, 0.0 = very old)
        timestamp = datetime.fromisoformat(timestamp_str)
        age_days = (datetime.now() - timestamp).days
        recency_score = max(0.0, 1.0 - (age_days / 365))  # Decay over 1 year
        result['recency_score'] = round(recency_score, 3)

        return result

    def boost_recent_in_search(self, results: List[Dict], boost_factor: float = 0.2) -> List[Dict]:
        """
        Boost search scores for recent results.

        Args:
            results: Search results with scores
            boost_factor: How much to boost (0.0 = none, 1.0 = double)

        Returns:
            Results with boosted scores
        """
        for result in results:
            if 'score' in result and 'recency_score' in result:
                # Apply recency boost
                boost = result['recency_score'] * boost_factor
                result['score'] = result['score'] * (1.0 + boost)
                result['boosted_by_recency'] = True

        # Re-sort by new scores
        return sorted(results, key=lambda x: x.get('score', 0), reverse=True)
