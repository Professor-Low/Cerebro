"""
Pattern Detector - Detect recurring patterns across conversations
Part of Agent 9: Code Understanding & Pattern Detection
"""
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List


class PatternDetector:
    """
    Detect recurring patterns across conversations.
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

        # Create directories
        self.patterns_path.mkdir(parents=True, exist_ok=True)

        # Output files
        self.recurring_topics_file = self.patterns_path / "recurring_topics.json"
        self.recurring_problems_file = self.patterns_path / "recurring_problems.json"

    def detect_recurring_topics(self, threshold: int = 3) -> List[Dict]:
        """
        Find topics mentioned multiple times.

        Args:
            threshold: Minimum occurrences to consider recurring

        Returns:
            List of recurring topics with stats
        """
        topic_counts = defaultdict(lambda: {'count': 0, 'conversations': [], 'first_seen': None, 'last_seen': None})

        # Scan all conversations
        for conv_file in self.conversations_path.glob('*.json'):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv = json.load(f)

                topics = conv.get('metadata', {}).get('topics', [])
                timestamp = conv.get('timestamp')

                for topic in topics:
                    topic_counts[topic]['count'] += 1
                    topic_counts[topic]['conversations'].append(conv['id'])

                    if topic_counts[topic]['first_seen'] is None:
                        topic_counts[topic]['first_seen'] = timestamp
                    topic_counts[topic]['last_seen'] = timestamp

            except (json.JSONDecodeError, KeyError):
                continue

        # Filter by threshold
        recurring = [
            {
                'topic': topic,
                'count': data['count'],
                'conversations': data['conversations'][:10],  # Limit to first 10
                'first_seen': data['first_seen'],
                'last_seen': data['last_seen'],
                'suggestion': f"Topic '{topic}' appears {data['count']} times - consider creating reference documentation"
            }
            for topic, data in topic_counts.items()
            if data['count'] >= threshold
        ]

        # Sort by count
        recurring.sort(key=lambda x: x['count'], reverse=True)

        # Save to file
        with open(self.recurring_topics_file, 'w', encoding='utf-8') as f:
            json.dump(recurring, f, indent=2, ensure_ascii=False)

        return recurring

    def detect_recurring_problems(self, keywords: List[str] = None) -> List[Dict]:
        """
        Find problems that appear multiple times.

        Args:
            keywords: Keywords indicating problems (default: error, issue, problem, etc.)

        Returns:
            List of recurring problems
        """
        if keywords is None:
            keywords = ['error', 'issue', 'problem', 'fail', 'broken', 'bug', 'not working']

        problem_patterns = defaultdict(lambda: {'count': 0, 'conversations': [], 'solutions': []})

        # Scan conversations
        for conv_file in self.conversations_path.glob('*.json'):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv = json.load(f)

                # Check problems_solved
                problems = conv.get('extracted_data', {}).get('problems_solved', [])

                for problem in problems:
                    problem_desc = problem.get('problem', '').lower()

                    # Normalize problem description
                    normalized = self._normalize_problem(problem_desc)

                    # Skip garbage (empty after normalization)
                    if not normalized:
                        continue

                    problem_patterns[normalized]['count'] += 1
                    problem_patterns[normalized]['conversations'].append(conv['id'])

                    if problem.get('solution'):
                        problem_patterns[normalized]['solutions'].append(problem['solution'])

            except (json.JSONDecodeError, KeyError):
                continue

        # Filter recurring (2+ occurrences)
        recurring = [
            {
                'problem': problem,
                'count': data['count'],
                'conversations': data['conversations'],
                'solutions': data['solutions'],
                'suggestion': f"Problem occurs {data['count']} times - consider fixing root cause or documenting solution"
            }
            for problem, data in problem_patterns.items()
            if data['count'] >= 2
        ]

        # Sort by count
        recurring.sort(key=lambda x: x['count'], reverse=True)

        # Save to file
        with open(self.recurring_problems_file, 'w', encoding='utf-8') as f:
            json.dump(recurring, f, indent=2, ensure_ascii=False)

        return recurring

    def _normalize_problem(self, problem: str) -> str:
        """Normalize problem description for matching"""
        import re

        if not problem:
            return ""

        # Remove file paths
        problem = re.sub(r'[A-Za-z]:\\[\w\\\.\-]+', '[PATH]', problem)
        problem = re.sub(r'/[\w/\.\-]+', '[PATH]', problem)

        # Remove numbers
        problem = re.sub(r'\d+', '[NUM]', problem)

        # Lowercase and trim
        problem = problem.lower().strip()

        # Filter out garbage patterns
        # Must have at least 12 chars after normalization
        if len(problem) < 12:
            return ""

        # Must have at least 2 meaningful words
        words = [w for w in problem.split() if len(w) > 2]
        if len(words) < 2:
            return ""

        # Filter out mostly punctuation/special chars (need at least 50% alphanumeric)
        alnum_count = sum(1 for c in problem if c.isalnum() or c.isspace())
        if alnum_count < len(problem) * 0.5:
            return ""

        # Filter out obvious garbage patterns
        garbage_patterns = [
            r'^\*+\s*$',  # Just asterisks
            r'^[\*\-\>\|\:\)\(\[\]\"\'\\n]+\s*$',  # Just punctuation
            r'^\s*s\s+and\s+fixes',  # Extraction artifact
            r'messages.*transcript',  # Transcript noise
            r'^\s*\d+\s*$',  # Just numbers
            r'^not explicitly stated',  # Placeholder
            r'^\*\*\s*$',  # Bold markers
            r'^\\n',  # Starts with newline escape
        ]
        for pattern in garbage_patterns:
            if re.search(pattern, problem, re.IGNORECASE):
                return ""

        # Strip leading/trailing punctuation and whitespace
        problem = re.sub(r'^[\*\-\:\|\>\s]+', '', problem)
        problem = re.sub(r'[\*\-\:\|\>\s]+$', '', problem)

        # Final length check after cleanup
        if len(problem) < 12:
            return ""

        return problem

    def find_knowledge_gaps(self, min_explanations: int = 3) -> List[Dict]:
        """
        Find topics user keeps explaining (should be in permanent context).
        AGENT 15 enhancement.

        Strategy:
        - Track topics user explains frequently
        - Identify concepts mentioned 3+ times by user
        - These should become part of user profile or system context

        Returns:
            [
                {
                    'topic': 'NAS IP address',
                    'explanation_count': 5,
                    'last_explained': '2025-12-20T...',
                    'suggestion': 'Add to permanent context',
                    'examples': ['my nas ip is 10.0.0.100', ...]
                }
            ]
        """

        # Track user explanations
        topic_tracker = defaultdict(lambda: {
            'count': 0,
            'last_seen': None,
            'examples': []
        })

        for conv_file in self.conversations_path.glob("*.json"):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv = json.load(f)

                # Look for user messages with explanations
                for msg in conv.get('messages', []):
                    if msg.get('role') != 'user':
                        continue

                    content = msg.get('content', '').lower()

                    # Detect explanation patterns
                    if any(phrase in content for phrase in [
                        'my nas ip is', 'the ip is', 'located at',
                        'i use', 'i have', 'it\'s called', 'the path is',
                        'my setup', 'i\'m using', 'runs on', 'it is'
                    ]):
                        # Extract topics
                        topics = self._extract_topics_from_explanation(content)
                        for topic in topics:
                            topic_tracker[topic]['count'] += 1
                            topic_tracker[topic]['last_seen'] = conv.get('timestamp')
                            if len(topic_tracker[topic]['examples']) < 5:
                                topic_tracker[topic]['examples'].append(content[:100])

            except Exception:
                continue

        # Find gaps
        gaps = []
        for topic, data in topic_tracker.items():
            if data['count'] >= min_explanations:
                gaps.append({
                    'topic': topic,
                    'explanation_count': data['count'],
                    'last_explained': data['last_seen'],
                    'suggestion': 'Add to permanent context - user explains this frequently',
                    'examples': data['examples'][:3]
                })

        gaps.sort(key=lambda x: x['explanation_count'], reverse=True)
        return gaps

    def _extract_topics_from_explanation(self, content: str) -> List[str]:
        """Extract topics from user explanation (AGENT 15)"""
        import re
        topics = []

        # Pattern: "my X is Y"
        matches = re.findall(r'my (\w+(?:\s+\w+){0,2})\s+is', content)
        topics.extend(matches)

        # Pattern: "the X is Y"
        matches = re.findall(r'the (\w+(?:\s+\w+){0,2})\s+is', content)
        topics.extend(matches)

        # Pattern: "I use X"
        matches = re.findall(r'i use (\w+(?:\s+\w+){0,2})', content)
        topics.extend(matches)

        # Pattern: "I have X"
        matches = re.findall(r'i have (\w+(?:\s+\w+){0,2})', content)
        topics.extend(matches)

        # Pattern: "I'm using X"
        matches = re.findall(r"i'm using (\w+(?:\s+\w+){0,2})", content)
        topics.extend(matches)

        return list(set(topics))[:5]  # Top 5 unique topics

    def detect_stale_projects(self, stale_days: int = 14) -> List[Dict]:
        """
        Find projects that haven't been updated recently.
        AGENT 15 enhancement.

        Args:
            stale_days: Days of inactivity to consider stale

        Returns:
            [
                {
                    'project_id': 'cerebral-interface',
                    'last_active': '2025-12-06T...',
                    'days_inactive': 14,
                    'last_topic': 'WebSocket improvements',
                    'suggestion': 'Check if still active'
                }
            ]
        """
        datetime.now() - timedelta(days=stale_days)
        stale = []

        # Try to load project states from ProjectTracker (Agent 3)
        try:
            from project_tracker import ProjectTracker

            tracker = ProjectTracker()
            projects = tracker.projects

            for project_id, project in projects.items():
                last_active = project.get('last_worked')
                if last_active:
                    try:
                        timestamp = datetime.fromisoformat(last_active)
                        days_inactive = (datetime.now() - timestamp).days

                        if days_inactive >= stale_days:
                            stale.append({
                                'project_id': project_id,
                                'name': project.get('name', project_id),
                                'last_active': last_active,
                                'days_inactive': days_inactive,
                                'last_topic': project.get('current_focus', 'Unknown'),
                                'status': project.get('status', 'unknown'),
                                'priority': project.get('priority', 'medium'),
                                'suggestion': 'Check if still active'
                            })
                    except ValueError:
                        # Invalid timestamp, skip
                        continue

        except ImportError:
            print("[PatternDetector] ProjectTracker not available, using fallback method")
            # Fallback: Scan conversations for project mentions
            stale = self._detect_stale_projects_fallback(stale_days)
        except Exception as e:
            print(f"[PatternDetector] Could not load projects: {e}")
            stale = self._detect_stale_projects_fallback(stale_days)

        # Sort by days inactive
        stale.sort(key=lambda x: x['days_inactive'], reverse=True)
        return stale

    def _detect_stale_projects_fallback(self, stale_days: int) -> List[Dict]:
        """Fallback method to detect stale projects from conversation topics"""
        datetime.now() - timedelta(days=stale_days)
        project_activity = defaultdict(lambda: {
            'last_seen': None,
            'mentions': 0,
            'topics': []
        })

        for conv_file in self.conversations_path.glob("*.json"):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv = json.load(f)

                timestamp = conv.get('timestamp')
                if not timestamp:
                    continue

                try:
                    conv_time = datetime.fromisoformat(timestamp)
                except ValueError:
                    continue

                topics = conv.get('metadata', {}).get('topics', [])

                for topic in topics:
                    if project_activity[topic]['last_seen'] is None or conv_time > datetime.fromisoformat(project_activity[topic]['last_seen']):
                        project_activity[topic]['last_seen'] = timestamp
                    project_activity[topic]['mentions'] += 1
                    project_activity[topic]['topics'] = topics[:3]

            except Exception:
                continue

        # Identify stale
        stale = []
        for project, data in project_activity.items():
            if data['last_seen']:
                try:
                    last_time = datetime.fromisoformat(data['last_seen'])
                    days_inactive = (datetime.now() - last_time).days

                    if days_inactive >= stale_days and data['mentions'] >= 2:
                        stale.append({
                            'project_id': project,
                            'name': project,
                            'last_active': data['last_seen'],
                            'days_inactive': days_inactive,
                            'last_topic': ', '.join(data['topics']),
                            'status': 'unknown',
                            'priority': 'unknown',
                            'suggestion': 'Check if still active (detected from topics)'
                        })
                except ValueError:
                    continue

        return stale

    def get_pattern_summary(self) -> Dict:
        """Get summary of detected patterns"""
        summary = {
            'recurring_topics': 0,
            'recurring_problems': 0,
            'last_analyzed': datetime.now().isoformat()
        }

        if self.recurring_topics_file.exists():
            with open(self.recurring_topics_file, 'r', encoding='utf-8') as f:
                topics = json.load(f)
                summary['recurring_topics'] = len(topics)

        if self.recurring_problems_file.exists():
            with open(self.recurring_problems_file, 'r', encoding='utf-8') as f:
                problems = json.load(f)
                summary['recurring_problems'] = len(problems)

        return summary
