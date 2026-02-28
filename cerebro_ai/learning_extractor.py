"""
Learning Extractor - Automatic Learning from Conversations
Analyzes conversations to extract learnings, corrections, and patterns.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


class LearningExtractor:
    """
    Automatically extract learnings from conversations.

    Detects:
    - Problems and their solutions
    - Corrections (when something was wrong)
    - What worked vs what didn't
    - Patterns and anti-patterns
    """

    def __init__(self, base_path: str = None):
        if base_path is None:

            from config import AI_MEMORY_BASE

            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.learnings_path = self.base_path / "learnings"
        self.corrections_path = self.base_path / "corrections"

        self.learnings_path.mkdir(parents=True, exist_ok=True)

        # Patterns that indicate learning moments
        self.correction_patterns = [
            r"(?:that|this) (?:didn't|doesn't|won't) work",
            r"(?:actually|wait|oops|sorry),? (?:that's|that was|I was) wrong",
            r"the (?:issue|problem|error) (?:is|was) (?:actually|because)",
            r"(?:let me|I need to) (?:fix|correct|update) that",
            r"(?:that|this) (?:caused|creates?|leads? to) (?:an? )?(?:error|problem|issue)",
            r"(?:instead|rather),? (?:we should|you should|use|try)",
            r"the (?:correct|right|proper) (?:way|approach|solution) is",
            r"(?:I|we) (?:should have|shouldn't have)",
        ]

        self.solution_patterns = [
            r"(?:the|a) (?:fix|solution|answer) (?:is|was)",
            r"(?:to fix|to solve|to resolve) this",
            r"(?:here's|here is) (?:how|what) (?:to|you)",
            r"(?:this|that) (?:fixes|solves|resolves)",
            r"(?:now it|that) (?:works|should work)",
            r"(?:the issue was|problem was solved by)",
            # Success indicators
            r"(?:fixed|resolved|solved|working now)",
            # Action patterns
            r"(?:changed|updated|modified|set|added)\s+.+?\s+(?:to|from|in)",
            r"(?:need(?:ed)?\s+to|had\s+to|should|must)",
            # Tool-based solutions
            r"(?:edited|wrote|created|deleted|configured|enabled|disabled)",
        ]

        self.problem_patterns = [
            r"(?:the|a) (?:problem|issue|error|bug) (?:is|was)",
            r"(?:getting|seeing|having) (?:an? )?(?:error|issue|problem|timeout|crash)",
            r"(?:it|this|that) (?:doesn't|didn't|won't|isn't) (?:work|working)",
            r"(?:failed|failing|broken|broke)",
            r"(?:can't|cannot|couldn't) (?:.*?) (?:because|due to)",
            # Natural speech patterns
            r"(?:doesn't|won't|can't|isn't|not)\s+(?:work|working|loading|connecting|running)",
            # Question-based problems
            r"why\s+(?:is|does|won't|can't|isn't)",
            # Frustration patterns
            r"(?:stuck|confused|frustrated|struggling)\s+(?:with|on|about)",
            # Timeout/connection issues
            r"(?:timeout|timed?\s*out|connection\s+(?:refused|failed|reset))",
        ]

    def analyze_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a conversation and extract all learnings.

        Returns:
            Dict with extracted problems, solutions, corrections, and learnings
        """
        messages = conversation.get('messages', [])
        conv_id = conversation.get('id', 'unknown')

        extracted = {
            'conversation_id': conv_id,
            'timestamp': datetime.now().isoformat(),
            'problems_found': [],
            'solutions_found': [],
            'corrections_found': [],
            'learnings': [],
            'what_worked': [],
            'what_didnt_work': []
        }

        # Analyze message pairs (user problem → assistant solution)
        for i, msg in enumerate(messages):
            content = msg.get('content', '')
            if isinstance(content, list):
                content = ' '.join(c.get('text', '') for c in content if isinstance(c, dict))

            role = msg.get('role', '')

            # Look for problems in user messages
            if role == 'user':
                problems = self._extract_problems(content)
                for p in problems:
                    extracted['problems_found'].append({
                        'text': p,
                        'message_index': i
                    })

            # Look for solutions and corrections in assistant messages
            if role == 'assistant':
                solutions = self._extract_solutions(content)
                for s in solutions:
                    extracted['solutions_found'].append({
                        'text': s,
                        'message_index': i
                    })

                corrections = self._extract_corrections(content)
                for c in corrections:
                    extracted['corrections_found'].append({
                        'text': c,
                        'message_index': i,
                        'type': 'self_correction'
                    })

        # Detect what worked vs what didn't by analyzing conversation flow
        worked, didnt_work = self._analyze_outcomes(messages)
        extracted['what_worked'] = worked
        extracted['what_didnt_work'] = didnt_work

        # Generate high-level learnings
        extracted['learnings'] = self._generate_learnings(extracted)

        return extracted

    def _extract_problems(self, text: str) -> List[str]:
        """Extract problem statements from text."""
        problems = []

        for pattern in self.problem_patterns:
            matches = re.findall(f"[^.]*{pattern}[^.]*\\.", text, re.IGNORECASE)
            problems.extend(matches)

        # Deduplicate and clean
        seen = set()
        cleaned = []
        for p in problems:
            p_clean = p.strip()[:300]
            if p_clean not in seen and len(p_clean) > 20:
                seen.add(p_clean)
                cleaned.append(p_clean)

        return cleaned[:5]  # Limit to 5

    def _extract_solutions(self, text: str) -> List[str]:
        """Extract solution statements from text."""
        solutions = []

        for pattern in self.solution_patterns:
            matches = re.findall(f"[^.]*{pattern}[^.]*\\.", text, re.IGNORECASE)
            solutions.extend(matches)

        # Deduplicate and clean
        seen = set()
        cleaned = []
        for s in solutions:
            s_clean = s.strip()[:300]
            if s_clean not in seen and len(s_clean) > 20:
                seen.add(s_clean)
                cleaned.append(s_clean)

        return cleaned[:5]

    def _extract_corrections(self, text: str) -> List[str]:
        """Extract correction statements (when something was wrong)."""
        corrections = []

        for pattern in self.correction_patterns:
            matches = re.findall(f"[^.]*{pattern}[^.]*\\.", text, re.IGNORECASE)
            corrections.extend(matches)

        # Deduplicate and clean
        seen = set()
        cleaned = []
        for c in corrections:
            c_clean = c.strip()[:300]
            if c_clean not in seen and len(c_clean) > 20:
                seen.add(c_clean)
                cleaned.append(c_clean)

        return cleaned[:5]

    def _analyze_outcomes(self, messages: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Analyze conversation to determine what worked and what didn't.
        Looks for patterns like "that worked!" or "still getting error".
        """
        worked = []
        didnt_work = []

        success_indicators = [
            r"(?:that|it) (?:worked|works|fixed)",
            r"(?:thanks|perfect|great|awesome)",
            r"(?:problem|issue) (?:solved|resolved|fixed)",
            r"(?:now it|that) (?:works|is working)",
        ]

        failure_indicators = [
            r"(?:still|same) (?:error|issue|problem)",
            r"(?:that|it) (?:didn't|doesn't) work",
            r"(?:nope|no|not working)",
            r"(?:another|new) (?:error|issue|problem)",
        ]

        for msg in messages:
            if msg.get('role') != 'user':
                continue

            content = msg.get('content', '')
            if isinstance(content, list):
                content = ' '.join(c.get('text', '') for c in content if isinstance(c, dict))

            content_lower = content.lower()

            # Check for success
            for pattern in success_indicators:
                if re.search(pattern, content_lower):
                    # Look at previous assistant message for what worked
                    worked.append(content[:200])
                    break

            # Check for failure
            for pattern in failure_indicators:
                if re.search(pattern, content_lower):
                    didnt_work.append(content[:200])
                    break

        return worked[:3], didnt_work[:3]

    def _generate_learnings(self, extracted: Dict) -> List[Dict[str, Any]]:
        """Generate high-level learnings from extracted data."""
        learnings = []

        # Learning from corrections
        for correction in extracted['corrections_found']:
            learnings.append({
                'type': 'correction',
                'content': correction['text'],
                'importance': 'high',
                'lesson': 'Previous approach was incorrect, updated method used'
            })

        # Learning from what didn't work
        for failure in extracted['what_didnt_work']:
            learnings.append({
                'type': 'failure',
                'content': failure,
                'importance': 'medium',
                'lesson': 'This approach did not work'
            })

        # Learning from problem→solution pairs
        if extracted['problems_found'] and extracted['solutions_found']:
            learnings.append({
                'type': 'problem_solution',
                'problem': extracted['problems_found'][0]['text'][:150],
                'solution': extracted['solutions_found'][0]['text'][:150],
                'importance': 'high',
                'lesson': 'Documented solution for this problem'
            })

        return learnings

    def save_learnings(self, learnings: Dict[str, Any]) -> str:
        """Save extracted learnings to file."""
        conv_id = learnings.get('conversation_id', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        filename = f"learning_{conv_id}_{timestamp}.json"
        filepath = self.learnings_path / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(learnings, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def compare_conversations(self, conv1_id: str, conv2_id: str) -> Dict[str, Any]:
        """
        Compare two conversations to find contradictions or updates.

        This is key for detecting when a later conversation contradicts
        or updates something from an earlier conversation.
        """
        conv_path = self.base_path / "conversations"

        conv1_file = conv_path / f"{conv1_id}.json"
        conv2_file = conv_path / f"{conv2_id}.json"

        if not conv1_file.exists() or not conv2_file.exists():
            return {'error': 'One or both conversations not found'}

        with open(conv1_file, 'r', encoding='utf-8') as f:
            conv1 = json.load(f)
        with open(conv2_file, 'r', encoding='utf-8') as f:
            conv2 = json.load(f)

        # Extract learnings from both
        learnings1 = self.analyze_conversation(conv1)
        learnings2 = self.analyze_conversation(conv2)

        # Look for contradictions
        contradictions = []

        # Check if conv2 mentions fixing something from conv1
        for solution2 in learnings2['solutions_found']:
            for solution1 in learnings1['solutions_found']:
                # If solution2 mentions the approach from solution1 as wrong
                if self._mentions_as_wrong(solution2['text'], solution1['text']):
                    contradictions.append({
                        'original': solution1['text'],
                        'correction': solution2['text'],
                        'type': 'solution_updated'
                    })

        return {
            'conv1_id': conv1_id,
            'conv2_id': conv2_id,
            'contradictions_found': len(contradictions) > 0,
            'contradictions': contradictions,
            'conv1_learnings': len(learnings1['learnings']),
            'conv2_learnings': len(learnings2['learnings'])
        }

    def _mentions_as_wrong(self, new_text: str, old_text: str) -> bool:
        """Check if new_text mentions old_text approach as wrong."""
        # Simple heuristic: check for correction patterns + similar words
        new_lower = new_text.lower()
        old_words = set(old_text.lower().split())

        # Check for correction language
        correction_found = any(
            re.search(pattern, new_lower)
            for pattern in self.correction_patterns
        )

        # Check for word overlap
        new_words = set(new_lower.split())
        overlap = len(old_words.intersection(new_words)) / max(len(old_words), 1)

        return correction_found and overlap > 0.2

    def get_recent_learnings(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent learnings."""
        learning_files = sorted(
            self.learnings_path.glob("learning_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:limit]

        learnings = []
        for lf in learning_files:
            try:
                with open(lf, 'r', encoding='utf-8') as f:
                    learnings.append(json.load(f))
            except:
                continue

        return learnings

    def get_all_corrections(self) -> List[Dict[str, Any]]:
        """Get all corrections/updates from learnings."""
        corrections = []

        for lf in self.learnings_path.glob("learning_*.json"):
            try:
                with open(lf, 'r', encoding='utf-8') as f:
                    learning = json.load(f)

                for c in learning.get('corrections_found', []):
                    corrections.append({
                        'correction': c['text'],
                        'conversation_id': learning.get('conversation_id'),
                        'timestamp': learning.get('timestamp')
                    })

                for l in learning.get('learnings', []):
                    if l.get('type') == 'correction':
                        corrections.append({
                            'correction': l.get('content'),
                            'conversation_id': learning.get('conversation_id'),
                            'timestamp': learning.get('timestamp')
                        })

            except:
                continue

        return corrections


if __name__ == "__main__":
    extractor = LearningExtractor()

    # Test with a sample conversation
    test_conv = {
        'id': 'test_conv_001',
        'messages': [
            {'role': 'user', 'content': 'The NAS mount is failing at boot with error 101'},
            {'role': 'assistant', 'content': 'The problem is that the network isn\'t ready. The fix is to add the reconnect option to fstab.'},
            {'role': 'user', 'content': 'Still getting an error: Unknown parameter reconnect'},
            {'role': 'assistant', 'content': 'Actually, that was wrong. The reconnect option isn\'t supported on this kernel. Instead, we should use x-systemd.automount which handles reconnection differently.'},
            {'role': 'user', 'content': 'That worked! Thanks!'}
        ]
    }

    result = extractor.analyze_conversation(test_conv)
    print(json.dumps(result, indent=2))

    # Save the learnings
    filepath = extractor.save_learnings(result)
    print(f"\nSaved to: {filepath}")
