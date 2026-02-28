#!/usr/bin/env python3
"""
Emotional Pattern Detector - Auto-detect emotional patterns from conversations.
Identifies frustrations, excitements, preferences, and communication style.
"""
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Emotional signal patterns
FRUSTRATION_SIGNALS = [
    r'\bthis is frustrating\b',
    r'\bwhy doesn\'t this work\b',
    r'\bwhy isn\'t this working\b',
    r'\bugh\b',
    r'\bannoying\b',
    r'\bfrustrating\b',
    r'\bhate when\b',
    r'\btired of\b',
    r'\bsick of\b',
    r'\bwaste of time\b',
    r'\bnot working\b',
    r'\bkeeps failing\b',
    r'\bstill broken\b',
]

EXCITEMENT_SIGNALS = [
    r'\bawesome[!\s]',
    r'\bperfect[!\s]',
    r'\blove this\b',
    r'\bexactly what i wanted\b',
    r'\bthis is great\b',
    r'\bworks perfectly\b',
    r'\bamazing[!\s]',
    r'\bexcellent[!\s]',
    r'\bfantastic[!\s]',
    r'\bbrilliantly?\b',
    r'\bthank you so much\b',
    r'\bthis is incredible\b',
]

PREFERENCE_SIGNALS = {
    'prefers': [
        r'\bi prefer\b',
        r'\bi like\b',
        r'\bi want\b',
        r'\bbetter to\b',
        r'\bi\'d rather\b',
        r'\bplease use\b',
        r'\balways use\b',
    ],
    'dislikes': [
        r'\bi don\'t like\b',
        r'\bi hate\b',
        r'\bplease don\'t\b',
        r'\bavoid\b',
        r'\bnever use\b',
        r'\bdon\'t use\b',
        r'\bstop\b.*\b(using|doing)\b',
    ]
}

# Communication style signals
COMMUNICATION_STYLE_SIGNALS = {
    'concise': [
        r'\bkeep it short\b',
        r'\bjust the code\b',
        r'\bbriefly\b',
        r'\bquick summary\b',
        r'\bin a nutshell\b',
        r'\bshort answer\b',
    ],
    'detailed': [
        r'\bexplain in detail\b',
        r'\bwalk me through\b',
        r'\bstep by step\b',
        r'\bdetailed explanation\b',
        r'\btell me more\b',
    ],
    'code_focused': [
        r'\bjust show me the code\b',
        r'\bcode only\b',
        r'\bskip the explanation\b',
        r'\bno need to explain\b',
    ],
    'explanation_focused': [
        r'\bexplain how\b',
        r'\bwhy does\b',
        r'\bhelp me understand\b',
        r'\bwhat\'s the difference\b',
    ]
}


class EmotionalDetector:
    """Detects emotional patterns and communication preferences from conversations."""

    def __init__(self, base_path: str = ""):
        if not base_path:
            from config import DATA_DIR
            base_path = str(DATA_DIR)
        self.base_path = Path(base_path)
        self.profile_path = self.base_path / "user" / "profile.json"

    def detect_emotional_patterns(self, conversation: Dict) -> Dict[str, List[Dict]]:
        """
        Scan conversation for emotional signals.

        Returns:
            Dictionary with 'frustrations', 'excitements', and 'preferences'
        """
        patterns = {
            'frustrations': [],
            'excitements': [],
            'preferences': {'prefers': [], 'dislikes': []}
        }

        messages = conversation.get('messages', [])

        for msg in messages:
            if msg.get('role') != 'user':
                continue

            content = msg.get('content', '')
            if not content:
                continue

            # Detect frustrations
            frustrations = self._detect_frustrations(content)
            patterns['frustrations'].extend(frustrations)

            # Detect excitements
            excitements = self._detect_excitements(content)
            patterns['excitements'].extend(excitements)

            # Detect preferences
            prefs = self._detect_preferences(content)
            patterns['preferences']['prefers'].extend(prefs['prefers'])
            patterns['preferences']['dislikes'].extend(prefs['dislikes'])

        return patterns

    def _detect_frustrations(self, text: str) -> List[Dict]:
        """Detect frustration signals in text."""
        frustrations = []
        text_lower = text.lower()

        for pattern in FRUSTRATION_SIGNALS:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # Extract context around the match (±50 chars)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()

                # Try to extract the trigger
                trigger = self._extract_frustration_trigger(text, match.start())

                frustrations.append({
                    'signal': match.group(),
                    'trigger': trigger,
                    'context': context,
                    'intensity': self._estimate_intensity(text_lower, match.start()),
                    'timestamp': datetime.now().isoformat()
                })

        return frustrations

    def _detect_excitements(self, text: str) -> List[Dict]:
        """Detect excitement signals in text."""
        excitements = []
        text_lower = text.lower()

        for pattern in EXCITEMENT_SIGNALS:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                # Extract context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()

                # Try to extract what they're excited about
                trigger = self._extract_excitement_trigger(text, match.start())

                excitements.append({
                    'signal': match.group(),
                    'trigger': trigger,
                    'context': context,
                    'intensity': self._estimate_intensity(text_lower, match.start()),
                    'timestamp': datetime.now().isoformat()
                })

        return excitements

    def _detect_preferences(self, text: str) -> Dict[str, List[str]]:
        """Detect preference signals in text."""
        preferences = {'prefers': [], 'dislikes': []}
        text_lower = text.lower()

        # Detect preferences
        for pattern in PREFERENCE_SIGNALS['prefers']:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Extract what comes after the preference signal
                pref_text = self._extract_preference_object(text, match.end())
                if pref_text:
                    preferences['prefers'].append(pref_text)

        # Detect dislikes
        for pattern in PREFERENCE_SIGNALS['dislikes']:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                dislike_text = self._extract_preference_object(text, match.end())
                if dislike_text:
                    preferences['dislikes'].append(dislike_text)

        return preferences

    def detect_communication_style(self, conversation: Dict) -> Dict[str, any]:
        """Detect user's communication style preferences."""
        style = {
            'tone_preference': 'professional but friendly',  # Default
            'detail_level': 'balanced',  # concise/detailed/balanced
            'code_focus': 'balanced',  # code_focused/explanation_focused/balanced
            'signals_detected': defaultdict(int)
        }

        messages = conversation.get('messages', [])
        user_messages = [m for m in messages if m.get('role') == 'user']

        if not user_messages:
            return style

        # Aggregate all user text
        all_text = ' '.join(m.get('content', '') for m in user_messages).lower()

        # Detect detail level
        concise_count = sum(
            len(re.findall(pattern, all_text))
            for pattern in COMMUNICATION_STYLE_SIGNALS['concise']
        )
        detailed_count = sum(
            len(re.findall(pattern, all_text))
            for pattern in COMMUNICATION_STYLE_SIGNALS['detailed']
        )

        if concise_count > detailed_count and concise_count > 0:
            style['detail_level'] = 'concise'
        elif detailed_count > concise_count and detailed_count > 0:
            style['detail_level'] = 'detailed'

        # Detect code focus
        code_focus_count = sum(
            len(re.findall(pattern, all_text))
            for pattern in COMMUNICATION_STYLE_SIGNALS['code_focused']
        )
        explanation_focus_count = sum(
            len(re.findall(pattern, all_text))
            for pattern in COMMUNICATION_STYLE_SIGNALS['explanation_focused']
        )

        if code_focus_count > explanation_focus_count and code_focus_count > 0:
            style['code_focus'] = 'code_focused'
        elif explanation_focus_count > code_focus_count and explanation_focus_count > 0:
            style['code_focus'] = 'explanation_focused'

        # Track signals for confidence
        style['signals_detected']['concise'] = concise_count
        style['signals_detected']['detailed'] = detailed_count
        style['signals_detected']['code_focused'] = code_focus_count
        style['signals_detected']['explanation_focused'] = explanation_focus_count

        return style

    def _extract_frustration_trigger(self, text: str, position: int) -> str:
        """Extract what the user is frustrated about."""
        # Look for object after frustration signal
        remaining = text[position:position+100]

        # Common patterns: "X is frustrating", "frustrated with X", "why doesn't X work"
        patterns = [
            r'with\s+([^,.!?]+)',
            r'about\s+([^,.!?]+)',
            r"doesn't\s+([^,.!?]+)\s+work",
            r"isn't\s+([^,.!?]+)\s+working",
        ]

        for pattern in patterns:
            match = re.search(pattern, remaining, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return "unspecified"

    def _extract_excitement_trigger(self, text: str, position: int) -> str:
        """Extract what the user is excited about."""
        # Look backward and forward for context
        before = text[max(0, position-100):position]
        after = text[position:position+50]

        # Try to find subject in surrounding context
        context = before + after

        # Look for common patterns
        patterns = [
            r'(?:this|that|it)\s+(\w+(?:\s+\w+){0,3})',
            r'the\s+(\w+(?:\s+\w+){0,3})',
        ]

        for pattern in patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return "achievement or feature"

    def _extract_preference_object(self, text: str, position: int) -> Optional[str]:
        """Extract the object of a preference statement."""
        remaining = text[position:position+150]

        # Extract until next sentence boundary
        match = re.match(r'\s*([^.!?\n]+)', remaining)
        if match:
            preference = match.group(1).strip()
            if len(preference) > 3:  # Filter out very short matches
                return preference

        return None

    def _estimate_intensity(self, text: str, position: int) -> str:
        """Estimate emotional intensity based on context."""
        # Check for intensity markers
        context = text[max(0, position-50):position+50]

        high_intensity_markers = ['!!!', 'very', 'extremely', 'really', 'so', 'absolutely']
        medium_intensity_markers = ['quite', 'pretty', 'somewhat']

        for marker in high_intensity_markers:
            if marker in context:
                return 'high'

        for marker in medium_intensity_markers:
            if marker in context:
                return 'medium'

        return 'low'

    def aggregate_patterns(self, patterns_list: List[Dict]) -> Dict:
        """
        Aggregate patterns from multiple conversations.

        Args:
            patterns_list: List of pattern dictionaries from multiple conversations

        Returns:
            Aggregated and deduplicated patterns
        """
        aggregated = {
            'frustrations': [],
            'excitements': [],
            'preferences': {'prefers': [], 'dislikes': []}
        }

        # Track occurrences for deduplication
        frustration_triggers = defaultdict(list)
        excitement_triggers = defaultdict(list)

        for patterns in patterns_list:
            # Aggregate frustrations
            for frust in patterns.get('frustrations', []):
                trigger = frust['trigger']
                frustration_triggers[trigger].append(frust)

            # Aggregate excitements
            for excite in patterns.get('excitements', []):
                trigger = excite['trigger']
                excitement_triggers[trigger].append(excite)

            # Aggregate preferences (deduplicate)
            for pref in patterns.get('preferences', {}).get('prefers', []):
                if pref not in aggregated['preferences']['prefers']:
                    aggregated['preferences']['prefers'].append(pref)

            for dislike in patterns.get('preferences', {}).get('dislikes', []):
                if dislike not in aggregated['preferences']['dislikes']:
                    aggregated['preferences']['dislikes'].append(dislike)

        # Convert to final format with occurrence counts
        for trigger, occurrences in frustration_triggers.items():
            aggregated['frustrations'].append({
                'trigger': trigger,
                'intensity': max(occ['intensity'] for occ in occurrences),
                'occurrences': len(occurrences),
                'first_noted': min(occ['timestamp'] for occ in occurrences),
                'last_noted': max(occ['timestamp'] for occ in occurrences)
            })

        for trigger, occurrences in excitement_triggers.items():
            aggregated['excitements'].append({
                'trigger': trigger,
                'intensity': max(occ['intensity'] for occ in occurrences),
                'occurrences': len(occurrences),
                'first_noted': min(occ['timestamp'] for occ in occurrences),
                'last_noted': max(occ['timestamp'] for occ in occurrences)
            })

        return aggregated


if __name__ == "__main__":
    # Test the detector
    detector = EmotionalDetector()

    test_conversation = {
        'messages': [
            {
                'role': 'user',
                'content': 'This is so frustrating! Why doesn\'t the NAS connection work?'
            },
            {
                'role': 'assistant',
                'content': 'Let me help you debug that.'
            },
            {
                'role': 'user',
                'content': 'Perfect! That works exactly as I wanted. I love this feature!'
            },
            {
                'role': 'user',
                'content': 'I prefer concise answers with just the code. Please don\'t add long explanations.'
            }
        ]
    }

    patterns = detector.detect_emotional_patterns(test_conversation)
    style = detector.detect_communication_style(test_conversation)

    print("Detected Emotional Patterns:")
    print(json.dumps(patterns, indent=2))
    print("\nDetected Communication Style:")
    print(json.dumps(style, indent=2, default=str))
