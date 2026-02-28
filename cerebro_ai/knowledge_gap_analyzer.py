"""
Knowledge Gap Analyzer - Identify things the user keeps explaining
Part of Agent 9: Code Understanding & Pattern Detection
"""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


class KnowledgeGapAnalyzer:
    """
    Identify things the user keeps explaining (knowledge gaps).
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
        self.gaps_file = self.patterns_path / "knowledge_gaps.json"

        self.patterns_path.mkdir(parents=True, exist_ok=True)

    def find_knowledge_gaps(self, threshold: int = 3) -> List[Dict]:
        """
        Find concepts user explains repeatedly.

        Args:
            threshold: Min times mentioned to be a gap

        Returns:
            List of knowledge gaps
        """
        # Track user explanations of concepts
        entity_mentions = defaultdict(lambda: {'user_explains': 0, 'conversations': []})

        for conv_file in self.conversations_path.glob('*.json'):
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conv = json.load(f)

                # Analyze user messages
                for msg in conv.get('messages', []):
                    if msg.get('role') != 'user':
                        continue

                    content = msg.get('content', '')

                    # Check for explanation patterns
                    if self._is_explanation(content):
                        # Extract entities from this explanation
                        entities = self._extract_entities_from_text(content, conv)

                        for entity in entities:
                            entity_mentions[entity]['user_explains'] += 1
                            if conv['id'] not in entity_mentions[entity]['conversations']:
                                entity_mentions[entity]['conversations'].append(conv['id'])

            except (json.JSONDecodeError, KeyError):
                continue

        # Filter gaps (repeatedly explained)
        gaps = [
            {
                'concept': entity,
                'explanation_count': data['user_explains'],
                'conversations': data['conversations'][:5],
                'gap_type': 'repeated_explanation',
                'suggestion': f"User explained '{entity}' {data['user_explains']} times - should be in permanent context"
            }
            for entity, data in entity_mentions.items()
            if data['user_explains'] >= threshold
        ]

        # Sort by frequency
        gaps.sort(key=lambda x: x['explanation_count'], reverse=True)

        # Save
        with open(self.gaps_file, 'w', encoding='utf-8') as f:
            json.dump(gaps, f, indent=2, ensure_ascii=False)

        return gaps

    def _is_explanation(self, text: str) -> bool:
        """Check if text is an explanation"""
        explanation_patterns = [
            'is a', 'is the', 'means', 'refers to', 'basically',
            'in other words', 'to clarify', 'what i mean is',
            'let me explain', 'the way it works'
        ]

        text_lower = text.lower()
        return any(pattern in text_lower for pattern in explanation_patterns)

    def _extract_entities_from_text(self, text: str, conversation: Dict) -> List[str]:
        """Extract entities from explanation text"""
        entities = []

        # Get entities from conversation metadata
        conv_entities = conversation.get('extracted_data', {}).get('entities', {})

        for entity_type, entity_list in conv_entities.items():
            if isinstance(entity_list, list):
                entities.extend(entity_list)
            elif isinstance(entity_list, dict):
                entities.extend(entity_list.keys())

        # Filter entities mentioned in this text
        text_lower = text.lower()
        mentioned = [e for e in entities if e.lower() in text_lower]

        return mentioned
