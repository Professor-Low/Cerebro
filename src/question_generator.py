#!/usr/bin/env python3
"""
Question Generator - AGI-Like Learning System
Uses information theory (Shannon entropy) to algorithmically identify knowledge gaps
and generate intelligent, contextually-aware questions.
"""

import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Field importance weights (1-10 scale)
FIELD_WEIGHTS = {
    # Identity (Core Understanding)
    'identity.name': 10,
    'identity.username': 10,
    'identity.aliases': 6,
    'identity.roles': 9,
    'identity.location': 7,
    'identity.contact.email': 5,
    'identity.contact.phone': 4,

    # Relationships (Social Context)
    'relationships.pets': 8,
    'relationships.family': 9,
    'relationships.colleagues': 6,
    'relationships.friends': 5,

    # Projects (Active Work)
    'projects.companies_owned': 10,  # Highest priority
    'projects.active_projects': 9,
    'projects.clients': 7,

    # Preferences (Personalization)
    'preferences.technical': 6,
    'preferences.personal': 5,
    'preferences.dislikes': 4,

    # Goals (Future Direction)
    'goals': 10,

    # Technical Environment
    'technical_environment.nas': 8,
    'technical_environment.operating_system': 5,
}

# Category multipliers
CATEGORY_MULTIPLIERS = {
    'identity': 1.0,
    'relationships': 0.9,
    'projects': 1.0,
    'preferences': 0.7,
    'goals': 0.95,
    'technical_environment': 0.6,
}

# Sensitive fields that require more conversations before asking
SENSITIVE_FIELDS = [
    'relationships.family',
    'preferences.dislikes',
    'identity.contact.email',
    'identity.contact.phone',
]

# Field dependencies (must have prerequisite before asking)
FIELD_DEPENDENCIES = {
    'relationships.pets.breed': 'relationships.pets',
    'projects.active_projects.status': 'projects.active_projects',
    'projects.companies_owned.industry': 'projects.companies_owned',
}


class QuestionGenerator:
    """Generates intelligent questions using information-theoretic gap analysis."""

    def __init__(self, base_path: str = ""):
        if not base_path:
            from config import DATA_DIR
            base_path = str(DATA_DIR)
        self.base_path = Path(base_path)
        self.profile_path = self.base_path / "user" / "profile.json"
        self.history_path = self.base_path / "user" / "question_history.json"
        self.conversations_path = self.base_path / "conversations"

    def load_profile(self) -> Dict:
        """Load user profile."""
        if not self.profile_path.exists():
            raise FileNotFoundError(f"Profile not found at {self.profile_path}")

        with open(self.profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_question_history(self) -> Dict:
        """Load question history, create if doesn't exist."""
        if not self.history_path.exists():
            history = {
                'asked_fields': {},
                'answered_fields': {},
                'last_tier4_timestamp': None,
                'tier4_categories_used': [],
                'statistics': {
                    'total_questions_asked': 0,
                    'total_answers_received': 0,
                    'answer_rate': 0.0
                }
            }
            self.save_question_history(history)
            return history

        with open(self.history_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_question_history(self, history: Dict):
        """Save question history."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def get_recent_conversations(self, limit: int = 5) -> List[Dict]:
        """Load most recent conversations."""
        if not self.conversations_path.exists():
            return []

        conv_files = sorted(self.conversations_path.glob("*.json"),
                          key=lambda p: p.stat().st_mtime,
                          reverse=True)

        conversations = []
        for conv_file in conv_files[:limit]:
            try:
                with open(conv_file, 'r', encoding='utf-8') as f:
                    conversations.append(json.load(f))
            except Exception as e:
                print(f"Error loading {conv_file}: {e}")
                continue

        return conversations

    def get_field_value(self, profile: Dict, field_path: str):
        """Get value from profile using dot notation path."""
        parts = field_path.split('.')
        value = profile

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None

            if value is None:
                return None

        return value

    def calculate_information_gain(self, field_path: str, profile: Dict) -> float:
        """
        Calculate information gain using Shannon entropy.

        Returns:
            1.0 = Empty field (maximum uncertainty/gain)
            0.3-0.7 = Partially filled
            0.0 = Complete field (no uncertainty)
        """
        value = self.get_field_value(profile, field_path)

        # Empty field = maximum information gain
        if value is None or value == [] or value == {}:
            return 1.0

        # List fields - saturation curve
        if isinstance(value, list):
            # 1 item = 0.8, 3 items = 0.5, 5 items = 0.2, 10+ items = 0.0
            saturation = min(len(value) / 10, 1.0)
            return max(0.0, 1.0 - saturation)

        # Dict fields - check sub-field completeness
        if isinstance(value, dict):
            if not value:  # Empty dict
                return 1.0

            # Expected subfields (simplified - could be more sophisticated)
            expected_count = 3  # Assume 3 expected subfields on average
            filled_count = len([v for v in value.values() if v])
            completeness = min(filled_count / expected_count, 1.0)
            return max(0.0, 1.0 - completeness)

        # Scalar field with value = no gain
        return 0.0

    def calculate_recency_factor(self, field_path: str, question_history: Dict) -> float:
        """
        Calculate recency factor using exponential decay.

        Returns:
            0.5 = Just asked (deprioritize)
            0.68 = 30 days ago
            0.86 = 60 days ago
            1.0 = 90+ days ago or never asked (fully ripened)
        """
        asked_history = question_history.get('asked_fields', {}).get(field_path, [])

        if not asked_history:
            return 1.0  # Never asked

        # Get most recent ask
        last_asked_str = asked_history[-1]['timestamp']
        last_asked = datetime.fromisoformat(last_asked_str)

        days_since = (datetime.now() - last_asked).days

        # Exponential decay: e^(-t/30)
        # This creates a smooth curve from 0.5 (just asked) to 1.0 (90+ days)
        decay = math.exp(-days_since / 30)

        return 0.5 + (0.5 * (1 - decay))

    def calculate_context_relevance(self, field_path: str, recent_conversations: List[Dict]) -> float:
        """
        Calculate context relevance based on recent conversation topics.

        Returns:
            1.3 = Highly relevant (mentioned 3+ times)
            1.1 = Somewhat relevant (mentioned 1-2 times)
            0.7 = Not relevant
        """
        # Extract topics from recent conversations
        recent_topics = []
        for conv in recent_conversations:
            topics = conv.get('metadata', {}).get('topics', [])
            tags = conv.get('metadata', {}).get('tags', [])
            recent_topics.extend(topics)
            recent_topics.extend(tags)

        # Map field paths to keywords
        relevance_keywords = {
            'relationships.pets': ['pet', 'dog', 'cat', 'puppy', 'kitten', 'animal'],
            'relationships.family': ['family', 'wife', 'husband', 'child', 'parent', 'sibling'],
            'projects.companies_owned': ['company', 'business', 'startup', 'enterprise'],
            'projects.active_projects': ['project', 'development', 'building', 'working'],
            'technical_environment.nas': ['nas', 'storage', 'synology', 'drive', 'backup'],
            'preferences.technical': ['language', 'framework', 'tool', 'technology'],
            'goals': ['goal', 'objective', 'plan', 'want', 'aim'],
            'identity.location': ['location', 'city', 'country', 'live', 'based'],
        }

        keywords = relevance_keywords.get(field_path, [])
        if not keywords:
            return 1.0  # Neutral if no mapping

        # Count matches
        matches = sum(1 for topic in recent_topics
                     if any(kw in topic.lower() for kw in keywords))

        if matches >= 3:
            return 1.3  # Strong relevance boost
        elif matches >= 1:
            return 1.1  # Some relevance
        else:
            return 0.7  # No relevance - deprioritize

    def calculate_question_fatigue(self, field_path: str, question_history: Dict) -> float:
        """
        Calculate question fatigue to avoid repeatedly asking same gaps.

        Returns:
            1.0 = Never asked
            1.3 = Asked once
            1.6 = Asked twice
            1.9 = Asked 3+ times
        """
        asked_count = len(question_history.get('asked_fields', {}).get(field_path, []))
        return 1.0 + (asked_count * 0.3)

    def calculate_gap_score(self, field_path: str, profile: Dict,
                           recent_conversations: List[Dict],
                           question_history: Dict) -> float:
        """
        Calculate comprehensive gap score using information theory.

        Formula:
        GapScore = (FieldWeight × CategoryMultiplier × InformationGain ×
                   RecencyFactor × ContextRelevance) / QuestionFatigue
        """
        # Get components
        field_weight = FIELD_WEIGHTS.get(field_path, 5)  # Default 5 if not mapped

        # Category multiplier
        category = field_path.split('.')[0]
        category_mult = CATEGORY_MULTIPLIERS.get(category, 0.8)

        info_gain = self.calculate_information_gain(field_path, profile)
        recency = self.calculate_recency_factor(field_path, question_history)
        context = self.calculate_context_relevance(field_path, recent_conversations)
        fatigue = self.calculate_question_fatigue(field_path, question_history)

        # Calculate final score
        score = (field_weight * category_mult * info_gain * recency * context) / fatigue

        return score

    def is_gap_ripe(self, field_path: str, profile: Dict,
                   question_history: Dict) -> bool:
        """
        Determine if a gap is "ripe" (ready to ask about).

        Checks:
        - Dependencies met
        - Comfort level (enough conversations for sensitive topics)
        - Not too many questions recently
        """
        # Check dependencies
        if field_path in FIELD_DEPENDENCIES:
            prereq = FIELD_DEPENDENCIES[field_path]
            prereq_value = self.get_field_value(profile, prereq)
            if not prereq_value:
                return False  # Dependency not met

        # Check comfort level for sensitive fields
        if field_path in SENSITIVE_FIELDS:
            total_convs = profile['metadata']['total_conversations_analyzed']
            if total_convs < 10:
                return False  # Too early for sensitive questions

        # Check question rate (don't ask too many at once)
        # This is enforced in select_questions_to_display by limiting to 2-3 questions

        return True

    def analyze_all_gaps(self, profile: Dict) -> List[Dict]:
        """Analyze all possible field gaps."""
        gaps = []

        for field_path in FIELD_WEIGHTS.keys():
            info_gain = self.calculate_information_gain(field_path, profile)

            # Only consider gaps with some information gain
            if info_gain > 0.1:
                gaps.append({
                    'field_path': field_path,
                    'information_gain': info_gain,
                    'category': field_path.split('.')[0]
                })

        return gaps

    def infer_tier(self, gap: Dict, profile: Dict) -> int:
        """
        Infer question tier based on gap characteristics.

        Tier 1: Basic empty fields
        Tier 2: Fields with some data that need enrichment
        Tier 3: Cross-field patterns
        Tier 4: Meta-learning
        """
        info_gain = gap['information_gain']

        if info_gain >= 0.9:
            return 1  # Completely empty = tier 1
        elif info_gain >= 0.3:
            return 2  # Partially filled = tier 2
        else:
            return 3  # Mostly filled but could be enriched = tier 3

    def generate_tier1_question(self, field_path: str) -> Optional[Dict]:
        """Generate tier 1 question (fill basic gaps)."""
        templates = {
            'identity.location': {
                'question': 'Where do you live?',
                'context': 'This helps me understand your local context and provide more relevant assistance.',
                'input_type': 'text'
            },
            'identity.roles': {
                'question': 'What is your professional role or occupation?',
                'context': 'Knowing your role helps me tailor information to your expertise level.',
                'input_type': 'text'
            },
            'relationships.pets': {
                'question': 'Do you have any pets? If yes, what are their names?',
                'context': "I'd love to remember your pets and ask about them in the future.",
                'input_type': 'text'
            },
            'relationships.family': {
                'question': 'Do you have family members you\'d like me to remember?',
                'context': 'This helps me understand your personal context better.',
                'input_type': 'text'
            },
            'projects.companies_owned': {
                'question': 'Do you run any companies or businesses?',
                'context': 'This helps me understand your professional endeavors and priorities.',
                'input_type': 'text'
            },
            'goals': {
                'question': 'What are your current goals or objectives?',
                'context': 'Understanding your goals helps me provide more relevant assistance.',
                'input_type': 'text'
            },
            'preferences.technical': {
                'question': 'What programming languages or technologies do you prefer working with?',
                'context': 'This helps me provide code examples and recommendations you\'ll prefer.',
                'input_type': 'text'
            },
        }

        template = templates.get(field_path)
        if template:
            return {
                **template,
                'tier': 1,
                'target_field': field_path
            }

        return None

    def generate_tier2_question(self, field_path: str, profile: Dict) -> Optional[Dict]:
        """Generate tier 2 question (deepen understanding)."""
        # Pet breed questions
        if 'pets' in field_path:
            pets = profile.get('relationships', {}).get('pets', [])
            if pets:
                for pet in pets:
                    if pet.get('name') and not pet.get('breed'):
                        return {
                            'question': f"What breed is {pet['name']}?",
                            'context': 'This helps me remember your pets better.',
                            'input_type': 'text',
                            'tier': 2,
                            'target_field': f"relationships.pets.{pet['name']}.breed"
                        }

        # Project status questions
        if 'active_projects' in field_path:
            projects = profile.get('projects', {}).get('active_projects', [])
            if projects:
                project = projects[0]
                project_name = project.get('name', 'your project')
                return {
                    'question': f'Is "{project_name}" still active?',
                    'context': 'This helps me track your current work priorities.',
                    'input_type': 'text',
                    'tier': 2,
                    'target_field': f"projects.active_projects.{project_name}.status"
                }

        # Company industry questions
        if 'companies_owned' in field_path:
            companies = profile.get('projects', {}).get('companies_owned', [])
            if companies:
                company = companies[0]
                company_name = company.get('name', 'your company')
                return {
                    'question': f'What industry is {company_name} in?',
                    'context': 'This helps me understand your business context.',
                    'input_type': 'text',
                    'tier': 2,
                    'target_field': f"projects.companies_owned.{company_name}.industry"
                }

        return None

    def generate_tier3_question(self, profile: Dict) -> Optional[Dict]:
        """Generate tier 3 question (discover patterns via cross-field analysis)."""
        # Pattern: Has NAS + Python skills but no automation projects
        nas_config = profile.get('technical_environment', {}).get('nas')
        active_projects = profile.get('projects', {}).get('active_projects', [])

        if nas_config and not any('automation' in str(p).lower() for p in active_projects):
            return {
                'question': 'You have a NAS setup and mention Python often - are you building any automation with them?',
                'context': 'This helps me discover projects you might be working on.',
                'input_type': 'text',
                'tier': 3,
                'target_field': 'projects.active_projects.nas_automation'
            }

        # Pattern: Has company but no clients
        companies = profile.get('projects', {}).get('companies_owned', [])
        clients = profile.get('projects', {}).get('clients', [])

        if companies and not clients:
            company_name = companies[0].get('name', 'your company')
            return {
                'question': f'Who are the primary customers or clients for {company_name}?',
                'context': 'This helps me understand your business relationships.',
                'input_type': 'text',
                'tier': 3,
                'target_field': 'projects.clients'
            }

        # Pattern: Multiple projects but no priority indication
        if len(active_projects) >= 3:
            return {
                'question': f'You have {len(active_projects)} active projects - which one is your top priority?',
                'context': 'This helps me understand what matters most to you right now.',
                'input_type': 'text',
                'tier': 3,
                'target_field': 'projects.active_projects.priority'
            }

        return None

    def generate_tier4_question(self, profile: Dict, question_history: Dict) -> Optional[Dict]:
        """Generate tier 4 question (meta-learning - recursive improvement)."""
        total_convs = profile['metadata']['total_conversations_analyzed']

        # Only ask tier 4 every 20 conversations
        if total_convs < 20 or total_convs % 20 != 0:
            return None

        # Avoid repeating same tier 4 category
        used_categories = question_history.get('tier4_categories_used', [])

        tier4_questions = [
            {
                'category': 'priority_discovery',
                'question': 'What kind of information would be most useful for me to remember about you?',
                'context': 'This helps me learn what matters most to you.',
            },
            {
                'category': 'learning_style',
                'question': 'Would you like me to ask more questions, or learn primarily through observation?',
                'context': 'This helps me understand your preferred interaction style.',
            },
            {
                'category': 'gap_awareness',
                'question': 'Is there anything important about you that I haven\'t asked about yet?',
                'context': 'This helps me discover gaps I might not be aware of.',
            },
            {
                'category': 'frequency_preference',
                'question': 'How often would you like me to ask you questions like these?',
                'context': 'This helps me adjust my question frequency to your preference.',
            },
        ]

        # Filter out used categories
        available = [q for q in tier4_questions if q['category'] not in used_categories]

        if not available:
            # Reset if all used
            available = tier4_questions

        selected = random.choice(available)

        return {
            'question': selected['question'],
            'context': selected['context'],
            'input_type': 'text',
            'tier': 4,
            'target_field': f"meta_learning.{selected['category']}"
        }

    def select_questions_to_display(self, profile: Dict,
                                    recent_conversations: List[Dict],
                                    question_history: Dict) -> List[Dict]:
        """
        Select 2-3 optimal questions to display.

        Strategy:
        - Always 1 tier 1/2 (core gaps)
        - Sometimes tier 3 (based on maturity)
        - Rarely tier 4 (every 20 conversations)
        """
        selected_questions = []

        # Step 1: Analyze all gaps and score them
        all_gaps = self.analyze_all_gaps(profile)
        gap_scores = []

        for gap in all_gaps:
            score = self.calculate_gap_score(
                gap['field_path'],
                profile,
                recent_conversations,
                question_history
            )

            if self.is_gap_ripe(gap['field_path'], profile, question_history):
                tier = self.infer_tier(gap, profile)
                gap_scores.append({
                    'gap': gap,
                    'score': score,
                    'tier': tier
                })

        # Sort by score (highest first)
        gap_scores.sort(key=lambda x: x['score'], reverse=True)

        # Step 2: Select top tier 1/2 question
        tier12_gaps = [g for g in gap_scores if g['tier'] in [1, 2]]

        if tier12_gaps:
            top_gap = tier12_gaps[0]

            if top_gap['tier'] == 1:
                question = self.generate_tier1_question(top_gap['gap']['field_path'])
            else:
                question = self.generate_tier2_question(top_gap['gap']['field_path'], profile)

            if question:
                selected_questions.append(question)

        # Step 3: Maybe add tier 3 question (pattern discovery)
        total_convs = profile['metadata']['total_conversations_analyzed']
        tier3_probability = min(0.6, (total_convs - 10) / 100)  # Increases with maturity

        if random.random() < tier3_probability:
            tier3_q = self.generate_tier3_question(profile)
            if tier3_q:
                selected_questions.append(tier3_q)

        # Step 4: Maybe add tier 4 question (meta-learning)
        tier4_q = self.generate_tier4_question(profile, question_history)
        if tier4_q:
            selected_questions.append(tier4_q)

        # Step 5: Fill to 2-3 questions with more tier 1/2
        while len(selected_questions) < 2 and len(tier12_gaps) > len(selected_questions):
            next_gap = tier12_gaps[len(selected_questions)]

            if next_gap['tier'] == 1:
                question = self.generate_tier1_question(next_gap['gap']['field_path'])
            else:
                question = self.generate_tier2_question(next_gap['gap']['field_path'], profile)

            if question and question not in selected_questions:
                selected_questions.append(question)

        return selected_questions[:3]  # Maximum 3 questions


# Test function
if __name__ == "__main__":
    generator = QuestionGenerator()

    try:
        profile = generator.load_profile()
        recent_convs = generator.get_recent_conversations(limit=5)
        history = generator.load_question_history()

        questions = generator.select_questions_to_display(profile, recent_convs, history)

        print("Generated Questions:")
        print("=" * 60)
        for i, q in enumerate(questions, 1):
            print(f"\n{i}. [{q['tier']}] {q['question']}")
            print(f"   Context: {q['context']}")
            print(f"   Target: {q['target_field']}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
