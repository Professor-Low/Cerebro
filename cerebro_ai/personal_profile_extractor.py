#!/usr/bin/env python3
"""
Personal Profile Extractor for NAS Cerebral Interface
Extracts comprehensive personal information about the user
from conversations to build a living personal knowledge profile.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class PersonalProfileExtractor:
    """Extract personal information about the user from conversations."""

    # Invalid names - pronouns, common words, etc.
    INVALID_NAMES = {
        'i', 'me', 'my', 'mine', 'you', 'your', 'we', 'our', 'they', 'them',
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ok', 'okay', 'yes', 'no', 'not', 'and', 'or', 'but', 'if', 'then',
        'so', 'as', 'at', 'by', 'for', 'in', 'of', 'on', 'to', 'with',
        'it', 'its', 'this', 'that', 'these', 'those', 'here', 'there',
        'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
        'such', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also'
    }

    @classmethod
    def _is_valid_name(cls, name: str) -> bool:
        """Check if a name is valid (not a pronoun or common word)."""
        if not name:
            return False
        if len(name) < 2:
            return False
        if name.lower() in cls.INVALID_NAMES:
            return False
        # Names should start with a letter
        if not name[0].isalpha():
            return False
        return True

    def __init__(self):
        """Initialize the personal profile extractor."""
        # Personal pronouns that indicate user talking about themselves
        self.personal_indicators = [
            r'\bmy\b', r'\bmine\b', r'\bi\b', r'\bme\b', r"\bi'm\b",
            r"\bi've\b", r"\bi'll\b", r"\bi'd\b"
        ]

        # Known user identities (will be populated from profile)
        self.user_identities = {
            'names': [],
            'aliases': [],
            'username': None
        }

    # Common pronouns/articles that should never be extracted as entities
    INVALID_EXTRACTIONS = {
        'she', 'he', 'his', 'her', 'the', 'a', 'an', 'it', 'they', 'them',
        'you', 'your', 'we', 'our', 'my', 'i'
    }

    # Regex to detect API keys, tokens, passwords, and other secrets
    _SECRET_PATTERN = re.compile(
        r'(sk_|pk_|api_|token_|key_|secret_|password|bearer)\S{10,}',
        re.IGNORECASE
    )

    def _is_valid_extraction(self, value: str) -> bool:
        """Validate that an extracted value is not garbage.

        Returns False for:
        - Empty or too short/long values
        - Values ending/starting with ellipsis
        - Instruction-like text
        - Strings that look like API keys/tokens/secrets
        - Common pronouns/articles
        - Single characters or pure numbers
        - Strings with excessive special characters (>30% non-alphanumeric)
        """
        if not value or len(value) < 2:
            return False
        if len(value) > 100:  # Too long = garbage
            return False
        if value.endswith('...') or value.startswith('...'):
            return False

        # Reject single characters or pure numbers
        stripped = value.strip()
        if len(stripped) <= 1:
            return False
        if stripped.isdigit():
            return False

        # Reject common pronouns/articles
        if stripped.lower() in self.INVALID_EXTRACTIONS:
            return False

        # Reject strings that look like API keys/tokens/secrets
        if self._SECRET_PATTERN.search(stripped):
            return False

        # Reject strings with excessive special characters (>30% non-alphanumeric)
        if len(stripped) > 0:
            alnum_count = sum(1 for c in stripped if c.isalnum() or c.isspace())
            if alnum_count / len(stripped) < 0.7:
                return False

        # Reject instruction-like text
        lower_val = stripped.lower()
        if lower_val.startswith(('you to ', 'i want you to', 'please ', 'can you ', 'could you ')):
            return False
        return True

    def extract_all(self, messages: List[Dict], conversation_id: str) -> Dict[str, Any]:
        """
        Extract all personal information from a conversation.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            conversation_id: Unique conversation identifier

        Returns:
            Dictionary containing all extracted personal data
        """
        # Only analyze user messages for personal info
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        full_text = ' '.join([msg.get('content', '') for msg in user_messages])

        return {
            'identity': self._extract_identity(full_text, conversation_id),
            'relationships': self._extract_relationships(full_text, conversation_id),
            'projects': self._extract_projects(full_text, conversation_id),
            'preferences': self._extract_preferences(full_text, conversation_id),
            'goals': self._extract_goals(full_text, conversation_id),
            'technical_environment': self._extract_tech_env(full_text, conversation_id),
            'is_personal_conversation': self._is_personal_conversation(full_text)
        }

    def _extract_identity(self, text: str, conv_id: str) -> Dict[str, Any]:
        """Extract identity information (name, role, location, etc.)."""
        identity = {}

        # Extract roles/titles
        role_patterns = [
            r'i(?:\s+am|\s*\'m)\s+(?:a|an|the)\s+([a-z]+(?:\s+[a-z]+)?)',  # "I'm a developer"
            r'my\s+(?:job|role|position|title)\s+is\s+([a-z\s]+)',  # "my role is CEO"
            r'i\s+work\s+as\s+(?:a|an)\s+([a-z\s]+)',  # "I work as a developer"
        ]

        roles = []
        for pattern in role_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                role = match.group(1).strip()
                # Filter out common false positives
                if role and len(role) > 2 and role not in ['user', 'person', 'one']:
                    roles.append(role.title())

        if roles:
            identity['roles'] = list(set(roles))

        # Extract location mentions
        location_patterns = [
            r'i\s+live\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # "I live in New York"
            r'i(?:\s+am|\s*\'m)\s+from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # "I'm from California"
            r'my\s+(?:city|location|home)\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        ]

        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                identity['location'] = match.group(1).strip()
                break

        # Extract contact info
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails and any(indicator in text.lower() for indicator in ['my email', 'contact me', 'reach me']):
            identity['email'] = emails[0]

        return identity if identity else None

    def _extract_relationships(self, text: str, conv_id: str) -> Dict[str, List[Dict]]:
        """Extract relationships (pets, family, friends, colleagues)."""
        relationships = {
            'pets': [],
            'family': [],
            'colleagues': [],
            'friends': []
        }

        # Extract pets
        pet_patterns = [
            # "i have 2 dogs teddy and tobi" - simple list format
            (r'(?:i\s+have|i\s+own|my)\s+\d*\s*(dogs?|cats?|pets?)\s+(?:named\s+|called\s+)?([a-z]+)(?:\s+and\s+([a-z]+))?', 'names_list', None),
            # "my dog is named Teddy" or "my dog Teddy"
            (r'my\s+(dog|cat|pet|puppy|kitten)(?:\s+(?:is\s+)?(?:named|called))?\s+([A-Za-z]+)', 'type', 'name'),
            # "my Teddy the dog"
            (r'my\s+([A-Za-z]+)\s+(?:the\s+)?(dog|cat|pet|puppy|kitten)', 'name', 'type'),
            # "my dog is a golden retriever"
            (r'my\s+(dog|cat|pet)(?:\s+is\s+a)?\s+([a-z\s]+\s+(?:retriever|shepherd|terrier|spaniel|lab|labrador|poodle|bulldog|beagle))', 'type_general', 'breed'),
        ]

        for pattern, group1_type, group2_type in pet_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if group1_type == 'names_list':
                    # Handle "i have 2 dogs teddy and tobi" format
                    # Group 1 is the pet type (dogs/cats/pets)
                    # Group 2 is the first name
                    # Group 3 is the second name (if present)
                    pet_type_raw = match.group(1).lower() if match.group(1) else 'pet'
                    # Normalize to singular
                    if pet_type_raw.endswith('s'):
                        pet_type = pet_type_raw[:-1]
                    else:
                        pet_type = pet_type_raw

                    # Extract names
                    name1 = match.group(2).capitalize() if match.group(2) else None
                    name2 = match.group(3).capitalize() if len(match.groups()) >= 3 and match.group(3) else None

                    # Add first pet (validate name)
                    if name1 and self._is_valid_name(name1):
                        pet_info1 = {
                            'name': name1,
                            'type': pet_type,
                            'conversation_id': conv_id
                        }
                        if pet_info1 not in relationships['pets']:
                            relationships['pets'].append(pet_info1)

                    # Add second pet (validate name)
                    if name2 and self._is_valid_name(name2):
                        pet_info2 = {
                            'name': name2,
                            'type': pet_type,
                            'conversation_id': conv_id
                        }
                        if pet_info2 not in relationships['pets']:
                            relationships['pets'].append(pet_info2)
                else:
                    # Handle other patterns
                    pet_info = {'conversation_id': conv_id}
                    name_to_validate = None

                    if group1_type == 'type':
                        pet_info['type'] = match.group(1).lower()
                        if len(match.groups()) > 1 and match.group(2):
                            name_to_validate = match.group(2).capitalize()
                            pet_info['name'] = name_to_validate
                    elif group1_type == 'name':
                        name_to_validate = match.group(1).capitalize()
                        pet_info['name'] = name_to_validate
                        pet_info['type'] = match.group(2).lower()
                    elif group1_type == 'type_general':
                        pet_info['type'] = match.group(1).lower()
                        pet_info['breed'] = match.group(2).strip()

                    # Only add if name is valid (or no name required like breed-only)
                    if 'name' not in pet_info or self._is_valid_name(pet_info.get('name', '')):
                        if pet_info not in relationships['pets']:
                            relationships['pets'].append(pet_info)

        # Extract family members
        family_patterns = [
            r'my\s+(wife|husband|spouse|partner|mother|mom|father|dad|sister|brother|son|daughter|child|kid)',
            r'my\s+(wife|husband|spouse|partner)(?:\s+(?:is\s+)?(?:named|called))?\s+([A-Z][a-z]+)',
        ]

        for pattern in family_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                relationship_type = match.group(1).lower()
                name = match.group(2) if len(match.groups()) > 1 else None

                family_member = {
                    'relationship': relationship_type,
                    'conversation_id': conv_id
                }
                if name:
                    family_member['name'] = name

                if family_member not in relationships['family']:
                    relationships['family'].append(family_member)

        # Extract colleagues/coworkers
        colleague_patterns = [
            r'my\s+(?:colleague|coworker|teammate|partner|boss|manager)\s+([A-Z][a-z]+)',
            r'i\s+work\s+with\s+([A-Z][a-z]+)',
        ]

        for pattern in colleague_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                name = match.group(1)
                colleague = {
                    'name': name,
                    'conversation_id': conv_id
                }
                if colleague not in relationships['colleagues']:
                    relationships['colleagues'].append(colleague)

        # Extract friends
        friend_patterns = [
            r'my\s+friend\s+([A-Z][a-z]+)',
            r'my\s+best\s+friend\s+([A-Z][a-z]+)',
        ]

        for pattern in friend_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                name = match.group(1)
                friend = {
                    'name': name,
                    'conversation_id': conv_id
                }
                if friend not in relationships['friends']:
                    relationships['friends'].append(friend)

        # Remove empty categories
        return {k: v for k, v in relationships.items() if v}

    def _extract_projects(self, text: str, conv_id: str) -> Dict[str, List[Dict]]:
        """Extract projects, companies, and professional work."""
        projects = {
            'companies_owned': [],
            'clients': [],
            'active_projects': []
        }

        # Extract companies owned/run
        company_patterns = [
            r'my\s+company(?:\s+(?:is\s+)?(?:named|called))?\s+([A-Z][A-Za-z\s&]+)',
            r'i\s+(?:own|run|founded)\s+(?:a\s+company\s+called\s+)?([A-Z][A-Za-z\s&]+)',
            r'i(?:\s+am|\s*\'m)\s+(?:the\s+)?(?:CEO|founder|owner)\s+of\s+([A-Z][A-Za-z\s&]+)',
        ]

        for pattern in company_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                company_name = match.group(1).strip()
                # Filter out generic words and validate extraction
                if company_name and len(company_name) > 2 and company_name not in ['Company', 'Business']:
                    if not self._is_valid_extraction(company_name):
                        continue
                    company = {
                        'name': company_name,
                        'conversation_id': conv_id
                    }
                    if company not in projects['companies_owned']:
                        projects['companies_owned'].append(company)

        # Extract client mentions
        client_patterns = [
            r'my\s+client\s+([A-Z][A-Za-z\s&]+)',
            r'(?:working|work)\s+(?:with|for)\s+(?:a\s+client\s+called\s+)?([A-Z][A-Za-z\s&]+)',
        ]

        for pattern in client_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                client_name = match.group(1).strip()
                if client_name and len(client_name) > 2:
                    if not self._is_valid_extraction(client_name):
                        continue
                    client = {
                        'name': client_name,
                        'conversation_id': conv_id
                    }
                    if client not in projects['clients']:
                        projects['clients'].append(client)

        # Extract project mentions
        project_patterns = [
            r'(?:working\s+on|building|creating|developing)\s+(?:a\s+)?([a-z][a-z\s-]+(?:system|interface|platform|tool|app|application|project|pipeline|service))',
            r'my\s+project\s+(?:is\s+)?(?:called\s+)?([A-Z][A-Za-z\s-]+)',
        ]

        for pattern in project_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                project_name = match.group(1).strip()
                if project_name and len(project_name) > 5:
                    project = {
                        'name': project_name.title(),
                        'conversation_id': conv_id,
                        'status': 'active'
                    }
                    if project not in projects['active_projects']:
                        projects['active_projects'].append(project)

        # Remove empty categories
        return {k: v for k, v in projects.items() if v}

    def _extract_preferences(self, text: str, conv_id: str) -> Dict[str, List[Dict]]:
        """Extract user preferences (both technical and personal)."""
        preferences = {
            'technical': [],
            'personal': [],
            'dislikes': []
        }

        # Technical preferences
        tech_pref_patterns = [
            r'i\s+(?:prefer|like)\s+(?:to\s+use|using)\s+([A-Z][A-Za-z\s]+(?:IDE|editor|tool|language|framework|library))',
            r'i\s+(?:prefer|like)\s+([A-Z][A-Za-z]+)\s+(?:for|over|to)',
            r'my\s+(?:favorite|preferred)\s+([a-z]+)\s+is\s+([A-Z][A-Za-z]+)',
        ]

        for pattern in tech_pref_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                pref_text = match.group(1) if len(match.groups()) == 1 else match.group(2)
                pref_text = pref_text.strip()
                # Validate extraction
                if not self._is_valid_extraction(pref_text):
                    continue
                preference = {
                    'preference': pref_text,
                    'category': 'technical',
                    'conversation_id': conv_id
                }
                if preference not in preferences['technical']:
                    preferences['technical'].append(preference)

        # Personal/lifestyle preferences
        personal_pref_patterns = [
            r'i\s+(?:prefer|like|enjoy|love)\s+(?:to\s+)?([a-z][a-z\s]+(?:ing)?)',
            r'i(?:\s+am|\s*\'m)\s+(?:a|an)\s+([a-z\s]+)\s+(?:person|type)',
        ]

        for pattern in personal_pref_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                pref_text = match.group(1).strip()
                # Validate extraction
                if not self._is_valid_extraction(pref_text):
                    continue
                # Filter out overly generic or technical terms
                if len(pref_text) > 5 and not any(tech in pref_text for tech in ['python', 'code', 'programming', 'develop']):
                    preference = {
                        'preference': pref_text,
                        'category': 'personal',
                        'conversation_id': conv_id
                    }
                    if preference not in preferences['personal']:
                        preferences['personal'].append(preference)

        # Dislikes
        dislike_patterns = [
            r'i\s+(?:don\'t|dont|do not)\s+(?:like|want|enjoy)\s+([a-z][a-z\s]+)',
            r'i\s+(?:hate|dislike)\s+([a-z][a-z\s]+)',
        ]

        for pattern in dislike_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                dislike_text = match.group(1).strip()
                # Validate extraction
                if not self._is_valid_extraction(dislike_text):
                    continue
                dislike = {
                    'dislike': dislike_text,
                    'conversation_id': conv_id
                }
                if dislike not in preferences['dislikes']:
                    preferences['dislikes'].append(dislike)

        # Remove empty categories
        return {k: v for k, v in preferences.items() if v}

    # Stop-phrases that indicate task instructions, not real goals
    _GOAL_STOP_PHRASES = {"do this", "do that", "try it", "make it", "test it"}

    def _extract_goals(self, text: str, conv_id: str) -> List[Dict[str, Any]]:
        """Extract user goals and intentions."""
        goals = []

        goal_patterns = [
            r'(?:my\s+goal\s+is|i\s+want\s+to|i(?:\s+am|\s*\'m)\s+trying\s+to|i\s+plan\s+to)\s+([^.!?]+)',
            r'i\s+(?:need|have)\s+to\s+([^.!?]+)',
            r'(?:the\s+purpose\s+is\s+to|this\s+is\s+for)\s+([^.!?]+)',
        ]

        for pattern in goal_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                goal_text = match.group(1).strip()
                # Quality gate: length > 15 chars
                if len(goal_text) <= 15:
                    continue
                # Quality gate: not a question
                if goal_text.endswith('?'):
                    continue
                # Quality gate: alphanumeric ratio > 0.5
                if len(goal_text) > 0:
                    alnum_count = sum(1 for c in goal_text if c.isalnum() or c.isspace())
                    if alnum_count / len(goal_text) <= 0.5:
                        continue
                # Quality gate: not a stop-phrase
                if goal_text.strip() in self._GOAL_STOP_PHRASES:
                    continue
                # Quality gate: passes _is_valid_extraction
                if not self._is_valid_extraction(goal_text):
                    continue

                # Determine priority based on language intensity
                priority = 'medium'
                if any(word in goal_text for word in ['must', 'critical', 'essential', 'important', 'need']):
                    priority = 'high'
                elif any(word in goal_text for word in ['would like', 'maybe', 'consider', 'might']):
                    priority = 'low'

                goal = {
                    'goal': goal_text,
                    'priority': priority,
                    'conversation_id': conv_id,
                    'status': 'active'
                }
                if goal not in goals:
                    goals.append(goal)

        # Cap at 30 entries (keep most recent, i.e. last 30)
        if len(goals) > 30:
            goals = goals[-30:]

        return goals

    def _extract_tech_env(self, text: str, conv_id: str) -> Dict[str, Any]:
        """Extract technical environment details."""
        tech_env = {}

        # Extract NAS details
        nas_patterns = [
            r'(?:NAS|Synology).*?(\d+\.\d+\.\d+\.\d+)',  # IP address
            r'(?:NAS|Synology).*?(\d+)\s*TB',  # Capacity
            r'(?:NAS.*?|hostname.*?)([A-Z][A-Z0-9-]+)',  # Hostname
        ]

        for pattern in nas_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if 'nas' not in tech_env:
                    tech_env['nas'] = {}

                value = match.group(1)
                if re.match(r'\d+\.\d+\.\d+\.\d+', value):
                    tech_env['nas']['ip'] = value
                elif value.isdigit():
                    tech_env['nas']['capacity'] = f"{value}TB"
                else:
                    tech_env['nas']['hostname'] = value

        # Extract OS
        os_keywords = {
            'windows': ['windows', 'win10', 'win11'],
            'macos': ['mac', 'macos', 'osx'],
            'linux': ['linux', 'ubuntu', 'debian', 'arch']
        }

        text_lower = text.lower()
        for os_name, keywords in os_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                tech_env['operating_system'] = os_name.title()
                break

        return tech_env if tech_env else None

    def _is_personal_conversation(self, text: str) -> bool:
        """
        Determine if a conversation is about the user personally
        (vs. purely technical discussion).
        """
        # Count personal pronouns
        personal_count = sum(
            len(re.findall(pattern, text.lower()))
            for pattern in self.personal_indicators
        )

        # Check for personal context keywords
        personal_keywords = [
            'my dog', 'my cat', 'my pet', 'my family', 'my wife', 'my husband',
            'my company', 'my business', 'my project', 'my goal', 'my client',
            'i live', 'i work', 'i prefer', 'i like', 'i want', 'i need'
        ]

        keyword_count = sum(1 for keyword in personal_keywords if keyword in text.lower())

        # If significant personal pronouns + personal keywords, it's a personal conversation
        return personal_count >= 3 and keyword_count >= 1


def update_user_profile(conversation_id: str, user_data: Dict[str, Any],
                        base_path: str = "") -> bool:
    if not base_path:
        from config import DATA_DIR
        base_path = str(DATA_DIR)
    """
    Update the aggregated user profile with new personal data.

    Args:
        conversation_id: Unique conversation identifier
        user_data: Extracted personal data from conversation
        base_path: Base path to AI_MEMORY on NAS

    Returns:
        True if update successful, False otherwise
    """
    profile_path = Path(base_path) / "user" / "profile.json"

    # Create directory if it doesn't exist
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing profile or create new one
    if profile_path.exists():
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile = json.load(f)
    else:
        # Initialize new profile
        profile = {
            'identity': {
                'name': None,
                'username': None,
                'aliases': [],
                'roles': [],
                'location': None,
                'contact': {},
                'first_mentioned': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            },
            'relationships': {
                'pets': [],
                'family': [],
                'colleagues': [],
                'friends': []
            },
            'projects': {
                'companies_owned': [],
                'clients': [],
                'active_projects': []
            },
            'preferences': {
                'technical': [],
                'personal': [],
                'dislikes': []
            },
            'goals': [],
            'technical_environment': {},
            'metadata': {
                'total_conversations_analyzed': 0,
                'last_full_scan': None,
                'profile_completeness': 0.0
            }
        }

    # Merge new data
    timestamp = datetime.now().isoformat()

    # Update identity
    if user_data.get('identity'):
        identity_data = user_data['identity']
        if 'roles' in identity_data:
            existing_roles = set(profile['identity'].get('roles', []))
            existing_roles.update(identity_data['roles'])
            profile['identity']['roles'] = list(existing_roles)

        if 'location' in identity_data:
            profile['identity']['location'] = identity_data['location']

        if 'email' in identity_data:
            profile['identity']['contact']['email'] = identity_data['email']

    # Update relationships (merge lists, avoid duplicates)
    if user_data.get('relationships'):
        for rel_type, rel_list in user_data['relationships'].items():
            if rel_type in profile['relationships']:
                for new_item in rel_list:
                    # Check if already exists
                    exists = False
                    for existing_item in profile['relationships'][rel_type]:
                        if _items_match(existing_item, new_item):
                            # Update conversation_ids
                            if 'conversation_ids' not in existing_item:
                                existing_item['conversation_ids'] = [existing_item.get('conversation_id', '')]
                            if conversation_id not in existing_item['conversation_ids']:
                                existing_item['conversation_ids'].append(conversation_id)
                            existing_item['last_mentioned'] = timestamp
                            exists = True
                            break

                    if not exists:
                        new_item['first_mentioned'] = timestamp
                        new_item['last_mentioned'] = timestamp
                        new_item['conversation_ids'] = [conversation_id]
                        profile['relationships'][rel_type].append(new_item)

    # Update projects
    if user_data.get('projects'):
        for proj_type, proj_list in user_data['projects'].items():
            if proj_type in profile['projects']:
                for new_item in proj_list:
                    exists = False
                    for existing_item in profile['projects'][proj_type]:
                        if _items_match(existing_item, new_item):
                            if 'conversation_ids' not in existing_item:
                                existing_item['conversation_ids'] = [existing_item.get('conversation_id', '')]
                            if conversation_id not in existing_item['conversation_ids']:
                                existing_item['conversation_ids'].append(conversation_id)
                            existing_item['last_mentioned'] = timestamp
                            exists = True
                            break

                    if not exists:
                        new_item['first_mentioned'] = timestamp
                        new_item['last_mentioned'] = timestamp
                        new_item['conversation_ids'] = [conversation_id]
                        profile['projects'][proj_type].append(new_item)

    # Update preferences
    if user_data.get('preferences'):
        for pref_type, pref_list in user_data['preferences'].items():
            if pref_type in profile['preferences']:
                for new_pref in pref_list:
                    exists = False
                    for existing_pref in profile['preferences'][pref_type]:
                        if _items_match(existing_pref, new_pref):
                            if 'conversation_ids' not in existing_pref:
                                existing_pref['conversation_ids'] = [existing_pref.get('conversation_id', '')]
                            if conversation_id not in existing_pref['conversation_ids']:
                                existing_pref['conversation_ids'].append(conversation_id)
                            exists = True
                            break

                    if not exists:
                        new_pref['conversation_ids'] = [conversation_id]
                        profile['preferences'][pref_type].append(new_pref)

    # Update goals
    if user_data.get('goals'):
        for new_goal in user_data['goals']:
            exists = False
            for existing_goal in profile['goals']:
                if _items_match(existing_goal, new_goal):
                    if 'conversation_ids' not in existing_goal:
                        existing_goal['conversation_ids'] = [existing_goal.get('conversation_id', '')]
                    if conversation_id not in existing_goal['conversation_ids']:
                        existing_goal['conversation_ids'].append(conversation_id)
                    exists = True
                    break

            if not exists:
                new_goal['conversation_ids'] = [conversation_id]
                new_goal['first_stated'] = timestamp
                profile['goals'].append(new_goal)

    # Update technical environment
    if user_data.get('technical_environment'):
        profile['technical_environment'].update(user_data['technical_environment'])

    # Update metadata
    profile['identity']['last_updated'] = timestamp
    profile['metadata']['total_conversations_analyzed'] += 1

    # Calculate profile completeness
    profile['metadata']['profile_completeness'] = _calculate_completeness(profile)

    # Save updated profile
    with open(profile_path, 'w', encoding='utf-8') as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    return True


def _items_match(item1: Dict, item2: Dict) -> bool:
    """Check if two items represent the same entity (case-insensitive)."""
    # Check by name first (case-insensitive)
    if 'name' in item1 and 'name' in item2:
        return item1['name'].lower().strip() == item2['name'].lower().strip()

    # Check by preference/goal/dislike text (case-insensitive)
    for key in ['preference', 'goal', 'dislike']:
        if key in item1 and key in item2:
            return item1[key].lower().strip() == item2[key].lower().strip()

    # Check by relationship type (case-insensitive)
    if 'relationship' in item1 and 'relationship' in item2:
        return item1['relationship'].lower().strip() == item2['relationship'].lower().strip()

    # Check by type (for pets without names, case-insensitive)
    if 'type' in item1 and 'type' in item2 and 'name' not in item1 and 'name' not in item2:
        return item1['type'].lower().strip() == item2['type'].lower().strip()

    return False


def _calculate_completeness(profile: Dict) -> float:
    """
    Calculate profile completeness score (0-1).

    Checks how many fields are populated vs. total possible fields.
    """
    total_fields = 0
    filled_fields = 0

    def _is_filled(value) -> bool:
        """Check if a value is meaningfully filled (not None, not empty list/dict)."""
        if value is None:
            return False
        if isinstance(value, (list, dict)):
            return len(value) > 0
        return bool(value)

    # Identity fields (6 possible)
    identity_fields = ['name', 'username', 'aliases', 'roles', 'location', 'contact']
    total_fields += len(identity_fields)
    for field in identity_fields:
        value = profile['identity'].get(field)
        if _is_filled(value):
            filled_fields += 1

    # Relationships (4 categories)
    total_fields += 4
    for category in ['pets', 'family', 'colleagues', 'friends']:
        if profile['relationships'].get(category):
            filled_fields += 1

    # Projects (3 categories)
    total_fields += 3
    for category in ['companies_owned', 'clients', 'active_projects']:
        if profile['projects'].get(category):
            filled_fields += 1

    # Preferences (3 categories)
    total_fields += 3
    for category in ['technical', 'personal', 'dislikes']:
        if profile['preferences'].get(category):
            filled_fields += 1

    # Goals (1 field)
    total_fields += 1
    if profile.get('goals'):
        filled_fields += 1

    # Technical environment (1 field)
    total_fields += 1
    if profile.get('technical_environment'):
        filled_fields += 1

    return round(filled_fields / total_fields, 2)


def sync_profile_from_quick_facts(base_path: str = "") -> Dict:
    if not base_path:
        from config import DATA_DIR
        base_path = str(DATA_DIR)
    """
    Sync identity data from quick_facts.json to profile.json.

    This ensures that known user data (name, username, etc.) from quick_facts
    is reflected in the profile for completeness scoring.

    Returns:
        Dict with sync status and updated fields
    """
    qf_path = Path(base_path) / "quick_facts.json"
    profile_path = Path(base_path) / "user" / "profile.json"

    if not qf_path.exists():
        return {"synced": False, "error": "quick_facts.json not found"}

    if not profile_path.exists():
        return {"synced": False, "error": "profile.json not found"}

    try:
        with open(qf_path, 'r', encoding='utf-8') as f:
            qf = json.load(f)

        with open(profile_path, 'r', encoding='utf-8') as f:
            profile = json.load(f)

        updated_fields = []

        # Sync from quick_facts.user
        user_data = qf.get('user', {})

        if user_data.get('name') and not profile['identity'].get('name'):
            profile['identity']['name'] = user_data['name']
            updated_fields.append('name')

        if user_data.get('github_username') and not profile['identity'].get('username'):
            profile['identity']['username'] = user_data['github_username']
            updated_fields.append('username')

        if user_data.get('nickname'):
            if 'aliases' not in profile['identity']:
                profile['identity']['aliases'] = []
            if user_data['nickname'] not in profile['identity']['aliases']:
                profile['identity']['aliases'].append(user_data['nickname'])
                updated_fields.append('aliases')

        # Sync emails
        emails = user_data.get('emails', [])
        if emails:
            if 'contact' not in profile['identity']:
                profile['identity']['contact'] = {}
            if not profile['identity']['contact'].get('emails'):
                profile['identity']['contact']['emails'] = emails
                updated_fields.append('emails')

        # Sync location
        if user_data.get('location') and not profile['identity'].get('location'):
            profile['identity']['location'] = user_data['location']
            updated_fields.append('location')

        # Sync pets from quick_facts
        pets = qf.get('user', {}).get('pets', [])
        if pets:
            if 'relationships' not in profile:
                profile['relationships'] = {}
            if 'pets' not in profile['relationships']:
                profile['relationships']['pets'] = []

            for pet in pets:
                # Check if pet already exists
                existing_names = [p.get('name', '').lower() for p in profile['relationships']['pets']]
                pet_name = pet.get('name', '') if isinstance(pet, dict) else pet
                if pet_name.lower() not in existing_names:
                    profile['relationships']['pets'].append(
                        pet if isinstance(pet, dict) else {'name': pet, 'type': 'dog'}
                    )
                    updated_fields.append(f'pet:{pet_name}')

        # Recalculate completeness
        profile['metadata']['profile_completeness'] = _calculate_completeness(profile)
        profile['identity']['last_updated'] = datetime.now().isoformat()

        # Save updated profile
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

        return {
            "synced": True,
            "profile_path": str(profile_path),
            "updated_fields": updated_fields,
            "new_completeness": profile['metadata']['profile_completeness']
        }

    except Exception as e:
        return {"synced": False, "error": str(e)}


if __name__ == "__main__":
    # Test the extractor
    extractor = PersonalProfileExtractor()

    test_messages = [
        {
            "role": "user",
            "content": "My dog Buddy loves playing fetch. He's a golden retriever and he's 3 years old."
        },
        {
            "role": "assistant",
            "content": "That's wonderful! Golden retrievers are great dogs."
        },
        {
            "role": "user",
            "content": "Yeah, I'm the CEO of my company and we're working on automated lead generation."
        }
    ]

    result = extractor.extract_all(test_messages, "test-123")
    print(json.dumps(result, indent=2))
