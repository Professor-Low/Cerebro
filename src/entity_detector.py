"""
Entity Detector - Extract entities from user prompts
Fast, lightweight detection for real-time use in hooks

AGENT 13: SMART AUTO-CONTEXT INJECTION
Design goals:
- FAST: Must run in <100ms for hook usage
- Lightweight: No heavy NLP models
- High recall: Better to over-detect than miss
"""
import json
import re
from typing import Dict, List


class EntityDetector:
    """
    Detect entities in user prompts for auto-context triggering.

    This is the intelligence layer that determines WHEN to inject context
    based on what entities are mentioned in the user's prompt.
    """

    def __init__(self):
        # Load known entities from memory system
        self.known_tools = set()
        self.known_projects = set()
        self.known_networks = set()
        self.known_technologies = set()
        self.known_file_patterns = []

        # Common technical keywords that warrant context
        self.technical_keywords = {
            'nas', 'mcp', 'server', 'docker', 'python', 'hook',
            'config', 'api', 'database', 'error', 'bug', 'fix',
            'memory', 'conversation', 'embedding', 'search',
            'agent', 'integration', 'system', 'file', 'script',
            'install', 'setup', 'configure', 'update', 'modify',
            'cerebral', 'interface', 'brain', 'visualization'
        }

        # Load from entity database (created by memory system)
        self._load_known_entities()

    def _load_known_entities(self):
        """Load known entities from data directory."""
        try:
            from config import DATA_DIR
            entities_path = DATA_DIR / "entities"

            if not entities_path.exists():
                print(f"[EntityDetector] Entities path not found: {entities_path}")
                return

            # Load tools
            tools_file = entities_path / "tools.json"
            if tools_file.exists():
                with open(tools_file, 'r') as f:
                    data = json.load(f)
                    self.known_tools = set(data.keys())

            # Load technologies
            tech_file = entities_path / "technologies.json"
            if tech_file.exists():
                with open(tech_file, 'r') as f:
                    data = json.load(f)
                    self.known_technologies = set(data.keys())

            # Load networks
            networks_file = entities_path / "networks.json"
            if networks_file.exists():
                with open(networks_file, 'r') as f:
                    data = json.load(f)
                    self.known_networks = set(data.keys())

            print(f"[EntityDetector] Loaded {len(self.known_tools)} tools, "
                  f"{len(self.known_technologies)} technologies, "
                  f"{len(self.known_networks)} networks")

        except Exception as e:
            print(f"[EntityDetector] Warning: Could not load entities: {e}")

    def detect(self, prompt: str) -> Dict[str, List[str]]:
        """
        Detect entities in prompt.

        Args:
            prompt: User's message text

        Returns:
            {
                'tools': ['NAS', 'MCP'],
                'projects': ['cerebral-interface'],
                'file_paths': ['/data/config.json'],
                'networks': ['10.0.0.100'],
                'keywords': ['config', 'error'],
                'confidence': 'high'  # high if specific entities, low if just keywords
            }
        """
        entities = {
            'tools': [],
            'projects': [],
            'file_paths': [],
            'networks': [],
            'technologies': [],
            'keywords': [],
            'confidence': 'low'
        }

        prompt_lower = prompt.lower()

        # Detect known tools (case-insensitive)
        for tool in self.known_tools:
            if tool.lower() in prompt_lower:
                entities['tools'].append(tool)

        # Detect known technologies
        for tech in self.known_technologies:
            if tech.lower() in prompt_lower:
                entities['technologies'].append(tech)

        # Detect file paths (Windows and Unix)
        # Windows: C:\path\to\file
        win_paths = re.findall(r'[A-Z]:\\(?:[^\s\\/:*?"<>|]+\\)*[^\s\\/:*?"<>|]*', prompt)
        entities['file_paths'].extend(win_paths)

        # Unix: /path/to/file
        unix_paths = re.findall(r'/(?:[^\s/:*?"<>|]+/)*[^\s/:*?"<>|]+', prompt)
        entities['file_paths'].extend(unix_paths)

        # Detect IP addresses / networks
        ips = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', prompt)
        entities['networks'].extend(ips)

        # Detect known networks (hostnames, etc.)
        for network in self.known_networks:
            if network.lower() in prompt_lower:
                entities['networks'].append(network)

        # Detect technical keywords
        for keyword in self.technical_keywords:
            if keyword in prompt_lower:
                entities['keywords'].append(keyword)

        # Detect project references (common patterns)
        project_patterns = [
            r'cerebral[-\s]interface',
            r'brain[-\s]visualization',
            r'nas[-\s]cerebral',
            r'memory[-\s]system',
            r'agent[-\s]\d+',
            r'v\d+[-_]agents'
        ]

        for pattern in project_patterns:
            matches = re.findall(pattern, prompt_lower)
            for match in matches:
                entities['projects'].append(match)

        # Determine confidence
        entity_count = (
            len(entities['tools']) +
            len(entities['projects']) +
            len(entities['file_paths']) +
            len(entities['networks']) +
            len(entities['technologies'])
        )

        if entity_count >= 2:
            entities['confidence'] = 'high'
        elif entity_count >= 1:
            entities['confidence'] = 'medium'
        elif len(entities['keywords']) >= 3:
            entities['confidence'] = 'medium'

        # Remove duplicate networks
        entities['networks'] = list(set(entities['networks']))

        return entities

    def should_inject_context(self, entities: Dict) -> bool:
        """
        Decide if we should inject context based on detected entities.

        Returns: True if we found specific entities worth searching for
        """
        # Inject if we found any specific entities
        has_entities = (
            len(entities['tools']) > 0 or
            len(entities['projects']) > 0 or
            len(entities['file_paths']) > 0 or
            len(entities['networks']) > 0 or
            len(entities['technologies']) > 0
        )

        # Or if we found 3+ technical keywords (likely technical discussion)
        many_keywords = len(entities['keywords']) >= 3

        return has_entities or many_keywords


# Example usage / testing
if __name__ == "__main__":
    detector = EntityDetector()

    # Test cases
    test_prompts = [
        "Update the config at /path/to/data/config.json",
        "What's my NAS IP address?",
        "How do I configure the MCP server with Docker?",
        "The cerebral-interface project has a bug in the Python code",
        "Hello, how are you?",  # Should not trigger
        "Can you help me with my config file setup and database connection?"
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        entities = detector.detect(prompt)
        should_inject = detector.should_inject_context(entities)
        print(f"Should inject: {should_inject}")
        print(f"Confidence: {entities['confidence']}")
        print(f"Entities found: {sum(len(v) if isinstance(v, list) else 0 for k, v in entities.items() if k != 'confidence')}")
        if should_inject:
            print(f"  Tools: {entities['tools']}")
            print(f"  File paths: {entities['file_paths']}")
            print(f"  Networks: {entities['networks']}")
            print(f"  Technologies: {entities['technologies']}")
            print(f"  Keywords: {entities['keywords'][:5]}")  # First 5
