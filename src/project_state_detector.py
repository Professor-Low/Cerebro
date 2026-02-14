"""
Project State Detection Engine
Identifies project mentions and state changes in conversations.

MULTI-AGENT NOTICE: This is Agent 3's exclusive domain.
Agent 4 (memory backend) and Agent 6 (MCP tools) will use this.
"""

import re
from typing import Dict, List


class ProjectStateDetector:
    """Detects project mentions and state changes."""

    # Known project patterns
    KNOWN_PROJECTS = {
        "cerebral-interface": {
            "names": ["cerebral", "brain", "visualization", "nas-cerebral", "cerebral interface"],
            "paths": ["NAS-cerebral-interface", "visualization", "cerebral-interface"],
            "technologies": ["Python", "WebSocket", "Three.js", "FAISS", "sentence-transformers"]
        },
        "lead-enrichment": {
            "names": ["lead enrichment", "scraper", "linkedin"],
            "paths": ["Scraper", "lead-enrichment"],
            "technologies": ["Playwright", "PostgreSQL", "Docker"]
        }
    }

    # State change patterns
    STATE_PATTERNS = {
        "started": r"(?:start|begin|initiat)(?:ed|ing)\s+(?:work on|working on)?\s*(.+)",
        "finished": r"(?:finish|complete|done with)(?:ed)?\s+(.+)",
        "next_step": r"(?:next|then)\s+(?:we need to|should|will)\s+(.+)",
        "blocker": r"(?:blocked by|can't|unable to|issue with)\s+(.+)",
        "in_progress": r"(?:working on|currently|now doing)\s+(.+)",
    }

    def __init__(self):
        self.state_regexes = {
            k: re.compile(v, re.IGNORECASE)
            for k, v in self.STATE_PATTERNS.items()
        }

    def detect_project_mention(self, text: str) -> List[Dict]:
        """
        Detect project mentions in text.

        Returns: List of {project_id, confidence, match_type}
        """
        text_lower = text.lower()
        mentions = []

        for project_id, project_info in self.KNOWN_PROJECTS.items():
            confidence = 0
            match_types = []

            # Check name matches
            for name in project_info["names"]:
                if name in text_lower:
                    confidence += 0.4
                    match_types.append(f"name:{name}")

            # Check path matches
            for path in project_info["paths"]:
                if path.lower() in text_lower:
                    confidence += 0.5
                    match_types.append(f"path:{path}")

            # Check technology matches
            tech_matches = sum(1 for tech in project_info["technologies"] if tech.lower() in text_lower)
            if tech_matches >= 2:
                confidence += 0.3 * tech_matches
                match_types.append(f"tech:{tech_matches}")

            if confidence > 0.3:  # Threshold
                mentions.append({
                    "project_id": project_id,
                    "confidence": min(1.0, confidence),
                    "match_types": match_types
                })

        return sorted(mentions, key=lambda x: x["confidence"], reverse=True)

    def detect_state_changes(self, text: str) -> Dict[str, List[str]]:
        """
        Detect state changes in text.

        Returns: {
            "started": [...],
            "finished": [...],
            "next_step": [...],
            "blocker": [...],
            "in_progress": [...]
        }
        """
        changes = {key: [] for key in self.STATE_PATTERNS.keys()}

        for state_type, regex in self.state_regexes.items():
            matches = regex.findall(text)
            if matches:
                changes[state_type].extend(matches)

        return {k: v for k, v in changes.items() if v}

    def detect_file_activity(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """
        Map file paths to projects.

        Returns: {project_id: [file_paths]}
        """
        project_files = {}

        for file_path in file_paths:
            path_lower = file_path.lower()

            for project_id, project_info in self.KNOWN_PROJECTS.items():
                for project_path in project_info["paths"]:
                    if project_path.lower() in path_lower:
                        if project_id not in project_files:
                            project_files[project_id] = []
                        project_files[project_id].append(file_path)
                        break

        return project_files

    def infer_priority(self, text: str) -> str:
        """Infer project priority from language."""
        text_lower = text.lower()

        if any(word in text_lower for word in ["urgent", "critical", "asap", "immediately"]):
            return "critical"
        elif any(word in text_lower for word in ["important", "high priority", "soon"]):
            return "high"
        elif any(word in text_lower for word in ["low priority", "later", "eventually"]):
            return "low"
        else:
            return "medium"
