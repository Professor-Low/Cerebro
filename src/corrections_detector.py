"""
Correction Detection Engine
Identifies when user corrects Claude's mistakes.

MULTI-AGENT NOTICE: This is Agent 2's exclusive domain.
Part of the NAS Cerebral Interface - Learning from Corrections System
"""

import re
from datetime import datetime
from typing import Dict, List, Optional


class CorrectionDetector:
    """Detects correction patterns in conversation messages."""

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Fix common encoding issues from speech-to-text."""
        if not text:
            return text

        # Common UTF-8 corruption patterns (as byte sequences decoded as latin-1)
        replacements = [
            # Smart quotes
            ("\xe2\x80\x99", "'"),   # Right single quote
            ("\xe2\x80\x98", "'"),   # Left single quote
            ("\xe2\x80\x9c", '"'),   # Left double quote
            ("\xe2\x80\x9d", '"'),   # Right double quote
            # Dashes
            ("\xe2\x80\x93", "-"),   # En dash
            ("\xe2\x80\x94", "--"),  # Em dash
            # Ellipsis
            ("\xe2\x80\xa6", "..."),
            # Common garbled patterns
            ("\xc3\xa2\xe2\x82\xac\xe2\x84\xa2", "'"),  # Garbled apostrophe
        ]

        for bad, good in replacements:
            if bad in text:
                text = text.replace(bad, good)

        # Also clean up any remaining high-byte characters
        import re
        # Remove strings that look like encoding artifacts
        text = re.sub(r'[\xc0-\xff]{2,4}', '', text)

        return text

    # Correction trigger phrases
    CORRECTION_PATTERNS = [
        r"^(?:no|nope),?\s+the\s+\w+\s+\w+\s+is\s+(.+)",  # "No, the NAS IP is..."
        r"^(?:no|nope),?\s+it'?s\s+(.+)",  # "No, it's..."
        r"actually,?\s+(?:it'?s|the\s+\w+\s+is)\s+(.+)",
        r"(?:not|isn'?t)\s+([A-Za-z0-9_./-]{3,50}),?\s+(?:it'?s|but)\s+([A-Za-z0-9_./-]{3,50})",
        r"the\s+correct\s+(?:answer|value)\s+is\s+(.+)",
        r"you'?re?\s+(?:wrong|mistaken|incorrect)",
        r"fix:?\s+(.+)",
        r"correction:?\s+(.+)",
        r"should\s+be\s+([A-Za-z0-9_./-]{3,50}),?\s+not\s+([A-Za-z0-9_./-]{3,50})",
    ]

    # Importance keywords
    HIGH_IMPORTANCE = ["critical", "important", "always", "never", "must"]
    MEDIUM_IMPORTANCE = ["should", "better", "prefer"]

    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.CORRECTION_PATTERNS]

    def detect_correction(self, user_message: str, assistant_message: str = None) -> Optional[Dict]:
        """
        Detect if user message contains a correction.

        Args:
            user_message: The user's message
            assistant_message: Previous assistant message (for context)

        Returns:
            Correction dict or None
        """
        # Sanitize inputs first to fix encoding issues
        user_message = self._sanitize_text(user_message)
        assistant_message = self._sanitize_text(assistant_message) if assistant_message else None

        # Check for correction patterns
        for pattern in self.patterns:
            match = pattern.search(user_message)
            if match:
                # Extract corrected value
                corrected = match.group(1) if match.groups() else None

                # Try to extract what was wrong from assistant message
                mistake = self._extract_mistake(assistant_message, user_message) if assistant_message else None

                # Determine importance
                importance = self._determine_importance(user_message)

                # Detect topic
                topic = self._detect_topic(user_message, assistant_message)

                return {
                    "detected": True,
                    "correction_text": corrected,
                    "mistake_text": mistake,
                    "importance": importance,
                    "topic": topic,
                    "full_message": user_message,
                    "detected_at": datetime.now().isoformat()
                }

        return None

    def _extract_mistake(self, assistant_msg: str, user_msg: str) -> Optional[str]:
        """Extract what Claude said that was wrong - with better heuristics."""
        if not assistant_msg:
            return None

        user_msg.lower()

        # 1. Look for explicit quoted text in user message
        quotes = re.findall(r'"([^"]+)"', user_msg)
        if quotes:
            # Verify quote appears in assistant message
            for q in quotes:
                if q.lower() in assistant_msg.lower():
                    return q

        # 2. Look for "not X" pattern where X appears in assistant message
        not_patterns = [
            r"not\s+([A-Za-z0-9_./:@-]{3,50})",
            r"isn't\s+([A-Za-z0-9_./:@-]{3,50})",
            r"wasn't\s+([A-Za-z0-9_./:@-]{3,50})",
        ]
        for pattern in not_patterns:
            match = re.search(pattern, user_msg, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if len(candidate) < 3:
                    continue
                if candidate.lower() in assistant_msg.lower():
                    return candidate

        # 3. Look for specific value patterns (IPs, ports, paths)
        # If user provides a correction value, look for similar but different value in assistant
        ip_user = re.findall(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', user_msg)
        ip_assistant = re.findall(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', assistant_msg)
        if ip_user and ip_assistant:
            for ip in ip_assistant:
                if ip not in ip_user:
                    return ip  # Found the wrong IP

        # 4. Look for port numbers
        port_user = re.findall(r'\bport\s*(\d+)\b', user_msg, re.IGNORECASE)
        port_assistant = re.findall(r'\bport\s*(\d+)\b', assistant_msg, re.IGNORECASE)
        if port_user and port_assistant:
            for port in port_assistant:
                if port not in port_user:
                    return f"port {port}"

        # 5. DO NOT fall back to last sentence - return None instead
        # The old behavior of returning last sentence caused garbage corrections
        return None

    def _determine_importance(self, message: str) -> str:
        """Determine correction importance from language."""
        message_lower = message.lower()

        if any(word in message_lower for word in self.HIGH_IMPORTANCE):
            return "high"
        elif any(word in message_lower for word in self.MEDIUM_IMPORTANCE):
            return "medium"
        else:
            return "low"

    def _detect_topic(self, user_msg: str, assistant_msg: str = None) -> str:
        """Detect the topic of the correction."""
        combined = user_msg + " " + (assistant_msg or "")

        # Topic keywords
        topics = {
            "network": ["ip", "network", "nas", "192.168", "localhost", "port"],
            "file_system": ["path", "directory", "file", "folder", "drive", "z:\\", "c:\\"],
            "configuration": ["config", "setting", "setup", "configure"],
            "database": ["database", "postgres", "sql", "docker", "container"],
            "code": ["function", "class", "method", "variable", "code"],
            "api": ["api", "endpoint", "request", "response"],
            "project": ["project", "repository", "repo", "git"],
        }

        for topic, keywords in topics.items():
            if any(kw in combined.lower() for kw in keywords):
                return topic

        return "general"

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities mentioned in correction."""
        entities = {
            "ips": re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text),
            "paths": re.findall(r'[A-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*', text),
            "ports": re.findall(r'\bport\s+(\d+)\b', text, re.IGNORECASE),
            "urls": re.findall(r'https?://[^\s]+', text),
        }
        return {k: v for k, v in entities.items() if v}
