"""
Privacy Filter - Phase 5: Privacy Layer
Handles redaction, sensitive tagging, and optional encryption.

Author: Claude
Created: 2026-01-18
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from secret_detector import DetectedSecret, SecretDetector, SecretType


class SensitivityLevel(Enum):
    """Sensitivity levels for data classification."""
    PUBLIC = "public"           # No restrictions
    INTERNAL = "internal"       # Internal use only
    CONFIDENTIAL = "confidential"  # Contains PII or sensitive business data
    SECRET = "secret"           # Contains credentials or secrets
    TOP_SECRET = "top_secret"   # Highly sensitive, encryption required


@dataclass
class RedactionResult:
    """Result of a redaction operation."""
    original_text: str
    redacted_text: str
    secrets_found: List[DetectedSecret]
    redaction_count: int
    sensitivity_level: SensitivityLevel
    hashes: Dict[str, str] = field(default_factory=dict)  # placeholder -> hash of original

    def to_dict(self) -> dict:
        return {
            "redacted_text": self.redacted_text,
            "secrets_count": self.redaction_count,
            "sensitivity_level": self.sensitivity_level.value,
            "secret_types": [s.secret_type.value for s in self.secrets_found],
            "hashes": self.hashes
        }


@dataclass
class SensitiveFactTag:
    """Tag indicating a fact contains sensitive information."""
    is_sensitive: bool
    sensitivity_level: SensitivityLevel
    reasons: List[str]
    secret_types: List[SecretType]
    redacted: bool
    encrypted: bool = False

    def to_dict(self) -> dict:
        return {
            "sensitive": self.is_sensitive,
            "level": self.sensitivity_level.value,
            "reasons": self.reasons,
            "types": [t.value for t in self.secret_types],
            "redacted": self.redacted,
            "encrypted": self.encrypted
        }


# Patterns for sensitive but non-secret data (PII, paths, IPs)
SENSITIVE_PATTERNS = [
    # IP Addresses (internal networks especially)
    (r"\b(?:192\.168|10\.|172\.(?:1[6-9]|2[0-9]|3[01]))\.\d{1,3}\.\d{1,3}\b", "internal_ip"),
    (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "ip_address"),

    # File paths (Windows and Unix)
    (r"[A-Z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*", "windows_path"),
    (r"/(?:home|Users|var|etc|opt)/[^\s:*?\"<>|]+", "unix_path"),

    # Email addresses
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),

    # Phone numbers
    (r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "phone"),

    # Social Security Numbers
    (r"\b\d{3}-\d{2}-\d{4}\b", "ssn"),

    # Credit card numbers
    (r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b", "credit_card"),

    # UNC paths
    (r"\\\\[A-Za-z0-9_.$-]+\\[A-Za-z0-9_.$\\-]+", "unc_path"),

    # Hostnames
    (r"\b[a-zA-Z][a-zA-Z0-9-]*\.(?:local|internal|corp|lan)\b", "internal_hostname"),
]


class PrivacyFilter:
    """
    Filters sensitive information from text and facts.

    Features:
    - Secret detection and redaction
    - Sensitive data tagging
    - Hash-based deduplication of redacted content
    - Optional encryption (placeholder for now)

    Usage:
        filter = PrivacyFilter()
        result = filter.redact("My API key is sk-abc123...")
        print(result.redacted_text)  # "My API key is [REDACTED:api_key]"
    """

    def __init__(self,
                 redact_secrets: bool = True,
                 min_confidence: float = 0.70,
                 tag_sensitive: bool = True,
                 custom_patterns: Optional[List[Tuple[str, str]]] = None):
        """
        Initialize the privacy filter.

        Args:
            redact_secrets: Whether to replace secrets with placeholders
            min_confidence: Minimum confidence for secret detection
            tag_sensitive: Whether to tag facts with sensitivity info
            custom_patterns: Additional (pattern, name) tuples for sensitive data
        """
        self.redact_secrets = redact_secrets
        self.min_confidence = min_confidence
        self.tag_sensitive = tag_sensitive
        self.secret_detector = SecretDetector()

        # Compile sensitive patterns
        self.sensitive_patterns: List[Tuple[re.Pattern, str]] = []
        for pattern, name in SENSITIVE_PATTERNS:
            try:
                self.sensitive_patterns.append((re.compile(pattern), name))
            except re.error:
                pass

        if custom_patterns:
            for pattern, name in custom_patterns:
                try:
                    self.sensitive_patterns.append((re.compile(pattern), name))
                except re.error:
                    pass

        # Redaction log for audit
        self.redaction_log: List[Dict] = []

    def redact(self, text: str) -> RedactionResult:
        """
        Redact secrets from text, replacing with placeholders.

        Args:
            text: Text to redact

        Returns:
            RedactionResult with redacted text and metadata
        """
        if not text:
            return RedactionResult(
                original_text="",
                redacted_text="",
                secrets_found=[],
                redaction_count=0,
                sensitivity_level=SensitivityLevel.PUBLIC
            )

        # Detect secrets
        secrets = self.secret_detector.scan(text, self.min_confidence)

        if not secrets:
            # Check for sensitive (non-secret) data
            sensitivity = self._classify_sensitivity(text, [])
            return RedactionResult(
                original_text=text,
                redacted_text=text,
                secrets_found=[],
                redaction_count=0,
                sensitivity_level=sensitivity
            )

        # Sort secrets by position (reverse order for replacement)
        secrets_sorted = sorted(secrets, key=lambda x: x.start, reverse=True)

        redacted_text = text
        hashes: Dict[str, str] = {}

        for secret in secrets_sorted:
            # Create placeholder
            placeholder = f"[REDACTED:{secret.secret_type.value}]"

            # Hash the original for deduplication
            secret_hash = hashlib.sha256(secret.value.encode()).hexdigest()[:16]
            hashes[placeholder] = secret_hash

            # Replace in text
            redacted_text = redacted_text[:secret.start] + placeholder + redacted_text[secret.end:]

            # Log the redaction
            self.redaction_log.append({
                "timestamp": datetime.now().isoformat(),
                "type": secret.secret_type.value,
                "hash": secret_hash,
                "confidence": secret.confidence,
                "pattern": secret.pattern_name
            })

        # Classify overall sensitivity
        sensitivity = self._classify_sensitivity(text, secrets)

        return RedactionResult(
            original_text=text,
            redacted_text=redacted_text,
            secrets_found=secrets,
            redaction_count=len(secrets),
            sensitivity_level=sensitivity,
            hashes=hashes
        )

    def _classify_sensitivity(self, text: str, secrets: List[DetectedSecret]) -> SensitivityLevel:
        """Classify the sensitivity level of text."""

        # If we found high-confidence secrets, it's SECRET level
        high_confidence_secrets = [s for s in secrets if s.confidence >= 0.90]
        if high_confidence_secrets:
            # Private keys and connection strings are TOP_SECRET
            top_secret_types = {SecretType.PRIVATE_KEY, SecretType.SSH_KEY, SecretType.PGP_KEY}
            if any(s.secret_type in top_secret_types for s in high_confidence_secrets):
                return SensitivityLevel.TOP_SECRET
            return SensitivityLevel.SECRET

        # Lower confidence secrets
        if secrets:
            return SensitivityLevel.CONFIDENTIAL

        # Check for sensitive patterns (PII, IPs, paths)
        has_sensitive = False
        for pattern, name in self.sensitive_patterns:
            if pattern.search(text):
                has_sensitive = True
                # SSN and credit cards are confidential
                if name in ("ssn", "credit_card"):
                    return SensitivityLevel.CONFIDENTIAL
                break

        if has_sensitive:
            return SensitivityLevel.INTERNAL

        return SensitivityLevel.PUBLIC

    def tag_fact(self, fact_content: str, fact_type: str = "unknown") -> SensitiveFactTag:
        """
        Analyze a fact and return sensitivity tags.

        Args:
            fact_content: The fact text
            fact_type: Type of fact (for context)

        Returns:
            SensitiveFactTag with classification
        """
        reasons: List[str] = []
        secret_types: List[SecretType] = []

        # Detect secrets
        secrets = self.secret_detector.scan(fact_content, self.min_confidence)
        if secrets:
            secret_types = list(set(s.secret_type for s in secrets))
            reasons.append(f"Contains {len(secrets)} detected secret(s)")

        # Check sensitive patterns
        sensitive_matches: List[str] = []
        for pattern, name in self.sensitive_patterns:
            if pattern.search(fact_content):
                sensitive_matches.append(name)

        if sensitive_matches:
            reasons.append(f"Contains sensitive data: {', '.join(sensitive_matches)}")

        # Determine sensitivity level
        sensitivity = self._classify_sensitivity(fact_content, secrets)

        is_sensitive = sensitivity != SensitivityLevel.PUBLIC

        return SensitiveFactTag(
            is_sensitive=is_sensitive,
            sensitivity_level=sensitivity,
            reasons=reasons,
            secret_types=secret_types,
            redacted=False,
            encrypted=False
        )

    def process_fact(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a fact through the privacy filter.

        Args:
            fact: Fact dictionary with at least 'content' key

        Returns:
            Fact dictionary with privacy tags and optional redaction
        """
        content = fact.get("content", "")
        if not content:
            return fact

        # Tag the fact
        tag = self.tag_fact(content, fact.get("type", "unknown"))

        # Apply redaction if configured and secrets found
        if self.redact_secrets and tag.secret_types:
            result = self.redact(content)
            fact["content"] = result.redacted_text
            tag.redacted = True

        # Add privacy metadata
        fact["privacy"] = tag.to_dict()

        return fact

    def process_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an entire conversation through the privacy filter.

        Args:
            conversation: Conversation dict with 'messages' and optional 'extracted_data'

        Returns:
            Conversation with redacted content and privacy tags
        """
        result = conversation.copy()
        overall_sensitivity = SensitivityLevel.PUBLIC
        total_secrets = 0

        # Process messages
        messages = result.get("messages", [])
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if content:
                redaction = self.redact(content)
                messages[i]["content"] = redaction.redacted_text
                total_secrets += redaction.redaction_count
                if redaction.sensitivity_level.value > overall_sensitivity.value:
                    overall_sensitivity = redaction.sensitivity_level

        # Process extracted facts
        extracted = result.get("extracted_data", {})
        facts = extracted.get("facts", [])
        for i, fact in enumerate(facts):
            facts[i] = self.process_fact(fact)
            fact_privacy = facts[i].get("privacy", {})
            if fact_privacy.get("level") == "secret":
                overall_sensitivity = SensitivityLevel.SECRET

        # Add conversation-level privacy metadata
        result["privacy"] = {
            "sensitivity_level": overall_sensitivity.value,
            "secrets_redacted": total_secrets,
            "processed_at": datetime.now().isoformat()
        }

        return result

    def scan_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Scan a file for secrets and sensitive data.

        Args:
            file_path: Path to file to scan

        Returns:
            Scan results including secrets found and recommendations
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {"error": str(e), "file": str(file_path)}

        secrets = self.secret_detector.scan(content, self.min_confidence)
        tag = self.tag_fact(content)

        return {
            "file": str(file_path),
            "size_bytes": len(content),
            "secrets_found": len(secrets),
            "secret_types": [s.secret_type.value for s in secrets],
            "sensitivity_level": tag.sensitivity_level.value,
            "recommendations": self._get_recommendations(secrets, tag)
        }

    def _get_recommendations(self, secrets: List[DetectedSecret],
                             tag: SensitiveFactTag) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations: List[str] = []

        if secrets:
            recommendations.append("Remove or rotate exposed secrets before storing")

            secret_types = set(s.secret_type for s in secrets)
            if SecretType.PRIVATE_KEY in secret_types or SecretType.SSH_KEY in secret_types:
                recommendations.append("CRITICAL: Private keys should never be stored in memory system")

            if SecretType.PASSWORD in secret_types:
                recommendations.append("Consider using environment variables or secret managers")

        if tag.sensitivity_level in (SensitivityLevel.SECRET, SensitivityLevel.TOP_SECRET):
            recommendations.append("Enable encryption for this content")

        if tag.sensitivity_level == SensitivityLevel.CONFIDENTIAL:
            recommendations.append("Review if this content needs to be stored")

        return recommendations

    def get_redaction_stats(self) -> Dict[str, Any]:
        """Get statistics about redactions performed."""
        if not self.redaction_log:
            return {"total_redactions": 0}

        type_counts: Dict[str, int] = {}
        for entry in self.redaction_log:
            t = entry["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_redactions": len(self.redaction_log),
            "by_type": type_counts,
            "unique_secrets": len(set(e["hash"] for e in self.redaction_log))
        }

    def clear_log(self):
        """Clear the redaction log."""
        self.redaction_log = []


# Convenience functions
def redact_text(text: str, min_confidence: float = 0.70) -> str:
    """Convenience function to redact secrets from text."""
    filter = PrivacyFilter(min_confidence=min_confidence)
    result = filter.redact(text)
    return result.redacted_text


def is_sensitive(text: str, min_confidence: float = 0.70) -> bool:
    """Check if text contains sensitive information."""
    filter = PrivacyFilter(min_confidence=min_confidence)
    tag = filter.tag_fact(text)
    return tag.is_sensitive


def classify_sensitivity(text: str) -> str:
    """Get the sensitivity level of text."""
    filter = PrivacyFilter()
    tag = filter.tag_fact(text)
    return tag.sensitivity_level.value


if __name__ == "__main__":
    # Test the filter
    filter = PrivacyFilter()

    test_cases = [
        "My API key is sk-abc123def456ghi789jkl012mno345pqr",
        "Connect to postgres://admin:supersecret@10.0.0.100:5432/mydb",
        "Email me at john.doe@example.com or call 555-123-4567",
        "The server is at C:\\Users\\admin\\secret\\config.json",
        "Just some normal text with no secrets",
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    ]

    print("Privacy Filter Tests\n" + "=" * 50)

    for text in test_cases:
        print(f"\nOriginal: {text[:60]}...")
        result = filter.redact(text)
        print(f"Redacted: {result.redacted_text[:60]}...")
        print(f"Sensitivity: {result.sensitivity_level.value}")
        print(f"Secrets found: {result.redaction_count}")

    print("\n\nRedaction Stats:")
    print(json.dumps(filter.get_redaction_stats(), indent=2))
