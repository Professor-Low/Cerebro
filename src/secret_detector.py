"""
Secret Detector - Phase 5: Privacy Layer
Detects secrets, API keys, tokens, and sensitive data in text.

Author: Claude
Created: 2026-01-18
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SecretType(Enum):
    """Types of secrets that can be detected."""
    API_KEY = "api_key"
    TOKEN = "token"
    PASSWORD = "password"
    PRIVATE_KEY = "private_key"
    CONNECTION_STRING = "connection_string"
    CREDENTIAL = "credential"
    SECRET_KEY = "secret_key"
    AWS_KEY = "aws_key"
    GITHUB_TOKEN = "github_token"
    JWT = "jwt"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    SSH_KEY = "ssh_key"
    PGP_KEY = "pgp_key"
    ENCRYPTION_KEY = "encryption_key"
    UNKNOWN = "unknown"


@dataclass
class DetectedSecret:
    """Represents a detected secret in text."""
    secret_type: SecretType
    value: str
    start: int
    end: int
    confidence: float  # 0.0 to 1.0
    pattern_name: str
    context: str = ""  # Surrounding text for verification

    def to_dict(self) -> dict:
        return {
            "type": self.secret_type.value,
            "value": self.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "pattern": self.pattern_name,
            "context": self.context
        }


# Secret detection patterns
# Format: (name, pattern, secret_type, confidence)
SECRET_PATTERNS: List[Tuple[str, str, SecretType, float]] = [
    # OpenAI / Anthropic API Keys
    ("openai_api_key", r"sk-[a-zA-Z0-9]{20,}", SecretType.API_KEY, 0.95),
    ("anthropic_api_key", r"sk-ant-[a-zA-Z0-9\-]{20,}", SecretType.API_KEY, 0.95),

    # GitHub Tokens (allow 30-50 chars for flexibility)
    ("github_pat", r"ghp_[a-zA-Z0-9]{30,50}", SecretType.GITHUB_TOKEN, 0.95),
    ("github_oauth", r"gho_[a-zA-Z0-9]{30,50}", SecretType.GITHUB_TOKEN, 0.95),
    ("github_app", r"ghu_[a-zA-Z0-9]{30,50}", SecretType.GITHUB_TOKEN, 0.95),
    ("github_refresh", r"ghr_[a-zA-Z0-9]{30,50}", SecretType.GITHUB_TOKEN, 0.95),
    ("github_fine_grained", r"github_pat_[a-zA-Z0-9]{20,}_[a-zA-Z0-9]{50,}", SecretType.GITHUB_TOKEN, 0.95),

    # AWS Keys
    ("aws_access_key", r"AKIA[0-9A-Z]{16}", SecretType.AWS_KEY, 0.95),
    ("aws_secret_key", r"(?<![a-zA-Z0-9/+])[a-zA-Z0-9/+]{40}(?![a-zA-Z0-9/+])", SecretType.AWS_KEY, 0.60),

    # Google Cloud
    ("google_api_key", r"AIza[0-9A-Za-z\-_]{35}", SecretType.API_KEY, 0.90),
    ("google_oauth", r"[0-9]+-[0-9A-Za-z_]{32}\.apps\.googleusercontent\.com", SecretType.CREDENTIAL, 0.90),

    # Generic API Keys
    ("generic_api_key", r"(?i)api[_-]?key['\"]?\s*[:=]\s*['\"]?([a-zA-Z0-9\-_]{20,})['\"]?", SecretType.API_KEY, 0.80),
    ("generic_secret", r"(?i)secret[_-]?key['\"]?\s*[:=]\s*['\"]?([a-zA-Z0-9\-_]{20,})['\"]?", SecretType.SECRET_KEY, 0.80),

    # JWT Tokens (header.payload.signature) - MUST be before bearer_token
    ("jwt_token", r"eyJ[a-zA-Z0-9\-_]+\.eyJ[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+", SecretType.JWT, 0.95),

    # Bearer/Auth Tokens
    ("bearer_token", r"(?i)bearer\s+([a-zA-Z0-9\-_\.]+)", SecretType.BEARER_TOKEN, 0.85),
    ("basic_auth", r"(?i)basic\s+([a-zA-Z0-9+/=]{20,})", SecretType.BASIC_AUTH, 0.85),

    # Private Keys
    ("rsa_private_key", r"-----BEGIN RSA PRIVATE KEY-----", SecretType.PRIVATE_KEY, 0.99),
    ("ec_private_key", r"-----BEGIN EC PRIVATE KEY-----", SecretType.PRIVATE_KEY, 0.99),
    ("openssh_private_key", r"-----BEGIN OPENSSH PRIVATE KEY-----", SecretType.SSH_KEY, 0.99),
    ("pgp_private_key", r"-----BEGIN PGP PRIVATE KEY BLOCK-----", SecretType.PGP_KEY, 0.99),

    # Connection Strings
    ("postgres_conn", r"postgres(?:ql)?://[^:]+:[^@]+@[^\s]+", SecretType.CONNECTION_STRING, 0.90),
    ("mysql_conn", r"mysql://[^:]+:[^@]+@[^\s]+", SecretType.CONNECTION_STRING, 0.90),
    ("mongodb_conn", r"mongodb(?:\+srv)?://[^:]+:[^@]+@[^\s]+", SecretType.CONNECTION_STRING, 0.90),
    ("redis_conn", r"redis://[^:]+:[^@]+@[^\s]+", SecretType.CONNECTION_STRING, 0.90),

    # Password patterns in context
    ("password_assignment", r"(?i)password['\"]?\s*[:=]\s*['\"]([^'\"]{8,})['\"]", SecretType.PASSWORD, 0.85),
    ("passwd_assignment", r"(?i)passwd['\"]?\s*[:=]\s*['\"]([^'\"]{8,})['\"]", SecretType.PASSWORD, 0.85),
    ("pwd_assignment", r"(?i)(?:^|[^a-z])pwd['\"]?\s*[:=]\s*['\"]([^'\"]{8,})['\"]", SecretType.PASSWORD, 0.80),

    # Slack Tokens
    ("slack_token", r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}", SecretType.TOKEN, 0.95),
    ("slack_webhook", r"https://hooks\.slack\.com/services/T[a-zA-Z0-9_]+/B[a-zA-Z0-9_]+/[a-zA-Z0-9_]+", SecretType.TOKEN, 0.95),

    # Discord
    ("discord_token", r"[MN][A-Za-z\d]{23,}\.[\w-]{6}\.[\w-]{27}", SecretType.TOKEN, 0.90),
    ("discord_webhook", r"https://discord(?:app)?\.com/api/webhooks/[0-9]+/[a-zA-Z0-9_\-]+", SecretType.TOKEN, 0.95),

    # Stripe
    ("stripe_live_key", r"sk_live_[a-zA-Z0-9]{24,}", SecretType.API_KEY, 0.95),
    ("stripe_test_key", r"sk_test_[a-zA-Z0-9]{24,}", SecretType.API_KEY, 0.90),
    ("stripe_restricted", r"rk_live_[a-zA-Z0-9]{24,}", SecretType.API_KEY, 0.95),

    # Twilio
    ("twilio_api_key", r"SK[a-f0-9]{32}", SecretType.API_KEY, 0.85),

    # SendGrid
    ("sendgrid_api_key", r"SG\.[a-zA-Z0-9\-_]{22}\.[a-zA-Z0-9\-_]{43}", SecretType.API_KEY, 0.95),

    # NPM Token
    ("npm_token", r"npm_[a-zA-Z0-9]{36}", SecretType.TOKEN, 0.95),

    # PyPI Token
    ("pypi_token", r"pypi-AgEIcHlwaS5vcmc[a-zA-Z0-9\-_]{50,}", SecretType.TOKEN, 0.95),

    # Docker Hub
    ("docker_auth", r"(?i)docker[_-]?(?:hub)?[_-]?(?:password|token|auth)['\"]?\s*[:=]\s*['\"]([^'\"]{8,})['\"]", SecretType.CREDENTIAL, 0.80),

    # Encryption keys (hex)
    ("encryption_key_hex", r"(?i)(?:encryption|aes|des)[_-]?key['\"]?\s*[:=]\s*['\"]?([a-fA-F0-9]{32,})['\"]?", SecretType.ENCRYPTION_KEY, 0.85),

    # Generic high-entropy strings in sensitive context
    ("env_secret", r"(?i)(?:SECRET|TOKEN|KEY|PASSWORD|CREDENTIAL|AUTH)[_A-Z]*['\"]?\s*[:=]\s*['\"]?([a-zA-Z0-9\-_\.]{20,})['\"]?", SecretType.UNKNOWN, 0.70),
]


class SecretDetector:
    """
    Detects secrets and sensitive information in text.

    Usage:
        detector = SecretDetector()
        secrets = detector.scan("My API key is sk-abc123...")
        for secret in secrets:
            print(f"Found {secret.secret_type}: {secret.value[:10]}...")
    """

    def __init__(self, custom_patterns: Optional[List[Tuple[str, str, SecretType, float]]] = None):
        """
        Initialize the secret detector.

        Args:
            custom_patterns: Additional patterns to detect (name, regex, type, confidence)
        """
        self.patterns = list(SECRET_PATTERNS)
        if custom_patterns:
            self.patterns.extend(custom_patterns)

        # Pre-compile patterns for efficiency
        self._compiled_patterns: List[Tuple[str, re.Pattern, SecretType, float]] = []
        for name, pattern, secret_type, confidence in self.patterns:
            try:
                compiled = re.compile(pattern)
                self._compiled_patterns.append((name, compiled, secret_type, confidence))
            except re.error as e:
                print(f"Warning: Invalid pattern '{name}': {e}")

    def scan(self, text: str, min_confidence: float = 0.0) -> List[DetectedSecret]:
        """
        Scan text for secrets.

        Args:
            text: The text to scan
            min_confidence: Minimum confidence threshold (0.0 to 1.0)

        Returns:
            List of DetectedSecret objects
        """
        if not text:
            return []

        detected: List[DetectedSecret] = []
        seen_positions: set = set()  # Avoid duplicates at same position

        for name, pattern, secret_type, confidence in self._compiled_patterns:
            if confidence < min_confidence:
                continue

            for match in pattern.finditer(text):
                start = match.start()
                end = match.end()

                # Skip if we've already found something at this position
                pos_key = (start, end)
                if pos_key in seen_positions:
                    continue

                # Get the matched value (use group 1 if available, else full match)
                try:
                    value = match.group(1) if match.lastindex else match.group(0)
                except IndexError:
                    value = match.group(0)

                # Get context (50 chars before and after)
                context_start = max(0, start - 50)
                context_end = min(len(text), end + 50)
                context = text[context_start:context_end]

                # Adjust confidence based on context
                adjusted_confidence = self._adjust_confidence(text, match, confidence)

                if adjusted_confidence >= min_confidence:
                    seen_positions.add(pos_key)
                    detected.append(DetectedSecret(
                        secret_type=secret_type,
                        value=value,
                        start=start,
                        end=end,
                        confidence=adjusted_confidence,
                        pattern_name=name,
                        context=context
                    ))

        # Sort by position
        detected.sort(key=lambda x: x.start)
        return detected

    def _adjust_confidence(self, text: str, match: re.Match, base_confidence: float) -> float:
        """
        Adjust confidence based on surrounding context.

        Increases confidence if:
        - Found in a config/env file context
        - Near keywords like "secret", "password", "key"

        Decreases confidence if:
        - Looks like example/placeholder text
        - Contains obvious fake values
        """
        value = match.group(0)
        start = match.start()

        # Get surrounding context
        context_start = max(0, start - 100)
        context = text[context_start:start].lower()

        adjustment = 0.0

        # Increase for sensitive keywords nearby
        sensitive_keywords = ["secret", "password", "key", "token", "credential", "auth", "api"]
        if any(kw in context for kw in sensitive_keywords):
            adjustment += 0.05

        # Decrease for example/placeholder indicators
        example_indicators = ["example", "sample", "test", "demo", "placeholder", "your_", "xxx", "fake"]
        value_lower = value.lower()
        if any(ind in value_lower for ind in example_indicators):
            adjustment -= 0.30

        # Decrease for obviously fake values
        if value_lower in ["password", "secret", "token", "apikey", "12345678", "abcdefgh"]:
            adjustment -= 0.50

        # Increase for .env or config file indicators in context
        if ".env" in context or "config" in context or "credential" in context:
            adjustment += 0.05

        return max(0.0, min(1.0, base_confidence + adjustment))

    def has_secrets(self, text: str, min_confidence: float = 0.70) -> bool:
        """Quick check if text contains any secrets above confidence threshold."""
        secrets = self.scan(text, min_confidence)
        return len(secrets) > 0

    def get_secret_types(self, text: str, min_confidence: float = 0.70) -> List[SecretType]:
        """Get list of unique secret types found in text."""
        secrets = self.scan(text, min_confidence)
        return list(set(s.secret_type for s in secrets))

    def scan_dict(self, data: dict, min_confidence: float = 0.70) -> Dict[str, List[DetectedSecret]]:
        """
        Scan a dictionary recursively for secrets.

        Args:
            data: Dictionary to scan
            min_confidence: Minimum confidence threshold

        Returns:
            Dict mapping key paths to detected secrets
        """
        results: Dict[str, List[DetectedSecret]] = {}
        self._scan_dict_recursive(data, "", results, min_confidence)
        return results

    def _scan_dict_recursive(self, data, path: str, results: Dict[str, List[DetectedSecret]],
                             min_confidence: float):
        """Recursively scan dictionary values."""
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path else key
                self._scan_dict_recursive(value, new_path, results, min_confidence)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_path = f"{path}[{i}]"
                self._scan_dict_recursive(item, new_path, results, min_confidence)
        elif isinstance(data, str):
            secrets = self.scan(data, min_confidence)
            if secrets:
                results[path] = secrets


# Convenience function
def detect_secrets(text: str, min_confidence: float = 0.70) -> List[DetectedSecret]:
    """
    Convenience function to detect secrets in text.

    Args:
        text: Text to scan
        min_confidence: Minimum confidence threshold

    Returns:
        List of detected secrets
    """
    detector = SecretDetector()
    return detector.scan(text, min_confidence)


if __name__ == "__main__":
    # Test the detector
    test_texts = [
        "My OpenAI key is sk-abc123def456ghi789jkl012mno345",
        "GitHub token: ghp_1234567890abcdefghijklmnopqrstuvwxyz",
        "Set password='SuperSecret123!' in config",
        "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
        "postgres://user:password123@localhost:5432/mydb",
        "This is just normal text with no secrets",
        "Example: api_key=your_api_key_here",  # Should be low confidence
    ]

    detector = SecretDetector()

    for text in test_texts:
        print(f"\nScanning: {text[:50]}...")
        secrets = detector.scan(text, min_confidence=0.5)
        if secrets:
            for s in secrets:
                print(f"  Found {s.secret_type.value}: {s.value[:20]}... (conf: {s.confidence:.2f})")
        else:
            print("  No secrets found")
