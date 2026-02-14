"""
Content Filter - Filter system-generated content from extraction.

Prevents session continuation summaries and other system-injected content
from being extracted as user corrections or meaningful conversation data.
"""

import re
from typing import Any, Dict, List

# System-generated content patterns to detect and skip
SYSTEM_CONTENT_PATTERNS = [
    # Session continuation markers
    r"This session is being continued from a previous conversation",
    r"The conversation is summarized below",
    r"\[SESSION CONTINUATION AVAILABLE\]",
    r"I found you were working on",
    r"Would you like to pick up where you left off",
    r"I found recent work you may want to continue",

    # Memory context markers
    r"\[MEMORY CONTEXT\]",
    r"\[/MEMORY CONTEXT\]",
    r"USER: \w+ \| direct",
    r"PROJECTS:.*\(active",
    r"CORRECTIONS:",
    r"LAST:.*Did:",
    r"DEVICE:.*\(Linux\)",

    # Hook output markers
    r"UserPromptSubmit hook success",
    r"system-reminder",

    # Claude-specific system messages
    r"Co-Authored-By: Claude",
    r"claude\.ai/claude-code",

    # Session metadata
    r"Session ID:",
    r"Session involving:",
    r"Confidence: \d+%",
    r"match\)\..*Would you like",
]

# Compiled patterns for efficiency
_COMPILED_PATTERNS = None


def _get_compiled_patterns():
    """Get compiled regex patterns (cached)."""
    global _COMPILED_PATTERNS
    if _COMPILED_PATTERNS is None:
        _COMPILED_PATTERNS = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in SYSTEM_CONTENT_PATTERNS
        ]
    return _COMPILED_PATTERNS


def is_system_generated_content(text: str) -> bool:
    """
    Check if text appears to be system-generated content.

    Args:
        text: Text content to check

    Returns:
        True if the content appears to be system-generated, False otherwise
    """
    if not text:
        return False

    # Check against all patterns
    for pattern in _get_compiled_patterns():
        if pattern.search(text):
            return True

    # Check for high density of system markers
    system_markers = ["[", "]", "|", "CONTEXT", "SESSION", "HOOK"]
    marker_count = sum(1 for marker in system_markers if marker in text)

    # If more than 3 system markers in a short message, likely system content
    if marker_count >= 3 and len(text) < 500:
        return True

    return False


def filter_messages_for_extraction(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter out messages with system-generated content from a message list.

    This is used before extracting corrections, summaries, or other data
    to prevent system-injected content from polluting the extraction.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Filtered list excluding messages with system-generated content
    """
    filtered = []

    for msg in messages:
        content = msg.get("content", "")

        # Handle content as list (Claude's format)
        if isinstance(content, list):
            # Extract text from content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = " ".join(text_parts)

        # Skip if system-generated
        if is_system_generated_content(content):
            continue

        # Also skip very short messages that are just system artifacts
        if len(content.strip()) < 10:
            continue

        filtered.append(msg)

    return filtered


def extract_user_content_only(text: str) -> str:
    """
    Extract only the user's actual content from text that may contain
    system-generated prefixes or suffixes.

    Args:
        text: Raw text that may contain system content

    Returns:
        Cleaned text with system content removed
    """
    if not text:
        return ""

    # Remove common system prefixes
    # These appear at the start of user messages due to hooks
    system_prefixes = [
        r"<system-reminder>.*?</system-reminder>\s*",
        r"\[MEMORY CONTEXT\].*?\[/MEMORY CONTEXT\]\s*",
        r"\[SESSION CONTINUATION AVAILABLE\].*?(?=\n\n|\Z)",
    ]

    cleaned = text
    for prefix_pattern in system_prefixes:
        cleaned = re.sub(prefix_pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    return cleaned.strip()


def get_message_hash(content: str) -> str:
    """
    Generate a hash for message content for deduplication.

    Args:
        content: Message content

    Returns:
        MD5 hash string
    """
    import hashlib

    # Normalize content for hashing
    normalized = content.lower().strip()
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)

    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


if __name__ == "__main__":
    # Test the filter
    test_messages = [
        {"role": "user", "content": "[MEMORY CONTEXT]\nUSER: ExampleUser\n[/MEMORY CONTEXT]\nHello, help me debug this."},
        {"role": "assistant", "content": "I'll help you debug. Let me look at the code."},
        {"role": "user", "content": "No, that's wrong. Use Python 3."},
        {"role": "user", "content": "[SESSION CONTINUATION AVAILABLE]\nI found recent work..."},
        {"role": "assistant", "content": "I understand, let me use Python 3 instead."},
    ]

    print("=== Content Filter Test ===\n")

    print("Original messages:")
    for i, msg in enumerate(test_messages):
        content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
        print(f"  {i+1}. [{msg['role']}] {content}")

    print("\nFiltered messages:")
    filtered = filter_messages_for_extraction(test_messages)
    for i, msg in enumerate(filtered):
        content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
        system = is_system_generated_content(msg["content"])
        print(f"  {i+1}. [{msg['role']}] {content} (system: {system})")

    print(f"\nFiltered out {len(test_messages) - len(filtered)} system messages")
