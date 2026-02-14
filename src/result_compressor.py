#!/usr/bin/env python3
"""
Result Compressor - Token-limited search result formatting.
Compresses search results to fit within token budgets.

Output format:
• Topic/Key → Value [CONFIDENCE, DATE]
"""

import re
from datetime import datetime
from typing import Any, Dict, List

# Approximate tokens per word (conservative)
TOKENS_PER_WORD = 1.3


def count_tokens(text: str) -> int:
    """Estimate token count from text."""
    if not text:
        return 0
    words = len(text.split())
    return int(words * TOKENS_PER_WORD)


def compress_to_bullet(result: Dict[str, Any]) -> str:
    """
    Compress a single search result to bullet format.

    Format: • Key → Value [CONFIDENCE, DATE]
    Target: ~50 tokens per result max
    """
    # Extract relevant fields
    content = result.get("content", result.get("text", ""))
    chunk_type = result.get("chunk_type", result.get("type", "unknown"))
    score = result.get("score", result.get("similarity", 0))
    timestamp = result.get("timestamp", result.get("date", ""))
    result.get("conversation_id", "")

    # Clean content
    content = content.strip()
    content = re.sub(r'\s+', ' ', content)  # Normalize whitespace

    # Determine confidence label
    if score >= 0.9:
        confidence = "CONFIRMED"
    elif score >= 0.7:
        confidence = "HIGH"
    elif score >= 0.5:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Format date (compact)
    date_str = ""
    if timestamp:
        try:
            if isinstance(timestamp, str):
                # Try to parse ISO format
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
            elif isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp)
                date_str = dt.strftime("%Y-%m-%d")
        except:
            date_str = str(timestamp)[:10]

    # Determine key based on chunk type
    if chunk_type == "fact":
        key = "Fact"
    elif chunk_type == "file_path":
        key = "File"
    elif chunk_type == "problem_solution":
        key = "Solution"
    elif chunk_type == "goal":
        key = "Goal"
    elif chunk_type == "correction":
        key = "Correction"
    else:
        key = "Info"

    # Compress content to ~40 words max
    words = content.split()
    if len(words) > 40:
        content = " ".join(words[:40]) + "..."

    # Build bullet
    meta_parts = []
    meta_parts.append(confidence)
    if date_str:
        meta_parts.append(date_str)

    meta_str = ", ".join(meta_parts)

    return f"• {key}: {content} [{meta_str}]"


def compress_results(results: List[Dict[str, Any]], max_tokens: int = 500,
                    max_items: int = 5) -> Dict[str, Any]:
    """
    Compress search results to fit within token budget.

    Args:
        results: Raw search results
        max_tokens: Maximum tokens for all results combined
        max_items: Maximum number of items to return

    Returns:
        {
            "compressed": List[str],  # Bullet-point strings
            "token_count": int,
            "items_returned": int,
            "items_truncated": int
        }
    """
    compressed = []
    token_count = 0
    items_truncated = 0

    for i, result in enumerate(results):
        if i >= max_items:
            items_truncated = len(results) - max_items
            break

        bullet = compress_to_bullet(result)
        bullet_tokens = count_tokens(bullet)

        # Check if adding this would exceed budget
        if token_count + bullet_tokens > max_tokens:
            # Try to fit a shorter version
            words = bullet.split()
            while len(words) > 10 and count_tokens(" ".join(words)) + token_count > max_tokens:
                words = words[:-5]  # Remove 5 words at a time

            if len(words) > 10:
                bullet = " ".join(words) + "...]"
                bullet_tokens = count_tokens(bullet)
            else:
                items_truncated = len(results) - i
                break

        compressed.append(bullet)
        token_count += bullet_tokens

    return {
        "compressed": compressed,
        "token_count": token_count,
        "items_returned": len(compressed),
        "items_truncated": items_truncated
    }


def format_compressed_output(compression_result: Dict[str, Any]) -> str:
    """
    Format compressed results as a single string for Claude.
    """
    lines = compression_result["compressed"]

    if not lines:
        return "No results found."

    output = "\n".join(lines)

    # Add truncation note if applicable
    if compression_result["items_truncated"] > 0:
        output += f"\n[{compression_result['items_truncated']} more results truncated]"

    return output


def compress_for_mcp(results: List[Dict[str, Any]], max_tokens: int = 500) -> Dict[str, Any]:
    """
    Main entry point for MCP search compression.
    Returns both structured data and formatted string.

    Args:
        results: Raw search results from memory system
        max_tokens: Token budget (default 500 per query)

    Returns:
        {
            "results": List[Dict],  # Compressed results with metadata
            "formatted": str,        # Ready-to-use bullet string
            "token_count": int,
            "stats": {
                "items_returned": int,
                "items_truncated": int
            }
        }
    """
    # Compress results
    compression = compress_results(results, max_tokens=max_tokens, max_items=5)

    # Build structured results with original data + compressed text
    structured_results = []
    for i, bullet in enumerate(compression["compressed"]):
        if i < len(results):
            structured_results.append({
                "bullet": bullet,
                "original": results[i]
            })

    return {
        "results": structured_results,
        "formatted": format_compressed_output(compression),
        "token_count": compression["token_count"],
        "stats": {
            "items_returned": compression["items_returned"],
            "items_truncated": compression["items_truncated"]
        }
    }


if __name__ == "__main__":
    # Test with sample data
    sample_results = [
        {
            "content": "The NAS IP address is 10.0.0.100. It's a Synology NAS with 16TB available storage.",
            "chunk_type": "fact",
            "score": 0.92,
            "timestamp": "2026-01-10T14:30:00",
            "conversation_id": "test-conv-1"
        },
        {
            "content": "To fix SSH timeouts, add ServerAliveInterval=60 to your SSH config. This keeps the connection alive.",
            "chunk_type": "problem_solution",
            "score": 0.85,
            "timestamp": "2026-01-08T10:00:00",
            "conversation_id": "test-conv-2"
        },
        {
            "content": "User prefers automation over manual processes. They dislike having to repeat context.",
            "chunk_type": "fact",
            "score": 0.78,
            "timestamp": "2026-01-05T16:00:00",
            "conversation_id": "test-conv-3"
        },
        {
            "content": "The AI Memory project path is ~/cerebro-memory",
            "chunk_type": "file_path",
            "score": 0.71,
            "timestamp": "2025-12-28T12:00:00",
            "conversation_id": "test-conv-4"
        },
        {
            "content": "Long-term goal: Have Claude with persistent memory and unique personality",
            "chunk_type": "goal",
            "score": 0.65,
            "timestamp": "2026-01-11T18:00:00",
            "conversation_id": "test-conv-5"
        },
    ]

    print("=== Token Budget: 500 ===")
    result = compress_for_mcp(sample_results, max_tokens=500)
    print(result["formatted"])
    print(f"\nToken count: {result['token_count']}")
    print(f"Items: {result['stats']['items_returned']}, Truncated: {result['stats']['items_truncated']}")

    print("\n=== Token Budget: 200 ===")
    result = compress_for_mcp(sample_results, max_tokens=200)
    print(result["formatted"])
    print(f"\nToken count: {result['token_count']}")
    print(f"Items: {result['stats']['items_returned']}, Truncated: {result['stats']['items_truncated']}")
