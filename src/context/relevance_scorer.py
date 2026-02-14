"""
Relevance Scorer - Score context items by relevance to current situation.

Part of Phase 1 Enhancement in the All-Knowing Brain PRD.
Scores context items on a 0-1 scale based on:
- Semantic similarity (0-0.4)
- Recency boost (0-0.2)
- Same project boost (0-0.2)
- Solution success boost (0-0.2)
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


class RelevanceScorer:
    """
    Scores context items by relevance to the current situation.
    Used to prioritize what context gets injected.
    """

    # Weight configuration (should sum to 1.0)
    WEIGHTS = {
        "semantic": 0.40,      # Semantic/text similarity
        "recency": 0.20,       # How recent the item is
        "project": 0.20,       # Same project boost
        "success": 0.20,       # Solution confirmation boost
    }

    # Recency decay - items older than this get minimal boost
    RECENCY_HALF_LIFE_DAYS = 14

    def __init__(self, base_path: str = None):
        if base_path is None:

            from config import AI_MEMORY_BASE

            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self._embeddings_available = None

    def score_item(self,
                   item: Dict[str, Any],
                   current_context: Dict[str, Any]) -> float:
        """
        Score a single context item against the current context.

        Args:
            item: The context item to score (conversation, fact, solution, etc.)
            current_context: Current situation context with:
                - query: str - The current search query or user message
                - project: str - Current project name (optional)
                - topics: List[str] - Current topics (optional)
                - cwd: str - Current working directory (optional)

        Returns:
            Relevance score between 0.0 and 1.0
        """
        scores = {
            "semantic": self._score_semantic(item, current_context),
            "recency": self._score_recency(item),
            "project": self._score_project(item, current_context),
            "success": self._score_success(item),
        }

        # Calculate weighted sum
        total = sum(
            scores[key] * self.WEIGHTS[key]
            for key in self.WEIGHTS
        )

        # Clamp to 0-1
        return max(0.0, min(1.0, total))

    def score_items(self,
                    items: List[Dict[str, Any]],
                    current_context: Dict[str, Any],
                    min_score: float = 0.0) -> List[Tuple[Dict[str, Any], float]]:
        """
        Score multiple items and return sorted by relevance.

        Args:
            items: List of context items to score
            current_context: Current situation context
            min_score: Minimum score threshold (items below this are filtered)

        Returns:
            List of (item, score) tuples sorted by score descending
        """
        scored = []
        for item in items:
            score = self.score_item(item, current_context)
            if score >= min_score:
                scored.append((item, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _score_semantic(self,
                        item: Dict[str, Any],
                        current_context: Dict[str, Any]) -> float:
        """
        Score semantic similarity between item and current context.
        Returns 0-1 score.
        """
        query = current_context.get("query", "")
        if not query:
            return 0.0

        # Get item text (try various fields)
        item_text = self._get_item_text(item)
        if not item_text:
            return 0.0

        # Calculate text similarity (Jaccard-based)
        similarity = self._text_similarity(query, item_text)

        # Also check topic overlap
        query_topics = set(current_context.get("topics", []))
        item_topics = set(item.get("topics", []))
        if not item_topics:
            item_topics = set(item.get("metadata", {}).get("topics", []))

        if query_topics and item_topics:
            topic_overlap = len(query_topics & item_topics) / max(len(query_topics), 1)
            # Blend text similarity with topic overlap
            similarity = similarity * 0.6 + topic_overlap * 0.4

        return similarity

    def _score_recency(self, item: Dict[str, Any]) -> float:
        """
        Score based on how recent the item is.
        Uses exponential decay with half-life.
        Returns 0-1 score.
        """
        timestamp = item.get("timestamp") or item.get("created_at")
        if not timestamp:
            return 0.5  # Default if no timestamp

        try:
            if isinstance(timestamp, str):
                # Parse ISO format
                if "T" in timestamp:
                    item_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                else:
                    item_time = datetime.fromisoformat(timestamp)
            else:
                item_time = timestamp

            # Make timezone-naive for comparison
            if hasattr(item_time, 'tzinfo') and item_time.tzinfo is not None:
                item_time = item_time.replace(tzinfo=None)

            days_old = (datetime.now() - item_time).days

            # Exponential decay: score = 0.5^(days_old / half_life)
            decay = 0.5 ** (days_old / self.RECENCY_HALF_LIFE_DAYS)

            # Scale to 0-1 (items from today get 1.0, very old items approach 0)
            return max(0.0, min(1.0, decay))

        except Exception:
            return 0.5  # Default on parse error

    def _score_project(self,
                       item: Dict[str, Any],
                       current_context: Dict[str, Any]) -> float:
        """
        Score based on project match.
        Returns 1.0 for same project, 0.0 for different.
        """
        current_project = current_context.get("project", "").lower()
        if not current_project:
            return 0.5  # No project context, neutral score

        # Check various project fields
        item_project = (
            item.get("project") or
            item.get("metadata", {}).get("project") or
            item.get("detected_project") or
            ""
        ).lower()

        if not item_project:
            return 0.3  # No project info in item

        # Exact match
        if current_project == item_project:
            return 1.0

        # Partial match (one contains the other)
        if current_project in item_project or item_project in current_project:
            return 0.7

        return 0.0

    def _score_success(self, item: Dict[str, Any]) -> float:
        """
        Score based on solution success/confirmation.
        Higher score for confirmed solutions.
        Returns 0-1 score.
        """
        # Check for solution confirmation count
        confirmations = item.get("success_confirmations", 0)
        failures = item.get("failure_count", 0)

        if confirmations > 0:
            # Net positive confirmations get high score
            net = confirmations - failures
            if net > 0:
                return min(1.0, 0.5 + net * 0.1)

        # Check for quality score
        quality = item.get("quality_score", 0)
        if quality > 0:
            return min(1.0, quality)

        # Check for has_solution flag
        if item.get("has_solution"):
            return 0.7

        # Default neutral
        return 0.5

    def _get_item_text(self, item: Dict[str, Any]) -> str:
        """Extract searchable text from an item."""
        text_parts = []

        # Try various fields
        for field in ["content", "text", "summary", "problem", "solution",
                      "description", "message", "query"]:
            if field in item and item[field]:
                text_parts.append(str(item[field]))

        # Also check metadata
        metadata = item.get("metadata", {})
        if metadata.get("summary"):
            text_parts.append(metadata["summary"])
        if metadata.get("topics"):
            text_parts.extend(metadata["topics"])

        # Check extracted_data
        extracted = item.get("extracted_data", {})
        if extracted.get("topics"):
            text_parts.extend(extracted["topics"])

        return " ".join(text_parts)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts.
        Returns 0-1 score.
        """
        if not text1 or not text2:
            return 0.0

        # Tokenize by words, remove stopwords
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'i', 'my', 'me',
            'it', 'this', 'that', 'to', 'in', 'on', 'at', 'for', 'with', 'and',
            'or', 'but', 'of', 'by', 'as', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can'
        }

        words1 = {w.lower() for w in re.findall(r'\b\w+\b', text1) if w.lower() not in stopwords}
        words2 = {w.lower() for w in re.findall(r'\b\w+\b', text2) if w.lower() not in stopwords}

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return 0.0

        return intersection / union


def score_relevance(item: Dict[str, Any], current_context: Dict[str, Any]) -> float:
    """
    Convenience function to score a single item.

    Args:
        item: Context item to score
        current_context: Current situation context

    Returns:
        Relevance score 0-1
    """
    scorer = RelevanceScorer()
    return scorer.score_item(item, current_context)


def rank_by_relevance(items: List[Dict[str, Any]],
                      current_context: Dict[str, Any],
                      top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Rank items by relevance and return top K.

    Args:
        items: List of context items
        current_context: Current situation context
        top_k: Number of items to return

    Returns:
        List of top K items (with 'relevance_score' added)
    """
    scorer = RelevanceScorer()
    scored = scorer.score_items(items, current_context)

    result = []
    for item, score in scored[:top_k]:
        item_copy = item.copy()
        item_copy["relevance_score"] = round(score, 3)
        result.append(item_copy)

    return result


if __name__ == "__main__":
    # Test the scorer
    scorer = RelevanceScorer()

    # Test item
    test_item = {
        "content": "Fixed the MCP timeout error by increasing the connection timeout to 30 seconds",
        "topics": ["mcp", "timeout", "error", "fix"],
        "timestamp": datetime.now().isoformat(),
        "project": "ai-memory-mcp",
        "success_confirmations": 2,
    }

    # Test context
    test_context = {
        "query": "MCP server timeout problem",
        "topics": ["mcp", "timeout"],
        "project": "ai-memory-mcp",
    }

    score = scorer.score_item(test_item, test_context)
    print(f"Test score: {score:.3f}")
    print("Expected: high score (>0.7) due to topic match + project match + recent + confirmed")
