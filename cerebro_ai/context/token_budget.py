"""
Token Budget Manager - Fit context within token limits intelligently.

Part of Phase 1 Enhancement in the All-Knowing Brain PRD.
Manages token allocation across different context types:
- Identity core: 200 tokens (fixed)
- Corrections: 150 tokens (high priority)
- Relevant context: 350 tokens (by relevance score)
- Total: 700 tokens max
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TokenBudget:
    """Token allocation for different context types."""
    identity: int = 200      # Fixed identity core
    corrections: int = 150   # High priority corrections
    context: int = 350       # Relevance-scored context
    total: int = 700         # Total budget

    def __post_init__(self):
        # Ensure components don't exceed total
        component_sum = self.identity + self.corrections + self.context
        if component_sum > self.total:
            # Scale down context to fit
            self.context = self.total - self.identity - self.corrections


class TokenBudgetManager:
    """
    Manages token budget for context injection.
    Ensures context fits within limits while prioritizing important items.
    """

    # Approximate tokens per character (conservative estimate)
    CHARS_PER_TOKEN = 4

    def __init__(self, budget: TokenBudget = None):
        self.budget = budget or TokenBudget()

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if not text:
            return 0
        return len(text) // self.CHARS_PER_TOKEN + 1

    def fit_to_budget(self,
                      identity_items: List[Dict[str, Any]] = None,
                      corrections: List[Dict[str, Any]] = None,
                      context_items: List[Tuple[Dict[str, Any], float]] = None
                      ) -> Dict[str, Any]:
        """
        Fit items within the token budget.

        Args:
            identity_items: Core identity items (always included up to budget)
            corrections: High-priority corrections (included by priority)
            context_items: List of (item, relevance_score) tuples sorted by score

        Returns:
            Dict with:
                - identity: List of included identity items
                - corrections: List of included corrections
                - context: List of included context items
                - tokens_used: Token counts per category
                - tokens_remaining: Remaining budget
        """
        result = {
            "identity": [],
            "corrections": [],
            "context": [],
            "tokens_used": {
                "identity": 0,
                "corrections": 0,
                "context": 0,
                "total": 0
            },
            "tokens_remaining": self.budget.total
        }

        # 1. Identity items (fixed budget)
        if identity_items:
            result["identity"], result["tokens_used"]["identity"] = self._fit_items(
                identity_items,
                self.budget.identity,
                sort_by_relevance=False
            )

        # 2. Corrections (high priority budget)
        if corrections:
            # Sort corrections by importance level
            sorted_corrections = sorted(
                corrections,
                key=lambda x: self._correction_priority(x),
                reverse=True
            )
            result["corrections"], result["tokens_used"]["corrections"] = self._fit_items(
                sorted_corrections,
                self.budget.corrections,
                sort_by_relevance=False
            )

        # 3. Context items (relevance-scored budget)
        if context_items:
            # Already sorted by relevance score
            items_only = [item for item, _ in context_items]
            result["context"], result["tokens_used"]["context"] = self._fit_items(
                items_only,
                self.budget.context,
                sort_by_relevance=False  # Already sorted
            )

        # Calculate totals
        result["tokens_used"]["total"] = sum(
            result["tokens_used"][k] for k in ["identity", "corrections", "context"]
        )
        result["tokens_remaining"] = self.budget.total - result["tokens_used"]["total"]

        return result

    def _fit_items(self,
                   items: List[Dict[str, Any]],
                   budget: int,
                   sort_by_relevance: bool = False
                   ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Fit items within a token budget.

        Args:
            items: Items to fit
            budget: Token budget for these items
            sort_by_relevance: Whether to sort by relevance_score first

        Returns:
            Tuple of (included_items, tokens_used)
        """
        if not items:
            return [], 0

        if sort_by_relevance:
            items = sorted(
                items,
                key=lambda x: x.get("relevance_score", 0),
                reverse=True
            )

        included = []
        tokens_used = 0

        for item in items:
            item_text = self._get_item_text(item)
            item_tokens = self.estimate_tokens(item_text)

            if tokens_used + item_tokens <= budget:
                included.append(item)
                tokens_used += item_tokens
            elif budget - tokens_used > 50:
                # Try to fit a truncated version
                remaining_chars = (budget - tokens_used) * self.CHARS_PER_TOKEN
                truncated_item = self._truncate_item(item, remaining_chars)
                if truncated_item:
                    included.append(truncated_item)
                    tokens_used = budget
                break
            else:
                break

        return included, tokens_used

    def _correction_priority(self, correction: Dict[str, Any]) -> int:
        """Get priority score for a correction."""
        level = correction.get("level", "").upper()
        priority_map = {
            "CRITICAL": 100,
            "HIGH": 80,
            "MEDIUM": 50,
            "MED": 50,
            "LOW": 20,
        }
        return priority_map.get(level, 30)

    def _get_item_text(self, item: Dict[str, Any]) -> str:
        """Get text representation of an item for token counting."""
        if isinstance(item, str):
            return item

        # Try various fields
        for field in ["text", "content", "summary", "formatted"]:
            if field in item and item[field]:
                return str(item[field])

        # Fall back to JSON-like representation
        parts = []
        for key in ["text", "content", "summary", "level", "decision", "reasoning"]:
            if key in item and item[key]:
                parts.append(f"{key}: {item[key]}")

        return " | ".join(parts) if parts else str(item)

    def _truncate_item(self,
                       item: Dict[str, Any],
                       max_chars: int) -> Optional[Dict[str, Any]]:
        """Truncate an item to fit within character limit."""
        if max_chars < 50:
            return None

        item_copy = item.copy()
        text_fields = ["text", "content", "summary"]

        for field in text_fields:
            if field in item_copy and item_copy[field]:
                text = str(item_copy[field])
                if len(text) > max_chars:
                    item_copy[field] = text[:max_chars - 3] + "..."
                    item_copy["truncated"] = True
                    return item_copy
                return item_copy

        return item_copy


class ContextFormatter:
    """Formats context items for injection into Claude."""

    def __init__(self, budget_manager: TokenBudgetManager = None):
        self.budget_manager = budget_manager or TokenBudgetManager()

    def format_context(self,
                       fitted_result: Dict[str, Any],
                       include_token_info: bool = False) -> str:
        """
        Format fitted context items as injection text.

        Args:
            fitted_result: Output from TokenBudgetManager.fit_to_budget()
            include_token_info: Whether to include token usage info

        Returns:
            Formatted context string ready for injection
        """
        parts = []

        # Identity section
        if fitted_result.get("identity"):
            identity_text = self._format_identity(fitted_result["identity"])
            if identity_text:
                parts.append(identity_text)

        # Corrections section
        if fitted_result.get("corrections"):
            corrections_text = self._format_corrections(fitted_result["corrections"])
            if corrections_text:
                parts.append(corrections_text)

        # Relevant context section
        if fitted_result.get("context"):
            context_text = self._format_relevant_context(fitted_result["context"])
            if context_text:
                parts.append(context_text)

        result = "\n".join(parts)

        if include_token_info:
            tokens = fitted_result.get("tokens_used", {})
            result += f"\n<!-- Tokens: {tokens.get('total', 0)}/{self.budget_manager.budget.total} -->"

        return result

    def _format_identity(self, items: List[Dict[str, Any]]) -> str:
        """Format identity items."""
        if not items:
            return ""

        # Typically already formatted, just join
        return "\n".join(
            item.get("formatted") or item.get("text") or str(item)
            for item in items
        )

    def _format_corrections(self, corrections: List[Dict[str, Any]]) -> str:
        """Format corrections."""
        if not corrections:
            return ""

        lines = ["CORRECTIONS:"]
        for c in corrections:
            level = c.get("level", "")
            text = c.get("text", str(c))
            lines.append(f"- [{level}] {text}")

        return "\n".join(lines)

    def _format_relevant_context(self, items: List[Dict[str, Any]]) -> str:
        """Format relevant context items."""
        if not items:
            return ""

        lines = ["[RELEVANT CONTEXT]"]
        for item in items:
            # Get summary or content
            summary = (
                item.get("summary") or
                item.get("content", "")[:150] or
                str(item)[:150]
            )
            score = item.get("relevance_score", "")
            score_str = f" ({score:.0%})" if isinstance(score, float) else ""

            lines.append(f"- {summary}{score_str}")

        lines.append("[/RELEVANT CONTEXT]")
        return "\n".join(lines)


def fit_context_to_budget(identity: List[Dict] = None,
                          corrections: List[Dict] = None,
                          context: List[Tuple[Dict, float]] = None,
                          budget: TokenBudget = None) -> Dict[str, Any]:
    """
    Convenience function to fit context within budget.

    Args:
        identity: Identity items
        corrections: Correction items
        context: List of (item, relevance_score) tuples
        budget: Optional custom budget

    Returns:
        Fitted result dict
    """
    manager = TokenBudgetManager(budget)
    return manager.fit_to_budget(identity, corrections, context)


if __name__ == "__main__":
    # Test the budget manager
    manager = TokenBudgetManager()

    # Test identity
    identity = [{"text": "USER: ExampleUser | direct | technical"}]

    # Test corrections
    corrections = [
        {"level": "HIGH", "text": "NAS must maintain constant connection"},
        {"level": "MED", "text": "Spit out plan first before implementing"},
        {"level": "LOW", "text": "Prefer concise responses"},
    ]

    # Test context (with relevance scores)
    context = [
        ({"summary": "Fixed MCP timeout by increasing connection timeout", "relevance_score": 0.85}, 0.85),
        ({"summary": "Session continuation implementation details", "relevance_score": 0.72}, 0.72),
        ({"summary": "Unrelated conversation about weather", "relevance_score": 0.15}, 0.15),
    ]

    result = manager.fit_to_budget(identity, corrections, context)

    print("=== Token Budget Test ===")
    print(f"Identity items: {len(result['identity'])}")
    print(f"Corrections: {len(result['corrections'])}")
    print(f"Context items: {len(result['context'])}")
    print(f"Tokens used: {result['tokens_used']}")
    print(f"Tokens remaining: {result['tokens_remaining']}")

    # Format the result
    formatter = ContextFormatter(manager)
    formatted = formatter.format_context(result, include_token_info=True)
    print("\n=== Formatted Context ===")
    print(formatted)
