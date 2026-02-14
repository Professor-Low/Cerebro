"""
Context Enhancement Module - Phase 1 of All-Knowing Brain PRD.

Provides:
- RelevanceScorer: Score context items by relevance
- TokenBudgetManager: Fit context within token limits
- CWDDetector: Detect project from current working directory
"""

from .cwd_detector import CWDDetector, detect_from_cwd, get_cwd_context
from .relevance_scorer import RelevanceScorer, rank_by_relevance, score_relevance
from .token_budget import ContextFormatter, TokenBudget, TokenBudgetManager, fit_context_to_budget

__all__ = [
    # Relevance scoring
    "RelevanceScorer",
    "score_relevance",
    "rank_by_relevance",
    # Token budgeting
    "TokenBudgetManager",
    "TokenBudget",
    "ContextFormatter",
    "fit_context_to_budget",
    # CWD detection
    "CWDDetector",
    "detect_from_cwd",
    "get_cwd_context",
]
