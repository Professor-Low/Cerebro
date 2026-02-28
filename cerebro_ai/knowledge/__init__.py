"""
Knowledge Module - Phase 2 of All-Knowing Brain PRD.

Provides:
- FactDeduplicator: Find and merge duplicate facts
- FactLinker: Create relationships between facts
"""

from .fact_deduplicator import FactDeduplicator, deduplicate_facts
from .fact_linker import LINK_TYPES, FactLinker, link_all_facts

__all__ = [
    # Fact deduplication
    "FactDeduplicator",
    "deduplicate_facts",
    # Fact linking
    "FactLinker",
    "link_all_facts",
    "LINK_TYPES",
]
