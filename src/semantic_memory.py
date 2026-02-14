"""
Semantic Memory Manager - Claude.Me v6.0
Stores general knowledge abstracted from episodes.

Part of Phase 1: Episodic vs Semantic Separation
"""
import hashlib
import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from memory_types import SemanticMemory, extract_domain


class SemanticMemoryManager:
    """
    Manage semantic memories - general knowledge abstracted from episodes.

    Semantic memories answer: "What do we know about X?"
    They are timeless facts generalized from specific episodes.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.semantic_path = self.base_path / "semantic"
        self.index_path = self.semantic_path / "_index.json"
        self._index_lock = threading.Lock()
        self._ensure_directories()

    def _ensure_directories(self):
        """Create directories if they don't exist."""
        self.semantic_path.mkdir(parents=True, exist_ok=True)

    def _generate_id(self, fact: str) -> str:
        """Generate unique semantic memory ID based on fact content."""
        # Create deterministic ID from content (helps detect duplicates)
        content_hash = hashlib.sha256(fact.lower().strip().encode()).hexdigest()[:10]
        return f"sem_{content_hash}"

    @staticmethod
    def _safe_confidence(value) -> float:
        """Convert confidence to float, handling string values like 'high'."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            mapping = {"high": 0.9, "medium": 0.7, "low": 0.5}
            return mapping.get(value.lower(), 0.5)
        return 0.5

    def _load_index(self) -> Dict:
        """Load the semantic memory index."""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {"entries": [], "by_domain": {}, "by_keyword": {}}

    def _save_index(self, index: Dict):
        """Save the semantic memory index."""
        with self._index_lock:
            with open(self.index_path, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2)

    def _extract_keywords(self, fact: str) -> List[str]:
        """Extract searchable keywords from a fact."""
        # Remove common words and keep meaningful terms
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "that", "which", "who", "whom", "this", "these", "those", "it", "its"
        }

        # Extract words (including technical terms with underscores/hyphens)
        words = re.findall(r'[a-zA-Z][a-zA-Z0-9_-]*', fact.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        # Keep unique, preserve order
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)

        return unique[:20]  # Limit to 20 keywords

    def save_semantic(self, semantic: SemanticMemory) -> str:
        """
        Save a semantic memory.

        Returns:
            The semantic memory ID
        """
        # Auto-generate ID if not set
        if not semantic.id or semantic.id.startswith("sem_temp"):
            semantic.id = self._generate_id(semantic.fact)

        # Auto-extract keywords if not set
        if not semantic.keywords:
            semantic.keywords = self._extract_keywords(semantic.fact)

        # Auto-detect domain if generic
        if semantic.domain == "general":
            semantic.domain = extract_domain(semantic.fact)

        # Set created_at if not set
        if not semantic.created_at:
            semantic.created_at = datetime.now().isoformat()

        # Save the semantic file
        semantic_file = self.semantic_path / f"{semantic.id}.json"
        with open(semantic_file, 'w', encoding='utf-8') as f:
            json.dump(semantic.to_dict(), f, indent=2)

        # Update index
        index = self._load_index()

        entry = {
            "id": semantic.id,
            "fact_summary": semantic.fact[:200],
            "domain": semantic.domain,
            "confidence": self._safe_confidence(semantic.confidence),
            "keywords": semantic.keywords[:5]  # Top 5 for index
        }

        # Check for existing entry (update vs add)
        existing_idx = None
        for i, e in enumerate(index.get("entries", [])):
            if e["id"] == semantic.id:
                existing_idx = i
                break

        if existing_idx is not None:
            index["entries"][existing_idx] = entry
        else:
            index["entries"].append(entry)

        # Index by domain
        if semantic.domain not in index["by_domain"]:
            index["by_domain"][semantic.domain] = []
        if semantic.id not in index["by_domain"][semantic.domain]:
            index["by_domain"][semantic.domain].append(semantic.id)

        # Index by keywords
        for kw in semantic.keywords:
            if kw not in index["by_keyword"]:
                index["by_keyword"][kw] = []
            if semantic.id not in index["by_keyword"][kw]:
                index["by_keyword"][kw].append(semantic.id)

        self._save_index(index)
        return semantic.id

    def get_semantic(self, semantic_id: str) -> Optional[SemanticMemory]:
        """Retrieve a semantic memory by ID."""
        semantic_file = self.semantic_path / f"{semantic_id}.json"
        if not semantic_file.exists():
            return None

        try:
            with open(semantic_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return SemanticMemory.from_dict(data)
        except Exception as e:
            print(f"[SemanticMemory] Error loading {semantic_id}: {e}")
            return None

    def query_by_domain(self, domain: str, limit: int = 20) -> List[SemanticMemory]:
        """Get semantic memories by domain."""
        index = self._load_index()
        semantic_ids = index.get("by_domain", {}).get(domain, [])
        memories = [self.get_semantic(sid) for sid in semantic_ids[:limit] if self.get_semantic(sid)]
        # Sort by confidence descending
        memories.sort(key=lambda m: m.confidence, reverse=True)
        return memories

    def query_by_keywords(self, keywords: List[str], limit: int = 10) -> List[SemanticMemory]:
        """Get semantic memories matching keywords."""
        index = self._load_index()
        scores = {}  # id -> match count

        for kw in keywords:
            kw_lower = kw.lower()
            matching_ids = index.get("by_keyword", {}).get(kw_lower, [])
            for sid in matching_ids:
                scores[sid] = scores.get(sid, 0) + 1

        # Sort by match count
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        memories = []
        for sid in sorted_ids[:limit]:
            mem = self.get_semantic(sid)
            if mem:
                memories.append(mem)

        return memories

    def search(self, query: str, limit: int = 10) -> List[SemanticMemory]:
        """
        Search semantic memories.

        Combines keyword matching with content search.
        """
        # Extract keywords from query
        keywords = self._extract_keywords(query)

        # Get keyword matches
        keyword_results = self.query_by_keywords(keywords, limit=limit * 2)

        # Also do content search
        query_lower = query.lower()
        index = self._load_index()
        content_matches = []

        for entry in index.get("entries", []):
            if query_lower in entry.get("fact_summary", "").lower():
                mem = self.get_semantic(entry["id"])
                if mem and mem not in keyword_results:
                    content_matches.append(mem)

        # Combine and dedupe
        results = keyword_results + content_matches

        # Score by relevance (keyword matches + confidence)
        def score(mem):
            kw_score = sum(1 for kw in keywords if kw in " ".join(mem.keywords))
            return (kw_score, mem.confidence)

        results.sort(key=score, reverse=True)
        return results[:limit]

    def record_access(self, semantic_id: str):
        """Record that a semantic memory was accessed."""
        mem = self.get_semantic(semantic_id)
        if mem:
            mem.access_count += 1
            mem.last_accessed = datetime.now().isoformat()
            self.save_semantic(mem)

    def boost_confidence(self, semantic_id: str, amount: float = 0.05, reason: str = None):
        """Boost confidence of a semantic memory (confirmed correct)."""
        mem = self.get_semantic(semantic_id)
        if mem:
            mem.confidence = min(1.0, mem.confidence + amount)
            self.save_semantic(mem)

    def decay_confidence(self, semantic_id: str, amount: float = 0.05, reason: str = None):
        """Reduce confidence of a semantic memory (found incorrect/outdated)."""
        mem = self.get_semantic(semantic_id)
        if mem:
            mem.confidence = max(0.0, mem.confidence - amount)
            self.save_semantic(mem)

    def find_or_create(self, fact: str, domain: str = None) -> Tuple[SemanticMemory, bool]:
        """
        Find existing semantic memory or create new one.

        Returns:
            (SemanticMemory, created) - the memory and whether it was newly created
        """
        # Generate ID to check for existing
        potential_id = self._generate_id(fact)
        existing = self.get_semantic(potential_id)

        if existing:
            # Boost confidence - seeing it again confirms it
            self.boost_confidence(potential_id, 0.02)
            return (existing, False)

        # Create new
        mem = SemanticMemory(
            id=potential_id,
            fact=fact,
            domain=domain or extract_domain(fact),
            confidence=0.7,  # Start at moderate confidence
            created_at=datetime.now().isoformat()
        )
        self.save_semantic(mem)
        return (mem, True)

    def generalize_from_episodes(
        self,
        episode_ids: List[str],
        generalized_fact: str,
        domain: str = None
    ) -> SemanticMemory:
        """
        Create a semantic memory generalized from multiple episodes.

        This is the key abstraction process - turning specific events into general knowledge.
        """
        mem = SemanticMemory(
            id=self._generate_id(generalized_fact),
            fact=generalized_fact,
            generalized_from=episode_ids,
            domain=domain or extract_domain(generalized_fact),
            confidence=0.6 + (0.05 * min(len(episode_ids), 8)),  # More episodes = higher confidence
            created_at=datetime.now().isoformat()
        )
        self.save_semantic(mem)
        return mem

    def get_high_confidence(self, threshold: float = 0.8, limit: int = 50) -> List[SemanticMemory]:
        """Get semantic memories with high confidence."""
        index = self._load_index()
        results = []

        for entry in index.get("entries", []):
            if self._safe_confidence(entry.get("confidence", 0)) >= threshold:
                mem = self.get_semantic(entry["id"])
                if mem:
                    results.append(mem)

        results.sort(key=lambda m: m.confidence, reverse=True)
        return results[:limit]

    def get_frequently_accessed(self, limit: int = 20) -> List[SemanticMemory]:
        """Get most frequently accessed semantic memories."""
        index = self._load_index()
        memories = []

        for entry in index.get("entries", []):
            mem = self.get_semantic(entry["id"])
            if mem:
                memories.append(mem)

        memories.sort(key=lambda m: m.access_count, reverse=True)
        return memories[:limit]

    def get_stats(self) -> Dict:
        """Get statistics about semantic memory."""
        index = self._load_index()

        domain_counts = {d: len(ids) for d, ids in index.get("by_domain", {}).items()}

        return {
            "total_facts": len(index.get("entries", [])),
            "domains": domain_counts,
            "unique_keywords": len(index.get("by_keyword", {})),
            "avg_confidence": sum(self._safe_confidence(e.get("confidence", 0)) for e in index.get("entries", [])) / max(len(index.get("entries", [])), 1)
        }

    def migrate_from_facts(self, facts_path: Path = None) -> int:
        """
        Migrate existing facts from canonical_facts.json to semantic memory.

        Returns number of facts migrated.
        """
        if facts_path is None:
            facts_path = self.base_path / "canonical_facts.json"

        if not facts_path.exists():
            return 0

        try:
            with open(facts_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            return 0

        migrated = 0
        facts = data.get("facts", [])

        for fact_data in facts:
            content = fact_data.get("content", "")
            if not content:
                continue

            # Map old fact types to domains
            old_type = fact_data.get("type", "general")
            domain_map = {
                "technical_limitation": "debugging",
                "technical_capability": "development",
                "discovery": "general",
                "configuration": "configuration",
                "usage": "workflow",
                "location": "infrastructure"
            }
            domain = domain_map.get(old_type, "general")

            # Create semantic memory
            mem = SemanticMemory(
                id=self._generate_id(content),
                fact=content,
                domain=domain,
                confidence=self._safe_confidence(fact_data.get("confidence", 0.7)),
                created_at=fact_data.get("extracted_at", datetime.now().isoformat())
            )
            self.save_semantic(mem)
            migrated += 1

        return migrated
