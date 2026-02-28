"""
Fact Deduplicator - Merge duplicate facts across conversations.

Part of Phase 2 Enhancement in the All-Knowing Brain PRD.
Finds and merges duplicate facts using text similarity clustering.

Merge Rules:
- Keep most recent content
- Aggregate all sources
- Boost confidence by confirmation count
- Track first_seen and last_confirmed
"""

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class FactDeduplicator:
    """
    Finds and merges duplicate facts across conversations.
    Uses text similarity to cluster similar facts.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:

            from config import AI_MEMORY_BASE

            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.conversations_path = self.base_path / "conversations"
        self.facts_index_path = self.base_path / "facts_index.json"
        self.canonical_facts_path = self.base_path / "canonical_facts.json"

    def load_all_facts(self) -> List[Dict[str, Any]]:
        """
        Load all facts from all conversations.

        Returns:
            List of facts with source conversation info
        """
        all_facts = []

        if not self.conversations_path.exists():
            return all_facts

        for conv_file in self.conversations_path.glob("*.json"):
            try:
                with open(conv_file, "r", encoding="utf-8") as f:
                    conv = json.load(f)

                conv_id = conv.get("id", conv_file.stem)
                timestamp = conv.get("timestamp", "")
                extracted = conv.get("extracted_data", {})

                # Get facts
                facts = extracted.get("facts", [])
                for i, fact in enumerate(facts):
                    if isinstance(fact, dict):
                        fact_entry = {
                            "content": fact.get("content", ""),
                            "type": fact.get("type", "unknown"),
                            "confidence": fact.get("confidence", "medium"),
                            "source_conversation": conv_id,
                            "source_file": conv_file.name,
                            "source_index": i,
                            "timestamp": fact.get("extracted_at", timestamp),
                            "fact_id": f"{conv_id}_fact_{i}",
                        }
                        if fact_entry["content"]:
                            all_facts.append(fact_entry)
                    elif isinstance(fact, str) and fact:
                        fact_entry = {
                            "content": fact,
                            "type": "unknown",
                            "confidence": "medium",
                            "source_conversation": conv_id,
                            "source_file": conv_file.name,
                            "source_index": i,
                            "timestamp": timestamp,
                            "fact_id": f"{conv_id}_fact_{i}",
                        }
                        all_facts.append(fact_entry)

            except Exception:
                continue

        return all_facts

    def cluster_by_similarity(self,
                              facts: List[Dict[str, Any]],
                              threshold: float = 0.85) -> List[List[Dict[str, Any]]]:
        """
        Group facts by text similarity.

        Args:
            facts: List of facts to cluster
            threshold: Similarity threshold (0-1)

        Returns:
            List of clusters (each cluster is a list of similar facts)
        """
        if not facts:
            return []

        # First pass: exact duplicate detection via hash
        content_to_facts = defaultdict(list)
        for fact in facts:
            content = self._normalize_content(fact.get("content", ""))
            content_hash = hashlib.md5(content.encode()).hexdigest()
            content_to_facts[content_hash].append(fact)

        # Group exact duplicates into clusters
        clusters = []
        processed_hashes = set()

        for content_hash, fact_group in content_to_facts.items():
            if content_hash in processed_hashes:
                continue
            processed_hashes.add(content_hash)

            if len(fact_group) > 1:
                # Exact duplicates
                clusters.append(fact_group)
            else:
                # Check for near-duplicates with other single facts
                fact = fact_group[0]
                cluster = [fact]

                for other_hash, other_group in content_to_facts.items():
                    if other_hash == content_hash or other_hash in processed_hashes:
                        continue
                    if len(other_group) == 1:
                        other_fact = other_group[0]
                        similarity = self._text_similarity(
                            fact.get("content", ""),
                            other_fact.get("content", "")
                        )
                        if similarity >= threshold:
                            cluster.append(other_fact)
                            processed_hashes.add(other_hash)

                clusters.append(cluster)

        return clusters

    def find_duplicates(self, threshold: float = 0.85) -> Dict[str, Any]:
        """
        Find all duplicate facts.

        Args:
            threshold: Similarity threshold

        Returns:
            Dict with duplicate info and statistics
        """
        facts = self.load_all_facts()

        if not facts:
            return {
                "total_facts": 0,
                "duplicate_clusters": 0,
                "facts_in_clusters": 0,
                "potential_savings": 0,
                "clusters": []
            }

        clusters = self.cluster_by_similarity(facts, threshold)

        # Filter to only clusters with >1 fact (actual duplicates)
        duplicate_clusters = [c for c in clusters if len(c) > 1]

        # Build preview
        cluster_previews = []
        for cluster in duplicate_clusters[:10]:  # First 10 clusters
            cluster_previews.append({
                "size": len(cluster),
                "sample_content": cluster[0].get("content", "")[:150],
                "types": list(set(f.get("type", "unknown") for f in cluster)),
                "sources": [f.get("source_conversation", "") for f in cluster],
            })

        facts_in_clusters = sum(len(c) for c in duplicate_clusters)

        return {
            "total_facts": len(facts),
            "duplicate_clusters": len(duplicate_clusters),
            "facts_in_clusters": facts_in_clusters,
            "potential_savings": facts_in_clusters - len(duplicate_clusters),
            "clusters": cluster_previews,
            "all_clusters": duplicate_clusters,  # For merge operation
        }

    def merge_facts(self, cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge a cluster of facts into a canonical fact.

        Args:
            cluster: List of similar facts

        Returns:
            Canonical fact with aggregated metadata
        """
        if not cluster:
            return {}

        if len(cluster) == 1:
            return cluster[0]

        # Sort by timestamp (most recent first) and content length (longer is better)
        sorted_cluster = sorted(
            cluster,
            key=lambda f: (
                f.get("timestamp", ""),
                len(f.get("content", ""))
            ),
            reverse=True
        )

        # Use most recent/detailed as base
        canonical = sorted_cluster[0].copy()

        # Aggregate sources
        canonical["sources"] = [
            {
                "conversation_id": f.get("source_conversation"),
                "fact_id": f.get("fact_id"),
                "timestamp": f.get("timestamp"),
            }
            for f in cluster
        ]

        # Boost confidence by confirmation count
        base_confidence = 0.5
        confirmation_boost = min(0.5, len(cluster) * 0.1)
        canonical["confidence_score"] = base_confidence + confirmation_boost

        # Track timestamps
        timestamps = [f.get("timestamp", "") for f in cluster if f.get("timestamp")]
        if timestamps:
            canonical["first_seen"] = min(timestamps)
            canonical["last_confirmed"] = max(timestamps)

        # Merge types (keep all unique types)
        canonical["types"] = list(set(f.get("type", "unknown") for f in cluster))

        # Generate canonical ID
        canonical["canonical_id"] = hashlib.md5(
            canonical.get("content", "").encode()
        ).hexdigest()[:12]

        return canonical

    def deduplicate(self,
                    threshold: float = 0.85,
                    dry_run: bool = True) -> Dict[str, Any]:
        """
        Find and optionally merge duplicate facts.

        Args:
            threshold: Similarity threshold
            dry_run: If True, only report without merging

        Returns:
            Deduplication results
        """
        duplicates = self.find_duplicates(threshold)

        if dry_run:
            return {
                "mode": "dry_run",
                "total_facts": duplicates["total_facts"],
                "duplicate_clusters_found": duplicates["duplicate_clusters"],
                "potential_savings": duplicates["potential_savings"],
                "preview": duplicates["clusters"],
            }

        # Merge duplicates
        all_clusters = duplicates.get("all_clusters", [])
        canonical_facts = []

        for cluster in all_clusters:
            canonical = self.merge_facts(cluster)
            canonical_facts.append(canonical)

        # Save canonical facts index
        self._save_canonical_facts(canonical_facts)

        return {
            "mode": "merge",
            "total_facts_processed": duplicates["total_facts"],
            "clusters_merged": len(canonical_facts),
            "facts_deduplicated": duplicates["facts_in_clusters"],
            "space_saved": duplicates["potential_savings"],
            "canonical_facts_created": len(canonical_facts),
        }

    def _save_canonical_facts(self, canonical_facts: List[Dict[str, Any]]) -> None:
        """Save canonical facts to index file."""
        try:
            index = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_facts": len(canonical_facts),
                "facts": canonical_facts,
            }

            with open(self.canonical_facts_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error saving canonical facts: {e}")

    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison."""
        if not content:
            return ""
        # Lowercase, remove extra whitespace
        content = content.lower().strip()
        content = re.sub(r'\s+', ' ', content)
        return content

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between texts."""
        if not text1 or not text2:
            return 0.0

        # Normalize
        text1 = self._normalize_content(text1)
        text2 = self._normalize_content(text2)

        # Tokenize
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        facts = self.load_all_facts()

        # Load canonical facts if they exist
        canonical_count = 0
        if self.canonical_facts_path.exists():
            try:
                with open(self.canonical_facts_path, "r") as f:
                    canonical = json.load(f)
                canonical_count = canonical.get("total_facts", 0)
            except:
                pass

        # Count fact types
        type_counts = defaultdict(int)
        for fact in facts:
            type_counts[fact.get("type", "unknown")] += 1

        return {
            "total_raw_facts": len(facts),
            "canonical_facts": canonical_count,
            "deduplication_ratio": (
                f"{(1 - canonical_count / len(facts)) * 100:.1f}%"
                if facts and canonical_count else "N/A"
            ),
            "fact_types": dict(type_counts),
            "conversations_with_facts": len(set(
                f.get("source_conversation") for f in facts
            )),
        }


def deduplicate_facts(threshold: float = 0.85, dry_run: bool = True) -> Dict[str, Any]:
    """
    Convenience function to deduplicate facts.

    Args:
        threshold: Similarity threshold
        dry_run: If True, only report

    Returns:
        Deduplication results
    """
    dedup = FactDeduplicator()
    return dedup.deduplicate(threshold, dry_run)


if __name__ == "__main__":
    dedup = FactDeduplicator()

    print("=== Fact Deduplication Stats ===")
    stats = dedup.get_stats()
    print(f"Total raw facts: {stats['total_raw_facts']}")
    print(f"Canonical facts: {stats['canonical_facts']}")
    print(f"Fact types: {stats['fact_types']}")

    print("\n=== Finding Duplicates (dry run) ===")
    result = dedup.deduplicate(threshold=0.85, dry_run=True)
    print(f"Duplicate clusters: {result['duplicate_clusters_found']}")
    print(f"Potential savings: {result['potential_savings']}")

    if result.get("preview"):
        print("\nPreview of duplicate clusters:")
        for i, cluster in enumerate(result["preview"][:5], 1):
            print(f"\n{i}. Size: {cluster['size']}")
            print(f"   Types: {cluster['types']}")
            print(f"   Content: {cluster['sample_content'][:80]}...")
