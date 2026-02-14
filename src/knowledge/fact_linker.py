"""
Fact Linker - Create relationships between facts.

Part of Phase 2 Enhancement in the All-Knowing Brain PRD.
Links facts based on:
- Same entity (NAS, user, etc.)
- Same project
- Related topic (semantically related)
- Causal relationships (A led to B)
"""

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Link types with their default weights
LINK_TYPES = {
    'same_entity': 0.9,      # Facts about same thing (NAS, user)
    'same_project': 0.7,     # Facts from same project
    'related_topic': 0.5,    # Semantically related
    'causal': 0.8,           # A led to B
    'temporal': 0.4,         # Same time period
    'contradicts': -0.5,     # Facts that contradict each other
}


class FactLinker:
    """
    Creates relationships between facts to build a knowledge graph.
    """

    def __init__(self, base_path: str = None):
        if base_path is None:

            from config import AI_MEMORY_BASE

            base_path = str(AI_MEMORY_BASE)

        self.base_path = Path(base_path)
        self.conversations_path = self.base_path / "conversations"
        self.links_path = self.base_path / "fact_links.json"
        self.entities_path = self.base_path / "entities"

        # Common entity patterns
        self.entity_patterns = {
            'server': r'\b(nas|server|dgx|spark|synology)\b',
            'tool': r'\b(claude|mcp|git|docker|python|node)\b',
            'path': r'(/[\w./]+|[A-Z]:\\[\w\\]+)',
            'ip': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'config': r'\b(config|settings|\.json|\.yaml|\.toml)\b',
        }

    def extract_entities(self, fact: Dict[str, Any]) -> Dict[str, Set[str]]:
        """
        Extract named entities from a fact.

        Args:
            fact: Fact dict with 'content' field

        Returns:
            Dict of entity_type -> set of entities
        """
        content = fact.get("content", "").lower()
        entities = defaultdict(set)

        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                entities[entity_type].add(match.lower())

        return dict(entities)

    def link_facts(self,
                   fact_a: Dict[str, Any],
                   fact_b: Dict[str, Any]) -> List[Tuple[str, float, Optional[str]]]:
        """
        Find all links between two facts.

        Args:
            fact_a: First fact
            fact_b: Second fact

        Returns:
            List of (link_type, weight, detail) tuples
        """
        links = []

        # Extract entities
        entities_a = self.extract_entities(fact_a)
        entities_b = self.extract_entities(fact_b)

        # Check entity overlap
        for entity_type in set(entities_a.keys()) | set(entities_b.keys()):
            shared = entities_a.get(entity_type, set()) & entities_b.get(entity_type, set())
            if shared:
                links.append((
                    'same_entity',
                    LINK_TYPES['same_entity'],
                    f"{entity_type}: {', '.join(list(shared)[:3])}"
                ))

        # Check project match
        project_a = fact_a.get("project") or self._infer_project(fact_a)
        project_b = fact_b.get("project") or self._infer_project(fact_b)
        if project_a and project_b and project_a.lower() == project_b.lower():
            links.append((
                'same_project',
                LINK_TYPES['same_project'],
                project_a
            ))

        # Check semantic similarity
        similarity = self._text_similarity(
            fact_a.get("content", ""),
            fact_b.get("content", "")
        )
        if 0.3 < similarity < 0.85:  # Not duplicate, but related
            links.append((
                'related_topic',
                LINK_TYPES['related_topic'] * similarity,
                f"similarity: {similarity:.2f}"
            ))

        # Check temporal proximity
        time_a = fact_a.get("timestamp", "")
        time_b = fact_b.get("timestamp", "")
        if time_a and time_b:
            try:
                dt_a = datetime.fromisoformat(time_a.replace("Z", "+00:00"))
                dt_b = datetime.fromisoformat(time_b.replace("Z", "+00:00"))
                hours_diff = abs((dt_a - dt_b).total_seconds()) / 3600
                if hours_diff < 24:  # Same day
                    links.append((
                        'temporal',
                        LINK_TYPES['temporal'],
                        f"{hours_diff:.1f} hours apart"
                    ))
            except:
                pass

        # Check for causal indicators
        content_a = fact_a.get("content", "").lower()
        content_b = fact_b.get("content", "").lower()
        causal_indicators = ['because', 'therefore', 'led to', 'caused', 'fixed by', 'solved by']
        for indicator in causal_indicators:
            if indicator in content_a or indicator in content_b:
                # Check if they reference each other's topic
                if similarity > 0.2:
                    links.append((
                        'causal',
                        LINK_TYPES['causal'],
                        f"indicator: {indicator}"
                    ))
                    break

        return links

    def _infer_project(self, fact: Dict[str, Any]) -> Optional[str]:
        """Infer project from fact content."""
        content = fact.get("content", "").lower()

        # Common project indicators
        project_indicators = {
            'ai-memory': ['memory', 'mcp', 'ai memory', 'ai_memory'],
            'claude-code': ['claude code', 'claude-code', 'hook'],
            'n8n': ['n8n', 'automation', 'workflow'],
        }

        for project, indicators in project_indicators.items():
            for indicator in indicators:
                if indicator in content:
                    return project

        return None

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity."""
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def build_link_graph(self,
                         facts: List[Dict[str, Any]],
                         min_weight: float = 0.3) -> Dict[str, Any]:
        """
        Build a graph of linked facts.

        Args:
            facts: List of facts to link
            min_weight: Minimum link weight to include

        Returns:
            Graph structure with nodes and edges
        """
        nodes = []
        edges = []

        # Create nodes
        for i, fact in enumerate(facts):
            fact_id = fact.get("fact_id", f"fact_{i}")
            nodes.append({
                "id": fact_id,
                "content": fact.get("content", "")[:100],
                "type": fact.get("type", "unknown"),
                "entities": self.extract_entities(fact),
            })

        # Create edges
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                links = self.link_facts(facts[i], facts[j])

                for link_type, weight, detail in links:
                    if weight >= min_weight:
                        edges.append({
                            "source": facts[i].get("fact_id", f"fact_{i}"),
                            "target": facts[j].get("fact_id", f"fact_{j}"),
                            "type": link_type,
                            "weight": round(weight, 3),
                            "detail": detail,
                        })

        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "built_at": datetime.now().isoformat(),
        }

    def save_links(self, graph: Dict[str, Any]) -> None:
        """Save link graph to file."""
        try:
            with open(self.links_path, "w", encoding="utf-8") as f:
                json.dump(graph, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving links: {e}")

    def load_links(self) -> Optional[Dict[str, Any]]:
        """Load existing link graph."""
        if not self.links_path.exists():
            return None

        try:
            with open(self.links_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return None

    def get_related_facts(self,
                          fact_id: str,
                          min_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Get facts related to a given fact.

        Args:
            fact_id: ID of the fact to find relations for
            min_weight: Minimum link weight

        Returns:
            List of related facts with link info
        """
        graph = self.load_links()
        if not graph:
            return []

        related = []
        for edge in graph.get("edges", []):
            if edge.get("weight", 0) < min_weight:
                continue

            if edge.get("source") == fact_id:
                related.append({
                    "fact_id": edge.get("target"),
                    "link_type": edge.get("type"),
                    "weight": edge.get("weight"),
                    "detail": edge.get("detail"),
                })
            elif edge.get("target") == fact_id:
                related.append({
                    "fact_id": edge.get("source"),
                    "link_type": edge.get("type"),
                    "weight": edge.get("weight"),
                    "detail": edge.get("detail"),
                })

        # Sort by weight
        related.sort(key=lambda x: x["weight"], reverse=True)

        return related

    def get_link_stats(self) -> Dict[str, Any]:
        """Get statistics about fact links."""
        graph = self.load_links()

        if not graph:
            return {
                "has_links": False,
                "node_count": 0,
                "edge_count": 0,
            }

        # Count link types
        link_type_counts = defaultdict(int)
        for edge in graph.get("edges", []):
            link_type_counts[edge.get("type", "unknown")] += 1

        return {
            "has_links": True,
            "node_count": graph.get("node_count", 0),
            "edge_count": graph.get("edge_count", 0),
            "link_types": dict(link_type_counts),
            "built_at": graph.get("built_at"),
        }


def link_all_facts(facts: List[Dict[str, Any]], save: bool = True) -> Dict[str, Any]:
    """
    Convenience function to link all facts.

    Args:
        facts: List of facts to link
        save: Whether to save the graph

    Returns:
        Link graph
    """
    linker = FactLinker()
    graph = linker.build_link_graph(facts)

    if save:
        linker.save_links(graph)

    return graph


if __name__ == "__main__":
    # Test with sample facts
    linker = FactLinker()

    # Load facts from deduplicator
    from fact_deduplicator import FactDeduplicator
    dedup = FactDeduplicator()
    facts = dedup.load_all_facts()

    print("=== Fact Linker Test ===")
    print(f"Loaded {len(facts)} facts")

    if facts:
        # Build link graph
        graph = linker.build_link_graph(facts[:20])  # Test with first 20
        print("\nGraph built:")
        print(f"  Nodes: {graph['node_count']}")
        print(f"  Edges: {graph['edge_count']}")

        if graph['edges']:
            print("\nSample links:")
            for edge in graph['edges'][:5]:
                print(f"  {edge['source']} --[{edge['type']}]--> {edge['target']}")
                print(f"    Weight: {edge['weight']}, Detail: {edge['detail']}")
