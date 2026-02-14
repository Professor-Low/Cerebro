"""
Causal Model Builder - Claude.Me v6.0
Stores WHY things happen, enables "what if" reasoning.

Part of Phase 3: Causal Model Building
"""
import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class CausalLink:
    """
    A causal relationship between events/conditions.

    Example:
    - Cause: "NAS sleep mode enabled"
    - Effect: "First query takes 30+ seconds"
    - Mechanism: "NAS must wake before responding"
    - Counterfactual: "If NAS kept awake, response <1s"
    - Interventions: ["Disable sleep", "Add keep-alive ping"]
    """

    def __init__(
        self,
        link_id: str,
        cause: str,
        effect: str,
        mechanism: str = None,
        counterfactual: str = None,
        interventions: List[str] = None,
        confidence: float = 0.7,
        evidence_count: int = 1
    ):
        self.link_id = link_id
        self.cause = cause
        self.effect = effect
        self.mechanism = mechanism
        self.counterfactual = counterfactual
        self.interventions = interventions or []
        self.confidence = confidence
        self.evidence_count = evidence_count
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at

    def to_dict(self) -> Dict:
        return {
            "link_id": self.link_id,
            "cause": self.cause,
            "effect": self.effect,
            "mechanism": self.mechanism,
            "counterfactual": self.counterfactual,
            "interventions": self.interventions,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CausalLink":
        link = cls(
            link_id=data["link_id"],
            cause=data["cause"],
            effect=data["effect"],
            mechanism=data.get("mechanism"),
            counterfactual=data.get("counterfactual"),
            interventions=data.get("interventions", []),
            confidence=data.get("confidence", 0.7),
            evidence_count=data.get("evidence_count", 1)
        )
        link.created_at = data.get("created_at", link.created_at)
        link.updated_at = data.get("updated_at", link.updated_at)
        return link

    def reinforce(self, amount: float = 0.05):
        """Increase confidence when link is confirmed."""
        self.confidence = min(1.0, self.confidence + amount)
        self.evidence_count += 1
        self.updated_at = datetime.now().isoformat()

    def weaken(self, amount: float = 0.1):
        """Decrease confidence when link is contradicted."""
        self.confidence = max(0.1, self.confidence - amount)
        self.updated_at = datetime.now().isoformat()


class CausalModelManager:
    """
    Manage causal models - graphs of cause-effect relationships.

    The causal model enables:
    - Understanding WHY problems occur
    - Predicting outcomes of changes
    - Suggesting interventions
    - "What if" reasoning
    """

    def __init__(self, base_path: str = None):
        if base_path is None:
            try:
                from .config import get_base_path
            except ImportError:
                from config import get_base_path
            base_path = get_base_path()
        self.base_path = Path(base_path)
        self.causal_path = self.base_path / "causal"
        self.links_path = self.causal_path / "links"
        self.graph_path = self.causal_path / "graph.json"
        self._lock = threading.Lock()
        self._ensure_directories()

    def _ensure_directories(self):
        """Create directories if they don't exist."""
        self.causal_path.mkdir(parents=True, exist_ok=True)
        self.links_path.mkdir(parents=True, exist_ok=True)

    def _generate_link_id(self, cause: str, effect: str) -> str:
        """Generate deterministic ID from cause-effect pair."""
        combined = f"{cause.lower().strip()}::{effect.lower().strip()}"
        return f"cl_{hashlib.sha256(combined.encode()).hexdigest()[:10]}"

    def _load_graph(self) -> Dict:
        """Load the causal graph."""
        if self.graph_path.exists():
            try:
                with open(self.graph_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {
            "nodes": {},
            "edges": [],
            "updated_at": datetime.now().isoformat()
        }

    def _save_graph(self, graph: Dict):
        """Save the causal graph."""
        graph["updated_at"] = datetime.now().isoformat()
        with self._lock:
            with open(self.graph_path, 'w', encoding='utf-8') as f:
                json.dump(graph, f, indent=2)

    def add_link(
        self,
        cause: str,
        effect: str,
        mechanism: str = None,
        counterfactual: str = None,
        interventions: List[str] = None,
        confidence: float = 0.7
    ) -> CausalLink:
        """
        Add a causal link.

        If link already exists, reinforces it.
        """
        link_id = self._generate_link_id(cause, effect)
        link_file = self.links_path / f"{link_id}.json"

        # Check if exists
        if link_file.exists():
            existing = self.get_link(link_id)
            if existing:
                existing.reinforce()
                # Update with new info if provided
                if mechanism and not existing.mechanism:
                    existing.mechanism = mechanism
                if counterfactual and not existing.counterfactual:
                    existing.counterfactual = counterfactual
                if interventions:
                    for iv in interventions:
                        if iv not in existing.interventions:
                            existing.interventions.append(iv)
                self._save_link(existing)
                return existing

        # Create new link
        link = CausalLink(
            link_id=link_id,
            cause=cause,
            effect=effect,
            mechanism=mechanism,
            counterfactual=counterfactual,
            interventions=interventions,
            confidence=confidence
        )
        self._save_link(link)

        # Update graph
        self._update_graph(link)

        return link

    def _save_link(self, link: CausalLink):
        """Save a causal link to disk."""
        link_file = self.links_path / f"{link.link_id}.json"
        with open(link_file, 'w', encoding='utf-8') as f:
            json.dump(link.to_dict(), f, indent=2)

    def _update_graph(self, link: CausalLink):
        """Update the graph with a new/updated link."""
        graph = self._load_graph()

        # Add/update cause node
        cause_id = hashlib.sha256(link.cause.lower().encode()).hexdigest()[:8]
        graph["nodes"][cause_id] = {
            "id": cause_id,
            "type": "cause",
            "description": link.cause
        }

        # Add/update effect node
        effect_id = hashlib.sha256(link.effect.lower().encode()).hexdigest()[:8]
        graph["nodes"][effect_id] = {
            "id": effect_id,
            "type": "effect",
            "description": link.effect
        }

        # Add/update mechanism node if present
        mechanism_id = None
        if link.mechanism:
            mechanism_id = hashlib.sha256(link.mechanism.lower().encode()).hexdigest()[:8]
            graph["nodes"][mechanism_id] = {
                "id": mechanism_id,
                "type": "mechanism",
                "description": link.mechanism
            }

        # Add/update edge
        edge = {
            "from": cause_id,
            "to": effect_id,
            "via": mechanism_id,
            "link_id": link.link_id,
            "confidence": link.confidence,
            "evidence_count": link.evidence_count
        }

        # Update or add edge
        edge_updated = False
        for i, e in enumerate(graph["edges"]):
            if e["link_id"] == link.link_id:
                graph["edges"][i] = edge
                edge_updated = True
                break
        if not edge_updated:
            graph["edges"].append(edge)

        self._save_graph(graph)

    def get_link(self, link_id: str) -> Optional[CausalLink]:
        """Get a causal link by ID."""
        link_file = self.links_path / f"{link_id}.json"
        if not link_file.exists():
            return None

        try:
            with open(link_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return CausalLink.from_dict(data)
        except Exception as e:
            print(f"[CausalModel] Error loading link {link_id}: {e}")
            return None

    def find_causes(self, effect: str, threshold: float = 0.5) -> List[CausalLink]:
        """Find what causes a given effect."""
        effect_lower = effect.lower()
        results = []

        for link_file in self.links_path.glob("cl_*.json"):
            try:
                with open(link_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if effect_lower in data.get("effect", "").lower():
                    link = CausalLink.from_dict(data)
                    if link.confidence >= threshold:
                        results.append(link)
            except:
                continue

        results.sort(key=lambda l: l.confidence, reverse=True)
        return results

    def find_effects(self, cause: str, threshold: float = 0.5) -> List[CausalLink]:
        """Find what effects a given cause produces."""
        cause_lower = cause.lower()
        results = []

        for link_file in self.links_path.glob("cl_*.json"):
            try:
                with open(link_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if cause_lower in data.get("cause", "").lower():
                    link = CausalLink.from_dict(data)
                    if link.confidence >= threshold:
                        results.append(link)
            except:
                continue

        results.sort(key=lambda l: l.confidence, reverse=True)
        return results

    def get_interventions(self, effect: str) -> List[Dict]:
        """Get possible interventions for an undesired effect."""
        causes = self.find_causes(effect)
        interventions = []

        for link in causes:
            for intervention in link.interventions:
                interventions.append({
                    "intervention": intervention,
                    "addresses_cause": link.cause,
                    "mechanism": link.mechanism,
                    "confidence": link.confidence
                })

        # Sort by confidence
        interventions.sort(key=lambda i: i["confidence"], reverse=True)
        return interventions

    def simulate_what_if(self, intervention: str) -> List[Dict]:
        """
        Simulate "what if" we apply an intervention.

        Traces through the causal graph to predict outcomes.
        """
        # Find links where intervention matches cause or counterfactual
        intervention_lower = intervention.lower()
        predictions = []

        for link_file in self.links_path.glob("cl_*.json"):
            try:
                with open(link_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                link = CausalLink.from_dict(data)

                # Check if intervention matches any intervention listed
                for iv in link.interventions:
                    if intervention_lower in iv.lower():
                        predictions.append({
                            "prediction": f"Would prevent: {link.effect}",
                            "mechanism": link.mechanism,
                            "confidence": link.confidence,
                            "original_cause": link.cause
                        })

                # Check if intervention is mentioned in counterfactual
                if link.counterfactual and intervention_lower in link.counterfactual.lower():
                    predictions.append({
                        "prediction": link.counterfactual,
                        "confidence": link.confidence * 0.9,  # Slightly lower for indirect match
                        "original_effect": link.effect
                    })

            except:
                continue

        predictions.sort(key=lambda p: p["confidence"], reverse=True)
        return predictions

    def search_links(self, query: str, limit: int = 10) -> List[CausalLink]:
        """Search causal links by keyword."""
        query_lower = query.lower()
        results = []

        for link_file in self.links_path.glob("cl_*.json"):
            try:
                with open(link_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Search in cause, effect, mechanism
                searchable = f"{data.get('cause', '')} {data.get('effect', '')} {data.get('mechanism', '')}"
                if query_lower in searchable.lower():
                    results.append(CausalLink.from_dict(data))
            except:
                continue

        results.sort(key=lambda l: l.confidence, reverse=True)
        return results[:limit]

    def reinforce_link(self, link_id: str, reason: str = None) -> Optional[CausalLink]:
        """Reinforce a causal link (confirmed correct)."""
        link = self.get_link(link_id)
        if link:
            link.reinforce()
            self._save_link(link)
            self._update_graph(link)
            return link
        return None

    def weaken_link(self, link_id: str, reason: str = None) -> Optional[CausalLink]:
        """Weaken a causal link (found incorrect/outdated)."""
        link = self.get_link(link_id)
        if link:
            link.weaken()
            self._save_link(link)
            self._update_graph(link)
            return link
        return None

    def get_graph_stats(self) -> Dict:
        """Get statistics about the causal graph."""
        graph = self._load_graph()

        node_types = {}
        for node in graph.get("nodes", {}).values():
            ntype = node.get("type", "unknown")
            node_types[ntype] = node_types.get(ntype, 0) + 1

        confidences = [e.get("confidence", 0) for e in graph.get("edges", [])]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return {
            "total_nodes": len(graph.get("nodes", {})),
            "total_edges": len(graph.get("edges", [])),
            "node_types": node_types,
            "average_confidence": round(avg_confidence, 2),
            "high_confidence_links": len([c for c in confidences if c >= 0.8]),
            "low_confidence_links": len([c for c in confidences if c < 0.5])
        }

    def get_stats(self) -> Dict:
        """Get overall causal model statistics."""
        link_count = len(list(self.links_path.glob("cl_*.json")))
        graph_stats = self.get_graph_stats()

        return {
            "total_links": link_count,
            "graph": graph_stats
        }
